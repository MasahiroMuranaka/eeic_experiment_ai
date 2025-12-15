import os
import argparse
from collections import deque
from typing import Dict, Tuple, List, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import SafetyConfig, load_config
from model import SafetyNet


# -----------------------------
# Utilities
# -----------------------------
def cfg_get(cfg: SafetyConfig, name: str, default):
    return getattr(cfg, name, default)


def camera_intrinsics_from_fov(h: int, w: int, fov_y_deg: float) -> Tuple[float, float, float, float]:
    fov_y = np.deg2rad(float(fov_y_deg))
    fy = (h / 2.0) / np.tan(fov_y / 2.0)
    fx = fy * (w / float(h))
    cx = w / 2.0
    cy = h / 2.0
    return float(fx), float(fy), float(cx), float(cy)


def estimate_root_xyz_from_bbox(
    box_xyxy: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    assumed_person_height_m: float,
    eps: float = 1e-6,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    u = (x1 + x2) * 0.5
    v = (y1 + y2) * 0.5
    h_px = max((y2 - y1), 1.0)

    z = fy * float(assumed_person_height_m) / (h_px + eps)
    x = (u - cx) / (fx + eps) * z
    y = (v - cy) / (fy + eps) * z
    return np.array([x, y, z], dtype=np.float32)


def pseudo3d_pose_from_keypoints(
    kpts_xy: np.ndarray,      # [J,2]
    kpts_conf: np.ndarray,    # [J]
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z: float,
    kp_conf_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    J = int(kpts_xy.shape[0])
    pose3d = np.zeros((J, 3), dtype=np.float32)
    conf = kpts_conf.astype(np.float32).copy()

    for j in range(J):
        if float(conf[j]) < float(kp_conf_thresh):
            continue
        uj = float(kpts_xy[j, 0])
        vj = float(kpts_xy[j, 1])
        pose3d[j, 2] = float(z)
        pose3d[j, 0] = (uj - cx) / float(fx) * float(z)
        pose3d[j, 1] = (vj - cy) / float(fy) * float(z)

    return pose3d.reshape(-1), conf


# -----------------------------
# Ego Motion (same as preprocess)
# -----------------------------
class EgoMotionTracker:
    def __init__(self):
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.prev_gray = None
        self.prev_pts = None

    def update(self, frame, exclude_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ego_vx, ego_vy = 0.0, 0.0

        if self.prev_gray is not None:
            if self.prev_pts is None or len(self.prev_pts) < 50:
                self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)

            if self.prev_pts is not None and len(self.prev_pts) > 0:
                good_prev_pts = []
                for pt in self.prev_pts:
                    x, y = pt.ravel()
                    is_in_person = False
                    for box in exclude_boxes:
                        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                            is_in_person = True
                            break
                    if not is_in_person:
                        good_prev_pts.append(pt)

                if len(good_prev_pts) == 0:
                    self.prev_pts = None
                else:
                    good_prev_pts = np.array(good_prev_pts, dtype=np.float32)
                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, good_prev_pts, None, **self.lk_params
                    )

                    fb_thresh = 1.5
                    good_prev = []
                    good_next = []
                    for i in range(len(good_prev_pts)):
                        if status is None or status[i][0] == 0:
                            continue
                        p_next = next_pts[i : i + 1]
                        p_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
                            gray, self.prev_gray, p_next, None, **self.lk_params
                        )
                        if st_back is None or st_back[0][0] == 0:
                            continue
                        err = np.linalg.norm(good_prev_pts[i] - p_back[0])
                        if err <= fb_thresh:
                            good_prev.append(good_prev_pts[i])
                            good_next.append(next_pts[i])

                    if len(good_prev) == 0:
                        self.prev_pts = None
                    else:
                        prev_arr = np.array(good_prev, dtype=np.float32).reshape(-1, 1, 2)
                        next_arr = np.array(good_next, dtype=np.float32).reshape(-1, 1, 2)

                        pts_prev = prev_arr.reshape(-1, 2)
                        pts_next = next_arr.reshape(-1, 2)

                        if len(pts_prev) >= 3:
                            M, inliers = cv2.estimateAffinePartial2D(
                                pts_prev, pts_next, method=cv2.RANSAC, ransacReprojThreshold=3.0
                            )
                            if M is not None:
                                ego_vx, ego_vy = float(M[0, 2]), float(M[1, 2])
                                if inliers is not None:
                                    in_mask = inliers.ravel() == 1
                                    if np.sum(in_mask) > 0:
                                        self.prev_pts = pts_next[in_mask].reshape(-1, 1, 2)
                                    else:
                                        self.prev_pts = pts_next.reshape(-1, 1, 2)
                                else:
                                    self.prev_pts = pts_next.reshape(-1, 1, 2)
                            else:
                                motion = pts_next - pts_prev
                                med = np.median(motion, axis=0)
                                ego_vx, ego_vy = float(med[0]), float(med[1])
                                self.prev_pts = pts_next.reshape(-1, 1, 2)
                        else:
                            motion = pts_next - pts_prev
                            med = np.median(motion, axis=0)
                            ego_vx, ego_vy = float(med[0]), float(med[1])
                            self.prev_pts = pts_next.reshape(-1, 1, 2)

        self.prev_gray = gray.copy()
        return float(ego_vx), float(ego_vy)


# -----------------------------
# YOLO tracking + pose
# -----------------------------
def yolo_track_pose(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float,
    iou: float,
    tracker: str,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kwargs = dict(
        persist=True,
        verbose=False,
        tracker=tracker,
        conf=float(conf),
        iou=float(iou),
        classes=[0],
    )
    if device:
        kwargs["device"] = device

    res = model.track(frame_bgr, **kwargs)
    r0 = res[0]

    if r0.boxes is None or len(r0.boxes) == 0 or r0.keypoints is None:
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.int32),
            np.zeros((0,), np.float32),
            np.zeros((0, 17, 2), np.float32),
            np.zeros((0, 17), np.float32),
        )

    boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = r0.boxes.conf.cpu().numpy().astype(np.float32)

    if r0.boxes.id is None:
        ids = np.arange(len(boxes), dtype=np.int32)
    else:
        ids = r0.boxes.id.int().cpu().numpy().astype(np.int32)

    kpts_xy = r0.keypoints.xy.cpu().numpy().astype(np.float32)
    kpts_conf = r0.keypoints.conf.cpu().numpy().astype(np.float32)
    return boxes, ids, confs, kpts_xy, kpts_conf


# -----------------------------
# Feature assembly (must match preprocess)
# -----------------------------
def build_feature_tensor(
    frames_state: List[Dict[int, Dict[str, np.ndarray]]],  # length T, each: tid->root/pose/conf
    frames_ego: List[Tuple[float, float]],                 # length T, each: ego_vx/vy pixel/frame
    fps: float,
    width: int,
    height: int,
    cfg: SafetyConfig,
    in_dim_expected: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      X: [1, T, Nmax, F]
      M: [1, T, Nmax]
    """
    eps = float(cfg_get(cfg, "eps", 1e-6))
    T = int(cfg_get(cfg, "T", 10))
    Nmax = int(cfg_get(cfg, "Nmax", 16))
    D_m = float(cfg_get(cfg, "D_m", 3.0))

    use_pose_delta = bool(cfg_get(cfg, "use_pose_delta", True))
    use_ego_motion = bool(cfg_get(cfg, "use_ego_motion", True))
    ego_as_feature = bool(cfg_get(cfg, "ego_as_feature", True))
    ego_normalize = bool(cfg_get(cfg, "ego_normalize", True))

    dt = 1.0 / float(fps if fps > 0 else 30.0)

    # Determine J3/J from current data (fallback to COCO17)
    J3 = None
    J = None
    for st in reversed(frames_state):
        if st:
            any_tid = next(iter(st.keys()))
            J3 = int(st[any_tid]["pose"].shape[0])
            J = int(st[any_tid]["conf"].shape[0])
            break
    if J3 is None or J is None:
        J = 17
        J3 = 17 * 3

    base_dim = 3 + 3 + 3 + 1 + 1 + 1
    ego_dim = 2 if (use_ego_motion and ego_as_feature) else 0
    dpose_dim = J3 if use_pose_delta else 0
    F = base_dim + ego_dim + J3 + dpose_dim + J

    if F != int(in_dim_expected):
        raise RuntimeError(
            f"Feature dim mismatch: built F={F} but ckpt expects in_dim={in_dim_expected}. "
            "You must run preprocess with the same config (use_ego_motion/ego_as_feature/use_pose_delta etc.)."
        )

    # Choose ids based on current frame distance within D_m
    cur = frames_state[-1]
    cand = []
    for tid, d in cur.items():
        dist = float(np.linalg.norm(d["root"]))
        if dist <= D_m:
            cand.append((tid, dist))
    cand.sort(key=lambda x: x[1])
    ids = [tid for tid, _ in cand[:Nmax]]

    X = np.zeros((1, T, Nmax, F), dtype=np.float32)
    M = np.zeros((1, T, Nmax), dtype=np.bool_)

    for ti in range(T):
        st = frames_state[ti]
        st_prev = frames_state[ti - 1] if ti - 1 >= 0 else {}

        ego_vx, ego_vy = (0.0, 0.0)
        if use_ego_motion and ti < len(frames_ego):
            ego_vx, ego_vy = frames_ego[ti]

        if ego_normalize:
            ego_vx = (float(ego_vx) / (float(width) + eps)) * float(fps)
            ego_vy = (float(ego_vy) / (float(height) + eps)) * float(fps)

        ego_feat = (
            np.array([ego_vx, ego_vy], dtype=np.float32)
            if (use_ego_motion and ego_as_feature)
            else np.zeros((0,), dtype=np.float32)
        )

        for n, tid in enumerate(ids):
            if tid not in st:
                continue

            root = st[tid]["root"].astype(np.float32)
            pose = st[tid]["pose"].astype(np.float32)
            conf = st[tid]["conf"].astype(np.float32)

            if tid in st_prev:
                root_prev = st_prev[tid]["root"].astype(np.float32)
                v3 = (root - root_prev) / float(dt)
                if use_pose_delta:
                    pose_prev = st_prev[tid]["pose"].astype(np.float32)
                    dpose = (pose - pose_prev).astype(np.float32)
                else:
                    dpose = np.zeros((0,), dtype=np.float32)
            else:
                v3 = np.zeros((3,), dtype=np.float32)
                dpose = np.zeros((J3,), dtype=np.float32) if use_pose_delta else np.zeros((0,), dtype=np.float32)

            speed = float(np.linalg.norm(v3))
            if speed > 1e-6:
                dvec = (v3 / (speed + 1e-6)).astype(np.float32)
            else:
                dvec = np.zeros((3,), dtype=np.float32)

            dist = float(np.linalg.norm(root))
            denom = float(speed * speed + 1e-6)
            ttc = float(-np.dot(root, v3) / denom)
            if ttc < 0:
                ttc = 999.0

            feat_vec = np.concatenate(
                [
                    root,
                    v3.astype(np.float32),
                    dvec,
                    np.array([speed], dtype=np.float32),
                    np.array([dist], dtype=np.float32),
                    np.array([ttc], dtype=np.float32),
                    ego_feat,
                    pose,
                    dpose,
                    conf,
                ],
                axis=0,
            )
            X[0, ti, n, :] = feat_vec
            M[0, ti, n] = True

    return X, M


# -----------------------------
# Visualization helpers
# -----------------------------
def draw_prob_bar(frame: np.ndarray, p: np.ndarray, x0=20, y0=40, w=320, h=12) -> np.ndarray:
    """
    Draw horizontal bar segments representing probability p over K bins.
    """
    K = int(p.shape[0])
    # outline
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    # fill per bin
    acc = 0
    for k in range(K):
        bw = int(round(w * float(p[k])))
        if bw <= 0:
            continue
        cv2.rectangle(frame, (x0 + acc, y0), (x0 + acc + bw, y0 + h), (0, 255, 0), -1)
        acc += bw
        if acc >= w:
            break
    return frame


# -----------------------------
# Inference main
# -----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--ckpt", required=True, help="trained checkpoint")
    ap.add_argument("--config", default="", help="config yaml path (use the same as preprocess/train)")
    ap.add_argument("--out-video", default="", help="optional output mp4 path with overlay")
    ap.add_argument("--out-csv", default="", help="optional output csv path with probabilities")
    ap.add_argument("--show", action="store_true", help="show window (may not work on headless)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=all)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else SafetyConfig()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    in_dim = int(ckpt["in_dim"])
    K = int(ckpt["K"])

    device_str = str(cfg_get(cfg, "device", "cpu"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model = SafetyNet(in_dim=in_dim, K=K).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # YOLO pose model
    pose_model_path = str(cfg_get(cfg, "yolo_pose_model", "yolov8n-pose.pt"))
    yolo_conf = float(cfg_get(cfg, "yolo_conf", 0.25))
    yolo_iou = float(cfg_get(cfg, "yolo_iou", 0.5))
    yolo_tracker = str(cfg_get(cfg, "yolo_tracker", "bytetrack.yaml"))
    yolo_device = str(cfg_get(cfg, "yolo_device", ""))

    pose_model = YOLO(pose_model_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    fov_y_deg = float(cfg_get(cfg, "fov_y_deg", 60.0))
    assumed_h = float(cfg_get(cfg, "assumed_person_height_m", 1.7))
    kp_conf_thresh = float(cfg_get(cfg, "kp_conf_thresh", 0.3))
    eps = float(cfg_get(cfg, "eps", 1e-6))

    fx, fy, cx, cy = camera_intrinsics_from_fov(height, width, fov_y_deg)

    T = int(cfg_get(cfg, "T", 10))
    use_ego_motion = bool(cfg_get(cfg, "use_ego_motion", True))
    ego_tracker = EgoMotionTracker() if use_ego_motion else None

    # rolling buffers
    frames_state: deque = deque(maxlen=T)   # each: tid->root/pose/conf
    frames_ego: deque = deque(maxlen=T)     # each: (ego_vx,ego_vy)

    # output video
    writer = None
    if args.out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))

    # output csv
    csv_f = None
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        csv_f = open(args.out_csv, "w", encoding="utf-8")
        header = "frame," + ",".join([f"p{k}" for k in range(K)]) + "\n"
        csv_f.write(header)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break

            boxes, ids, confs, kpts_xy, kpts_conf = yolo_track_pose(
                model=pose_model,
                frame_bgr=frame,
                conf=yolo_conf,
                iou=yolo_iou,
                tracker=yolo_tracker,
                device=yolo_device,
            )

            ego_vx, ego_vy = 0.0, 0.0
            if ego_tracker is not None:
                exclude = boxes.tolist() if boxes.shape[0] > 0 else []
                ego_vx, ego_vy = ego_tracker.update(frame, exclude)
            frames_ego.append((ego_vx, ego_vy))

            st: Dict[int, Dict[str, np.ndarray]] = {}
            for i in range(boxes.shape[0]):
                tid = int(ids[i])
                root = estimate_root_xyz_from_bbox(
                    boxes[i], fx, fy, cx, cy, assumed_h, eps=eps
                )
                pose_flat, conf_j = pseudo3d_pose_from_keypoints(
                    kpts_xy[i],
                    kpts_conf[i],
                    fx, fy, cx, cy,
                    z=float(root[2]),
                    kp_conf_thresh=kp_conf_thresh,
                )
                st[tid] = {"root": root, "pose": pose_flat, "conf": conf_j}
            frames_state.append(st)

            # inference only after we have T frames
            p = None
            if len(frames_state) == T and len(frames_ego) == T:
                X_np, M_np = build_feature_tensor(
                    frames_state=list(frames_state),
                    frames_ego=list(frames_ego),
                    fps=fps,
                    width=width,
                    height=height,
                    cfg=cfg,
                    in_dim_expected=in_dim,
                )
                X = torch.from_numpy(X_np).to(device)
                M = torch.from_numpy(M_np).to(device)
                p_t, _ = model(X, M)   # [1,K]
                p = p_t[0].detach().cpu().numpy()

            # overlay
            out_frame = frame
            if p is not None:
                k_best = int(np.argmax(p))
                cv2.putText(out_frame, f"best_bin={k_best}", (20, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                out_frame = draw_prob_bar(out_frame, p, x0=20, y0=40, w=360, h=14)

                if csv_f is not None:
                    csv_f.write(str(frame_idx) + "," + ",".join([f"{float(v):.6f}" for v in p]) + "\n")

            if writer is not None:
                writer.write(out_frame)

            if args.show:
                disp = out_frame
                if width > 1280:
                    disp = cv2.resize(out_frame, (1280, int(1280 * height / width)))
                cv2.imshow("infer", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if csv_f is not None:
            csv_f.close()
        if args.show:
            cv2.destroyAllWindows()

    print("[infer] done.")
    if args.out_video:
        print(f"[infer] wrote video: {args.out_video}")
    if args.out_csv:
        print(f"[infer] wrote csv: {args.out_csv}")


if __name__ == "__main__":
    main()
