import os
import glob
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import SafetyConfig, load_config, save_config


# -----------------------------
# Utilities
# -----------------------------
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def cfg_get(cfg: SafetyConfig, name: str, default):
    return getattr(cfg, name, default)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_videos(video_dir: str) -> List[str]:
    paths = []
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext.upper()}")))
    paths = sorted(set(paths))
    return paths


def camera_intrinsics_from_fov(h: int, w: int, fov_y_deg: float) -> Tuple[float, float, float, float]:
    fov_y = np.deg2rad(float(fov_y_deg))
    fy = (h / 2.0) / np.tan(fov_y / 2.0)
    fx = fy * (w / float(h))
    cx = w / 2.0
    cy = h / 2.0
    return float(fx), float(fy), float(cx), float(cy)

# 深度推定の簡易処理（後でちゃんとしたものに置き換える）
def estimate_root_xyz_from_bbox(
    box_xyxy: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    assumed_person_height_m: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    bbox + pinhole model -> pseudo 3D root (x,y,z) in meters (approx).
    """
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    u = (x1 + x2) * 0.5
    v = (y1 + y2) * 0.5
    h_px = max((y2 - y1), 1.0)

    z = fy * float(assumed_person_height_m) / (h_px + eps)
    x = (u - cx) / (fx + eps) * z
    y = (v - cy) / (fy + eps) * z
    return np.array([x, y, z], dtype=np.float32)


def pseudo3d_pose_from_keypoints(
    kpts_xy: np.ndarray,      # [J,2] in pixel
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


def bin_index(x: float, x_min: float, x_max: float, K: int) -> int:
    if x <= x_min:
        return 0
    if x >= x_max:
        return K - 1
    t = (x - x_min) / (x_max - x_min)
    k = int(np.floor(t * K))
    return max(0, min(K - 1, k))


def compute_soft_label_from_future_xz(
    future_people_xz: List[List[Tuple[float, float]]],  # tau -> list[(x,z)]
    K: int,
    x_min: float,
    x_max: float,
    alpha_depth: float,
    gamma_time: float,
    beta_risk: float,
) -> np.ndarray:
    R = np.zeros((K,), dtype=np.float32)
    for tau, xz_list in enumerate(future_people_xz):
        w_t = (float(gamma_time) ** float(tau))
        for (x, z) in xz_list:
            w_z = float(np.exp(-float(alpha_depth) * float(z)))
            k = bin_index(float(x), float(x_min), float(x_max), int(K))
            R[k] += float(w_t * w_z)

    S = np.exp(-float(beta_risk) * R)
    q = S / (S.sum() + 1e-6)
    return q.astype(np.float32)


# Ego Motion (as-is from your mechanism)
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
                        p_next = next_pts[i:i + 1]
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


# YOLO extraction
def yolo_track_pose(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float,
    iou: float,
    tracker: str,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      boxes_xyxy: [N,4] float32
      ids: [N] int32 (if none, use 0..N-1)
      confs: [N] float32
      kpts_xy: [N,J,2] float32
      kpts_conf: [N,J] float32
    """
    kwargs = dict(
        persist=True,
        verbose=False,
        tracker=tracker,
        conf=float(conf),
        iou=float(iou),
        classes=[0],   # person
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

    kpts_xy = r0.keypoints.xy.cpu().numpy().astype(np.float32)     # [N,J,2]
    kpts_conf = r0.keypoints.conf.cpu().numpy().astype(np.float32)  # [N,J]
    return boxes, ids, confs, kpts_xy, kpts_conf


# Build training tensors
def build_npz_from_video_buffers(
    frames_state: List[Dict[int, Dict[str, np.ndarray]]],    # per frame: tid -> {"root","pose","conf"}
    frames_ego: List[Tuple[float, float]],                   # per frame: (ego_vx, ego_vy) pixel/frame
    fps: float,
    width: int,
    height: int,
    cfg: SafetyConfig,
    out_npz_path: str,
    video_path: str,
) -> None:
    """
    Produce X/M/y:
      X: [N, T, Nmax, F]
      M: [N, T, Nmax]
      y: [N, K]
    """
    eps = float(cfg_get(cfg, "eps", 1e-6))
    K = int(cfg_get(cfg, "K", 9))
    T = int(cfg_get(cfg, "T", 10))
    H = int(cfg_get(cfg, "H", 10))
    Nmax = int(cfg_get(cfg, "Nmax", 16))
    D_m = float(cfg_get(cfg, "D_m", 3.0))

    x_min = float(cfg_get(cfg, "x_min", -2.0))
    x_max = float(cfg_get(cfg, "x_max", 2.0))
    alpha_depth = float(cfg_get(cfg, "alpha_depth", 1.0))
    gamma_time = float(cfg_get(cfg, "gamma_time", 1.0))
    beta_risk = float(cfg_get(cfg, "beta_risk", 1.0))

    use_pose_delta = bool(cfg_get(cfg, "use_pose_delta", True))
    use_ego_motion = bool(cfg_get(cfg, "use_ego_motion", True))
    ego_as_feature = bool(cfg_get(cfg, "ego_as_feature", True))
    ego_normalize = bool(cfg_get(cfg, "ego_normalize", True))

    dt = 1.0 / float(fps if fps > 0 else 30.0)

    J3 = None
    J = None
    for st in frames_state:
        if st:
            any_tid = next(iter(st.keys()))
            J3 = int(st[any_tid]["pose"].shape[0])
            J = int(st[any_tid]["conf"].shape[0])
            break
    if J3 is None or J is None:
        raise RuntimeError(f"No people detected in entire video: {video_path}")

    base_dim = 3 + 3 + 3 + 1 + 1 + 1  # root, vel, dir, speed, dist, ttc
    ego_dim = 2 if (use_ego_motion and ego_as_feature) else 0
    dpose_dim = J3 if use_pose_delta else 0
    F = base_dim + ego_dim + J3 + dpose_dim + J

    frames_xz: List[List[Tuple[float, float]]] = []
    for st in frames_state:
        xz_list = []
        for tid, d in st.items():
            root = d["root"]
            x = float(root[0])
            z = float(root[2])
            dist = float(np.linalg.norm(root))
            if dist <= D_m:
                xz_list.append((x, z))
        frames_xz.append(xz_list)

    X_list, M_list, y_list = [], [], []
    num_frames = len(frames_state)

    if num_frames < (T + H):
        raise RuntimeError(f"Video too short for T/H. frames={num_frames} T={T} H={H} video={video_path}")

    for t in range(T - 1, num_frames - H):
        # teacher from future H frames
        future_people = [frames_xz[t + tau + 1] for tau in range(H)]
        y = compute_soft_label_from_future_xz(
            future_people, K, x_min, x_max, alpha_depth, gamma_time, beta_risk
        )

        cur = frames_state[t]
        cand = []
        for tid, d in cur.items():
            dist = float(np.linalg.norm(d["root"]))
            if dist <= D_m:
                cand.append((tid, dist))
        cand.sort(key=lambda x: x[1])
        ids = [tid for tid, _ in cand[:Nmax]]

        X = np.zeros((T, Nmax, F), dtype=np.float32)
        M = np.zeros((T, Nmax), dtype=np.bool_)

        for ti in range(T):
            ft = t - (T - 1 - ti)  # absolute frame idx
            st = frames_state[ft]
            st_prev = frames_state[ft - 1] if ft - 1 >= 0 else {}

            ego_vx, ego_vy = (0.0, 0.0)
            if use_ego_motion and ft < len(frames_ego):
                ego_vx, ego_vy = frames_ego[ft]

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
                if feat_vec.shape[0] != F:
                    raise RuntimeError(f"Feature dim mismatch: got={feat_vec.shape[0]} expected={F}")

                X[ti, n, :] = feat_vec
                M[ti, n] = True

        X_list.append(X)
        M_list.append(M)
        y_list.append(y)

    X_arr = np.stack(X_list, axis=0)
    M_arr = np.stack(M_list, axis=0)
    y_arr = np.stack(y_list, axis=0)

    ensure_dir(os.path.dirname(out_npz_path))
    np.savez_compressed(
        out_npz_path,
        X=X_arr,
        M=M_arr,
        y=y_arr,
        video=np.array([video_path], dtype=object),
        fps=np.array([float(fps)], dtype=np.float32),
        cfg=np.array([asdict(cfg)], dtype=object),
    )

    print(f"[preprocess] saved npz: {out_npz_path}")
    print(f"[preprocess] X={X_arr.shape} M={M_arr.shape} y={y_arr.shape}")


def preprocess_one_video(
    video_path: str,
    out_dir: str,
    model: YOLO,
    cfg: SafetyConfig,
    skip_existing: bool = False,
) -> Optional[str]:
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_npz = os.path.join(out_dir, f"{base}.npz")

    if skip_existing and os.path.exists(out_npz):
        print(f"[preprocess] skip existing: {out_npz}")
        return out_npz

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[preprocess] cannot open: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if width <= 0 or height <= 0:
        cap.release()
        print(f"[preprocess] invalid video size: {video_path}")
        return None

    fov_y_deg = float(cfg_get(cfg, "fov_y_deg", 60.0))
    assumed_h = float(cfg_get(cfg, "assumed_person_height_m", 1.7))
    kp_conf_thresh = float(cfg_get(cfg, "kp_conf_thresh", 0.3))

    yolo_conf = float(cfg_get(cfg, "yolo_conf", 0.25))
    yolo_iou = float(cfg_get(cfg, "yolo_iou", 0.5))
    yolo_tracker = str(cfg_get(cfg, "yolo_tracker", "bytetrack.yaml"))
    yolo_device = str(cfg_get(cfg, "yolo_device", ""))

    use_ego_motion = bool(cfg_get(cfg, "use_ego_motion", True))

    fx, fy, cx, cy = camera_intrinsics_from_fov(height, width, fov_y_deg)

    ego_tracker = EgoMotionTracker() if use_ego_motion else None
    frames_ego: List[Tuple[float, float]] = []
    frames_state: List[Dict[int, Dict[str, np.ndarray]]] = []

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, ids, confs, kpts_xy, kpts_conf = yolo_track_pose(
                model=model,
                frame_bgr=frame,
                conf=yolo_conf,
                iou=yolo_iou,
                tracker=yolo_tracker,
                device=yolo_device,
            )

            # ego motion from background flow excluding person boxes
            ego_vx, ego_vy = 0.0, 0.0
            if ego_tracker is not None and boxes.shape[0] > 0:
                ego_vx, ego_vy = ego_tracker.update(frame, boxes.tolist())
            elif ego_tracker is not None:
                ego_vx, ego_vy = ego_tracker.update(frame, [])

            frames_ego.append((ego_vx, ego_vy))

            st: Dict[int, Dict[str, np.ndarray]] = {}
            for i in range(boxes.shape[0]):
                tid = int(ids[i])
                root = estimate_root_xyz_from_bbox(
                    boxes[i], fx, fy, cx, cy, assumed_h, eps=float(cfg_get(cfg, "eps", 1e-6))
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
            frame_idx += 1

    finally:
        cap.release()

    try:
        build_npz_from_video_buffers(
            frames_state=frames_state,
            frames_ego=frames_ego,
            fps=fps,
            width=width,
            height=height,
            cfg=cfg,
            out_npz_path=out_npz,
            video_path=video_path,
        )
        return out_npz
    except Exception as e:
        print(f"[preprocess] failed building npz for {video_path}: {e}")
        return None


def write_manifest(out_dir: str, npz_paths: List[str]) -> str:
    manifest = os.path.join(out_dir, "manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for p in npz_paths:
            f.write(p + "\n")
    print(f"[preprocess] wrote manifest: {manifest}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="directory containing videos")
    ap.add_argument("--out-dir", required=True, help="output directory for npz files")
    ap.add_argument("--config", default="", help="config yaml path")
    ap.add_argument("--save-config", default="", help="write default config yaml and exit")
    ap.add_argument("--skip-existing", action="store_true", help="skip if npz already exists")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else SafetyConfig()
    if args.save_config:
        ensure_dir(os.path.dirname(args.save_config) or ".")
        save_config(args.save_config, cfg)
        print(f"[preprocess] wrote config: {args.save_config}")
        return

    ensure_dir(args.out_dir)

    # YOLO pose model
    pose_model_path = str(cfg_get(cfg, "yolo_pose_model", "yolov8n-pose.pt"))
    print(f"[preprocess] loading YOLO pose model: {pose_model_path}")
    model = YOLO(pose_model_path)

    vids = list_videos(args.video_dir)
    if not vids:
        raise SystemExit(f"No videos found in: {args.video_dir}")

    npz_paths: List[str] = []
    for vp in vids:
        print(f"[preprocess] processing: {vp}")
        out_npz = preprocess_one_video(
            video_path=vp,
            out_dir=args.out_dir,
            model=model,
            cfg=cfg,
            skip_existing=args.skip_existing,
        )
        if out_npz:
            npz_paths.append(out_npz)

    if not npz_paths:
        raise SystemExit("No npz outputs were generated.")

    write_manifest(args.out_dir, npz_paths)
    print("[preprocess] done.")


if __name__ == "__main__":
    main()
