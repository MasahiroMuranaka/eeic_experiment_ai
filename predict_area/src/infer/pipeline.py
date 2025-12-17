# src/infer/pipeline.py

from __future__ import annotations

import os
import argparse
from collections import deque
from typing import Dict, Optional, Tuple

import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO  # type: ignore[import-not-found]

from ..config import SafetyConfig, load_config
from ..preprocess.io_utils import cfg_get
from ..preprocess.camera import camera_intrinsics_from_fov, pseudo3d_pose_from_keypoints, estimate_root_xyz_from_bbox
from ..preprocess.ego import EgoMotionTracker
from ..preprocess.yolo_pose import yolo_track_pose
from ..preprocess.depth_anything_v2 import DepthAnythingV2DepthEstimator
from ..preprocess.frame_source import iter_frames_from_dir, read_first_frame

from .features import build_feature_tensor
from ..model import SafetyNet
from .model_io import load_safetynet


def draw_prob_bar(frame: np.ndarray, p: np.ndarray, x0=20, y0=40, w=320, h=12) -> np.ndarray:
    K = int(p.shape[0])
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
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


@torch.no_grad()
def run_inference(
    video_path: str,
    ckpt_in_dim: int,
    K: int,
    cfg: SafetyConfig,
    device: torch.device,
    safety_model: SafetyNet,
    out_video: str = "",
    out_json: str = "",
    show: bool = False,
    max_frames: int = 0,
    frames_dir: str = "",
    fps_override: float = 0.0,
) -> None:
    """
    推論パイプライン（CLI から呼ぶための “分割版” のエントリポイント）。

    Parameters
    - video_path: 入力動画パス
    - ckpt_in_dim: checkpoint が期待する特徴次元 F
    - K: 出力bin数
    - cfg: SafetyConfig（preprocess/train と同一設定であること）
    - device: 推論に使うdevice
    - safety_model: `SafetyNet`（すでに重みロード済み）
    - out_video/out_csv/show/max_frames: 出力制御
    """
    model = safety_model
    model.eval()

    # YOLO pose
    pose_model_path = str(cfg_get(cfg, "yolo_pose_model", "yolov8n-pose.pt"))
    yolo_conf = float(cfg_get(cfg, "yolo_conf", 0.25))
    yolo_iou = float(cfg_get(cfg, "yolo_iou", 0.5))
    yolo_tracker = str(cfg_get(cfg, "yolo_tracker", "bytetrack.yaml"))
    yolo_device = str(cfg_get(cfg, "yolo_device", ""))

    pose_model = YOLO(pose_model_path)

    if bool(video_path) == bool(frames_dir):
        raise ValueError("Specify exactly one of video_path or frames_dir")

    cap = None
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or float(cfg_get(cfg, "fps", 30.0)) or 30.0)
        frame_iter = None
    else:
        first_name, first_frame = read_first_frame(frames_dir)
        height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
        fps = float(fps_override) if fps_override and fps_override > 0 else float(cfg_get(cfg, "fps", 30.0)) or 30.0
        frame_iter = iter_frames_from_dir(frames_dir, max_frames=max_frames)

    # camera + pose params
    fov_y_deg = float(cfg_get(cfg, "fov_y_deg", 60.0))
    assumed_h = float(cfg_get(cfg, "assumed_person_height_m", 1.7))
    kp_conf_thresh = float(cfg_get(cfg, "kp_conf_thresh", 0.3))
    eps = float(cfg_get(cfg, "eps", 1e-6))

    fx, fy, cx, cy = camera_intrinsics_from_fov(height, width, fov_y_deg)

    T = int(cfg_get(cfg, "T", 10))

    # ego
    use_ego_motion = bool(cfg_get(cfg, "use_ego_motion", True))
    ego_tracker = EgoMotionTracker() if use_ego_motion else None

    # depth
    depth_mode = str(cfg_get(cfg, "depth_mode", "bbox")).lower()
    use_depth_anything = depth_mode in (
        "midas",
        "depth_anything",
        "dav2",
        "metric",
        "metric_depth",
        "depth_anything_metric",
    )
    depth_est = DepthAnythingV2DepthEstimator(cfg) if use_depth_anything else None

    frames_state: deque = deque(maxlen=T)
    frames_ego: deque = deque(maxlen=T)

    writer = None
    if out_video:
        os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    json_f = None
    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        json_f = open(out_json, "w", encoding="utf-8")
        ## csv_f.write("frame," + ",".join([f"p{k}" for k in range(K)]) + "\n")

    frame_idx = 0
    infer_result = {}
    try:
        while True:
            frame_name = f"{frame_idx}.json"
            if frame_iter is None:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_idx >= max_frames:
                    break
            else:
                try:
                    _, _, frame = next(frame_iter)  # type: ignore[assignment]
                except StopIteration:
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
            frames_ego.append((float(ego_vx), float(ego_vy)))

            depth_map = None
            if depth_est is not None:
                try:
                    depth_map = depth_est.infer_and_calibrate(
                        frame_bgr=frame,
                        boxes_xyxy=boxes,
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        assumed_person_height_m=assumed_h,
                    )
                except Exception as e:
                    print(f"[infer] depth_anything failed at frame={frame_idx}: {e}")
                    depth_map = None

            st: Dict[int, Dict[str, np.ndarray]] = {}
            for i in range(boxes.shape[0]):
                tid = int(ids[i])

                if depth_est is None:
                    root = estimate_root_xyz_from_bbox(
                        boxes[i], fx, fy, cx, cy, assumed_h, eps=eps
                    )
                else:
                    root = depth_est.root_xyz_from_bbox(
                        box_xyxy=boxes[i],
                        depth_map=depth_map,
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        assumed_person_height_m=assumed_h,
                    )

                pose_flat, conf_j = pseudo3d_pose_from_keypoints(
                    kpts_xy[i],
                    kpts_conf[i],
                    fx, fy, cx, cy,
                    z=float(root[2]),
                    kp_conf_thresh=kp_conf_thresh,
                )
                st[tid] = {
                    "root": root.astype(np.float32),
                    "pose": pose_flat.astype(np.float32),
                    "conf": conf_j.astype(np.float32),
                }

            frames_state.append(st)

            p = None
            if len(frames_state) == T and len(frames_ego) == T:
                X_np, M_np = build_feature_tensor(
                    frames_state=list(frames_state),
                    frames_ego=list(frames_ego),
                    fps=fps,
                    width=width,
                    height=height,
                    cfg=cfg,
                    in_dim_expected=int(ckpt_in_dim),
                )
                X = torch.from_numpy(X_np).to(device)
                M = torch.from_numpy(M_np).to(device)
                p_t = model.predict_proba(X, M)  # [1,K]
                p = p_t[0].detach().cpu().numpy()

            out_frame = frame
            if p is not None:
                k_best = int(np.argmax(p))
                cv2.putText(
                    out_frame, f"best_bin={k_best}", (20, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                out_frame = draw_prob_bar(out_frame, p, x0=20, y0=40, w=360, h=14)
                if json_f is not None:
                    ## csv_f.write(str(frame_idx) + "," + ",".join([f"{float(v):.6f}" for v in p]) + "\n")
                    infer_result[frame_name] = [float(v) for v in p]

            if writer is not None:
                writer.write(out_frame)

            if show:
                disp = out_frame
                if width > 1280:
                    disp = cv2.resize(out_frame, (1280, int(1280 * height / width)))
                cv2.imshow("infer", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if json_f is not None:
            json.dump(infer_result, json_f, indent=3)
            json_f.close()
        if show:
            cv2.destroyAllWindows()

    print("[infer] done.")
    if out_video:
        print(f"[infer] wrote video: {out_video}")
    if out_json:
        print(f"[infer] wrote json: {out_json}")


@torch.no_grad()
def run_infer(
    video_path: str,
    ckpt_path: str,
    cfg: SafetyConfig,
    out_video: str = "",
    out_csv: str = "",
    show: bool = False,
    max_frames: int = 0,
) -> None:
    model, in_dim, K, device = load_safetynet(ckpt_path, cfg)

    # core impl
    run_inference(
        video_path=video_path,
        frames_dir="",
        ckpt_in_dim=in_dim,
        K=K,
        cfg=cfg,
        device=device,
        safety_model=model,
        out_video=out_video,
        out_csv=out_csv,
        show=show,
        max_frames=max_frames,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--out-video", default="")
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else SafetyConfig()
    run_infer(
        video_path=args.video,
        ckpt_path=args.ckpt,
        cfg=cfg,
        out_video=args.out_video,
        out_csv=args.out_csv,
        show=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
