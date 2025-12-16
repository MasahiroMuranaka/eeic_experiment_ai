# src/preprocess/pipeline.py

from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from config import SafetyConfig
from preprocess.io_utils import cfg_get, ensure_dir
from preprocess.camera import camera_intrinsics_from_fov, pseudo3d_pose_from_keypoints
from preprocess.ego import EgoMotionTracker
from preprocess.yolo_pose import yolo_track_pose
from preprocess.build_npz import build_npz_from_video_buffers
from preprocess.camera import estimate_root_xyz_from_bbox  # fallback
from preprocess.depth_anything_v2 import DepthAnythingV2DepthEstimator


VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def list_videos(video_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def preprocess_one_video(
    video_path: str,
    out_dir: str,
    yolo_pose_model: YOLO,
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
    fps = float(cap.get(cv2.CAP_PROP_FPS) or float(cfg_get(cfg, "fps", 30.0)) or 30.0)
    if width <= 0 or height <= 0:
        cap.release()
        print(f"[preprocess] invalid video size: {video_path}")
        return None

    # camera + pose params
    fov_y_deg = float(cfg_get(cfg, "fov_y_deg", 60.0))
    assumed_h = float(cfg_get(cfg, "assumed_person_height_m", 1.7))
    kp_conf_thresh = float(cfg_get(cfg, "kp_conf_thresh", 0.3))
    eps = float(cfg_get(cfg, "eps", 1e-6))

    # yolo params
    yolo_conf = float(cfg_get(cfg, "yolo_conf", 0.25))
    yolo_iou = float(cfg_get(cfg, "yolo_iou", 0.5))
    yolo_tracker = str(cfg_get(cfg, "yolo_tracker", "bytetrack.yaml"))
    yolo_device = str(cfg_get(cfg, "yolo_device", ""))

    # ego motion
    use_ego_motion = bool(cfg_get(cfg, "use_ego_motion", True))
    ego_tracker = EgoMotionTracker() if use_ego_motion else None

    fx, fy, cx, cy = camera_intrinsics_from_fov(height, width, fov_y_deg)

    # depth
    depth_mode = str(cfg_get(cfg, "depth_mode", "bbox")).lower()
    use_depth_anything = (depth_mode == "midas")  # 既存 config を崩さず DAV2 を割り当て
    depth_est = DepthAnythingV2DepthEstimator(cfg) if use_depth_anything else None

    frames_ego: List[Tuple[float, float]] = []
    frames_state: List[Dict[int, Dict[str, np.ndarray]]] = []

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, ids, confs, kpts_xy, kpts_conf = yolo_track_pose(
                model=yolo_pose_model,
                frame_bgr=frame,
                conf=yolo_conf,
                iou=yolo_iou,
                tracker=yolo_tracker,
                device=yolo_device,
            )

            # ego motion (background flow excluding person boxes)
            ego_vx, ego_vy = 0.0, 0.0
            if ego_tracker is not None:
                exclude = boxes.tolist() if boxes.shape[0] > 0 else []
                ego_vx, ego_vy = ego_tracker.update(frame, exclude)
            frames_ego.append((float(ego_vx), float(ego_vy)))

            # depth map once per frame
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
                    # depth fails -> fallback to bbox
                    print(f"[preprocess] depth_anything failed at frame={frame_idx}: {e}")
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


def preprocess_video_dir(
    video_dir: str,
    out_dir: str,
    cfg: SafetyConfig,
    skip_existing: bool = False,
) -> List[str]:
    ensure_dir(out_dir)

    pose_model_path = str(cfg_get(cfg, "yolo_pose_model", "yolov8n-pose.pt"))
    print(f"[preprocess] loading YOLO pose model: {pose_model_path}")
    yolo_pose_model = YOLO(pose_model_path)

    vids = list_videos(video_dir)
    if not vids:
        raise RuntimeError(f"No videos found in: {video_dir}")

    npz_paths: List[str] = []
    for vp in vids:
        print(f"[preprocess] processing: {vp}")
        out_npz = preprocess_one_video(
            video_path=vp,
            out_dir=out_dir,
            yolo_pose_model=yolo_pose_model,
            cfg=cfg,
            skip_existing=skip_existing,
        )
        if out_npz:
            npz_paths.append(out_npz)
    return npz_paths
