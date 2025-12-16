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
from preprocess.frame_source import iter_frames_from_dir, read_first_frame
from preprocess.external_json import load_detection_json, load_prob_dist_json, boxes_xyxy_from_detection_list


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


def _boxes_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return float(inter / (area_a + area_b - inter + 1e-6))


class _SimpleIoUTracker:
    """
    Very small tracker to assign stable IDs to per-frame detections.
    This is only used when external detection JSON does not provide IDs.
    """

    def __init__(self, iou_thresh: float = 0.3, max_lost: int = 30):
        self.iou_thresh = float(iou_thresh)
        self.max_lost = int(max_lost)
        self._next_id = 1
        self._tracks: Dict[int, Dict[str, object]] = {}  # tid -> {box, lost}

    def update(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        n = int(boxes_xyxy.shape[0])
        ids = -np.ones((n,), dtype=np.int64)

        # age tracks
        for tid in list(self._tracks.keys()):
            self._tracks[tid]["lost"] = int(self._tracks[tid]["lost"]) + 1  # type: ignore[arg-type]
            if int(self._tracks[tid]["lost"]) > self.max_lost:  # type: ignore[arg-type]
                del self._tracks[tid]

        if n == 0:
            return ids

        track_items = list(self._tracks.items())
        used_tracks = set()
        for i in range(n):
            best_tid = None
            best_iou = 0.0
            for tid, st in track_items:
                if tid in used_tracks:
                    continue
                iou = _boxes_iou_xyxy(boxes_xyxy[i], np.asarray(st["box"], dtype=np.float32))  # type: ignore[index]
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None and best_iou >= self.iou_thresh:
                ids[i] = int(best_tid)
                used_tracks.add(best_tid)
                self._tracks[best_tid] = {"box": boxes_xyxy[i].copy(), "lost": 0}
            else:
                tid_new = int(self._next_id)
                self._next_id += 1
                ids[i] = tid_new
                self._tracks[tid_new] = {"box": boxes_xyxy[i].copy(), "lost": 0}
        return ids


def preprocess_one_frames_dir(
    frames_dir: str,
    out_dir: str,
    yolo_pose_model: YOLO,
    cfg: SafetyConfig,
    skip_existing: bool = False,
    y_json: str = "",
    det_json: str = "",
    fps_override: float = 0.0,
) -> Optional[str]:
    """
    Preprocess a directory of frame images as one sequence and save a single .npz.

    - If y_json is provided, it is used as the teacher distribution y for each frame (no need for H).
    - If det_json is provided, it is used as person boxes (and we assign IDs by IoU tracking).
      Note: the provided det_json does not include keypoints; pose/conf are filled with zeros.
    - Otherwise, we run YOLO pose tracking on each frame (same as video path).
    """
    base = os.path.basename(os.path.normpath(frames_dir))
    out_npz = os.path.join(out_dir, f"{base}.npz")
    if skip_existing and os.path.exists(out_npz):
        print(f"[preprocess] skip existing: {out_npz}")
        return out_npz

    first_name, first_frame = read_first_frame(frames_dir)
    height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
    fps = float(fps_override) if fps_override and fps_override > 0 else float(cfg_get(cfg, "fps", 30.0)) or 30.0

    # camera + pose params
    fov_y_deg = float(cfg_get(cfg, "fov_y_deg", 60.0))
    assumed_h = float(cfg_get(cfg, "assumed_person_height_m", 1.7))
    kp_conf_thresh = float(cfg_get(cfg, "kp_conf_thresh", 0.3))
    eps = float(cfg_get(cfg, "eps", 1e-6))
    fx, fy, cx, cy = camera_intrinsics_from_fov(height, width, fov_y_deg)

    # yolo params
    yolo_conf = float(cfg_get(cfg, "yolo_conf", 0.25))
    yolo_iou = float(cfg_get(cfg, "yolo_iou", 0.5))
    yolo_tracker = str(cfg_get(cfg, "yolo_tracker", "bytetrack.yaml"))
    yolo_device = str(cfg_get(cfg, "yolo_device", ""))

    # ego motion
    use_ego_motion = bool(cfg_get(cfg, "use_ego_motion", True))
    ego_tracker = EgoMotionTracker() if use_ego_motion else None

    # depth
    depth_mode = str(cfg_get(cfg, "depth_mode", "bbox")).lower()
    use_depth_anything = (depth_mode == "midas")
    depth_est = DepthAnythingV2DepthEstimator(cfg) if use_depth_anything else None

    # external y / det
    y_by_name = load_prob_dist_json(y_json) if y_json else None
    det_by_name = load_detection_json(det_json) if det_json else None
    iou_tracker = _SimpleIoUTracker() if det_by_name is not None else None

    frames_ego: List[Tuple[float, float]] = []
    frames_state: List[Dict[int, Dict[str, np.ndarray]]] = []
    frame_names: List[str] = []

    for frame_idx, frame_name, frame in iter_frames_from_dir(frames_dir):
        frame_names.append(frame_name)

        if det_by_name is not None:
            det_list = det_by_name.get(frame_name, [])
            boxes, confs = boxes_xyxy_from_detection_list(det_list, width=width, height=height)
            ids = iou_tracker.update(boxes) if iou_tracker is not None else -np.ones((boxes.shape[0],), dtype=np.int64)
            # no keypoints in det_json; fill dummy arrays
            kpts_xy = np.zeros((boxes.shape[0], 17, 2), dtype=np.float32)
            kpts_conf = np.zeros((boxes.shape[0], 17), dtype=np.float32)
        else:
            boxes, ids, confs, kpts_xy, kpts_conf = yolo_track_pose(
                model=yolo_pose_model,
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
                print(f"[preprocess] depth_anything failed at frame={frame_idx}: {e}")
                depth_map = None

        st: Dict[int, Dict[str, np.ndarray]] = {}
        for i in range(int(boxes.shape[0])):
            tid = int(ids[i]) if int(ids[i]) >= 0 else int(i + 1)
            if depth_est is None:
                root = estimate_root_xyz_from_bbox(boxes[i], fx, fy, cx, cy, assumed_h, eps=eps)
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

    try:
        build_npz_from_video_buffers(
            frames_state=frames_state,
            frames_ego=frames_ego,
            fps=fps,
            width=width,
            height=height,
            cfg=cfg,
            out_npz_path=out_npz,
            video_path=frames_dir,
            frame_names=frame_names,
            y_by_frame_name=y_by_name,
        )
        return out_npz
    except Exception as e:
        print(f"[preprocess] failed building npz for frames_dir={frames_dir}: {e}")
        return None

