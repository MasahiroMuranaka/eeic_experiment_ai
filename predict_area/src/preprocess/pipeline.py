# src/preprocess/pipeline.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import csv
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

    out_csv = os.path.join(out_dir, f"{base}.csv")
    save_tracking_csv(out_csv, frames_state, frames_ego, fps)
    run_stgcnn_save_separate(out_csv, out_dir, base)
    
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

def save_tracking_csv(
    csv_path: str,
    frames_state: List[Dict[int, Dict[str, np.ndarray]]],
    frames_ego: List[Tuple[float, float]],
    fps: float
):
    """
    時系列データから速度を計算し、CSVとして保存する
    """
    header = [
        "frame_id", "track_id", 
        "root_x", "root_y", "root_z",  # 3次元位置 (zが深度)
        "vel_x", "vel_y",              # 速度ベクトル (m/s)
        "ego_vx", "ego_vy",            # カメラの動き
    ]
    # ポーズ(関節)データのヘッダー追加 (34要素: x,y * 17点) ※簡易化のためconfは除く
    for i in range(17):
        header.extend([f"kpt_{i}_x", f"kpt_{i}_y"])

    rows = []
    prev_positions = {} # {track_id: (x, y)}

    for frame_idx, (st, ego) in enumerate(zip(frames_state, frames_ego)):
        ego_vx, ego_vy = ego
        
        for tid, info in st.items():
            # info["root"] には [x, y, z] が入っている (DepthAnything等で計算済み)
            root = info["root"]
            rx, ry, rz = root[0], root[1], root[2]
            
            # --- 速度ベクトルの計算 (今回追加するロジック) ---
            vx, vy = 0.0, 0.0
            if tid in prev_positions:
                px, py = prev_positions[tid]
                # (現在の位置 - 1フレーム前の位置) * FPS = 秒速
                vx = (rx - px) * fps
                vy = (ry - py) * fps
            
            # 位置を更新
            prev_positions[tid] = (rx, ry)

            # 行データの作成
            row = [
                frame_idx, tid,
                rx, ry, rz,
                vx, vy,
                ego_vx, ego_vy
            ]
            
            # 関節データ (info["pose"] はフラット化されている前提)
            # pose_flat は [x1, y1, conf1, x2, y2, conf2...] の並びなので座標だけ抜く
            pose = info["pose"]
            kpts_xy = []
            for k in range(17):
                # 3つ飛ばしでx, yを取得 (confは飛ばす)
                idx = k * 3
                if idx + 1 < len(pose):
                    kpts_xy.extend([pose[idx], pose[idx+1]])
                else:
                    kpts_xy.extend([0, 0])
            
            row.extend(kpts_xy)
            rows.append(row)

    # 書き出し
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"[preprocess] Saved CSV: {csv_path}")
    except Exception as e:
        print(f"[preprocess] Failed to save CSV: {e}")
        
# --- Social-STGCNN Logic (Separate NPZ Output) ---

class SimpleSTGCNN(nn.Module):
    def __init__(self, n_nodes, obs_len, pred_len, input_feat=2, kernel_size=3):
        super(SimpleSTGCNN, self).__init__()
        self.n_nodes = n_nodes
        self.obs_len = obs_len
        self.pred_len = pred_len
        
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(input_feat, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv_temporal = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, kernel_size), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, kernel_size), padding=(0, 1)),
            nn.ReLU()
        )
        self.output_layer = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, v):
        x = self.conv_spatial(v)
        x = self.conv_temporal(x)
        x = torch.mean(x, dim=3, keepdim=True) 
        out = self.output_layer(x)
        out = out.repeat(1, 1, 1, self.pred_len)
        return out

def run_stgcnn_save_separate(csv_path, out_dir, base_name):
    """
    STGCNNを実行し、結果を独立したNPZファイルとして保存する。
    保存形式は正規のNPZ (N, T, Nmax, F) に揃える。
    """
    target_npz = os.path.join(out_dir, f"{base_name}_stgcnn.npz")
    print(f"[STGCNN] Creating separate prediction file: {target_npz}")
    
    # 1. データ読み込み
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[STGCNN] Error loading CSV: {e}")
        return

    track_ids = df['track_id'].unique()
    if len(track_ids) == 0:
        return

    # 設定 (正規のNPZに合わせるパラメータ)
    obs_len = 8    # T
    pred_len = 12  # H (予測先)
    Nmax = 10      # 正規のフォーマットに合わせる人数
    
    max_len = df.groupby('track_id').size().max()
    if max_len <= obs_len + pred_len:
        print("[STGCNN] Sequence too short.")
        return

    n_nodes_total = len(track_ids)
    input_tensor = np.zeros((2, n_nodes_total, max_len)) 
    id_map = {tid: i for i, tid in enumerate(track_ids)}
    
    use_z = 'root_z' in df.columns
    for tid in track_ids:
        idx = id_map[tid]
        group = df[df['track_id'] == tid].sort_values('frame_id')
        if use_z:
            coords = group[['root_x', 'root_z']].values.T
        else:
            coords = group[['x', 'y']].values.T
        length = coords.shape[1]
        input_tensor[:, idx, :length] = coords

    # 2. 学習 & 推論
    data_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0) 
    
    model = SimpleSTGCNN(n_nodes=n_nodes_total, obs_len=obs_len, pred_len=pred_len)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    X_train_list, Y_train_list = [], []
    for i in range(max_len - obs_len - pred_len):
        X_train_list.append(data_tensor[:, :, :, i : i+obs_len])
        Y_train_list.append(data_tensor[:, :, :, i+obs_len : i+obs_len+pred_len])
    
    if X_train_list:
        X_train = torch.cat(X_train_list, dim=0)
        Y_train = torch.cat(Y_train_list, dim=0)
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, Y_train)
            loss.backward()
            optimizer.step()

    # 3. フォーマット整形 [Samples, T, Nmax, F]
    model.eval()
    
    X_list = []    # Input Features
    M_list = []    # Mask
    Pred_list = [] # Prediction Features

    for t in range(obs_len, max_len):
        input_window = data_tensor[:, :, :, t-obs_len:t]
        
        with torch.no_grad():
            output_traj = model(input_window)
            
        # Nmax人まででカットして整形
        last_pos = input_window[0, :, :, -1] 
        valid_indices = []
        for n_idx in range(n_nodes_total):
            if not (last_pos[0, n_idx] == 0 and last_pos[1, n_idx] == 0):
                valid_indices.append(n_idx)
        
        valid_indices = valid_indices[:Nmax]
        
        X_frame = np.zeros((obs_len, Nmax, 2), dtype=np.float32)
        M_frame = np.zeros((obs_len, Nmax), dtype=bool)
        Pred_frame = np.zeros((pred_len, Nmax, 2), dtype=np.float32)

        raw_obs = input_window.numpy()[0]
        raw_pred = output_traj.numpy()[0]

        for i, n_idx in enumerate(valid_indices):
            X_frame[:, i, :] = raw_obs[:, n_idx, :].T
            M_frame[:, i] = True
            Pred_frame[:, i, :] = raw_pred[:, n_idx, :].T

        X_list.append(X_frame)
        M_list.append(M_frame)
        Pred_list.append(Pred_frame)

    # 4. 別ファイルとして保存
    np.savez_compressed(
        target_npz,
        data=np.array(X_list),      # X [Samples, 8, 10, 2]
        mask=np.array(M_list),      # M [Samples, 8, 10]
        prediction=np.array(Pred_list), # Pred [Samples, 12, 10, 2]
        track_ids=track_ids
    )
    print(f"[STGCNN] Saved separate NPZ: {target_npz}")
