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
    run_stgcnn_auto(out_csv)
    
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

# --- Social-STGCNN Auto-Implementation (Append to bottom of pipeline.py) ---

class SimpleSTGCNN(nn.Module):
    """簡易版 Social-STGCNN モデル"""
    def __init__(self, n_nodes, obs_len, pred_len, input_feat=2, kernel_size=3):
        super(SimpleSTGCNN, self).__init__()
        self.n_nodes = n_nodes
        self.obs_len = obs_len
        self.pred_len = pred_len
        
        # 空間畳み込み (簡易グラフ畳み込み)
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(input_feat, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        # 時間畳み込み
        self.conv_temporal = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, kernel_size), padding=(0, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, kernel_size), padding=(0, 1)),
            nn.ReLU()
        )
        # 軌跡予測出力層 (平均と分散を出力するが今回は座標のみ予測)
        self.output_layer = nn.Conv2d(64, 2, kernel_size=1) # output (x, y)

    def forward(self, v):
        # v: (Batch, Feat, Nodes, Time)
        x = self.conv_spatial(v)
        x = self.conv_temporal(x)
        # 時間次元を圧縮して未来を予測
        x = torch.mean(x, dim=3, keepdim=True) 
        out = self.output_layer(x) # (Batch, 2, Nodes, 1)
        
        # 未来のステップ数分だけ単純線形補間で拡張 (簡易実装のため)
        # 本来はRNNやTCNで再帰的に出すが、ここではデモ用に簡略化
        out = out.repeat(1, 1, 1, self.pred_len)
        return out

def run_stgcnn_auto(csv_path):
    """CSVを読み込み、STGCNNを学習し、推論結果を保存する"""
    print(f"[STGCNN] Starting auto-training for {csv_path}...")
    
    # 1. データ読み込み
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[STGCNN] Error loading CSV: {e}")
        return

    # 必要なカラム (root_x, root_z) を使用
    # track_idごとにデータをまとめる
    track_ids = df['track_id'].unique()
    if len(track_ids) == 0:
        return

    # データをテンソル化 (Batch=1, Feat=2, Nodes=N, Time=T)
    # 簡易化のため、一番長い系列に合わせてゼロパディング
    max_len = df.groupby('track_id').size().max()
    if max_len < 10:
        print("[STGCNN] Data too short for training. Skipping.")
        return

    n_nodes = len(track_ids)
    input_tensor = np.zeros((1, 2, n_nodes, max_len))
    
    # IDマッピング
    id_map = {tid: i for i, tid in enumerate(track_ids)}
    
    for tid in track_ids:
        idx = id_map[tid]
        group = df[df['track_id'] == tid].sort_values('frame_id')
        # x, z (Top-down view)
        coords = group[['root_x', 'root_z']].values.T # (2, Time)
        length = coords.shape[1]
        input_tensor[0, :, idx, :length] = coords

    # 学習用設定
    obs_len = 8   # 観察フレーム数
    pred_len = 12 # 予測フレーム数
    
    if max_len <= obs_len + pred_len:
        print("[STGCNN] Sequence too short. Skipping.")
        return

    # PyTorchテンソル変換
    data_tensor = torch.tensor(input_tensor, dtype=torch.float32) # (1, 2, N, T)
    
    # 学習データの作成 (スライディングウィンドウ)
    X_train_list = []
    Y_train_list = []
    
    for i in range(max_len - obs_len - pred_len):
        X_train_list.append(data_tensor[:, :, :, i : i+obs_len])
        Y_train_list.append(data_tensor[:, :, :, i+obs_len : i+obs_len+pred_len])
    
    if not X_train_list:
        return

    X_train = torch.cat(X_train_list, dim=0) # (Batch, 2, N, 8)
    Y_train = torch.cat(Y_train_list, dim=0) # (Batch, 2, N, 12)

    # 2. モデル構築
    model = SimpleSTGCNN(n_nodes=n_nodes, obs_len=obs_len, pred_len=pred_len)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 3. 学習ループ (超高速完了のため5エポック)
    model.train()
    print("[STGCNN] Training model...")
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        # print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # 4. 推論 (最後のフレームから未来を予測)
    model.eval()
    last_obs = data_tensor[:, :, :, -obs_len:] # 最後の8フレーム
    if last_obs.shape[3] < obs_len:
        # 足りない場合はパディング
        pad = torch.zeros((1, 2, n_nodes, obs_len - last_obs.shape[3]))
        last_obs = torch.cat([last_obs, pad], dim=3)

    with torch.no_grad():
        future_pred = model(last_obs) # (1, 2, N, 12)

    # 5. 結果保存
    # 予測結果をCSV形式に戻す
    pred_numpy = future_pred.numpy()[0] # (2, N, 12)
    pred_rows = []
    
    start_frame = df['frame_id'].max() + 1
    
    for t in range(pred_len):
        frame_id = start_frame + t
        for tid in track_ids:
            idx = id_map[tid]
            x_pred = pred_numpy[0, idx, t]
            z_pred = pred_numpy[1, idx, t]
            
            # 予測値が0,0 (パディング領域) なら無視する簡易フィルタ
            if x_pred == 0 and z_pred == 0:
                continue

            pred_rows.append([frame_id, tid, x_pred, z_pred])
            
    out_pred_path = csv_path.replace(".csv", "_prediction.csv")
    pred_df = pd.DataFrame(pred_rows, columns=["frame_id", "track_id", "pred_x", "pred_z"])
    pred_df.to_csv(out_pred_path, index=False)
    
    print(f"[STGCNN] Prediction saved to: {out_pred_path}")
