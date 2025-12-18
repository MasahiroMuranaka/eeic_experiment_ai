import os
import json
import re
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Union
import glob

def load_data_map(path: str) -> Dict[int, np.ndarray]:
    """
    CSV/NPZ/JSONから確率分布を読み込み、{フレーム番号: 配列} の辞書にして返す。
    柔軟に対応するため、リスト形式・辞書形式の双方をサポート。
    """
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return {}

    ext = os.path.splitext(path)[1].lower()
    mapping = {}

    try:
        if ext == '.csv':
            # CSVは行=フレームとみなす
            data = np.loadtxt(path, delimiter=',')
            if data.ndim == 1: data = data[np.newaxis, :]
            for i, row in enumerate(data):
                mapping[i] = row

        elif ext == '.npy':
            data = np.load(path)
            if data.ndim == 1: data = data[np.newaxis, :]
            for i, row in enumerate(data):
                mapping[i] = row

        elif ext == '.npz':
            with np.load(path, allow_pickle=True) as f:
                # 代表的なキーを探す
                key = next((k for k in ['probs', 'data', 'arr_0'] if k in f), f.files[0])
                data = f[key]
                # 2D配列なら行=フレーム
                if hasattr(data, 'ndim') and data.ndim == 2:
                    for i, row in enumerate(data):
                        mapping[i] = row
                else:
                    # 辞書的な格納の可能性があればファイル名キー解析などを実装可能だがMVPでは省略
                    pass

        elif ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for i, row in enumerate(data):
                    mapping[i] = np.array(row)
            elif isinstance(data, dict):
                # "frame_123" 等のキーから数字を抽出
                for k, v in data.items():
                    nums = re.findall(r'\d+', k)
                    if nums:
                        mapping[int(nums[-1])] = np.array(v)

    except Exception as e:
        print(f"Error loading {path}: {e}")
    
    return mapping

def get_grid_dims(size: int) -> Tuple[int, int]:
    """配列長から正方形グリッドを推定。不一致なら1行とする。"""
    s = int(np.sqrt(size))
    if s * s == size:
        return s, s
    return 1, size

def draw_safe_rect(frame: np.ndarray, prob: np.ndarray, color: Tuple[int, int, int], label: str = ""):
    """確率最大の位置に矩形を描画"""
    h, w = frame.shape[:2]
    gh, gw = get_grid_dims(prob.size)
    # 安全対策: prob が空、または gw が 0 の場合は何もしないで返す
    arr = np.asarray(prob).flatten()
    if arr.size == 0 or gw == 0:
        return frame

    # 最大値のインデックス
    flat_idx = int(np.argmax(arr))
    # divmod の前に gw が正の整数であることを保証
    if gw <= 0:
        return frame
    r, c = divmod(flat_idx, gw)
    
    # 座標計算
    cell_h, cell_w = h / gh, w / gw
    x1, y1 = int(c * cell_w), int(r * cell_h)
    x2, y2 = int((c + 1) * cell_w), int((r + 1) * cell_h)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    if label:
        cv2.putText(frame, f"{label}:{prob[flat_idx]:.2f}", (x1, max(y1-5, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def create_viz_videos(video_path, model_path, answer_path):
    # 1. データ読み込み
    print("Loading data...")
    model_map = load_data_map(model_path)
    answer_map = load_data_map(answer_path)
    
    # 2. 動画or画像準備
    use_video = True
    frames = []
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        base = os.path.splitext(os.path.basename(video_path))[0]
    else:
        use_video = False

        image_paths = glob.glob(os.path.join(video_path, "*.jpg"))
        for path in image_paths:
            frames.append(cv2.imread(path))

        fps = 15.0
        w = frames[0].shape[1]
        h = frames[0].shape[0]

        base = os.path.basename(video_path)
    
    out_model = f"{base}_model_overlay.mp4"
    out_answ = f"{base}_answer_overlay.mp4"
    out_side = f"{base}_side_by_side.mp4"
    
    # VideoWriter codec を試行して動作するものを選ぶ
    def _select_working_fourcc(sample_path, fps_val, size_val):
        codecs = ['mp4v', 'XVID', 'avc1', 'H264']
        for code in codecs:
            try:
                f = cv2.VideoWriter_fourcc(*code) #type: ignore
            except Exception:
                continue
            wtest = cv2.VideoWriter(sample_path, f, fps_val, size_val)
            if wtest.isOpened():
                wtest.release()
                return f
            try:
                wtest.release()
            except Exception:
                pass
        raise RuntimeError('No available VideoWriter codec')

    size_single = (w, h)
    fourcc = _select_working_fourcc(out_model, fps, size_single)
    writer_m = cv2.VideoWriter(out_model, fourcc, fps, (w, h))
    writer_a = cv2.VideoWriter(out_answ, fourcc, fps, (w, h))
    writer_s = cv2.VideoWriter(out_side, fourcc, fps, (w * 2, h))
    
    print(f"Processing frames... ({w}x{h} @ {fps}fps)")
    
    frame_idx = 0
    while True:
        if use_video:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            if frame_idx < len(frames) -  1:
                frame = frames[frame_idx]
            else:
                break
            
        frame_m = frame.copy()
        frame_a = frame.copy()
        
        # ロジック: 現在フレーム(idx)に対し、
        # モデル予測は `idx` の出力を使用
        # 正解データは「次どこに行くべきか」なので `idx + 1` のデータを使用
        
        # モデルデータ描画 (青)
        if frame_idx in model_map:
            frame_m = draw_safe_rect(frame_m, model_map[frame_idx], (255, 0, 0), "Pred")
            
        # 正解データ描画 (緑)
        target_idx = frame_idx + 1
        if target_idx in answer_map:
            frame_a = draw_safe_rect(frame_a, answer_map[target_idx], (0, 255, 0), "GT")
            
        # 書き出し
        writer_m.write(frame_m)
        writer_a.write(frame_a)
        
        side_img = np.hstack([frame_m, frame_a])
        writer_s.write(side_img)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    if use_video:
        cap.release()
    writer_m.release()
    writer_a.release()
    writer_s.release()
    print("Done. Outputs:")
    print(f" - {out_model}\n - {out_answ}\n - {out_side}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        create_viz_videos(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python viz_mvp.py <video.mp4> <model_probs.json/npz> <answer.csv/npz>")