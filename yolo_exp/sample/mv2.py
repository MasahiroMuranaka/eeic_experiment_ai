#実行するときは、
# 「仮想環境をアクティベートすること」
# 「sampleフォルダに入ること」
#を忘れないで。

import cv2
import numpy as np
import random
from ultralytics import YOLO

# --- 1. 設定 (Configuration) ---
CONFIG = {
    "model_path": "yolov8n.pt",
    "input_path": "../mv/input_1.mp4",   # 入力ファイル
    "output_path": "../mv/output_1.mp4", # 出力ファイル
    "conf_threshold": 0.3,               # 検出の閾値
    "class_ids": [0],                    # 0: Person
    "show_window": True                  # 画面表示をするか
}

# --- 2. メモリ処理 (Data Extraction) ---
def extract_tracking_data(results):
    """
    YOLOの解析結果から、必要なデータ（ID, 座標, 信頼度）のみを抽出してリスト化する。
    ここでNumPy型から標準のint/float型へ変換し、バグを防ぐ。
    """
    tracked_objects = []
    
    # 何も検出されなかった場合は空リストを返す
    if results[0].boxes.id is None:
        return tracked_objects

    # データをCPU/NumPyへ転送
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    for box, track_id, conf in zip(boxes, track_ids, confs):
        # ★ここで型変換を済ませる（バグ回避の重要ポイント）
        obj = {
            "id": int(track_id),
            "box": [int(x) for x in box],  # [x1, y1, x2, y2]
            "conf": float(conf)
        }
        tracked_objects.append(obj)
        
    return tracked_objects

# --- 3. 描画関数 (Drawing) ---
def draw_tracking_info(frame, tracked_objects, id_color_map):
    """
    フレームに対し、抽出されたオブジェクト情報の枠とIDを描画する。
    """
    for obj in tracked_objects:
        tid = obj["id"]
        x1, y1, x2, y2 = obj["box"]
        conf = obj["conf"]

        # IDに基づいて色を決定（辞書になければ生成）
        if tid not in id_color_map:
            random.seed(tid)
            id_color_map[tid] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        color = id_color_map[tid]

        # 枠の描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # テキストラベルの描画
        label = f"ID:{tid} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # 文字が見やすいように背景を塗りつぶす
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# --- 4. 動画保存・準備関数 (Video I/O Setup) ---
def setup_video_io(config):
    """
    動画の読み込み(cap)と書き込み(writer)を初期化して返す。
    """
    cap = cv2.VideoCapture(config["input_path"])
    if not cap.isOpened():
        raise FileNotFoundError(f"動画が見つかりません: {config['input_path']}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 保存用ライターの作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config["output_path"], fourcc, fps, (width, height))
    
    print(f"--- 開始 ---")
    print(f"入力: {config['input_path']}")
    print(f"出力: {config['output_path']} ({width}x{height}, {fps:.2f}fps)")
    
    return cap, out

# --- メイン処理 (Main Loop) ---
def main():
    # 準備
    model = YOLO(CONFIG["model_path"])
    cap, out = setup_video_io(CONFIG)
    id_color_map = {} # 色管理用

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 推論 & トラッキング
            results = model.track(frame, persist=True, classes=CONFIG["class_ids"], 
                                  conf=CONFIG["conf_threshold"], verbose=False)

            # 2. メモリ処理（データの抽出・整形）
            objects = extract_tracking_data(results)

            # 3. 描画処理
            frame_drawn = draw_tracking_info(frame, objects, id_color_map)

            # 4. 動画保存
            out.write(frame_drawn)

            # 5. 画面表示（オプション）
            if CONFIG["show_window"]:
                # 画面が大きすぎる場合の表示用リサイズ
                disp_frame = frame_drawn
                if frame_drawn.shape[1] > 1280:
                     disp_frame = cv2.resize(frame_drawn, (1280, int(1280 * frame_drawn.shape[0] / frame_drawn.shape[1])))
                
                cv2.imshow('YOLOv8 Tracking', disp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("中断しました")
                    break

    finally:
        # 終了処理（必ず実行される）
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("--- 完了 ---")

if __name__ == "__main__":
    main()