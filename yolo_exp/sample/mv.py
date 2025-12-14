import cv2
import numpy as np
import random
from ultralytics import YOLO

# 1. モデルのロード
model = YOLO('yolov8n.pt')

# IDごとの色設定
id_color_map = {}

# 入力動画の設定
input_path = '../mv/input_1.mp4'
cap = cv2.VideoCapture(input_path)


# 元動画の幅・高さ・FPSを取得して、保存用設定に合わせる
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 保存先ファイル名と圧縮形式の設定
output_path = '../mv/output_1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"処理開始: {input_path}")
print(f"保存先: {output_path} ({width}x{height}, {fps:.2f}fps)")
print("処理中は画面を表示します。'q'キーで中断・終了できます。")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 2. YOLOv8で追跡
    results = model.track(frame, persist=True, classes=[0], conf=0.3, verbose=False)

    # 3. 描画処理
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)
            conf = float(conf)

            # 色の決定
            if track_id not in id_color_map:
                random.seed(track_id)
                id_color_map[track_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            color = id_color_map[track_id]

            # 枠とIDを描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"ID:{track_id}"
            # 文字背景（読みやすくするため）
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 描画済みのフレームを書き込む
    out.write(frame)

    # --- 画面表示 ---
    display_frame = frame
    if width > 1280:
        display_frame = cv2.resize(frame, (1280, int(1280 * height / width)))
    
    cv2.imshow('Processing...', display_frame)

    # 'q'キーで途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("ユーザー操作により中断しました。")
        break

# 後始末
cap.release()
out.release() 
cv2.destroyAllWindows()

print("完了しました。")