# 実行前の注意:
# - 仮想環境をアクティベートすること
# - このファイルは sample フォルダ内で実行することを想定しています

import os
import cv2
import csv
import math
import random
from ultralytics import YOLO

# --- 設定 ---
CONFIG = {
    "model_path": "yolov8n.pt",
    "input_path": "../mv/input_1.mp4",
    "output_path": "../mv/output_1.mp4",
    "csv_path": "../csv/output_1.csv",
    "conf_threshold": 0.3,
    "class_ids": [0],
    "show_window": True,
    "debug": False
}


def resolve_path(path):
    """スクリプト位置を基準に相対パスを絶対化する。"""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, path))


def extract_tracking_data(results):
    tracked_objects = []
    if results[0].boxes.id is None:
        return tracked_objects

    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy() if hasattr(results[0].boxes, 'cls') else [0]*len(track_ids)

    for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
        obj = {
            "id": int(track_id),
            "class_id": int(cls),
            "box": [int(x) for x in box],
            "conf": float(conf)
        }
        tracked_objects.append(obj)

    return tracked_objects


def compute_features(tracked_objects, frame_idx, fps, prev_state, frame_height):
    """
    各オブジェクトについて特徴量を計算しCSV用の行を返す。
    prev_state は track_id -> {'cx','cy','frame'} を保持し、速度計算に使う。
    """
    rows = []
    time_s = frame_idx / fps if fps and fps > 0 else 0.0

    for obj in tracked_objects:
        tid = obj['id']
        x1, y1, x2, y2 = obj['box']
        conf = obj['conf']
        cls = obj.get('class_id', 0)

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        area = w * h
        aspect = (w / h) if h != 0 else 0.0

        # 速度計算
        vx = vy = speed = 0.0
        prev = prev_state.get(tid)
        if prev is not None:
            dt = (frame_idx - prev['frame']) / fps if fps and fps > 0 else 0.0
            if dt > 0:
                vx = (cx - prev['cx']) / dt
                vy = (cy - prev['cy']) / dt
                speed = math.hypot(vx, vy)

        # 簡易深度推定（バウンディングボックス高さの逆数的指標）
        est_depth = (frame_height / h) if h > 0 else 0.0

        # CSV行
        row = {
            'frame': frame_idx,
            'time': round(time_s, 4),
            'id': tid,
            'class_id': cls,
            'conf': round(conf, 4),
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cx': round(cx, 2), 'cy': round(cy, 2),
            'w': w, 'h': h, 'area': area, 'aspect_ratio': round(aspect, 4),
            'vx': round(vx, 4), 'vy': round(vy, 4), 'speed': round(speed, 4),
            'est_depth': round(est_depth, 4)
        }

        rows.append(row)

        # 状態更新
        prev_state[tid] = {'cx': cx, 'cy': cy, 'frame': frame_idx}

    return rows


def setup_video_io(config):
    input_path = resolve_path(config['input_path'])
    output_path = resolve_path(config['output_path'])

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画が見つかりません: {config['input_path']} (resolved: {input_path})")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"--- 開始 ---")
    print(f"入力: {input_path}")
    print(f"出力: {output_path} ({width}x{height}, {fps:.2f}fps)")

    return cap, out, width, height, fps


def write_csv_header(csv_path):
    header = ['frame','time','id','class_id','conf','x1','y1','x2','y2','cx','cy','w','h','area','aspect_ratio','vx','vy','speed','est_depth']
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()


def append_rows_to_csv(csv_path, rows):
    if not rows:
        return
    fieldnames = rows[0].keys()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for r in rows:
            writer.writerow(r)


def draw_tracking_info(frame, tracked_objects, id_color_map):
    for obj in tracked_objects:
        tid = obj['id']
        x1, y1, x2, y2 = obj['box']
        conf = obj['conf']

        if tid not in id_color_map:
            random.seed(tid)
            id_color_map[tid] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
        color = id_color_map[tid]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"ID:{tid} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def main():
    model = YOLO(CONFIG['model_path'])
    cap, out, width, height, fps = setup_video_io(CONFIG)
    id_color_map = {}
    prev_state = {}  # track_id -> last state for velocity

    csv_path = resolve_path(CONFIG['csv_path'])
    write_csv_header(csv_path)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, classes=CONFIG['class_ids'], conf=CONFIG['conf_threshold'], verbose=False)

            objects = extract_tracking_data(results)

            # 特徴量計算
            rows = compute_features(objects, frame_idx, fps, prev_state, frame.shape[0])
            append_rows_to_csv(csv_path, rows)

            # 描画
            frame_drawn = draw_tracking_info(frame, objects, id_color_map)
            out.write(frame_drawn)

            if CONFIG['show_window']:
                disp_frame = frame_drawn

                if frame_drawn.shape[1] > 1280:
                    disp_frame = cv2.resize(frame_drawn, (1280, int(1280 * frame_drawn.shape[0] / frame_drawn.shape[1])))

                cv2.imshow('YOLOv8 Tracking (mv3)', disp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('中断しました')
                    break

            frame_idx += 1

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print('--- 完了 ---')


if __name__ == '__main__':
    main()
