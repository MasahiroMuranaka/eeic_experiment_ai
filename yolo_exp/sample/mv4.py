import os
import cv2
import csv
import math
import random
import numpy as np
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
}

def resolve_path(path):
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, path))

# --- カメラの動き（Ego-Motion）を推定するクラス ---
class EgoMotionTracker:
    def __init__(self):
        # オプティカルフローの設定
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.prev_pts = None

    def update(self, frame, exclude_boxes):
        """
        frame: 現在のフレーム
        exclude_boxes: 人物のBBoxリスト [[x1, y1, x2, y2], ...] -> この領域は無視する
        戻り値: (vx, vy) カメラの推定移動量（ピクセル/フレーム）
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ego_vx, ego_vy = 0.0, 0.0

        if self.prev_gray is not None:
            # 特徴点が少なくなったら補充
            if self.prev_pts is None or len(self.prev_pts) < 50:
                self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)

            if self.prev_pts is not None and len(self.prev_pts) > 0:
                # 人物領域にある点を除外（背景だけ残す）
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

                    # オプティカルフロー計算（forward）
                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, good_prev_pts, None, **self.lk_params)

                    # forward-backward チェックで信頼できる点だけ残す
                    fb_thresh = 1.5
                    good_prev = []
                    good_next = []
                    for i in range(len(good_prev_pts)):
                        if status is None or status[i][0] == 0:
                            continue
                        p_next = next_pts[i:i+1]
                        p_back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p_next, None, **self.lk_params)
                        if st_back is None or st_back[0][0] == 0:
                            continue
                        # 再投影誤差
                        err = np.linalg.norm(good_prev_pts[i] - p_back[0])
                        if err <= fb_thresh:
                            good_prev.append(good_prev_pts[i])
                            good_next.append(next_pts[i])

                    if len(good_prev) == 0:
                        self.prev_pts = None
                    else:
                        prev_arr = np.array(good_prev, dtype=np.float32).reshape(-1,1,2)
                        next_arr = np.array(good_next, dtype=np.float32).reshape(-1,1,2)

                        # 幾何変換（アフィン）をRANSACで推定して堅牢に平行移動を抽出
                        pts_prev = prev_arr.reshape(-1,2)
                        pts_next = next_arr.reshape(-1,2)
                        if len(pts_prev) >= 3:
                            M, inliers = cv2.estimateAffinePartial2D(pts_prev, pts_next, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                            if M is not None:
                                # 平行移動成分を使用（M の [0,2], [1,2]）
                                ego_vx, ego_vy = M[0,2], M[1,2]
                                # 更新点はinliersのnext点
                                if inliers is not None:
                                    in_mask = inliers.ravel()==1
                                    if np.sum(in_mask) > 0:
                                        self.prev_pts = pts_next[in_mask].reshape(-1,1,2)
                                    else:
                                        self.prev_pts = pts_next.reshape(-1,1,2)
                                else:
                                    self.prev_pts = pts_next.reshape(-1,1,2)
                            else:
                                # 推定失敗時は中央値でロバストにフォールバック
                                motion = pts_next - pts_prev
                                med = np.median(motion, axis=0)
                                ego_vx, ego_vy = float(med[0]), float(med[1])
                                self.prev_pts = pts_next.reshape(-1,1,2)
                        else:
                            # 点が少ない場合は中央値を使う
                            motion = pts_next - pts_prev
                            med = np.median(motion, axis=0)
                            ego_vx, ego_vy = float(med[0]), float(med[1])
                            self.prev_pts = pts_next.reshape(-1,1,2)

        self.prev_gray = gray.copy()
        # 背景が動いた方向と逆が、カメラの進んだ方向になる点に注意。
        # ここでは背景の流れ（ピクセル/フレーム）を出力する（符号ルールはドキュメントに記載予定）。
        return round(float(ego_vx), 4), round(float(ego_vy), 4)


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

def compute_features(tracked_objects, frame_idx, fps, prev_state, frame_height, ego_motion):
    """
    ego_motion: (ego_vx, ego_vy) カメラ自体の動き情報
    """
    rows = []
    time_s = frame_idx / fps if fps and fps > 0 else 0.0
    ego_vx, ego_vy = ego_motion

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

        vx = vy = speed = 0.0
        prev = prev_state.get(tid)
        if prev is not None:
            dt = (frame_idx - prev['frame']) / fps if fps and fps > 0 else 0.0
            if dt > 0:
                vx = (cx - prev['cx']) / dt
                vy = (cy - prev['cy']) / dt
                speed = math.hypot(vx, vy)

        est_depth = (frame_height / h) if h > 0 else 0.0

        row = {
            'frame': frame_idx,
            'time': round(time_s, 4),
            'id': tid,
            'class_id': cls,
            'conf': round(conf, 4),
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'cx': round(cx, 2), 'cy': round(cy, 2),
            'w': w, 'h': h, 
            'vx': round(vx, 4), 'vy': round(vy, 4), 'speed': round(speed, 4),
            'est_depth': round(est_depth, 4),
            # ★追加: カメラの動き情報
            'ego_vx': ego_vx, 'ego_vy': ego_vy
        }
        rows.append(row)
        prev_state[tid] = {'cx': cx, 'cy': cy, 'frame': frame_idx}

    return rows

def setup_video_io(config):
    input_path = resolve_path(config['input_path'])
    output_path = resolve_path(config['output_path'])
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"動画が見つかりません: {config['input_path']}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"--- 開始: {input_path} -> {output_path} ---")
    return cap, out, width, height, fps

def write_csv_header(csv_path):
    # ヘッダーに ego_vx, ego_vy を追加
    header = ['frame','time','id','class_id','conf','x1','y1','x2','y2','cx','cy','w','h',
              'vx','vy','speed','est_depth', 'ego_vx', 'ego_vy']
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

def append_rows_to_csv(csv_path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    # 追記モードで複数行をまとめて書く
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)

def draw_tracking_info(frame, tracked_objects, id_color_map, ego_motion):
    # カメラの動きを画面左上に矢印で表示
    cv2.arrowedLine(frame, (50, 50), (int(50 + ego_motion[0]*5), int(50 + ego_motion[1]*5)), (0, 0, 255), 3)
    cv2.putText(frame, "Ego Motion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for obj in tracked_objects:
        tid = obj['id']
        x1, y1, x2, y2 = obj['box']
        conf = obj['conf']
        if tid not in id_color_map:
            random.seed(tid)
            id_color_map[tid] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        color = id_color_map[tid]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"ID:{tid}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def main():
    model = YOLO(CONFIG['model_path'])
    cap, out, width, height, fps = setup_video_io(CONFIG)
    id_color_map = {}
    prev_state = {}
    
    # Ego Motion Trackerの初期化
    ego_tracker = EgoMotionTracker()

    csv_path = resolve_path(CONFIG['csv_path'])
    write_csv_header(csv_path)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 1. 人物検出
            results = model.track(frame, persist=True, classes=CONFIG['class_ids'], conf=CONFIG['conf_threshold'], verbose=False)
            objects = extract_tracking_data(results)

            # 2. Ego Motion (カメラの動き) の計算
            # 現在検出されている人物のBounding Boxをリスト化して渡す（そこは背景として計算しないため）
            person_boxes = [obj['box'] for obj in objects]
            ego_motion = ego_tracker.update(frame, person_boxes)

            # 3. 特徴量計算 (Ego Motionも渡す)
            rows = compute_features(objects, frame_idx, fps, prev_state, frame.shape[0], ego_motion)
            append_rows_to_csv(csv_path, rows)

            # 4. 描画
            frame_drawn = draw_tracking_info(frame, objects, id_color_map, ego_motion)
            out.write(frame_drawn)

            if CONFIG['show_window']:
                disp = cv2.resize(frame_drawn, (1280, int(1280 * frame_drawn.shape[0] / frame_drawn.shape[1]))) if frame_drawn.shape[1] > 1280 else frame_drawn
                cv2.imshow('YOLOv8 + EgoMotion', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            frame_idx += 1
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print('--- 完了 ---')

if __name__ == '__main__':
    main()