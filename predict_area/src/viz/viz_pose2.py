#type: ignore
import argparse
import cv2
import numpy as np
import sys
import csv
import os

# ---------------------------------------------------------
# インポートエラー回避策
# VSCodeなどで "ultralytics" が見つからないと警告される場合の対策
# ---------------------------------------------------------
try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    # 万が一通常のインポートが失敗した場合のフォールバック
    try:
        from ultralytics.yolo.engine.model import YOLO
    except ImportError:
        print("[Error] ultralyticsが見つかりません。'pip install ultralytics' を実行してください。")
        sys.exit(1)


# 骨格の接続定義 (COCO Keypoints format)
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12),  # 脚
    (11, 12),  # 腰
    (5, 11), (6, 12),  # 胴体
    (5, 6),  # 肩
    (5, 7), (7, 9), (6, 8), (8, 10),  # 腕
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)  # 顔周辺
]

# COCO Keypointの名前（CSVヘッダー用）
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 描画色設定 (BGR)
COLOR_BOX = (0, 255, 0)      # 緑
COLOR_TEXT = (255, 255, 255) # 白
COLOR_KPT = (0, 0, 255)      # 赤
COLOR_LIMB = (255, 0, 0)     # 青


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Pose特徴量の抽出と可視化・CSV出力")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力画像のパス")
    parser.add_argument("--output", "-o", type=str, default="output.jpg", help="出力画像のパス")
    parser.add_argument("--csv", "-c", type=str, default="confidence.csv", help="信頼度データの出力先CSVパス")
    parser.add_argument("--model", "-m", type=str, default="yolov8n-pose.pt", help="モデルファイルのパス")
    parser.add_argument("--conf", type=float, default=0.3, help="検出の信頼度閾値")
    parser.add_argument("--device", type=str, default="", help="デバイス (例: '0', 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. モデルの読み込み
    print(f"[Info] Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[Error] モデルの読み込みに失敗しました: {e}")
        sys.exit(1)

    # 2. 画像の読み込み
    if not os.path.exists(args.input):
        print(f"[Error] 画像が見つかりません: {args.input}")
        sys.exit(1)
    frame = cv2.imread(args.input)

    # 3. 推論実行
    kwargs = dict(
        persist=True,
        verbose=False,
        conf=args.conf,
        tracker="bytetrack.yaml"
    )
    if args.device:
        kwargs["device"] = args.device

    results = model.track(frame, **kwargs)
    r0 = results[0]

    # 結果が空の場合
    if r0.boxes is None or len(r0.boxes) == 0:
        print("[Info] 人物が検出されませんでした。画像をそのまま保存します。")
        cv2.imwrite(args.output, frame)
        # 空のCSVを作成しておく（ヘッダーのみ）
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ["filename", "person_id", "box_conf"] + KEYPOINT_NAMES
            writer.writerow(header)
        return

    # 4. データの取り出し
    boxes = r0.boxes.xyxy.cpu().numpy().astype(np.int32)
    box_confs = r0.boxes.conf.cpu().numpy().astype(np.float32)

    if r0.boxes.id is not None:
        ids = r0.boxes.id.int().cpu().numpy()
    else:
        ids = np.arange(len(boxes), dtype=np.int32)

    kpts_xy = r0.keypoints.xy.cpu().numpy().astype(np.float32)
    kpts_conf = r0.keypoints.conf.cpu().numpy().astype(np.float32)

    # 5. 可視化処理 & データ収集
    print(f"[Info] Detected {len(boxes)} persons.")
    vis_frame = frame.copy()
    
    csv_rows = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        track_id = int(ids[i])
        box_conf = float(box_confs[i])
        
        # --- CSV用データ準備 ---
        # 1人のデータを1行にまとめる
        # [ファイル名, ID, Box信頼度, 鼻の信頼度, 左目の信頼度, ...]
        this_kpts_conf = kpts_conf[i].tolist() # 17個の信頼度
        row = [os.path.basename(args.input), track_id, f"{box_conf:.4f}"]
        row.extend([f"{c:.4f}" for c in this_kpts_conf])
        csv_rows.append(row)

        # --- 描画処理 ---
        # Bounding Box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
        
        # ID & Confidence Text
        label = f"ID:{track_id} Conf:{box_conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_frame, (x1, y1 - 20), (x1 + w, y1), COLOR_BOX, -1)
        cv2.putText(vis_frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

        # Keypoints
        this_kpts = kpts_xy[i]
        
        for j, (kx, ky) in enumerate(this_kpts):
            if this_kpts_conf[j] < 0.5: continue
            cv2.circle(vis_frame, (int(kx), int(ky)), 4, COLOR_KPT, -1)

        for p1_idx, p2_idx in SKELETON_CONNECTIONS:
            if this_kpts_conf[p1_idx] > 0.5 and this_kpts_conf[p2_idx] > 0.5:
                pt1 = (int(this_kpts[p1_idx][0]), int(this_kpts[p1_idx][1]))
                pt2 = (int(this_kpts[p2_idx][0]), int(this_kpts[p2_idx][1]))
                cv2.line(vis_frame, pt1, pt2, COLOR_LIMB, 2)

    # 6. 保存処理
    # 画像保存
    cv2.imwrite(args.output, vis_frame)
    print(f"[Info] Saved image to: {args.output}")

    # CSV保存
    with open(args.csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ヘッダー書き込み
        header = ["filename", "person_id", "box_conf"] + KEYPOINT_NAMES
        writer.writerow(header)
        # データ書き込み
        writer.writerows(csv_rows)
    
    print(f"[Info] Saved confidence data to: {args.csv}")

if __name__ == "__main__":
    main()