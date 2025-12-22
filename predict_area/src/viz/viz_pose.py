#type: ignore

import argparse
import cv2
import numpy as np
from ultralytics import YOLO 
import sys

# 骨格の接続定義 (COCO Keypoints format for YOLOv8)
# 各数字は関節のインデックスを表します
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12),  # 脚
    (11, 12),  # 腰
    (5, 11), (6, 12),  # 胴体
    (5, 6),  # 肩
    (5, 7), (7, 9), (6, 8), (8, 10),  # 腕
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4)  # 顔周辺
]

# 描画色設定 (BGR)
COLOR_BOX = (0, 255, 0)      # 緑
COLOR_TEXT = (255, 255, 255) # 白
COLOR_KPT = (0, 0, 255)      # 赤
COLOR_LIMB = (255, 0, 0)     # 青

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Pose特徴量の抽出と可視化")
    parser.add_argument("--input", "-i", type=str, required=True, help="入力画像のパス")
    parser.add_argument("--output", "-o", type=str, default="output.jpg", help="出力画像のパス")
    parser.add_argument("--model", "-m", type=str, default="yolov8n-pose.pt", help="モデルファイルのパス")
    parser.add_argument("--conf", type=float, default=0.3, help="検出の信頼度閾値")
    parser.add_argument("--device", type=str, default="", help="デバイス (例: '0' for GPU, 'cpu')")
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
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"[Error] 画像が見つかりません: {args.input}")
        sys.exit(1)

    # 3. 推論実行 (yolo_pose.py のロジックを踏襲)
    # trackモードを使用することでID取得を試みます
    kwargs = dict(
        persist=True,
        verbose=False,
        conf=args.conf,
        tracker="bytetrack.yaml" # デフォルトトラッカー
    )
    if args.device:
        kwargs["device"] = args.device

    results = model.track(frame, **kwargs)
    r0 = results[0]

    # 結果が空の場合のハンドリング
    if r0.boxes is None or len(r0.boxes) == 0:
        print("[Info] 人物が検出されませんでした。画像をそのまま保存します。")
        cv2.imwrite(args.output, frame)
        return

    # 4. データの取り出し (CPU上のnumpy配列に変換)
    boxes = r0.boxes.xyxy.cpu().numpy().astype(np.int32)
    
    # IDは動画でないとNoneになる場合がありますが、取得できた場合は利用します
    if r0.boxes.id is not None:
        ids = r0.boxes.id.int().cpu().numpy()
    else:
        # IDがない場合は0から連番を振る
        ids = np.arange(len(boxes), dtype=np.int32)

    # キーポイント (x, y) と信頼度
    kpts_xy = r0.keypoints.xy.cpu().numpy().astype(np.float32)
    kpts_conf = r0.keypoints.conf.cpu().numpy().astype(np.float32)

    # 5. 可視化 (描画処理)
    print(f"[Info] Detected {len(boxes)} persons.")
    
    vis_frame = frame.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        track_id = ids[i]
        
        # Bounding Box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
        
        # ID Text
        label = f"ID: {track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis_frame, (x1, y1 - 20), (x1 + w, y1), COLOR_BOX, -1)
        cv2.putText(vis_frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

        # Keypoints & Skeleton
        # この人のキーポイントセット
        this_kpts = kpts_xy[i]      # shape: (17, 2)
        this_confs = kpts_conf[i]   # shape: (17,)

        # 関節点の描画
        for j, (kx, ky) in enumerate(this_kpts):
            if this_confs[j] < 0.5: continue # 信頼度が低い点は描画しない
            cv2.circle(vis_frame, (int(kx), int(ky)), 4, COLOR_KPT, -1)

        # 骨格（線）の描画
        for p1_idx, p2_idx in SKELETON_CONNECTIONS:
            # 両端の信頼度が高い場合のみ線を引く
            if this_confs[p1_idx] > 0.5 and this_confs[p2_idx] > 0.5:
                pt1 = (int(this_kpts[p1_idx][0]), int(this_kpts[p1_idx][1]))
                pt2 = (int(this_kpts[p2_idx][0]), int(this_kpts[p2_idx][1]))
                cv2.line(vis_frame, pt1, pt2, COLOR_LIMB, 2)

    # 6. 保存
    cv2.imwrite(args.output, vis_frame)
    print(f"[Info] Saved result to: {args.output}")

if __name__ == "__main__":
    main()