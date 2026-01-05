import os
import json
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

# ensure project src and infer directories are on sys.path so local modules can be imported
this_file = Path(__file__).resolve()
sys.path.insert(0, str(this_file.parent))
sys.path.insert(0, str(this_file.parent.parent))

from ultralytics import YOLO
from preprocess.yolo_pose import yolo_track_pose
from create_answer import get_answer_from_video


def detect_video_to_json(video_path: str, model_path: str, out_json: str, conf: float = 0.25, iou: float = 0.5, tracker: str = "bytetrack.yaml"):
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    detections = {}
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes, ids, confs, kpts_xy, kpts_conf = yolo_track_pose(
                model=model, frame_bgr=frame, conf=conf, iou=iou, tracker=tracker, device=""
            )

            entry = []
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[i].tolist()
                w = float(x2 - x1)
                h = float(y2 - y1)
                x = float(x1)
                score = float(confs[i]) if i < len(confs) else 0.0
                entry.append({"box": [x, float(y1), w, h], "score": score})

            detections[f"{frame_idx}.jpg"] = entry
            frame_idx += 1
    finally:
        cap.release()

    out = {"detections": detections}
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"wrote detections json: {out_json}")
    return out_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='predict_area/data/train_data/data_1.mp4')
    parser.add_argument('--model', default='yolov8n-pose.pt')
    parser.add_argument('--out_json', default='predict_area/detections/data_1_detections.json')
    parser.add_argument('--out_npz', default='npz/test_npz/data_1_from_detect.npz')
    parser.add_argument('--num_area', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_npz) or '.', exist_ok=True)

    detect_json = detect_video_to_json(args.video, args.model, args.out_json)

    # generate per-frame probabilities using existing function
    ans = get_answer_from_video(args.video, detect_json, args.num_area)

    # save json and npz
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(ans, f, indent=2)
    keys = sorted(ans.keys(), key=lambda x: int(os.path.splitext(x)[0]))
    arr = np.stack([np.array(ans[k], dtype=np.float32) for k in keys], axis=0)
    np.savez_compressed(args.out_npz, probs=arr)
    print(f"saved npz: {args.out_npz}")


if __name__ == '__main__':
    main()
