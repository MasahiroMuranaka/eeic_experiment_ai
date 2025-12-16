import os
import argparse
import numpy as np
import cv2

from create_answer import get_answer_from_video
from viz2 import load_prob_sequence, overlay_heatmap_from_grid, overlay_safe_area


def run(video_path: str, probs_npz: str, out_path: str, num_area: int, detections_json: str = None):
    # ensure probs npz exists; if not, try to generate via create_answer
    if not probs_npz or not os.path.exists(probs_npz):
        print("probs npz not found, generating from detections/video...")
        ans = get_answer_from_video(video_path, detections_json, num_area)
        keys = sorted(ans.keys(), key=lambda x: int(os.path.splitext(x)[0]))
        arr = np.stack([np.array(ans[k], dtype=np.float32) for k in keys], axis=0)
        probs_npz = os.path.splitext(out_path)[0] + "_probs.npz"
        np.savez_compressed(probs_npz, probs=arr)
        print(f"saved temporary probs to {probs_npz}")

    data = np.load(probs_npz)
    if 'probs' in data:
        probs = data['probs']
    else:
        # try first array
        probs = data[data.files[0]]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx = 0
    ema = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < probs.shape[0]:
            p = probs[frame_idx]
        else:
            p = np.ones((num_area,), dtype=np.float32) / num_area

        out_frame, _ = overlay_safe_area(frame, p, grid_size=(1, num_area), color=(0,255,0), thickness=4, pad=2)
        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"wrote overlay video to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='predict_area/data/train_data/data_1.mp4')
    parser.add_argument('--probs', default='')
    parser.add_argument('--detections', default=None)
    parser.add_argument('--num_area', type=int, default=8)
    parser.add_argument('--out', default='predict_area/result/test_result/viz_infer_output.mp4')
    args = parser.parse_args()

    # ensure target directories exist
    os.makedirs('npz/test_npz', exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    # if probs not provided, default will generate into npz/test_npz using video basename
    if not args.probs:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.probs = os.path.join('npz', 'test_npz', f"{base}.npz")

    run(args.video, args.probs, args.out, args.num_area, detections_json=args.detections)
