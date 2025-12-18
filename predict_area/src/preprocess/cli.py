import os
import sys
import argparse
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SafetyConfig, load_config, save_config
from preprocess.io_utils import cfg_get, ensure_dir, list_videos
from preprocess.pipeline import preprocess_one_video


def write_manifest(out_dir: str, paths: List[str]) -> str:
    manifest = os.path.join(out_dir, "manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"[preprocess] wrote manifest: {manifest}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="directory containing videos")
    ap.add_argument("--out-dir", required=True, help="output directory for npz files")
    ap.add_argument("--config", default="", help="config yaml path")
    ap.add_argument("--save-config", default="", help="write default config yaml and exit")
    ap.add_argument("--skip-existing", action="store_true", help="skip if npz already exists")
    args = ap.parse_args()

    # 設定読み込み
    cfg = load_config(args.config) if args.config else SafetyConfig()
    if args.save_config:
        ensure_dir(os.path.dirname(args.save_config) or ".")
        save_config(args.save_config, cfg)
        print(f"[preprocess] wrote config: {args.save_config}")
        return

    ensure_dir(args.out_dir)

    # YOLOモデル読み込み
    pose_model_path = str(cfg_get(cfg, "yolo_pose_model", "yolov8n-pose.pt"))
    print(f"[preprocess] loading YOLO pose model: {pose_model_path}")
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as e:
        raise SystemExit(
            "ultralytics が見つかりません。`pip install ultralytics` などを実行してください。"
        ) from e
    yolo_pose_model = YOLO(pose_model_path)

    # 動画リスト取得
    vids = list_videos(args.video_dir)
    if not vids:
        raise SystemExit(f"No videos found in: {args.video_dir}")

    # 実行ループ
    npz_paths: List[str] = []
    for vp in vids:
        print(f"[preprocess] processing: {vp}")
        out_npz = preprocess_one_video(
            video_path=vp,
            out_dir=args.out_dir,
            yolo_pose_model=yolo_pose_model,
            cfg=cfg,
            skip_existing=args.skip_existing,
        )
        if out_npz:
            npz_paths.append(out_npz)

    if not npz_paths:
        raise SystemExit("No npz outputs were generated.")

    write_manifest(args.out_dir, npz_paths)
    print("[preprocess] done.")


if __name__ == "__main__":
    main()