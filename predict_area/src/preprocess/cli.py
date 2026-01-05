import os
import sys
import argparse
from typing import List

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import SafetyConfig, load_config, save_config  # noqa: E402
from preprocess.io_utils import cfg_get, ensure_dir, list_videos  # noqa: E402


def write_manifest(out_dir: str, npz_paths: List[str]) -> str:
    manifest = os.path.join(out_dir, "manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for p in npz_paths:
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

    cfg = load_config(args.config) if args.config else SafetyConfig()
    if args.save_config:
        ensure_dir(os.path.dirname(args.save_config) or ".")
        save_config(args.save_config, cfg)
        print(f"[preprocess] wrote config: {args.save_config}")
        return

    ensure_dir(args.out_dir)

    pose_model_path = str(cfg_get(cfg, "yolo_pose_model", "yolov8n-pose.pt"))
    print(f"[preprocess] loading YOLO pose model: {pose_model_path}")
    try:
        from ultralytics import YOLO  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "ultralytics が見つかりません。依存関係をインストールしてください。\n"
            "例: `uv sync` もしくは `pip install ultralytics`\n"
            f"詳細: {e}"
        ) from e
    yolo_pose_model = YOLO(pose_model_path)

    try:
        from preprocess.pipeline import preprocess_one_video
    except ModuleNotFoundError as e:
        raise SystemExit(
            "前処理に必要な依存関係が見つかりません（opencv-python など）。依存関係をインストールしてください。\n"
            "例: `uv sync`\n"
            f"詳細: {e}"
        ) from e

    vids = list_videos(args.video_dir)
    if not vids:
        raise SystemExit(f"No videos found in: {args.video_dir}")

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
