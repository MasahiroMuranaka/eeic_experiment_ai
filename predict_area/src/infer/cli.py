import os
import sys
import argparse

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from config import SafetyConfig, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--ckpt", required=True, help="trained checkpoint")
    ap.add_argument("--config", default="", help="config yaml path (same as preprocess/train)")
    ap.add_argument("--out-video", default="", help="optional output mp4 path with overlay")
    ap.add_argument("--out-csv", default="", help="optional output csv path with probabilities")
    ap.add_argument("--show", action="store_true", help="show window (may not work on headless)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=all)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else SafetyConfig()
    try:
        from infer.model_io import load_safetynet
    except ModuleNotFoundError as e:
        raise SystemExit(
            "推論に必要な依存関係が見つかりません（torch など）。依存関係をインストールしてください。\n"
            "例: `uv sync`\n"
            f"詳細: {e}"
        ) from e

    model, in_dim, K, device = load_safetynet(args.ckpt, cfg)

    try:
        from infer.pipeline import run_inference
    except ModuleNotFoundError as e:
        raise SystemExit(
            "推論に必要な依存関係が見つかりません（torch/ultralytics/opencv-python など）。\n"
            "依存関係をインストールしてください。\n"
            "例: `uv sync` もしくは `pip install -r requirements.txt` 相当\n"
            f"詳細: {e}"
        ) from e

    run_inference(
        video_path=args.video,
        ckpt_in_dim=in_dim,
        K=K,
        cfg=cfg,
        device=device,
        safety_model=model,
        out_video=args.out_video,
        out_csv=args.out_csv,
        show=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
