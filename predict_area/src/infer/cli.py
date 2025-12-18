import argparse

from ..config import SafetyConfig, load_config


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", default="", help="input video path")
    g.add_argument("--frames-dir", default="", help="input frames directory (images)")
    g.add_argument("--npz", default="", help="input npz path (features X/M)")
    ap.add_argument("--ckpt", required=True, help="trained checkpoint")
    ap.add_argument("--config", default="", help="config yaml path (same as preprocess/train)")
    ap.add_argument("--out-video", default="", help="optional output mp4 path with overlay")
    ap.add_argument("--out-csv", default="", help="optional output csv path with probabilities")
    ap.add_argument("--out-json", default="", help="optional output json path with probabilities (frame_name -> [K])")
    ap.add_argument("--show", action="store_true", help="show window (may not work on headless)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0=all)")
    ap.add_argument("--fps", type=float, default=0.0, help="fps override (useful for frames-dir)")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else SafetyConfig()
    try:
        from .model_io import load_safetynet
    except ModuleNotFoundError as e:
        raise SystemExit(
            "推論に必要な依存関係が見つかりません（torch など）。依存関係をインストールしてください。\n"
            "例: `uv sync`\n"
            f"詳細: {e}"
        ) from e

    model, in_dim, K, device = load_safetynet(args.ckpt, cfg)

    try:
        from .pipeline import run_inference, run_inference_npz
    except ModuleNotFoundError as e:
        raise SystemExit(
            "推論に必要な依存関係が見つかりません（torch/ultralytics/opencv-python など）。\n"
            "依存関係をインストールしてください。\n"
            "例: `uv sync` もしくは `pip install -r requirements.txt` 相当\n"
            f"詳細: {e}"
        ) from e

    if args.npz:
        # Feature-only inference
        run_inference_npz(
            npz_path=args.npz,
            ckpt_in_dim=in_dim,
            K=K,
            device=device,
            safety_model=model,
            out_csv=args.out_csv,
        )
        return

    run_inference(
        video_path=args.video,
        frames_dir=args.frames_dir,
        ckpt_in_dim=in_dim,
        K=K,
        cfg=cfg,
        device=device,
        safety_model=model,
        out_video=args.out_video,
        out_csv=args.out_csv,
        out_json=args.out_json,
        show=args.show,
        max_frames=args.max_frames,
        fps_override=float(args.fps),
    )


if __name__ == "__main__":
    main()
