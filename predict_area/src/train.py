
import argparse
import os
import platform
import random
import time
from typing import Iterable, List

import numpy as np
import torch  # type: ignore[import-not-found]
from torch.utils.data import DataLoader  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]

from .config import SafetyConfig, load_config, save_config  # type: ignore[import-not-found]
from .dataset import (  # type: ignore[import-not-found]
    FileGroupedSampler,
    MultiNpzSafetyDataset,
    list_npz_in_dir,
    read_manifest,
)
from .model import SafetyNet, SafetyNet2  # type: ignore[import-not-found]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_ce_loss(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # q: [B,K], logits: [B,K]
    p = torch.softmax(logits, dim=-1)
    p = p.clamp(min=1e-8)
    q = q.clamp(min=1e-8)
    m = (p + q) * 0.5

    logm = torch.log(m)
    logp = torch.log(p)
    logq = torch.log(q)

    loss = 0.5 * (q * (logq - logm)).sum(dim=-1) + 0.5 * (p * (logp - logm)).sum(dim=-1)
    return loss.mean()
    # return -(q * logp).sum(dim=-1).mean()


def _unique_keep_order(xs: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def _norm_path(p: str) -> str:
    return os.path.abspath(os.path.expanduser(p))


def _load_state_dict_from_input_ckpt(path: str) -> dict:
    """
    Accept either:
      - a plain PyTorch state_dict (mapping param_name -> tensor), OR
      - a training checkpoint dict saved by this repo (with key "model_state").
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj and isinstance(obj["model_state"], dict):
        return obj["model_state"]
    if isinstance(obj, dict):
        # assume it's already a state_dict
        return obj
    raise SystemExit(f"--input-ckpt must be a state_dict or a dict with 'model_state': {path}")


def _expand_npz_inputs(inputs: List[str]) -> List[str]:
    """
    Accept a mix of:
      - .npz file paths
      - directories (recursively expanded to .npz files)
    """
    out: List[str] = []
    for raw in inputs:
        if not raw:
            continue
        p = _norm_path(raw)
        if os.path.isdir(p):
            out.extend(list_npz_in_dir(p))
            continue
        if not os.path.exists(p):
            raise SystemExit(f"path not found: {raw}")
        if not p.lower().endswith(".npz"):
            raise SystemExit(f"not an .npz file: {raw}")
        out.append(p)
    return out


def resolve_npz_paths(train_npz: List[str], npz_dir: List[str], manifest: str) -> List[str]:
    if train_npz:
        paths = _expand_npz_inputs(train_npz)
        paths = _unique_keep_order(paths)
        if not paths:
            raise SystemExit("--train-npz resolved to empty list")
        return paths
    if manifest:
        paths = [_norm_path(p) for p in read_manifest(manifest)]
        if not paths:
            raise SystemExit(f"manifest is empty: {manifest}")
        return _unique_keep_order(paths)
    if npz_dir:
        paths = _expand_npz_inputs(npz_dir)
        paths = _unique_keep_order(paths)
        if not paths:
            raise SystemExit(f"no npz found in: {npz_dir}")
        return paths
    raise SystemExit("one of --train-npz / --manifest / --npz-dir is required")


def default_num_workers() -> int:
    # macOS: DataLoader multi-process is often unstable in user environments.
    if platform.system().lower() == "darwin":
        return 0
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train-npz",
        nargs="+",
        default=[],
        help="one or more .npz paths (can also pass a directory to expand recursively)",
    )
    ap.add_argument(
        "--npz-dir",
        nargs="+",
        default=[],
        help="one or more directories containing multiple npz (recursive); .npz files are also accepted",
    )
    ap.add_argument("--manifest", default="", help="manifest.txt that lists npz paths (one per line)")

    ap.add_argument("--config", default="")
    ap.add_argument("--out-ckpt", default="safetynet.pt")
    ap.add_argument("--save-config", default="")
    ap.add_argument(
        "--temporal-model",
        default="",
        choices=["gru", "transformer"],
        help="override cfg.temporal_model (if not set, uses config/default)",
    )

    # loader knobs
    ap.add_argument("--num-workers", type=int, default=-1, help="default: macOS=0, otherwise=2")
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--no-pin-memory", action="store_true", help="force disable pin_memory")
    ap.add_argument(
        "--group-by-file",
        action="store_true",
        help="use a sampler that keeps samples from the same .npz together (recommended for .npz-compressed datasets)",
    )
    ap.add_argument(
        "--no-group-by-file",
        action="store_true",
        help="disable file-grouped sampling and use DataLoader(shuffle=True) instead",
    )
    ap.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="print training progress every N steps (0 to disable)",
    )
    ap.add_argument(
        "--sync-timing",
        action="store_true",
        help="synchronize CUDA for more accurate timing logs (slower; useful for debugging GPU usage)",
    )
    ap.add_argument(
        "--input-ckpt",
        default=None,
        help=(
            "追加学習用の初期重み。"
            "state_dict（param名->tensor）または、このtrain.pyが保存した ckpt（'model_state' を含むdict）のパス。"
        ),
    )

    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else SafetyConfig()
    if args.temporal_model:
        cfg.temporal_model = str(args.temporal_model)
    if args.save_config:
        save_config(args.save_config, cfg)
        print(f"[train] wrote config: {args.save_config}")

    set_seed(cfg.seed)

    # resolve inputs
    npz_paths = resolve_npz_paths(args.train_npz, args.npz_dir, args.manifest)
    print(f"[train] npz files: {len(npz_paths)}")
    for p in npz_paths[:10]:
        print(f"  - {p}")
    if len(npz_paths) > 10:
        print("  ...")

    # build dataset (lazy load per-npz)
    ds = MultiNpzSafetyDataset(npz_paths)
    print(f"[train] dataset ready: total={len(ds)} T={ds.T} Nmax={ds.Nmax} F={ds.F} K={ds.K}")

    # DataLoader defaults
    nw = default_num_workers() if args.num_workers < 0 else args.num_workers

    if args.no_pin_memory:
        pin = False
    else:
        # if user explicitly set --pin-memory, honor it; otherwise default True when CUDA is available
        pin = bool(args.pin_memory) if args.pin_memory else bool(torch.cuda.is_available())

    # default: enable file-grouped sampling (better locality for np.savez_compressed datasets)
    use_group = bool(args.group_by_file) if (args.group_by_file or args.no_group_by_file) else True
    if args.no_group_by_file:
        use_group = False

    sampler = FileGroupedSampler(ds, seed=cfg.seed) if use_group else None

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=(nw > 0),
    )
    print(
        f"[train] dataloader ready: batch={cfg.batch_size} num_workers={nw} pin_memory={pin} "
        f"group_by_file={sampler is not None}"
    )

    assert ds.F is not None and ds.K is not None
    in_dim = int(ds.F)
    K = int(ds.K)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown"
        print(
            f"[train] torch={torch.__version__} torch_cuda={getattr(torch.version, 'cuda', None)} "
            f"cuda_available=True device={device} gpu0={name}"
        )
    else:
        print(f"[train] torch={torch.__version__} cuda_available=False device={device}")
    model_kwargs = {
        "emb_dim": int(getattr(cfg, "emb_dim", 128)),
        "temporal": str(getattr(cfg, "temporal_model", "gru")),
        "rnn_hidden": int(getattr(cfg, "rnn_hidden", 256)),
        "rnn_layers": int(getattr(cfg, "rnn_layers", 2)),
        "tf_layers": int(getattr(cfg, "tf_layers", 2)),
        "tf_nhead": int(getattr(cfg, "tf_nhead", 4)),
        "tf_ff": int(getattr(cfg, "tf_ff", 512)),
        "tf_dropout": float(getattr(cfg, "tf_dropout", 0.1)),
        "tf_norm_first": bool(getattr(cfg, "tf_norm_first", True)),
    }

    if args.input_ckpt is None:
        model = SafetyNet(in_dim=in_dim, K=K, **model_kwargs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        state_dict = _load_state_dict_from_input_ckpt(args.input_ckpt)
        model = SafetyNet2(in_dim=in_dim, K=K, **model_kwargs).to(device)
        # residual calibration should start from "do nothing" if the ckpt doesn't have it.
        # (If the ckpt *does* include residual_layer params, load_state_dict will overwrite them.)
        for m in model.residual_layer:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("[train] input-ckpt: missing keys (ignored):")
            for k in missing[:50]:
                print(f"  - {k}")
            if len(missing) > 50:
                print("  ...")
        if unexpected:
            print("[train] input-ckpt: unexpected keys (ignored):")
            for k in unexpected[:50]:
                print(f"  - {k}")
            if len(unexpected) > 50:
                print("  ...")

        # Keep optimizer robust across temporal modes ("gru" vs "transformer"/st_transformer).
        # If you want fine-tuning with per-module LRs, add a dedicated flag later.
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = []
        steps_per_epoch = len(dl)
        print(f"[train] epoch {epoch:03d}/{cfg.epochs} start (steps={steps_per_epoch})")

        if sampler is not None:
            sampler.set_epoch(epoch)

        t_prev = time.perf_counter()
        data_s, step_s = 0.0, 0.0

        for step, (X, M, y) in enumerate(dl, start=1):
            t0 = time.perf_counter()
            data_s += (t0 - t_prev)

            X = X.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if step == 1:
                try:
                    pdev = next(model.parameters()).device
                except StopIteration:
                    pdev = torch.device("cpu")
                print(f"[train] debug devices: X={X.device} M={M.device} y={y.device} model={pdev}")

            _, logits = model(X, M)
            loss = soft_ce_loss(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if args.sync_timing and torch.cuda.is_available():
                torch.cuda.synchronize()

            loss_val = float(loss.item())
            running.append(loss_val)

            t1 = time.perf_counter()
            step_s += (t1 - t0)
            t_prev = t1

            if args.log_interval > 0 and (step % args.log_interval == 0 or step == 1):
                # moving average over recent interval (or fewer at the beginning)
                w = min(len(running), args.log_interval)
                recent_mean = float(np.mean(running[-w:]))
                mean_data = (data_s / float(step)) if step > 0 else 0.0
                mean_step = (step_s / float(step)) if step > 0 else 0.0
                print(
                    f"[train] epoch {epoch:03d} step {step:06d}/{steps_per_epoch} "
                    f"loss={loss_val:.6f} mean{w}={recent_mean:.6f} "
                    f"time(data/step)={mean_data:.4f}s time(step)={mean_step:.4f}s"
                )

        print(f"[train] epoch {epoch:03d} mean_loss={np.mean(running):.6f}")

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg.__dict__,
            "in_dim": in_dim,
            "K": K,
            "model_kwargs": model_kwargs,
            "npz_files": npz_paths,
        },
        args.out_ckpt,
    )
    print(f"[train] saved ckpt: {args.out_ckpt}")


if __name__ == "__main__":
    main()
