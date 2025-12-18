import argparse
import os
import platform
import random
from typing import Iterable, List

import numpy as np
import torch  # type: ignore[import-not-found]
from torch.utils.data import DataLoader  # type: ignore[import-not-found]

from .config import SafetyConfig, load_config, save_config  # type: ignore[import-not-found]
from .dataset import MultiNpzSafetyDataset, list_npz_in_dir, read_manifest  # type: ignore[import-not-found]
from .model import SafetyNet  # type: ignore[import-not-found]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_ce_loss(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # q: [B,K], logits: [B,K]
    logp = torch.log_softmax(logits, dim=-1)
    return -(q * logp).sum(dim=-1).mean()


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

    # loader knobs
    ap.add_argument("--num-workers", type=int, default=-1, help="default: macOS=0, otherwise=2")
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--no-pin-memory", action="store_true", help="force disable pin_memory")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else SafetyConfig()
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
        # if user explicitly set --pin-memory, honor it; otherwise default False on CPU-only
        pin = bool(args.pin_memory)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
    )
    print(f"[train] dataloader ready: batch={cfg.batch_size} num_workers={nw} pin_memory={pin}")

    assert ds.F is not None and ds.K is not None
    in_dim = int(ds.F)
    K = int(ds.K)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
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
    model = SafetyNet(in_dim=in_dim, K=K, **model_kwargs).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = []

        for step, (X, M, y) in enumerate(dl, start=1):
            X = X.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X, M)
            loss = soft_ce_loss(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_val = float(loss.item())
            running.append(loss_val)

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
