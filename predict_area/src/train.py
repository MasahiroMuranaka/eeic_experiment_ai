import argparse
import os
import platform
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import SafetyConfig, load_config, save_config
from dataset import MultiNpzSafetyDataset, list_npz_in_dir, read_manifest
from model import SafetyNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_ce_loss(logits: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # q: [B,K], logits: [B,K]
    logp = torch.log_softmax(logits, dim=-1)
    return -(q * logp).sum(dim=-1).mean()


def resolve_npz_paths(train_npz: str, npz_dir: str, manifest: str) -> List[str]:
    if train_npz:
        return [train_npz]
    if manifest:
        paths = read_manifest(manifest)
        if not paths:
            raise SystemExit(f"manifest is empty: {manifest}")
        return paths
    if npz_dir:
        paths = list_npz_in_dir(npz_dir)
        if not paths:
            raise SystemExit(f"no npz found in dir: {npz_dir}")
        return paths
    raise SystemExit("one of --train-npz / --manifest / --npz-dir is required")


def default_num_workers() -> int:
    # macOS: DataLoader multi-process is often unstable in user environments.
    if platform.system().lower() == "darwin":
        return 0
    return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", default="", help="single npz path (legacy)")
    ap.add_argument("--npz-dir", default="", help="directory containing multiple npz (recursive)")
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

    in_dim = int(ds.F)
    K = int(ds.K)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = SafetyNet(in_dim=in_dim, K=K).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = []

        pbar = tqdm(dl, desc=f"epoch {epoch:03d}/{cfg.epochs}", leave=True)
        for step, (X, M, y) in enumerate(pbar, start=1):
            X = X.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            _, logits = model(X, M)
            loss = soft_ce_loss(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            l = float(loss.item())
            running.append(l)

            # show batch loss and epoch mean
            pbar.set_postfix(loss=f"{l:.4f}", mean=f"{np.mean(running):.4f}")

        print(f"[train] epoch {epoch:03d} mean_loss={np.mean(running):.6f}")

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": cfg.__dict__,
            "in_dim": in_dim,
            "K": K,
            "npz_files": npz_paths,
        },
        args.out_ckpt,
    )
    print(f"[train] saved ckpt: {args.out_ckpt}")


if __name__ == "__main__":
    main()
