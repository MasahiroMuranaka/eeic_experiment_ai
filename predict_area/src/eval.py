import os
import argparse
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class EvalSummary:
    n: int
    K: int
    eps: float
    mean_ce: float
    mean_kl: float
    mean_js: float
    mean_emd_bins: float
    mean_emd_x: float
    top1_acc: float
    mean_abs_bin_expect: float
    mean_abs_x_expect: float


def _normalize_dist(a: np.ndarray, eps: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"Expected 1D distribution, got shape={a.shape}")
    a = np.maximum(a, 0.0)
    s = float(a.sum())
    if not np.isfinite(s) or s <= 0:
        # fallback to uniform (avoids NaN cascade)
        return np.ones_like(a, dtype=np.float64) / float(a.shape[0])
    a = a / s
    # avoid log(0)
    a = np.clip(a, eps, 1.0)
    a = a / float(a.sum())
    return a


def _metrics_one(q: np.ndarray, p: np.ndarray, x_min: float, x_max: float, eps: float) -> Dict[str, float]:
    q = _normalize_dist(q, eps)
    p = _normalize_dist(p, eps)
    K = int(q.shape[0])
    if p.shape[0] != K:
        raise ValueError(f"K mismatch: q(K={K}) vs p(K={p.shape[0]})")

    ce = -float(np.sum(q * np.log(p)))
    kl = float(np.sum(q * (np.log(q) - np.log(p))))
    m = 0.5 * (q + p)
    js = 0.5 * float(np.sum(q * (np.log(q) - np.log(m)))) + 0.5 * float(np.sum(p * (np.log(p) - np.log(m))))

    # 1D EMD on bin index line: sum |cumsum(q - p)|
    cdf_diff = np.cumsum(q - p)
    emd_bins = float(np.sum(np.abs(cdf_diff)))

    # translate to x-distance using bin width dx
    dx = float(x_max - x_min) / float(K)
    emd_x = emd_bins * dx

    kq = int(np.argmax(q))
    kp = int(np.argmax(p))
    top1 = 1.0 if kq == kp else 0.0

    ks = np.arange(K, dtype=np.float64)
    eq = float(np.sum(q * ks))
    ep = float(np.sum(p * ks))
    abs_bin_expect = float(abs(eq - ep))

    centers = (x_min + (ks + 0.5) * dx).astype(np.float64)
    ex_q = float(np.sum(q * centers))
    ex_p = float(np.sum(p * centers))
    abs_x_expect = float(abs(ex_q - ex_p))

    return {
        "ce": ce,
        "kl": kl,
        "js": js,
        "emd_bins": emd_bins,
        "emd_x": emd_x,
        "top1": top1,
        "abs_bin_expect": abs_bin_expect,
        "abs_x_expect": abs_x_expect,
    }


def _aggregate_metrics(
    pairs: Iterable[Tuple[np.ndarray, np.ndarray]],
    x_min: float,
    x_max: float,
    eps: float,
    K_hint: Optional[int] = None,
) -> EvalSummary:
    n = 0
    K = K_hint
    acc = {
        "ce": 0.0,
        "kl": 0.0,
        "js": 0.0,
        "emd_bins": 0.0,
        "emd_x": 0.0,
        "top1": 0.0,
        "abs_bin_expect": 0.0,
        "abs_x_expect": 0.0,
    }
    for q, p in pairs:
        if K is None:
            K = int(np.asarray(q).shape[0])
        m = _metrics_one(q, p, x_min=x_min, x_max=x_max, eps=eps)
        for k in acc.keys():
            acc[k] += float(m[k])
        n += 1

    if n <= 0 or K is None:
        raise ValueError("No samples to evaluate.")

    return EvalSummary(
        n=int(n),
        K=int(K),
        eps=float(eps),
        mean_ce=float(acc["ce"] / n),
        mean_kl=float(acc["kl"] / n),
        mean_js=float(acc["js"] / n),
        mean_emd_bins=float(acc["emd_bins"] / n),
        mean_emd_x=float(acc["emd_x"] / n),
        top1_acc=float(acc["top1"] / n),
        mean_abs_bin_expect=float(acc["abs_bin_expect"] / n),
        mean_abs_x_expect=float(acc["abs_x_expect"] / n),
    )


def _read_pred_csv(pred_csv: str) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    with open(pred_csv, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        if not header or len(header) < 2 or header[0].strip().lower() != "frame":
            raise ValueError(f"Invalid pred csv header: {pred_csv}")
        for row in r:
            if not row:
                continue
            frame_i = int(row[0])
            probs = np.asarray([float(x) for x in row[1:]], dtype=np.float32)
            out[frame_i] = probs
    if not out:
        raise ValueError(f"No predictions found in: {pred_csv}")
    return out


def _read_gt_json(gt_json: str) -> Dict[str, np.ndarray]:
    from .preprocess.external_json import load_prob_dist_json

    return load_prob_dist_json(gt_json)


def _index_to_frame_name(frames_dir: str) -> List[str]:
    # reuse the project's natural sort to match preprocessing
    from .preprocess.frame_source import list_frame_paths

    paths = list_frame_paths(frames_dir)
    return [os.path.basename(p) for p in paths]


def eval_from_csv_and_gt_json(
    gt_json: str,
    pred_csv: str,
    frames_dir: str,
    x_min: float,
    x_max: float,
    eps: float,
) -> EvalSummary:
    gt = _read_gt_json(gt_json)  # frame_name -> q[K]
    pred = _read_pred_csv(pred_csv)  # frame_idx -> p[K]
    names = _index_to_frame_name(frames_dir)

    def _pairs():
        for frame_i, p in pred.items():
            if frame_i < 0 or frame_i >= len(names):
                continue
            name = names[frame_i]
            if name not in gt:
                continue
            yield gt[name], p

    # infer K from any matching sample
    any_q = next(iter(gt.values()))
    return _aggregate_metrics(_pairs(), x_min=x_min, x_max=x_max, eps=eps, K_hint=int(any_q.shape[0]))


def eval_from_npz_with_model(
    ckpt: str,
    npz_paths: List[str],
    config_path: str,
    batch_size: int,
    num_workers: int,
    x_min: float,
    x_max: float,
    eps: float,
) -> EvalSummary:
    from .config import SafetyConfig, load_config
    from .dataset import MultiNpzSafetyDataset
    from .infer.model_io import load_safetynet, load_safetynet2

    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader  # type: ignore[import-not-found]

    cfg = load_config(config_path) if config_path else SafetyConfig()
    # model, in_dim, K, device = load_safetynet(ckpt, cfg)
    model, in_dim, K, device = load_safetynet2(ckpt, cfg)

    ds = MultiNpzSafetyDataset(npz_paths)
    assert ds.K is not None
    if int(ds.K) != int(K):
        raise ValueError(f"K mismatch: dataset K={ds.K} vs ckpt K={K}")

    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=False)
    model.eval()

    # streaming aggregation in torch
    n = 0
    sum_ce = 0.0
    sum_kl = 0.0
    sum_js = 0.0
    sum_emd_bins = 0.0
    sum_emd_x = 0.0
    sum_top1 = 0.0
    sum_abs_bin_expect = 0.0
    sum_abs_x_expect = 0.0

    dx = float(x_max - x_min) / float(K)
    ks = torch.arange(K, device=device, dtype=torch.float32)
    centers = (float(x_min) + (ks + 0.5) * float(dx)).to(device)

    with torch.no_grad():
        for X, M, y in dl:
            X = X.to(device)
            M = M.to(device)
            q = y.to(device)  # [B,K]
            p = model.predict_proba(X, M)  # [B,K]

            # normalize / clip
            q = torch.clamp(q, min=0.0)
            q = q / (q.sum(dim=-1, keepdim=True) + float(eps))
            q = torch.clamp(q, min=float(eps), max=1.0)
            q = q / (q.sum(dim=-1, keepdim=True) + float(eps))

            p = torch.clamp(p, min=float(eps), max=1.0)
            p = p / (p.sum(dim=-1, keepdim=True) + float(eps))

            logq = torch.log(q)
            logp = torch.log(p)

            ce = -(q * logp).sum(dim=-1)  # [B]
            kl = (q * (logq - logp)).sum(dim=-1)  # [B]
            m = 0.5 * (q + p)
            logm = torch.log(m)
            js = 0.5 * (q * (logq - logm)).sum(dim=-1) + 0.5 * (p * (logp - logm)).sum(dim=-1)

            cdf_diff = torch.cumsum(q - p, dim=-1)
            emd_bins = torch.abs(cdf_diff).sum(dim=-1)
            emd_x = emd_bins * float(dx)

            top1 = (torch.argmax(q, dim=-1) == torch.argmax(p, dim=-1)).float()

            eq = (q * ks).sum(dim=-1)
            ep = (p * ks).sum(dim=-1)
            abs_bin_expect = torch.abs(eq - ep)

            ex_q = (q * centers).sum(dim=-1)
            ex_p = (p * centers).sum(dim=-1)
            abs_x_expect = torch.abs(ex_q - ex_p)

            b = int(q.shape[0])
            n += b
            sum_ce += float(ce.sum().item())
            sum_kl += float(kl.sum().item())
            sum_js += float(js.sum().item())
            sum_emd_bins += float(emd_bins.sum().item())
            sum_emd_x += float(emd_x.sum().item())
            sum_top1 += float(top1.sum().item())
            sum_abs_bin_expect += float(abs_bin_expect.sum().item())
            sum_abs_x_expect += float(abs_x_expect.sum().item())

    if n <= 0:
        raise ValueError("No samples evaluated (empty dataloader?).")

    return EvalSummary(
        n=int(n),
        K=int(K),
        eps=float(eps),
        mean_ce=float(sum_ce / n),
        mean_kl=float(sum_kl / n),
        mean_js=float(sum_js / n),
        mean_emd_bins=float(sum_emd_bins / n),
        mean_emd_x=float(sum_emd_x / n),
        top1_acc=float(sum_top1 / n),
        mean_abs_bin_expect=float(sum_abs_bin_expect / n),
        mean_abs_x_expect=float(sum_abs_x_expect / n),
    )


def _resolve_npz_paths(npz: str, npz_dir: str, manifest: str) -> List[str]:
    from .dataset import list_npz_in_dir, read_manifest

    if npz:
        return [npz]
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
    raise SystemExit("one of --npz / --npz-dir / --manifest is required")


def main():
    ap = argparse.ArgumentParser(description="Evaluate predict_area model outputs as probability distributions.")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--ckpt",
        default="",
        help="(mode A) checkpoint path; evaluates directly on npz (teacher y in npz).",
    )
    mode.add_argument(
        "--pred-csv",
        default="",
        help="(mode B) prediction csv (frame,p0..pK-1). Requires --gt-json and --frames-dir.",
    )

    # mode A inputs
    ap.add_argument("--npz", default="", help="single npz path")
    ap.add_argument("--npz-dir", default="", help="directory containing npz (recursive)")
    ap.add_argument("--manifest", default="", help="manifest.txt that lists npz paths")
    ap.add_argument("--config", default="", help="config yaml path (same as preprocess/train)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)

    # mode B inputs
    ap.add_argument("--gt-json", default="", help="teacher distribution json (frame_name -> [K])")
    ap.add_argument("--frames-dir", default="", help="frames directory to map csv frame index -> frame_name")

    # common knobs
    ap.add_argument("--x-min", type=float, default=-2.0, help="bin x_min (for x-based metrics)")
    ap.add_argument("--x-max", type=float, default=2.0, help="bin x_max (for x-based metrics)")
    ap.add_argument("--eps", type=float, default=1e-6, help="numerical epsilon for logs/normalization")
    ap.add_argument("--out-json", default="", help="optional output json path for summary")
    args = ap.parse_args()

    if args.ckpt:
        npz_paths = _resolve_npz_paths(args.npz, args.npz_dir, args.manifest)
        summary = eval_from_npz_with_model(
            ckpt=args.ckpt,
            npz_paths=npz_paths,
            config_path=args.config,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            x_min=float(args.x_min),
            x_max=float(args.x_max),
            eps=float(args.eps),
        )
    else:
        if not args.gt_json or not args.frames_dir:
            raise SystemExit("--pred-csv mode requires --gt-json and --frames-dir")
        summary = eval_from_csv_and_gt_json(
            gt_json=args.gt_json,
            pred_csv=args.pred_csv,
            frames_dir=args.frames_dir,
            x_min=float(args.x_min),
            x_max=float(args.x_max),
            eps=float(args.eps),
        )

    d = asdict(summary)
    print("[eval] summary")
    for k, v in d.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        import json

        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        print(f"[eval] wrote: {args.out_json}")


if __name__ == "__main__":
    main()
