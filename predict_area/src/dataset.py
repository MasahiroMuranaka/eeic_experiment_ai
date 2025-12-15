import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiNpzSafetyDataset(Dataset):
    """
    Multiple .npz dataset (X, M, y) with lazy loading.

    Each npz must contain:
      X: [N, T, Nmax, F] float32
      M: [N, T, Nmax] bool
      y: [N, K] float32
    """

    def __init__(self, npz_paths: List[str]):
        if not npz_paths:
            raise ValueError("npz_paths is empty")

        self.npz_paths = npz_paths

        # lengths per file, and global prefix sums
        self.lengths: List[int] = []
        self.prefix: List[int] = [0]

        # cache (last opened file)
        self._cache_idx: Optional[int] = None
        self._cache_X = None
        self._cache_M = None
        self._cache_y = None

        # check consistency (T, Nmax, F, K)
        self.T = None
        self.Nmax = None
        self.F = None
        self.K = None

        for p in self.npz_paths:
            z = np.load(p, allow_pickle=True)
            X = z["X"]
            M = z["M"]
            y = z["y"]

            if X.ndim != 4 or M.ndim != 3 or y.ndim != 2:
                raise ValueError(f"invalid shapes in {p}: X{X.shape} M{M.shape} y{y.shape}")

            n = int(X.shape[0])
            self.lengths.append(n)
            self.prefix.append(self.prefix[-1] + n)

            T, Nmax, F = X.shape[1], X.shape[2], X.shape[3]
            K = y.shape[1]

            if self.T is None:
                self.T, self.Nmax, self.F, self.K = T, Nmax, F, K
            else:
                if (T, Nmax, F, K) != (self.T, self.Nmax, self.F, self.K):
                    raise ValueError(
                        "inconsistent dims across npz files.\n"
                        f"  first: T={self.T} Nmax={self.Nmax} F={self.F} K={self.K}\n"
                        f"  this : T={T} Nmax={Nmax} F={F} K={K}\n"
                        f"  file : {p}"
                    )

        self.total = self.prefix[-1]

    def __len__(self) -> int:
        return self.total

    def _locate(self, idx: int) -> Tuple[int, int]:
        # binary search over prefix sums
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        lo, hi = 0, len(self.lengths) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.prefix[mid]
            end = self.prefix[mid + 1]
            if start <= idx < end:
                return mid, idx - start
            if idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise RuntimeError("locate failed")

    def _load_file(self, file_i: int):
        if self._cache_idx == file_i:
            return
        p = self.npz_paths[file_i]
        z = np.load(p, allow_pickle=True)
        self._cache_X = z["X"].astype(np.float32, copy=False)
        self._cache_M = z["M"].astype(np.bool_, copy=False)
        self._cache_y = z["y"].astype(np.float32, copy=False)
        self._cache_idx = file_i

    def __getitem__(self, idx: int):
        file_i, local_i = self._locate(idx)
        self._load_file(file_i)

        X = torch.from_numpy(self._cache_X[local_i])  # [T,Nmax,F]
        M = torch.from_numpy(self._cache_M[local_i])  # [T,Nmax]
        y = torch.from_numpy(self._cache_y[local_i])  # [K]
        return X, M, y


def list_npz_in_dir(npz_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(npz_dir):
        for fn in files:
            if fn.lower().endswith(".npz"):
                paths.append(os.path.join(root, fn))
    paths.sort()
    return paths


def read_manifest(manifest_path: str) -> List[str]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    paths = [ln for ln in lines if ln and not ln.startswith("#")]
    return paths
