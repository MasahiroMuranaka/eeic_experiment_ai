from typing import List, Tuple
import numpy as np


def bin_index(x: float, x_min: float, x_max: float, K: int) -> int:
    if x <= x_min:
        return 0
    if x >= x_max:
        return K - 1
    t = (x - x_min) / (x_max - x_min)
    k = int(np.floor(t * K))
    return max(0, min(K - 1, k))


def compute_soft_label_from_future_xz(
    future_people_xz: List[List[Tuple[float, float]]],  # tau -> list[(x,z)]
    K: int,
    x_min: float,
    x_max: float,
    alpha_depth: float,
    gamma_time: float,
    beta_risk: float,
) -> np.ndarray:
    R = np.zeros((K,), dtype=np.float32)
    for tau, xz_list in enumerate(future_people_xz):
        w_t = (float(gamma_time) ** float(tau))
        for (x, z) in xz_list:
            w_z = float(np.exp(-float(alpha_depth) * float(z)))
            k = bin_index(float(x), float(x_min), float(x_max), int(K))
            R[k] += float(w_t * w_z)

    S = np.exp(-float(beta_risk) * R)
    q = S / (S.sum() + 1e-6)
    return q.astype(np.float32)
