from typing import Dict, List, Tuple
import numpy as np

from config import SafetyConfig, load_config
from preprocess.io_utils import cfg_get


def build_feature_tensor(
    frames_state: List[Dict[int, Dict[str, np.ndarray]]],  # length T
    frames_ego: List[Tuple[float, float]],                 # length T
    fps: float,
    width: int,
    height: int,
    cfg: SafetyConfig,
    in_dim_expected: int,
) -> Tuple[np.ndarray, np.ndarray]:
    eps = float(cfg_get(cfg, "eps", 1e-6))
    T = int(cfg_get(cfg, "T", 10))
    Nmax = int(cfg_get(cfg, "Nmax", 16))
    # 距離閾値（meters）
    D = float(cfg_get(cfg, "D", 3.0))

    use_pose_delta = bool(cfg_get(cfg, "use_pose_delta", True))
    use_ego_motion = bool(cfg_get(cfg, "use_ego_motion", True))
    ego_as_feature = bool(cfg_get(cfg, "ego_as_feature", True))
    ego_normalize = bool(cfg_get(cfg, "ego_normalize", True))

    dt = 1.0 / float(fps if fps > 0 else 30.0)

    # Determine J3/J (fallback to COCO17)
    J3 = None
    J = None
    for st in reversed(frames_state):
        if st:
            any_tid = next(iter(st.keys()))
            J3 = int(st[any_tid]["pose"].shape[0])
            J = int(st[any_tid]["conf"].shape[0])
            break
    if J3 is None or J is None:
        J = 17
        J3 = 17 * 3

    # base features (keep everything except vel(3) and speed(1)):
    # root(3) + dir(3) + dist(1) + ttc(1)
    base_dim = 3 + 3 + 1 + 1
    ego_dim = 2 if (use_ego_motion and ego_as_feature) else 0
    dpose_dim = J3 if use_pose_delta else 0
    F = base_dim + ego_dim + J3 + dpose_dim + J

    if F != int(in_dim_expected):
        raise RuntimeError(
            f"Feature dim mismatch: built F={F} but ckpt expects in_dim={in_dim_expected}. "
            "Run preprocess/train with identical config flags."
        )

    # Choose ids based on current frame distance within D
    cur = frames_state[-1]
    cand = []
    for tid, d in cur.items():
        dist = float(np.linalg.norm(d["root"]))
        if dist <= D:
            cand.append((tid, dist))
    cand.sort(key=lambda x: x[1])
    ids = [tid for tid, _ in cand[:Nmax]]

    X = np.zeros((1, T, Nmax, F), dtype=np.float32)
    M = np.zeros((1, T, Nmax), dtype=np.bool_)

    for ti in range(T):
        st = frames_state[ti]
        st_prev = frames_state[ti - 1] if ti - 1 >= 0 else {}

        ego_vx, ego_vy = (0.0, 0.0)
        if use_ego_motion and ti < len(frames_ego):
            ego_vx, ego_vy = frames_ego[ti]

        if ego_normalize:
            ego_vx = (float(ego_vx) / (float(width) + eps)) * float(fps)
            ego_vy = (float(ego_vy) / (float(height) + eps)) * float(fps)

        ego_feat = (
            np.array([ego_vx, ego_vy], dtype=np.float32)
            if (use_ego_motion and ego_as_feature)
            else np.zeros((0,), dtype=np.float32)
        )

        for n, tid in enumerate(ids):
            if tid not in st:
                continue

            root = st[tid]["root"].astype(np.float32)
            pose = st[tid]["pose"].astype(np.float32)
            conf = st[tid]["conf"].astype(np.float32)

            if tid in st_prev:
                root_prev = st_prev[tid]["root"].astype(np.float32)
                v3 = (root - root_prev) / float(dt)
                if use_pose_delta:
                    pose_prev = st_prev[tid]["pose"].astype(np.float32)
                    dpose = (pose - pose_prev).astype(np.float32)
                else:
                    dpose = np.zeros((0,), dtype=np.float32)
            else:
                v3 = np.zeros((3,), dtype=np.float32)
                dpose = np.zeros((J3,), dtype=np.float32) if use_pose_delta else np.zeros((0,), dtype=np.float32)

            speed = float(np.linalg.norm(v3))
            dvec = (v3 / (speed + 1e-6)).astype(np.float32) if speed > 1e-6 else np.zeros((3,), dtype=np.float32)

            dist = float(np.linalg.norm(root))
            denom = float(speed * speed + 1e-6)
            ttc = float(-np.dot(root, v3) / denom)
            if ttc < 0:
                ttc = 999.0

            feat_vec = np.concatenate(
                [
                    root,
                    dvec,
                    np.array([dist], dtype=np.float32),
                    np.array([ttc], dtype=np.float32),
                    ego_feat,
                    pose,
                    dpose,
                    conf,
                ],
                axis=0,
            )
            X[0, ti, n, :] = feat_vec
            M[0, ti, n] = True

    return X, M
