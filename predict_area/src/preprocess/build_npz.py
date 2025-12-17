import os
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import SafetyConfig  # type: ignore[import-not-found]
from .io_utils import cfg_get, ensure_dir
from .labels import compute_soft_label_from_future_xz


def build_npz_from_video_buffers(
    frames_state: List[Dict[int, Dict[str, np.ndarray]]],
    frames_ego: List[Tuple[float, float]],
    fps: float,
    width: int,
    height: int,
    cfg: SafetyConfig,
    out_npz_path: str,
    video_path: str,
    frame_names: Optional[List[str]] = None,
    y_by_frame_name: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    eps = float(cfg_get(cfg, "eps", 1e-6))
    K = int(cfg_get(cfg, "K", 9))
    T = int(cfg_get(cfg, "T", 10))
    H = int(cfg_get(cfg, "H", 10))
    Nmax = int(cfg_get(cfg, "Nmax", 16))
    # 距離閾値（meters）
    D = float(cfg_get(cfg, "D", 3.0))

    x_min = float(cfg_get(cfg, "x_min", -2.0))
    x_max = float(cfg_get(cfg, "x_max", 2.0))
    alpha_depth = float(cfg_get(cfg, "alpha_depth", 1.0))
    gamma_time = float(cfg_get(cfg, "gamma_time", 1.0))
    beta_risk = float(cfg_get(cfg, "beta_risk", 1.0))

    use_pose_delta = bool(cfg_get(cfg, "use_pose_delta", True))
    use_ego_motion = bool(cfg_get(cfg, "use_ego_motion", True))
    ego_as_feature = bool(cfg_get(cfg, "ego_as_feature", True))
    ego_normalize = bool(cfg_get(cfg, "ego_normalize", True))

    dt = 1.0 / float(fps if fps > 0 else 30.0)

    J3 = None
    J = None
    for st in frames_state:
        if st:
            any_tid = next(iter(st.keys()))
            J3 = int(st[any_tid]["pose"].shape[0])
            J = int(st[any_tid]["conf"].shape[0])
            break
    if J3 is None or J is None:
        raise RuntimeError(f"No people detected in entire video: {video_path}")

    # base features (keep everything except vel(3) and speed(1)):
    # root(3) + dir(3) + dist(1) + ttc(1)
    base_dim = 3 + 3 + 1 + 1
    ego_dim = 2 if (use_ego_motion and ego_as_feature) else 0
    dpose_dim = J3 if use_pose_delta else 0
    F = base_dim + ego_dim + J3 + dpose_dim + J

    use_external_y = y_by_frame_name is not None
    # (type narrowing helpers for static analyzers)
    names: List[str] = []
    y_map: Dict[str, np.ndarray] = {}
    frames_xz: List[List[Tuple[float, float]]] = []
    if use_external_y:
        if frame_names is None or y_by_frame_name is None:
            raise ValueError("frame_names and y_by_frame_name are required together")
        names = frame_names
        y_map = y_by_frame_name
        if len(names) != len(frames_state):
            raise ValueError(
                f"frame_names length mismatch: names={len(names)} frames_state={len(frames_state)}"
            )
        # validate K
        any_name = next(iter(y_map.keys()))
        ext_k = int(np.asarray(y_map[any_name]).shape[0])
        if ext_k != K:
            # When teacher distributions are provided externally, K must match that distribution length.
            # For convenience, we automatically align cfg.K to ext_k so preprocess can proceed.
            print(
                f"[preprocess] warning: external y has K={ext_k} but cfg.K={K}. "
                "Auto-adjusting cfg.K to match external y."
            )
            try:
                setattr(cfg, "K", int(ext_k))
            except Exception:
                # cfg is expected to be mutable, but keep running even if not.
                pass
            K = int(ext_k)
    else:
        for st in frames_state:
            xz_list = []
            for _, d in st.items():
                root = d["root"]
                x = float(root[0])
                z = float(root[2])
                dist = float(np.linalg.norm(root))
                if dist <= D:
                    xz_list.append((x, z))
            frames_xz.append(xz_list)

    X_list, M_list, y_list = [], [], []
    num_frames = len(frames_state)

    if use_external_y:
        if num_frames < T:
            raise RuntimeError(f"Sequence too short for T. frames={num_frames} T={T} video={video_path}")
        t_end = num_frames
    else:
        if num_frames < (T + H):
            raise RuntimeError(f"Video too short for T/H. frames={num_frames} T={T} H={H} video={video_path}")
        t_end = num_frames - H

    for t in range(T - 1, t_end):
        if use_external_y:
            # (narrowed above)
            name = names[t]
            if name not in y_map:
                raise KeyError(f"External y missing for frame: {name}")
            y = np.asarray(y_map[name], dtype=np.float32)
            if y.shape != (K,):
                raise ValueError(f"Invalid y shape for {name}: {y.shape} expected ({K},)")
        else:
            future_people = [frames_xz[t + tau + 1] for tau in range(H)]
            y = compute_soft_label_from_future_xz(
                future_people, K, x_min, x_max, alpha_depth, gamma_time, beta_risk
            )

        cur = frames_state[t]
        cand = []
        for tid, d in cur.items():
            dist = float(np.linalg.norm(d["root"]))
            if dist <= D:
                cand.append((tid, dist))
        cand.sort(key=lambda x: x[1])
        ids = [tid for tid, _ in cand[:Nmax]]

        X = np.zeros((T, Nmax, F), dtype=np.float32)
        M = np.zeros((T, Nmax), dtype=np.bool_)

        for ti in range(T):
            ft = t - (T - 1 - ti)
            st = frames_state[ft]
            st_prev = frames_state[ft - 1] if ft - 1 >= 0 else {}

            ego_vx, ego_vy = (0.0, 0.0)
            if use_ego_motion and ft < len(frames_ego):
                ego_vx, ego_vy = frames_ego[ft]

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
                if feat_vec.shape[0] != F:
                    raise RuntimeError(f"Feature dim mismatch: got={feat_vec.shape[0]} expected={F}")

                X[ti, n, :] = feat_vec
                M[ti, n] = True

        X_list.append(X)
        M_list.append(M)
        y_list.append(y)

    X_arr = np.stack(X_list, axis=0)
    M_arr = np.stack(M_list, axis=0)
    y_arr = np.stack(y_list, axis=0)

    ensure_dir(os.path.dirname(out_npz_path))
    np.savez_compressed(
        out_npz_path,
        X=X_arr,
        M=M_arr,
        y=y_arr,
        video=np.array([video_path], dtype=object),
        fps=np.array([float(fps)], dtype=np.float32),
        cfg=np.array([asdict(cfg)], dtype=object),
    )

    print(f"[preprocess] saved npz: {out_npz_path}")
    print(f"[preprocess] X={X_arr.shape} M={M_arr.shape} y={y_arr.shape}")
