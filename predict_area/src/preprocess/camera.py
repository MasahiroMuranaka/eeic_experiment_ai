from typing import Tuple
import numpy as np


def camera_intrinsics_from_fov(h: int, w: int, fov_y_deg: float) -> Tuple[float, float, float, float]:
    fov_y = np.deg2rad(float(fov_y_deg))
    fy = (h / 2.0) / np.tan(fov_y / 2.0)
    fx = fy * (w / float(h))
    cx = w / 2.0
    cy = h / 2.0
    return float(fx), float(fy), float(cx), float(cy)


def estimate_root_xyz_from_bbox(
    box_xyxy: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    assumed_person_height_m: float,
    eps: float = 1e-6,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    u = (x1 + x2) * 0.5
    v = (y1 + y2) * 0.5
    h_px = max((y2 - y1), 1.0)

    z = fy * float(assumed_person_height_m) / (h_px + eps)
    x = (u - cx) / (fx + eps) * z
    y = (v - cy) / (fy + eps) * z
    return np.array([x, y, z], dtype=np.float32)


def pseudo3d_pose_from_keypoints(
    kpts_xy: np.ndarray,      # [J,2]
    kpts_conf: np.ndarray,    # [J]
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z: float,
    kp_conf_thresh: float,
):
    J = int(kpts_xy.shape[0])
    pose3d = np.zeros((J, 3), dtype=np.float32)
    conf = kpts_conf.astype(np.float32).copy()

    for j in range(J):
        if float(conf[j]) < float(kp_conf_thresh):
            continue
        uj = float(kpts_xy[j, 0])
        vj = float(kpts_xy[j, 1])
        pose3d[j, 2] = float(z)
        pose3d[j, 0] = (uj - cx) / float(fx) * float(z)
        pose3d[j, 1] = (vj - cy) / float(fy) * float(z)

    return pose3d.reshape(-1), conf
