from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..config import SafetyConfig
from .io_utils import cfg_get
from .camera import estimate_root_xyz_from_bbox


# Depth Anything V2 (official repo) model configs (README)
# encoder: vits / vitb / vitl / vitg
# input_size default = 518
# See: https://github.com/DepthAnything/Depth-Anything-V2 (Usage section)
# and depth_anything_v2/dpt.py (infer_image signature)
_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def _ensure_depth_anything_repo_on_path(use_metric_depth: bool) -> str:
    """
    同梱している `Depth-Anything-V2/` を import 可能にするために sys.path を調整する。
    - 事前に `DEPTH_ANYTHING_V2_REPO` を指定すればそちらを優先
    - use_metric_depth=True のときは `Depth-Anything-V2/metric_depth` を優先して import する
    """
    repo_dir = os.environ.get(
        "DEPTH_ANYTHING_V2_REPO",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Depth-Anything-V2")),
    )
    metric_dir = os.path.join(repo_dir, "metric_depth")

    # sys.path は先頭ほど優先されるので、metric_depth を使う場合は metric_dir を最優先にする
    if use_metric_depth and metric_dir not in sys.path:
        sys.path.insert(0, metric_dir)

    if repo_dir not in sys.path:
        # metric_dir が先頭にあるなら、その“次”に repo_dir を入れて優先順位を崩さない
        if use_metric_depth and len(sys.path) > 0 and sys.path[0] == metric_dir:
            sys.path.insert(1, repo_dir)
        else:
            sys.path.insert(0, repo_dir)
    return repo_dir


def _auto_device() -> torch.device:
    # DepthAnythingV2.image2tensor internally uses:
    # cuda -> mps -> cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _bbox_center_uv(box_xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _root_xyz_from_uvz(u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float, eps: float) -> np.ndarray:
    z = float(z_m)
    x = (float(u) - float(cx)) / (float(fx) + eps) * z
    y = (float(v) - float(cy)) / (float(fy) + eps) * z
    return np.array([x, y, z], dtype=np.float32)


def _median_depth_in_bbox(depth_map: np.ndarray, box_xyxy: np.ndarray) -> float:
    if depth_map.size == 0:
        return float("nan")
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    xi1 = int(max(0, min(w - 1, np.floor(x1))))
    yi1 = int(max(0, min(h - 1, np.floor(y1))))
    xi2 = int(max(0, min(w - 1, np.ceil(x2))))
    yi2 = int(max(0, min(h - 1, np.ceil(y2))))
    if xi2 <= xi1 or yi2 <= yi1:
        return float("nan")
    patch = depth_map[yi1:yi2, xi1:xi2]
    if patch.size == 0:
        return float("nan")
    patch = patch[np.isfinite(patch)]
    if patch.size == 0:
        return float("nan")
    return float(np.median(patch))


@dataclass
class DepthCalibState:
    # z_m ≈ scale * depth_value   (or scale * inv_depth_value)
    use_inverse: bool = False
    scale_ema: float = 1.0


class DepthAnythingV2DepthEstimator:
    """
    Depth Anything V2 adapter:
      - depth_map = model.infer_image(frame_bgr, input_size=518)  -> HxW float (relative or metric depending on ckpt)
      - optionally calibrate relative depth to meters using bbox heuristic z (EMA scale)
    """

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.device = _auto_device()

        # depth_mode で "metric" を指定できるようにする（既存の "midas" 互換も維持）
        depth_mode = str(cfg_get(cfg, "depth_mode", "bbox")).lower()
        self.use_metric_depth = bool(cfg_get(cfg, "depth_anything_metric", False)) or depth_mode in (
            "metric",
            "metric_depth",
            "depth_anything_metric",
            "dav2_metric",
            "absolute",
            "abs",
        )

        # SafetyConfig には項目がないので cfg_get で “存在すれば使う” にしています
        self.encoder = str(cfg_get(cfg, "depth_anything_encoder", "vits"))
        self.input_size = int(cfg_get(cfg, "depth_anything_input_size", 518))

        # metric depth 用のデフォルト
        self.metric_dataset = str(cfg_get(cfg, "depth_anything_metric_dataset", "hypersim")).lower()
        default_max_depth = 20.0 if self.metric_dataset in ("hypersim", "indoor", "indoor_hypersim") else 80.0
        self.max_depth = float(cfg_get(cfg, "depth_anything_max_depth", default_max_depth))

        default_ckpt = (
            f"checkpoints/depth_anything_v2_metric_{self.metric_dataset}_{self.encoder}.pth"
            if self.use_metric_depth
            else f"checkpoints/depth_anything_v2_{self.encoder}.pth"
        )
        self.ckpt_path = str(cfg_get(cfg, "depth_anything_ckpt", default_ckpt))

        # 相対深度→m へスケール合わせをするか（デフォルト True）
        # metric depth の場合は meters を直接出すので、キャリブレーションは無効化する
        self.enable_calibration = (
            False
            if self.use_metric_depth
            else bool(cfg_get(cfg, "depth_anything_calibrate_to_meters", True))
        )
        self.calib_momentum = float(cfg_get(cfg, "depth_anything_calib_momentum", 0.95))  # EMA
        self.eps = float(cfg_get(cfg, "eps", 1e-6))

        self._state = DepthCalibState(use_inverse=False, scale_ema=1.0)

        try:
            _ensure_depth_anything_repo_on_path(use_metric_depth=self.use_metric_depth)
            # metric_depth を優先 sys.path に入れることで、DepthAnythingV2(max_depth=...) を利用可能にする
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Cannot import DepthAnythingV2. You likely need to clone Depth-Anything-V2 and "
                "make it importable (e.g., install requirements, and ensure PYTHONPATH includes it)."
            ) from e

        if self.encoder not in _MODEL_CONFIGS:
            raise ValueError(f"Unsupported encoder: {self.encoder}. Choose one of {list(_MODEL_CONFIGS.keys())}")

        if not os.path.exists(self.ckpt_path):
            if self.use_metric_depth:
                raise FileNotFoundError(
                    f"DepthAnythingV2 *metric* checkpoint not found: {self.ckpt_path}\n"
                    "Download a metric-depth checkpoint from Depth-Anything-V2/metric_depth/README.md and "
                    "put it under ./checkpoints (predict_area/checkpoints).\n"
                    "Example (indoor): checkpoints/depth_anything_v2_metric_hypersim_vitl.pth\n"
                    "Example (outdoor): checkpoints/depth_anything_v2_metric_vkitti_vitl.pth\n"
                )
            raise FileNotFoundError(
                f"DepthAnythingV2 checkpoint not found: {self.ckpt_path}\n"
                "Per official README, download the checkpoint and place it under ./checkpoints.\n"
                "Example: checkpoints/depth_anything_v2_vits.pth"
            )

        model_kwargs = dict(_MODEL_CONFIGS[self.encoder])
        if self.use_metric_depth:
            model_kwargs["max_depth"] = float(self.max_depth)
        self.model = DepthAnythingV2(**model_kwargs)
        sd = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(sd)
        self.model = self.model.to(self.device).eval()

    def infer_depth_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        # infer_image(raw_image, input_size=518) returns HxW depth map (numpy)
        depth = self.model.infer_image(frame_bgr, input_size=self.input_size)
        depth = depth.astype(np.float32)
        return depth

    def _update_calibration(
        self,
        depth_map: np.ndarray,
        boxes_xyxy: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        assumed_person_height_m: float,
    ) -> None:
        """
        Use bbox heuristic z (meters) as weak supervision to fit scale for this frame.
        We try both:
          z ≈ s * depth
          z ≈ s * (1/depth)
        and pick the one with more stable ratios across people.
        """
        if boxes_xyxy.size == 0:
            return

        ratios_direct: List[float] = []
        ratios_inv: List[float] = []

        for i in range(boxes_xyxy.shape[0]):
            box = boxes_xyxy[i]
            d_med = _median_depth_in_bbox(depth_map, box)
            if not np.isfinite(d_med):
                continue

            # fallback z from bbox (meters)
            root_bbox = estimate_root_xyz_from_bbox(
                box_xyxy=box,
                fx=fx, fy=fy, cx=cx, cy=cy,
                assumed_person_height_m=assumed_person_height_m,
                eps=self.eps,
            )
            z_bbox = float(root_bbox[2])
            if not np.isfinite(z_bbox) or z_bbox <= 0:
                continue

            ratios_direct.append(z_bbox / (float(d_med) + self.eps))
            ratios_inv.append(z_bbox * (float(d_med) + self.eps))  # since z ≈ s*(1/d) => s ≈ z*d

        if len(ratios_direct) < 2:
            return

        def rel_cv(xs: List[float]) -> float:
            arr = np.array(xs, dtype=np.float32)
            m = float(np.mean(arr))
            s = float(np.std(arr))
            if not np.isfinite(m) or abs(m) < 1e-6:
                return float("inf")
            return abs(s / m)

        cv_direct = rel_cv(ratios_direct)
        cv_inv = rel_cv(ratios_inv)

        if cv_inv < cv_direct:
            use_inverse = True
            scale_frame = float(np.median(np.array(ratios_inv, dtype=np.float32)))
        else:
            use_inverse = False
            scale_frame = float(np.median(np.array(ratios_direct, dtype=np.float32)))

        if not np.isfinite(scale_frame) or scale_frame <= 0:
            return

        # EMA update
        m = float(self.calib_momentum)
        self._state.use_inverse = use_inverse
        self._state.scale_ema = m * float(self._state.scale_ema) + (1.0 - m) * float(scale_frame)

    def infer_and_calibrate(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        assumed_person_height_m: float,
    ) -> np.ndarray:
        depth_map = self.infer_depth_map(frame_bgr)
        if self.enable_calibration:
            self._update_calibration(depth_map, boxes_xyxy, fx, fy, cx, cy, assumed_person_height_m)
        return depth_map

    def z_m_from_bbox(
        self,
        depth_map: np.ndarray,
        box_xyxy: np.ndarray,
        fallback_z_m: float,
    ) -> float:
        d_med = _median_depth_in_bbox(depth_map, box_xyxy)
        if not np.isfinite(d_med):
            return float(fallback_z_m)

        # metric depth の場合、depth_map は meters なので中央値をそのまま z として使う
        if self.use_metric_depth and not self.enable_calibration:
            z = float(d_med)
            if not np.isfinite(z) or z <= 0:
                return float(fallback_z_m)
            return z

        s = float(self._state.scale_ema)
        if self._state.use_inverse:
            z = s * (1.0 / (float(d_med) + self.eps))
        else:
            z = s * float(d_med)

        if not np.isfinite(z) or z <= 0:
            return float(fallback_z_m)
        return float(z)

    def root_xyz_from_bbox(
        self,
        box_xyxy: np.ndarray,
        depth_map: Optional[np.ndarray],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        assumed_person_height_m: float,
    ) -> np.ndarray:
        # fallback (bbox heuristic)
        root_bbox = estimate_root_xyz_from_bbox(
            box_xyxy=box_xyxy,
            fx=fx, fy=fy, cx=cx, cy=cy,
            assumed_person_height_m=assumed_person_height_m,
            eps=self.eps,
        )
        if depth_map is None:
            return root_bbox

        z = self.z_m_from_bbox(depth_map, box_xyxy, fallback_z_m=float(root_bbox[2]))
        u, v = _bbox_center_uv(box_xyxy)
        return _root_xyz_from_uvz(u, v, z, fx, fy, cx, cy, self.eps)
