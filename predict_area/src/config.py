from dataclasses import dataclass, asdict
import yaml
from typing import Any, Dict


@dataclass
class SafetyConfig:
    # ===== Output bins (横方向K分割) =====
    K: int = 9
    x_min: float = -2.0  # [m] 左端
    x_max: float = 2.0  # [m] 右端

    # ===== Temporal =====
    T: int = 10  # past frames
    H: int = 10  # future frames (for label)

    # ===== Person selection / padding =====
    D: float = 3.0       # [m] 入力に含める人物の距離閾値
    Nmax: int = 16       # 最大人数（超えたら近い順に切る）
    fps: float = 30.0    # 動画fps（不明なら推定or指定）

    # ===== Camera model (疑似距離復元) =====
    fov_y_deg: float = 60.0  # だいたいの縦FOV
    assumed_person_height_m: float = 1.7  # bbox高さ→距離換算用

    # ===== Pose / keypoints =====
    kp_conf_thresh: float = 0.3  # この未満は欠損扱い
    use_pose_delta: bool = True

    # ===== Area encoding (入力用: (x,z)粗区画) =====
    area_z_bins: int = 4     # 奥行き方向の分割数（入力だけに使う）
    z_min: float = 0.5       # [m]
    z_max: float = 6.0       # [m]

    # ===== Label shaping (占有ベース) =====
    alpha_depth: float = 1.0  # w_depth = exp(-alpha*z)
    gamma_time: float = 1.0   # w_time = gamma^(tau-1)
    beta_risk: float = 1.0    # S = exp(-beta*R)

    # ===== Depth mode =====
    depth_mode: str = "bbox"  # "bbox" or "midas"

    # ===== Training =====
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "cuda"

    # ===== Misc =====
    seed: int = 42
    eps: float = 1e-6

    # --- YOLO settings ---
    yolo_pose_model: str = "yolov8n-pose.pt"
    yolo_tracker: str = "bytetrack.yaml"
    yolo_conf: float = 0.25
    yolo_iou: float = 0.5
    yolo_device: str = ""   # "cpu" or "0" など。空なら自動

    # --- EgoMotion settings ---
    use_ego_motion: bool = True
    ego_as_feature: bool = True      # ★TrueならXに入れる（Fが増える）
    ego_normalize: bool = True       # ★(vx/width)*fps, (vy/height)*fps を使う


def save_config(path: str, cfg: SafetyConfig) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)


def load_config(path: str) -> SafetyConfig:
    with open(path, "r", encoding="utf-8") as f:
        d: Dict[str, Any] = yaml.safe_load(f)
    return SafetyConfig(**d)
