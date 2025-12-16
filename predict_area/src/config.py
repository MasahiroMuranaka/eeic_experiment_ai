from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class SafetyConfig:
    # ===== Output bins (横方向K分割) =====
    K: int = 8
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
    # "bbox": bbox高さから距離推定（簡易）
    # "midas": 既存互換（DepthAnythingV2の相対深度 + スケール合わせ）
    # "metric": DepthAnythingV2/metric_depth による絶対深度（meters）
    depth_mode: str = "bbox"

    # ===== DepthAnythingV2 settings (optional; used via cfg_get) =====
    # metric depth を使う場合は True 推奨（depth_mode="metric" でも自動的に True 扱い）
    depth_anything_metric: bool = False
    # metricモデルの種類: "hypersim"(indoor) or "vkitti"(outdoor)
    depth_anything_metric_dataset: str = "hypersim"
    # metricモデルの max_depth（hypersim推奨=20, vkitti推奨=80）
    depth_anything_max_depth: float = 20.0

    # ===== Training =====
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "cuda"

    # ===== Model (SafetyNet) =====
    # temporal encoder: "gru" or "transformer"
    temporal_model: str = "gru"
    emb_dim: int = 128

    # --- GRU params ---
    rnn_hidden: int = 256
    rnn_layers: int = 2

    # --- Transformer params ---
    tf_layers: int = 2
    tf_nhead: int = 4
    tf_ff: int = 512
    tf_dropout: float = 0.1
    tf_norm_first: bool = True

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
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyYAML が見つかりません。`pip install pyyaml` もしくは `uv sync` で依存関係を入れてください。"
        ) from e
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)


def load_config(path: str) -> SafetyConfig:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyYAML が見つかりません。`pip install pyyaml` もしくは `uv sync` で依存関係を入れてください。"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        d: Dict[str, Any] = yaml.safe_load(f)
    return SafetyConfig(**d)
