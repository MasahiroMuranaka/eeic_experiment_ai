import torch
from typing import Any, Dict, Tuple

from config import SafetyConfig
from model import SafetyNet
from preprocess.io_utils import cfg_get


def _model_kwargs_from_cfg(cfg: SafetyConfig) -> Dict[str, Any]:
    return {
        "emb_dim": int(cfg_get(cfg, "emb_dim", 128)),
        "temporal": str(cfg_get(cfg, "temporal_model", "gru")),
        "rnn_hidden": int(cfg_get(cfg, "rnn_hidden", 256)),
        "rnn_layers": int(cfg_get(cfg, "rnn_layers", 2)),
        "tf_layers": int(cfg_get(cfg, "tf_layers", 2)),
        "tf_nhead": int(cfg_get(cfg, "tf_nhead", 4)),
        "tf_ff": int(cfg_get(cfg, "tf_ff", 512)),
        "tf_dropout": float(cfg_get(cfg, "tf_dropout", 0.1)),
        "tf_norm_first": bool(cfg_get(cfg, "tf_norm_first", True)),
    }


def load_safetynet(ckpt_path: str, cfg: SafetyConfig) -> Tuple[SafetyNet, int, int, torch.device]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim = int(ckpt["in_dim"])
    K = int(ckpt["K"])

    device_str = str(cfg_get(cfg, "device", "cpu"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model_kwargs = ckpt.get("model_kwargs") or _model_kwargs_from_cfg(cfg)
    model = SafetyNet(in_dim=in_dim, K=K, **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, in_dim, K, device
