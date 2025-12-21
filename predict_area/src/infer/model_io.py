import torch  # type: ignore[import-not-found]
from typing import Any, Dict, Tuple

from ..config import SafetyConfig  # type: ignore[import-not-found]
from ..model import SafetyNet, SafetyNet2, SafetyNetV0  # type: ignore[import-not-found]


def _model_kwargs_from_cfg(cfg: SafetyConfig) -> Dict[str, Any]:
    return {
        "emb_dim": int(getattr(cfg, "emb_dim", 128)),
        "temporal": str(getattr(cfg, "temporal_model", "gru")),
        "rnn_hidden": int(getattr(cfg, "rnn_hidden", 256)),
        "rnn_layers": int(getattr(cfg, "rnn_layers", 2)),
        "tf_layers": int(getattr(cfg, "tf_layers", 2)),
        "tf_nhead": int(getattr(cfg, "tf_nhead", 4)),
        "tf_ff": int(getattr(cfg, "tf_ff", 512)),
        "tf_dropout": float(getattr(cfg, "tf_dropout", 0.1)),
        "tf_norm_first": bool(getattr(cfg, "tf_norm_first", True)),
    }


def load_safetynet(ckpt_path: str, cfg: SafetyConfig) -> Tuple[SafetyNet, int, int, torch.device]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim = int(ckpt["in_dim"])
    K = int(ckpt["K"])

    device_str = str(getattr(cfg, "device", "cpu"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model_kwargs = ckpt.get("model_kwargs") or _model_kwargs_from_cfg(cfg)
    model = SafetyNet(in_dim=in_dim, K=K, **model_kwargs).to(device)
    try:
        model.load_state_dict(ckpt["model_state"], strict=True)
    except RuntimeError as e:
        # Keep user experience smooth: allow loading older checkpoints even if the model definition evolved.
        # Note: if keys are missing, those new parameters will stay randomly initialized.
        print(f"[infer] warning: strict load_state_dict failed: {e}")
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            print(f"[infer] warning: missing keys (initialized randomly): {len(missing)}")
        if unexpected:
            print(f"[infer] warning: unexpected keys (ignored): {len(unexpected)}")
    model.eval()
    return model, in_dim, K, device

def load_safetynet2(ckpt_path: str, cfg: SafetyConfig) -> Tuple[SafetyNet, int, int, torch.device]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    in_dim = int(ckpt["in_dim"])
    K = int(ckpt["K"])

    device_str = str(getattr(cfg, "device", "cpu"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    model_kwargs = ckpt.get("model_kwargs") or _model_kwargs_from_cfg(cfg)
    state: Dict[str, Any] = ckpt["model_state"]

    # Auto-detect which model definition the checkpoint was saved from.
    # - Very old checkpoints used `SafetyNetV0` (person.net / pool.* / temporal.encoder.*).
    # - SafetyNet2 adds `residual_layer.*`. Older checkpoints (or checkpoints trained as SafetyNet)
    #   won't have these keys, so instantiating SafetyNet2 and loading strictly will fail.
    is_v0 = any(str(k).startswith("person.net.") for k in state.keys())
    has_residual = any(str(k).startswith("residual_layer.") for k in state.keys())
    if is_v0:
        print("[infer] info: detected legacy checkpoint format (person.net.*); loading SafetyNetV0.")
        model = SafetyNetV0(in_dim=in_dim, K=K, **model_kwargs).to(device)
    elif has_residual:
        model = SafetyNet2(in_dim=in_dim, K=K, **model_kwargs).to(device)
    else:
        print(
            "[infer] info: checkpoint has no residual_layer.* keys; "
            "loading as SafetyNet (not SafetyNet2) for compatibility."
        )
        model = SafetyNet(in_dim=in_dim, K=K, **model_kwargs).to(device)

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # Keep user experience smooth: allow loading older checkpoints even if the model definition evolved.
        # Note: if keys are missing, those new parameters will stay randomly initialized.
        print(f"[infer] warning: strict load_state_dict failed: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[infer] warning: missing keys (initialized randomly): {len(missing)}")
        if unexpected:
            print(f"[infer] warning: unexpected keys (ignored): {len(unexpected)}")
    model.eval()
    return model, in_dim, K, device

