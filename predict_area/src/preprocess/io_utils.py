import os
import glob
from typing import List

from ..config import SafetyConfig, load_config

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def cfg_get(cfg: SafetyConfig, name: str, default):
    return getattr(cfg, name, default)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_videos(video_dir: str) -> List[str]:
    paths = []
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(video_dir, f"*{ext.upper()}")))
    return sorted(set(paths))
