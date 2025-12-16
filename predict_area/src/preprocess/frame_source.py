from __future__ import annotations

import glob
import os
import re
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


_NUM_RE = re.compile(r"(\d+)")


def _natural_key(path: str):
    """
    Natural-ish sort key for frame filenames like 000479.jpg or frame_12.png.
    """
    base = os.path.basename(path)
    parts = _NUM_RE.split(base)
    key = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def list_frame_paths(frames_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(frames_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(frames_dir, f"*{ext.upper()}")))
    paths = sorted(set(paths), key=_natural_key)
    return paths


def iter_frames_from_dir(
    frames_dir: str, max_frames: int = 0
) -> Iterable[Tuple[int, str, np.ndarray]]:
    """
    Yields (frame_idx, frame_name, frame_bgr) from a directory.
    """
    paths = list_frame_paths(frames_dir)
    if not paths:
        raise RuntimeError(f"No image frames found in: {frames_dir}")

    for i, p in enumerate(paths):
        if max_frames and i >= int(max_frames):
            break
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        yield i, os.path.basename(p), img


def read_first_frame(frames_dir: str) -> Tuple[str, np.ndarray]:
    paths = list_frame_paths(frames_dir)
    if not paths:
        raise RuntimeError(f"No image frames found in: {frames_dir}")
    p0 = paths[0]
    img = cv2.imread(p0, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {p0}")
    return os.path.basename(p0), img


