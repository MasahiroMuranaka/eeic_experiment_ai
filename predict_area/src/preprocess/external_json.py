from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_prob_dist_json(path: str) -> Dict[str, np.ndarray]:
    """
    Load mapping: frame_name -> probability list (length K).
    Example (tressider-2019-04-26_2.json):
      { "000000.jpg": [0,0,0,0,1.0,0,0,0], ... }
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"Invalid prob json (expected dict): {path}")
    out: Dict[str, np.ndarray] = {}
    for k, v in d.items():
        if not isinstance(k, str) or not isinstance(v, list):
            continue
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 1:
            continue
        out[k] = arr
    if not out:
        raise ValueError(f"No valid (frame->prob) entries found in: {path}")
    return out


def load_detection_json(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load detections JSON (tressider-2019-04-26_2_image8.json style):
      { "detections": { "000479.jpg": [ { "box": [x,y,w,h], "score": ... }, ... ], ... } }
    Returns mapping: frame_name -> list[det_dict]
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict) or "detections" not in d:
        raise ValueError(f"Invalid detection json (missing 'detections'): {path}")
    dets = d["detections"]
    if not isinstance(dets, dict):
        raise ValueError(f"Invalid detection json (detections not dict): {path}")
    out: Dict[str, List[Dict[str, Any]]] = {}
    for frame_name, arr in dets.items():
        if not isinstance(frame_name, str) or not isinstance(arr, list):
            continue
        out[frame_name] = [x for x in arr if isinstance(x, dict)]
    return out


def boxes_xyxy_from_detection_list(
    det_list: List[Dict[str, Any]], width: int, height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert detection dict list to (boxes_xyxy[N,4], scores[N]).
    Assumes det["box"] is [x,y,w,h] (as seen in sample).
    """
    boxes = []
    scores = []
    for det in det_list:
        box = det.get("box", None)
        if not isinstance(box, list) or len(box) != 4:
            continue
        x, y, w, h = box
        try:
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
        except Exception:
            continue
        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(float(width - 1), x + max(0.0, w))
        y2 = min(float(height - 1), y + max(0.0, h))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2, y2])
        s = det.get("score", 1.0)
        try:
            scores.append(float(s))
        except Exception:
            scores.append(1.0)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)


