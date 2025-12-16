from typing import Tuple
import numpy as np
from ultralytics import YOLO


def yolo_track_pose(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float,
    iou: float,
    tracker: str,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kwargs = dict(
        persist=True,
        verbose=False,
        tracker=tracker,
        conf=float(conf),
        iou=float(iou),
        classes=[0],   # person
    )
    if device:
        kwargs["device"] = device

    res = model.track(frame_bgr, **kwargs)
    r0 = res[0]

    if r0.boxes is None or len(r0.boxes) == 0 or r0.keypoints is None:
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.int32),
            np.zeros((0,), np.float32),
            np.zeros((0, 17, 2), np.float32),
            np.zeros((0, 17), np.float32),
        )

    boxes = r0.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = r0.boxes.conf.cpu().numpy().astype(np.float32)

    if r0.boxes.id is None:
        ids = np.arange(len(boxes), dtype=np.int32)
    else:
        ids = r0.boxes.id.int().cpu().numpy().astype(np.int32)

    kpts_xy = r0.keypoints.xy.cpu().numpy().astype(np.float32)
    kpts_conf = r0.keypoints.conf.cpu().numpy().astype(np.float32)
    return boxes, ids, confs, kpts_xy, kpts_conf
