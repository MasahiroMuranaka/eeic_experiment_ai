import os
from typing import Optional, TextIO, Tuple

import cv2


class OutputWriters:
    def __init__(self, writer: Optional[cv2.VideoWriter], csv_f: Optional[TextIO]):
        self.writer = writer
        self.csv_f = csv_f

    def close(self, show: bool):
        if self.writer is not None:
            self.writer.release()
        if self.csv_f is not None:
            self.csv_f.close()
        if show:
            cv2.destroyAllWindows()


def open_video_writer(out_video: str, fps: float, size_wh: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    if not out_video:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = size_wh
    return cv2.VideoWriter(out_video, fourcc, fps, (w, h))


def open_csv(out_csv: str, K: int) -> Optional[TextIO]:
    if not out_csv:
        return None
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    f = open(out_csv, "w", encoding="utf-8")
    header = "frame," + ",".join([f"p{k}" for k in range(K)]) + "\n"
    f.write(header)
    return f
