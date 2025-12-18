import cv2
import numpy as np


def draw_prob_bar(frame: np.ndarray, p: np.ndarray, x0=20, y0=40, w=320, h=12) -> np.ndarray:
    K = int(p.shape[0])
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    acc = 0
    for k in range(K):
        bw = int(round(w * float(p[k])))
        if bw <= 0:
            continue
        cv2.rectangle(frame, (x0 + acc, y0), (x0 + acc + bw, y0 + h), (0, 255, 0), -1)
        acc += bw
        if acc >= w:
            break
    return frame


def overlay_prediction(frame: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = frame
    k_best = int(np.argmax(p))
    cv2.putText(out, f"best_bin={k_best}", (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out = draw_prob_bar(out, p, x0=20, y0=40, w=360, h=14)
    return out
