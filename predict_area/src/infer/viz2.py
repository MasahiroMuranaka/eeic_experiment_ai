import os
from typing import Optional, Tuple

import cv2
import numpy as np


def load_prob_sequence(path: str) -> np.ndarray:
    """
    確率配列を読み込むユーティリティ。
    サポート: .npz (内部に配列が一つ)、.npy, .csv

    戻り値: numpy array of shape (N, K) または (K,)（フレーム毎/単一フレーム）
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        # 代表的キーを探す
        for key in ("probs", "prob", "arr_0", "data"):
            if key in data:
                return data[key]
        # それ以外は最初の配列を返す
        for k in data.files:
            return data[k]
        raise ValueError(f"no array found in {path}")
    elif ext == ".npy":
        return np.load(path)
    elif ext == ".csv":
        arr = np.loadtxt(path, delimiter=",")
        return arr
    else:
        raise ValueError(f"unsupported extension: {ext}")


def _get_colormap_const(name: str) -> int:
    name = name.lower()
    mapping = {
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", None),
        "jet": getattr(cv2, "COLORMAP_JET", cv2.COLORMAP_JET),
        "hot": getattr(cv2, "COLORMAP_HOT", None),
    }
    val = mapping.get(name)
    if val is None:
        # fallback
        return cv2.COLORMAP_JET
    return val


def overlay_heatmap_from_grid(
    frame: np.ndarray,
    prob: np.ndarray,
    grid_size: Optional[Tuple[int, int]] = None,
    colormap: str = "viridis",
    alpha: float = 0.5,
    blur: int = 3,
    normalize: Optional[bool] = None,
    p_min: float = 0.02,
    ema_prev: Optional[np.ndarray] = None,
    ema_alpha: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    フレームに確率グリッドをヒートマップとして重ねる。

    引数:
      - frame: BGR uint8 image (Hf, Wf, 3)
      - prob: 1D または 2D 確率配列。1D の場合は grid_size を指定するか自動で正方形化を試みる。
      - grid_size: (rows, cols) を与えると 1D->2D の変換に使用
      - colormap: 'viridis'|'jet' 等（OpenCV の対応に従う）
      - alpha: 元画像とヒートマップの合成比
      - blur: 空間ブラー半径（0 で無効）
      - normalize: None=自動 (max>1 なら max 正規化)、True=常に max 正規化、False=なし
      - p_min: しきい値未満を 0 にする
      - ema_prev/ema_alpha: フレーム間 EMA を使う場合に前フレームの確率マップを渡す

    戻り値:
      - out_frame: 合成済み BGR uint8
      - prob_resized: フレームサイズにリサイズした 2D 確率マップ (float32, range 0..1)
    """
    if frame is None:
        raise ValueError("frame is None")
    fh, fw = frame.shape[:2]

    p = np.array(prob, dtype=np.float32)
    # 1D -> 2D 変換
    if p.ndim == 1:
        if grid_size is not None:
            gh, gw = grid_size
            if gh * gw != p.size:
                raise ValueError("grid_size does not match prob length")
            p2 = p.reshape((gh, gw))
        else:
            # try square
            s = int(np.round(np.sqrt(p.size)))
            if s * s == p.size:
                p2 = p.reshape((s, s))
            else:
                # fallback: treat as single-row
                p2 = p.reshape((1, p.size))
    elif p.ndim == 2:
        p2 = p
    else:
        raise ValueError("prob must be 1D or 2D array")

    # 正規化
    if normalize is None:
        if p2.max() > 1.0:
            do_norm = True
        else:
            do_norm = False
    else:
        do_norm = bool(normalize)
    if do_norm:
        m = p2.max() if p2.max() > 0 else 1.0
        p2 = p2 / (m + 1e-12)

    # 閾値
    p2 = np.where(p2 >= p_min, p2, 0.0)

    # ブラー
    if blur and blur > 0:
        k = int(blur) * 2 + 1
        p2 = cv2.GaussianBlur(p2, (k, k), 0)

    # resize to frame
    p_resized = cv2.resize(p2.astype(np.float32), (fw, fh), interpolation=cv2.INTER_LINEAR)
    p_clipped = np.clip(p_resized, 0.0, 1.0)

    # EMA (frameサイズで行う: ema_prev はフレームサイズのマップを期待)
    if ema_prev is not None:
        if ema_prev.shape != p_clipped.shape:
            # try to resize ema_prev to match
            try:
                ema_prev = cv2.resize(np.array(ema_prev, dtype=np.float32), (fw, fh), interpolation=cv2.INTER_LINEAR)
            except Exception:
                raise ValueError("ema_prev shape mismatch and could not be resized")
        p_clipped = ema_alpha * p_clipped + (1 - ema_alpha) * ema_prev

    # colormap
    cmap_const = _get_colormap_const(colormap)
    heat_uint8 = (p_clipped * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cmap_const)

    # 合成
    out = cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0)

    return out, p_clipped


def overlay_safe_area(
    frame: np.ndarray,
    prob: np.ndarray,
    grid_size: Optional[Tuple[int, int]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
    pad: int = 2,
    label: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    フレーム上で最も安全（最大値）のグリッドセルを矩形で囲む。

    戻り値: (out_frame, (x1,y1,x2,y2)) で矩形座標を返す。
    """
    if frame is None:
        raise ValueError("frame is None")
    fh, fw = frame.shape[:2]

    p = np.array(prob, dtype=np.float32)
    # 1D -> 2D 変換（heatmap と同じルール）
    if p.ndim == 1:
        if grid_size is not None:
            gh, gw = grid_size
            if gh * gw != p.size:
                raise ValueError("grid_size does not match prob length")
            p2 = p.reshape((gh, gw))
        else:
            s = int(np.round(np.sqrt(p.size)))
            if s * s == p.size:
                p2 = p.reshape((s, s))
            else:
                p2 = p.reshape((1, p.size))
    elif p.ndim == 2:
        p2 = p
    else:
        raise ValueError("prob must be 1D or 2D array")

    # 「安全」= 最大値のセルを選択
    idx = int(np.argmax(p2))
    gh, gw = p2.shape
    r = idx // gw
    c = idx % gw

    # ブロックのピクセル範囲を計算
    bw = fw / gw
    bh = fh / gh
    x1 = int(max(0, np.floor(c * bw) - pad))
    y1 = int(max(0, np.floor(r * bh) - pad))
    x2 = int(min(fw - 1, np.ceil((c + 1) * bw) + pad))
    y2 = int(min(fh - 1, np.ceil((r + 1) * bh) + pad))

    out = frame.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    if label:
        txt = f"safe:{p2[r, c]:.2f}"
        cv2.putText(out, txt, (x1 + 4, max(y1 + 16, y1 + 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return out, (x1, y1, x2, y2)


if __name__ == "__main__":
    # 簡易デモ: 既存のビデオフレームにランダムグリッドを重ねて保存
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="viz2_demo.mp4")
    parser.add_argument("--w", type=int, default=640)
    parser.add_argument("--h", type=int, default=360)
    args = parser.parse_args()

    # 5x8 グリッドのランダム確率で 120 フレームを作る
    gh, gw = 5, 8
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, 20.0, (args.w, args.h))
    ema = None
    for i in range(120):
        frame = np.full((args.h, args.w, 3), 50, dtype=np.uint8)
        probs = np.random.rand(gh * gw).astype(np.float32)
        out, ema = overlay_heatmap_from_grid(frame, probs, grid_size=(gh, gw), alpha=0.5, ema_prev=ema)
        writer.write(out)
    writer.release()
    print("demo written to", args.out)
