## ファイル一覧（目的）
- `predict_area/src/viz/viz2.py` — 可視化ユーティリティ
- `predict_area/src/viz/create_answer.py` — 動画/検出 JSON からフレーム毎の確率分布を生成するロジック（動画対応を追加）
- `predict_area/src/viz/run_viz_infer.py` — 動画と `.npz` を受け取りオーバーレイ動画を生成するランナー
- `predict_area/src/viz/run_detect_and_create_answer.py` — Ultralytics YOLO を用いて動画→検出 JSON を作成し、`.npz` を生成する一括ランナー

---

## ファイル詳細と関数ドキュメント

- ファイル: [predict_area/src/viz/viz2.py](predict_area/src/viz/viz2.py)
  - 目的: フレームに対する確率分布（1D/2D）を読み込み、ヒートマップ合成や「最も安全」セルの矩形描画を行うユーティリティ群。
  - 関数:
    - `load_prob_sequence(path: str) -> np.ndarray`
      - 引数: `path` (.npz/.npy/.csv のパス)
      - 返り値: numpy 配列（shape=(N,K) か (K,)）
      - 動作: 指定ファイルから代表的な配列キー(`probs`,`prob`,`arr_0`等)を探して読み込む。

    - `overlay_heatmap_from_grid(frame: np.ndarray, prob: np.ndarray, grid_size: Optional[Tuple[int,int]] = None, colormap: str = "viridis", alpha: float = 0.5, blur: int = 3, normalize: Optional[bool] = None, p_min: float = 0.02, ema_prev: Optional[np.ndarray] = None, ema_alpha: float = 0.6) -> Tuple[np.ndarray, np.ndarray]`
      - 引数:
        - `frame`: BGR uint8 画像 (H,W,3)
        - `prob`: 1D (K,) または 2D (gh,gw) の確率配列
        - `grid_size`: (rows, cols) — 1D を 2D に変換する際の形
        - `colormap`: OpenCV カラーマップ名
        - `alpha`: 合成比（0..1）
        - `blur`: ガウシアンブラー半径（0で無効）
        - `normalize`: None=自動判定 / True=最大値で正規化 / False=正規化なし
        - `p_min`: 下限しきい値（未満は 0 にする）
        - `ema_prev`: 前フレームの確率マップ（フレーム解像度） — EMA を行う場合に渡す
        - `ema_alpha`: EMA の現在値係数
      - 返り値: `(out_frame, prob_resized)`
        - `out_frame`: ヒートマップを合成した BGR uint8 画像
        - `prob_resized`: フレーム解像度にリサイズされた 2D float32 マップ（0..1）
      - 備考: EMA はフレーム解像度で適用されるようになっており、`ema_prev` は同解像度で渡すか自動リサイズされます。

    - `overlay_safe_area(frame: np.ndarray, prob: np.ndarray, grid_size: Optional[Tuple[int,int]] = None, color: Tuple[int,int,int] = (0,255,0), thickness: int = 4, pad: int = 2, label: bool = True) -> Tuple[np.ndarray, Tuple[int,int,int,int]]`
      - 引数:
        - `frame`: BGR uint8 画像
        - `prob`: 1D/2D 確率配列
        - `grid_size`: 1D を 2D に変換するサイズ
        - `color`, `thickness`, `pad`, `label`：描画パラメータ
      - 返り値: `(out_frame, (x1,y1,x2,y2))` — 矩形描画した画像と矩形座標
      - 動作: `prob` の最大セルを選択し、フレーム上の対応ブロックを矩形で描画する。

- ファイル: [predict_area/src/viz/create_answer.py](predict_area/src/viz/create_answer.py)
  - 目的: 既存の検出データ（画像バッチ用の JSON）や動画の検出 JSON を受け、各フレームごとに K 分割した領域ごとの安全確率（分布）を計算して返す。今回、動画対応の `get_answer_from_video` を追加/拡張した。
  - 関数:
    - `calc_answer(box_list, score_list, num_area, im_w) -> List[float]`
      - 引数:
        - `box_list`: 各検出の `[x, y, w, h]`（左上座標 + 幅高さ）リスト
        - `score_list`: 各検出のスコアリスト
        - `num_area`: 横方向に分割する領域数 K
        - `im_w`: 画像幅（ピクセル）
      - 返り値: 長さ `num_area` のリスト。最小被覆（安全）領域に対して確率を均等に割り当てる実装（ヒューリスティック）。

    - `get_answer_from_video(video_path: str, detections_json: Optional[str], num_area: int) -> Dict[str, List[float]]`
      - 引数:
        - `video_path`: 入力動画パス
        - `detections_json`: フレーム単位検出 JSON（キー: `"{frame_idx}.jpg"`）または None
        - `num_area`: K
      - 返り値: `{ "{frame_idx}.jpg": [p0, p1, ..., p_{K-1}] }` の辞書（各フレームの確率配列）
      - 動作: `detections_json` が無い/そのフレームに検出が無い場合は一様分布を返す。動画をフレーム単位で読みながら `calc_answer` を適用する。

- ファイル: [predict_area/src/viz/run_viz_infer.py](predict_area/src/viz/run_viz_infer.py)
  - 目的: 動画と `.npz`（確率配列）を受け取り、各フレームに `overlay_safe_area` を適用してオーバーレイ動画を書き出す。`.npz` が無ければ `get_answer_from_video` で生成する簡易ランナー。
  - 関数:
    - `run(video_path: str, probs_npz: str, out_path: str, num_area: int, detections_json: str = None) -> None`
      - 引数:
        - `video_path`: 入力動画
        - `probs_npz`: 確率配列の .npz パス。存在しなければ自動生成される（出力先は out_path に基づく）
        - `out_path`: 出力動画パス
        - `num_area`: K
        - `detections_json`: 生成時に使う検出 JSON
      - 返り値: なし（ファイル出力: out_path へ動画を書き出す）

- ファイル: [predict_area/src/viz/run_detect_and_create_answer.py](predict_area/src/viz/run_detect_and_create_answer.py)
  - 目的: Ultralytics YOLO モデルを使って動画をフレーム単位に検出・追跡し、`detect_json` を出力、続けて `get_answer_from_video` を呼んで `.npz` を保存するワンステップスクリプト。
  - 関数:
    - `detect_video_to_json(video_path: str, model_path: str, out_json: str, conf: float = 0.25, iou: float = 0.5, tracker: str = "bytetrack.yaml") -> str`
      - 引数: 動画、モデルパス、出力 JSON、検出閾値等
      - 返り値: 書き出した `out_json` のパス
      - 副作用: `out_json` に `{"detections": { "{frame_idx}.jpg": [ {"box":[x,y,w,h], "score": s}, ... ] } }` 形式で保存する

---

## テスト／再現手順（最小手順）
以下はローカルで最小限に再現する順序です。PowerShell の例を示します。

1) 仮想環境（`.venv`）を作る・有効化し、依存を入れる

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install opencv-python numpy Pillow
# YOLO 検出を使う場合:
pip install ultralytics
```

2) YOLO 検出 → `.npz` 生成（モデルファイル `yolov8n-pose.pt` がリポジトリにある前提）

```powershell
python predict_area/src/viz/run_detect_and_create_answer.py --video predict_area/data/train_data/data_1.mp4 --model yolov8n-pose.pt --out_json predict_area/detections/data_1_detections.json --out_npz npz/test_npz/data_1_from_detect.npz --num_area 8
```

3) 生成済み `.npz` を使ってオーバーレイ動画を作る

```powershell
python predict_area/src/viz/run_viz_infer.py --video predict_area/data/train_data/data_1.mp4 --probs npz/test_npz/data_1_from_detect.npz --out predict_area/result/test_result/data_1_detect_overlay.mp4 --num_area 8
```

4) （ヒートマップ表示を確認したい場合）`viz2.py` のデモを単独で実行

```powershell
python predict_area/src/viz/viz2.py --out viz2_demo.mp4 --w 640 --h 360
```

---

## 注意点 / 既知の問題
- `run_detect_and_create_answer.py` は `ultralytics` に依存します。Tracker の設定（`tracker` 引数）は `bytetrack.yaml` 等の有効なトラッカーネームを渡してください。空文字だと内部でエラーになることがあります。
- `overlay_heatmap_from_grid` の `ema_prev` はフレーム解像度（H,W）で渡すか、関数が自動リサイズしますが、型/形状が正しいことを確認してください。

