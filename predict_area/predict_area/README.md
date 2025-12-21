### uvで仮想環境を構築して実行する手順（推奨）
このプロジェクトは `pyproject.toml` と `uv.lock` を同梱しているため、`uv` で **仮想環境作成 + 依存関係インストール**までをまとめて行えます。

#### 0) uv のインストール（未導入の場合）
macOS（Homebrew）:

```bash
brew install uv
```

#### 1) 仮想環境の作成
`predict_area/` に移動して、Python 3.11 で venv を作ります（`pyproject.toml` の `requires-python = ">=3.11"` に合わせます）。

```bash
cd /Users/muranakamasahiro/Dev/eeic_experiment_ai/predict_area
uv venv --python 3.11
```

#### 2) 依存関係のインストール（ロックファイルから）

```bash
source .venv/bin/activate
uv sync
```

（補足）もし `uv sync` がうまくいかない場合は、次で代替できます:

```bash
uv pip install -e .
```

#### 3) 実行例（学習 / 推論）
以降は仮想環境を有効化したまま、READMEにあるコマンドをそのまま実行できます（`python -m ...`）。

例: 学習

```bash
python -m src.train \
  --npz-dir /path/to/out_npz_dir \
  --config /path/to/config.yaml \
  --log-interval 100 \
  --out-ckpt /path/to/out_model.pt
```

例: 推論

```bash
python -m src.infer.cli \
  --video /path/to/input.mp4 \
  --ckpt /path/to/model.pt \
  --config /path/to/config.yaml \
  --out-csv /path/to/out.csv \
  --out-video /path/to/out.mp4
```

（任意）仮想環境の activate を省略したい場合は `uv run` でも実行できます:

```bash
uv run python -m src.infer.cli --help
```


### このドキュメントについて
`predict_area/` 配下の各ファイル（主に `src/`）について、**役割**・**主要な関数/クラス**・**引数/入出力（特にテンソル形状）**を短くまとめたガイドです。

本プロジェクトは大きく以下の流れで動きます。

- **前処理**: 動画 → YOLO Pose 追跡 +（任意で）DepthAnythingV2 → 特徴量 `X` とマスク `M` と教師 `y` を `.npz` に保存
- **学習**: `.npz`（複数可）→ `SafetyNet` を学習 → checkpoint (`.pt`) を保存
- **推論**: 動画 + checkpoint → フレームごとの `K` 分割確率 `p` を推定し、動画/CSVへ出力

---

### コマンドライン実行方法（動画/フレーム対応）
作業ディレクトリを `predict_area/` に移動して実行します。

```bash
cd /Users/muranakamasahiro/Dev/eeic_experiment_ai/predict_area
```

#### 前処理（npz生成）

- **（これ！）フレームフォルダ + bbox JSON + 正解分布JSON（新）**　

```bash
python -m src.preprocess.cli \
  --frames-dir /path/to/frames_dir \
  --out-dir /path/to/out_npz_dir \
  --config /path/to/config.yaml \
  --det-json /path/to/*_image8.json \
  --y-json /path/to/*.json \
  --fps 30
```


- **動画ディレクトリ → npz一括生成（従来どおり）**

```bash
python -m src.preprocess.cli \
  --video-dir /path/to/videos \
  --out-dir /path/to/out_npz_dir \
  --config /path/to/config.yaml \
  --skip-existing
```

- **フレームフォルダ（画像列）→ npz生成（新）**
  - 動画が無いので `--fps` を必要に応じて指定（未指定なら `config.yaml` の `fps` / デフォルト30）

```bash
python -m src.preprocess.cli \
  --frames-dir /path/to/frames_dir \
  --out-dir /path/to/out_npz_dir \
  --config /path/to/config.yaml \
  --fps 30 \
  --skip-existing
```

- **フレームフォルダ + 正解分布JSONで教師 `y` を使用（新）**
  - `tressider-2019-04-26_2.json` は `frame_name -> [K]` の確率分布です
  - **注意**: JSON内の分布長 `K` と `config.yaml` の `K` を一致させてください（不一致だと前処理でエラーになります）

```bash
python -m src.preprocess.cli \
  --frames-dir /path/to/frames_dir \
  --out-dir /path/to/out_npz_dir \
  --config /path/to/config.yaml \
  --y-json /Users/muranakamasahiro/Dev/eeic_experiment_ai/tressider-2019-04-26_2.json \
  --fps 30
```

- **フレームフォルダ + bbox JSONを使用（新）**
  - `tressider-2019-04-26_2_image8.json` は `detections[frame_name]` に bbox が入っています
  - bbox JSON には追跡IDが無い想定のため、内部で **IoU簡易トラッキング**してIDを割り当てます

```bash
python -m src.preprocess.cli \
  --frames-dir /path/to/frames_dir \
  --out-dir /path/to/out_npz_dir \
  --config /path/to/config.yaml \
  --det-json /Users/muranakamasahiro/Dev/eeic_experiment_ai/tressider-2019-04-26_2_image8.json \
  --fps 30
```


#### 学習（npz → ckpt）
（`src/train.py` を使用。npz単体/npz-dir/manifest のいずれかを指定できます）

```bash
python -m src.train \
  --npz-dir /path/to/out_npz_dir \
  --config /path/to/config.yaml \
  --out-ckpt /path/to/out_model.pt
```

#### 推論（ckpt + 入力 → CSV/動画）
- **動画で推論（従来どおり）**

```bash
python -m src.infer.cli \
  --video /path/to/input.mp4 \
  --ckpt /path/to/model.pt \
  --config /path/to/config.yaml \
  --out-csv /path/to/out.csv \
  --out-video /path/to/out.mp4
```

- **フレームフォルダで推論（新）**

```bash
python -m src.infer.cli \
  --frames-dir /path/to/frames_dir \ <= テストデータへのディレクトリ
  --ckpt /path/to/model.pt \ <= モデルのチェックポイント
  --config /path/to/config.yaml \
  --out-csv /path/to/out.csv \
  --out-video /path/to/out.mp4 \
  --fps 30
```

- **.npz（特徴量 X/M）で推論（新）**
  - 前処理で作った `.npz` の `X/M` をそのままモデルに入れて、サンプルごとの確率分布 \(p\) をCSVに出します

```bash
python -m src.infer.cli \
  --npz /path/to/features.npz \
  --ckpt /path/to/model.pt \
  --out-csv /path/to/out.csv
```

#### 評価（教師分布と予測分布の比較）
`src/eval.py` は、**教師データの確率分布 \(q\)** と **モデル予測の確率分布 \(p\)** を比較し、分布のズレを指標として出力します（分類の正解率だけでなく、**「どれだけ分布として近いか」**を見たいときに使います）。

評価指標（平均）:
- **CE（cross entropy）**: \(-\sum_k q_k \log p_k\)（小さいほど良い）
- **KL**: \(\sum_k q_k \log(q_k/p_k)\)（小さいほど良い）
- **JS**: Jensen–Shannon divergence（小さいほど良い、対称）
- **EMD（1D Earth Mover）**: \(\sum |\mathrm{cumsum}(q-p)|\)（小さいほど良い）
  - `mean_emd_bins`: bin単位
  - `mean_emd_x`: `x_min/x_max` を使って **実座標（x方向）**に換算した距離
- **top1_acc**: `argmax(q)==argmax(p)` の一致率（参考）
- **mean_abs_bin_expect / mean_abs_x_expect**: 期待値（bin / x座標）の差（小さいほど良い）

`src/eval.py` には 2つの評価モードがあります。

##### A) ckpt + npz で評価（推奨）
学習に使った `.npz` には **教師 `y`（=確率分布）**が入っているので、`--ckpt` を指定すると **npzの `X/M` をモデルに入れて予測 \(p\) を作り、npzの教師 \(q\) と比較**します。

```bash
python src/eval.py \
  --ckpt /Users/muranakamasahiro/Dev/eeic_experiment_ai/predict_area/ckpt \
  --npz-dir /path/to/out_npz_dir \
  --config /path/to/config.yaml
```

npzの指定方法（いずれか）:
- `--npz /path/to/file.npz`
- `--npz-dir /path/to/dir`（再帰で `.npz` を探索）
- `--manifest /path/to/manifest.txt`（1行1パス）

注意:
- **preprocess/train と同じ `config.yaml`** を使ってください（特徴量次元が一致しないと推論できません）。
- `predict_area/npz/` が空の場合は、まず前処理（npz生成）を実行して `.npz` を作ってください。

##### B) 予測CSV + 教師JSON で評価（frame対応）
すでに推論で出した `out.csv`（`frame,p0..pK-1`）と、教師 `answers/*.json`（`frame_name -> [K]`）を突き合わせて評価します。  
CSVの `frame` は **フレーム番号**なので、`--frames-dir` を指定して **frame番号 → frameファイル名（例: `000479.jpg`）** に変換して対応付けます。

```bash
python src/eval.py \
  --pred-csv /path/to/out.csv \
  --gt-json /path/to/tressider-2019-04-26_2.json \
  --frames-dir /path/to/frames_dir
```

（任意）結果をJSON保存:

```bash
python src/eval.py ... --out-json /path/to/summary.json
```

### 重要なデータ仕様（共通）
学習/推論で扱うテンソルは以下が前提です。

- **`X`**: `[B, T, Nmax, F]`（float32）
  - `T`: 過去フレーム長（入力系列長）
  - `B`: バッチサイズ
  - `Nmax`: 1フレーム内で使う最大人数（近い順に上位）
  - `F`: 1人あたりの特徴量次元
- **`M`**: `[N, T, Nmax]`（bool）
  - `True` の場所だけが有効人物
- **`y`**: `[N, K]`（float32）
  - 未来 `H` フレームの占有/リスクから作る **soft label**（`sum(y)=1`）

特徴量 `F` は以下（実装に準拠）です。

- `root(3) + dir(3) + dist(1) + ttc(1)`（※vel/speed は特徴量から除外）
- `+ ego(2)`（`use_ego_motion && ego_as_feature` のとき）
- `+ pose(J3)`（`J*3` のフラット 3D 擬似骨格）
- `+ dpose(J3)`（`use_pose_delta` のとき）
- `+ conf(J)`（キーポイント信頼度）

---

### Depth-Anything-V2 を “絶対深度（meters）” で使う（metric depth）
同梱の `Depth-Anything-V2/metric_depth/` には **絶対深度（meters）を直接出すモデル**が用意されています。  
使うには metric depth 用 checkpoint を `predict_area/checkpoints/` に置いて、`config.yaml` を以下のように指定してください。

- **Indoor（推奨）**: `depth_anything_v2_metric_hypersim_<encoder>.pth`（max_depth=20 推奨）
- **Outdoor**: `depth_anything_v2_metric_vkitti_<encoder>.pth`（max_depth=80 推奨）

`config.yaml` 例（Indoor）:

```yaml
depth_mode: metric
depth_anything_encoder: vitl
depth_anything_metric_dataset: hypersim
depth_anything_max_depth: 20
# depth_anything_ckpt: checkpoints/depth_anything_v2_metric_hypersim_vitl.pth
```

※ checkpoint は `Depth-Anything-V2/metric_depth/README.md` にある配布先から取得し、ファイル名を上の規約に合わせて配置してください。

---

### ルート直下（`predict_area/`）
- **`pyproject.toml`**
  - **役割**: 依存関係（`torch`, `opencv-python`, `ultralytics`, `numpy` など）とPython要件を定義。
- **`uv.lock`**
  - **役割**: `uv` 用ロックファイル（環境再現）。
- **`README.md`**
  - **役割**: プロジェクト説明（現状は空）。
- **`yolov8n-pose.pt`**
  - **役割**: YOLO Pose の重み（前処理/推論で使用）。
- **`checkpoints/`**
  - **役割**: DepthAnythingV2 の重み置き場（例: `depth_anything_v2_vits.pth`）。
- **`data/`**
  - **役割**: 入力動画データ置き場（例: `train_data/*.mp4`）。
- **`npz/`**
  - **役割**: 前処理出力（学習用 `.npz`）置き場の例。
- **`Depth-Anything-V2/`**
  - **役割**: DepthAnythingV2 の公式実装を同梱した外部コード。
  - **本プロジェクトからの主な参照点**: `src/preprocess/depth_anything_v2.py` が `depth_anything_v2.dpt.DepthAnythingV2` を import して推論します。
  - **注記**: 同梱ディレクトリ配下の各ファイル詳細は upstream の `README.md` を参照するのが推奨です。

---

### `src/config.py`
- **役割**: 学習/前処理/推論で共有するハイパーパラメータ定義（`dataclass`）。
- **主要クラス**
  - **`SafetyConfig`**: bin数 `K`、入力長 `T`、未来長 `H`、最大人数 `Nmax`、距離閾値 `D`、YOLO設定、Depth設定、学習設定などを保持。
    - **重要**: 人物選別の距離閾値は `D`（m）。
    - **モデル設定（新）**: `SafetyNet` の構成（GRU / TransformerEncoder など）もここで切り替える。
      - `temporal_model`: `"gru"` or `"transformer"`（時系列エンコーダの種類）
      - `emb_dim`: 人物特徴を埋め込みに落とす次元（E）
      - **GRU系**: `rnn_hidden`, `rnn_layers`
      - **Transformer系**: `tf_layers`, `tf_nhead`, `tf_ff`, `tf_dropout`, `tf_norm_first`
- **主要関数**
  - **`save_config(path, cfg)`**: `SafetyConfig` を YAML に保存。
  - **`load_config(path)`**: YAML から `SafetyConfig` を復元。

---

### モデル構成の切り替えと復元ルール（重要）
このプロジェクトでは **学習時のモデル構成（GRU/Transformer 等）を推論時に確実に再現**できるように、checkpoint に「モデル引数」を保存してあります。

- **学習時**（`src/train.py`）:
  - `SafetyConfig` から `model_kwargs`（例: `{"temporal": "transformer", "emb_dim": 128, ...}`）を組み立てて `SafetyNet(..., **model_kwargs)` を作る。
  - checkpoint に `model_kwargs` を保存する（後述）。
- **推論時**（`src/infer/model_io.py`）:
  - checkpoint 内に `model_kwargs` があれば **それを最優先**して `SafetyNet` を構築する（学習と完全一致）。
  - 古い checkpoint などで `model_kwargs` が無い場合は、`SafetyConfig` から `model_kwargs` 相当を組み立ててフォールバックする。

これにより、**推論側の `config.yaml` を間違えても、checkpoint に保存された構成が優先**され、学習時と異なるモデルでロードしてしまう事故を避けられます。

#### 設定例（`config.yaml`）
GRU / Transformer は `SafetyConfig.temporal_model` で切り替えます（学習で保存された checkpoint がある場合、推論は checkpoint 側の `model_kwargs` が優先されます）。

**GRU（デフォルト）例**:

```yaml
temporal_model: gru
emb_dim: 128
rnn_hidden: 256
rnn_layers: 2
```

**TransformerEncoder 例**:

```yaml
temporal_model: transformer
emb_dim: 128
tf_layers: 2
tf_nhead: 4
tf_ff: 512
tf_dropout: 0.1
tf_norm_first: true
```

**注意点**:
- Transformer の場合、`emb_dim % tf_nhead == 0` が必須です（一致しないとモデル生成時に例外）。
- `run_inference()` の入力系列長 `T` は固定長運用（rolling buffer が `T` 溜まったら推論）です。

---

### `src/dataset.py`
- **役割**: `.npz` を複数束ねて学習できる `torch.utils.data.Dataset`。
- **主要クラス**
  - **`MultiNpzSafetyDataset(npz_paths)`**
    - **入力**: `.npz` のパス配列（各npzは `X/M/y` を含む）
    - **挙動**: ファイルごとに長さを持ち、`__getitem__` で該当ファイルだけ lazy load（直前ファイルはキャッシュ）
    - **`__getitem__(idx)` 出力**: `(X[T,Nmax,F], M[T,Nmax], y[K])`（いずれも `torch.Tensor`）
- **主要関数**
  - **`list_npz_in_dir(npz_dir)`**: ディレクトリ配下の `.npz` を再帰探索してソートして返す。
  - **`read_manifest(manifest_path)`**: `manifest.txt`（1行1パス）から `.npz` パス一覧を読む。

---

### `src/model.py`
- **役割**: 人物集合を時系列で集約し、横方向 `K` 分割の確率を出すモデル（PyTorch）。
- **主要クラス**
  - **`PersonEncoder(in_dim, emb_dim=128)`**
    - **入力**: `x[B,T,N,F]`
    - **出力**: `e[B,T,N,E]`
  - **`AttentionPool(emb_dim=128)`**
    - **入力**: `e[B,T,N,E]`, `m[B,T,N]`（無効人物をmask）
    - **出力**: `s[B,T,E]`（人物集合を attention で集約）
  - **`TemporalEncoder(temporal, emb_dim, ...)`（新）**
    - **役割**: 時系列エンコーダを **GRU** と **TransformerEncoder** で切り替える薄いラッパ。
    - **入力**: `s[B,T,E]`
    - **出力**: `h[B,D]`（最後の時刻の表現）
      - GRU のとき `D=rnn_hidden`
      - Transformer のとき `D=emb_dim`
    - **注意**: Transformer の場合は位置エンコーディング（sinusoidal）を加える。
  - **`SafetyNet(in_dim, K, emb_dim=128, temporal="gru", ...)`**
    - **入力**: `X[B,T,N,F]`, `M[B,T,N]`
    - **出力**: `p[B,K]`（softmax確率）, `logits[B,K]`
    - **内部構成（概要）**:
      - `PersonEncoder`: `F -> E`
      - `AttentionPool`: `N` 人を `E` 次元で集約（mask対応）
      - `TemporalEncoder`: `T` を集約して 1ベクトル化（GRU/Transformer切替）
      - `head`: `D -> K`

---

### `src/train.py`
- **役割**: `.npz`（単体/ディレクトリ/manifest）から学習し、checkpoint を保存するCLI。
- **主要関数**
  - **`set_seed(seed)`**: 乱数シード固定（`random/numpy/torch`）。
  - **`soft_ce_loss(logits, q)`**: soft label `q[B,K]` に対するクロスエントロピー。
  - **`resolve_npz_paths(train_npz, npz_dir, manifest)`**: 入力npzの解決（優先度: `--train-npz` > `--manifest` > `--npz-dir`）。
  - **`default_num_workers()`**: macOSは `0`（安定性優先）それ以外は `2`。
  - **`main()`**:
    - **主な引数**: `--train-npz/--npz-dir/--manifest`, `--config`, `--out-ckpt`, `--save-config`, `--num-workers`, `--pin-memory`
    - **保存物**: `torch.save` する辞書（推論で必要）
      - `model_state`: `state_dict()`
      - `cfg`: `SafetyConfig` の中身（参考/再現用）
      - `in_dim`: 特徴次元 `F`
      - `K`: 出力bin数
      - `model_kwargs`（新）: 学習時に `SafetyNet` を構築した引数（`temporal` など）
      - `npz_files`: 学習に使った `.npz` 一覧
    - **重要**: `model_kwargs` があることで、推論は **学習時のモデル構成を自動復元**できる。

---

### `src/preprocess/`（分割版・推奨）
#### `src/preprocess/cli.py`
- **役割**: 前処理CLI（動画ディレクトリ→複数npz生成）。
- **主な引数**: `--video-dir`, `--out-dir`, `--config`, `--save-config`, `--skip-existing`
- **内部呼び出し**: `preprocess.pipeline.preprocess_one_video()` を各動画へ適用し、最後に `manifest.txt` を生成。

#### `src/preprocess/pipeline.py`
- **役割**: 前処理の本体（フレーム処理 + バッファ化 + npz化）。
- **主要関数**
  - **`list_videos(video_dir)`**: 対応拡張子の動画一覧を返す。
  - **`preprocess_one_video(video_path, out_dir, yolo_pose_model, cfg, skip_existing=False)`**
    - **入出力**: 1動画→1 `*.npz`（失敗時 `None`）
    - **中核**:
      - `yolo_track_pose()` で人物 bbox/追跡ID/COCO17 keypoints を取得
      - `EgoMotionTracker.update()` で背景オプティカルフローから ego motion を推定
      - `depth_mode` が `"midas"` のとき `DepthAnythingV2DepthEstimator` を使い、bboxの中心深度を meters にキャリブレーションして `root(x,y,z)` を推定（失敗時は bbox 推定へフォールバック）
      - `build_npz_from_video_buffers()` で `X/M/y` を保存
  - **`preprocess_video_dir(video_dir, out_dir, cfg, skip_existing=False)`**: 複数動画をまとめて処理して `.npz` パス一覧を返す。
- **主要引数（cfg）**: `T/H/Nmax/D`, `use_pose_delta`, `use_ego_motion`, `ego_as_feature`, `ego_normalize`, `depth_mode`, YOLO関連

#### `src/preprocess/yolo_pose.py`
- **役割**: Ultralytics YOLO の tracking + pose を薄くラップ。
- **主要関数**
  - **`yolo_track_pose(model, frame_bgr, conf, iou, tracker, device)`**
    - **戻り値形状**:
      - `boxes_xyxy[N,4]`, `ids[N]`, `confs[N]`, `kpts_xy[N,17,2]`, `kpts_conf[N,17]`

#### `src/preprocess/camera.py`
- **役割**: 簡易カメラモデル（FOV→内部パラメータ、bbox/2D→擬似3D）。
- **主要関数**
  - **`camera_intrinsics_from_fov(h, w, fov_y_deg)`**: `(fx,fy,cx,cy)` を返す。
  - **`estimate_root_xyz_from_bbox(box_xyxy, fx,fy,cx,cy, assumed_person_height_m, eps)`**
    - bbox高さから `z` を推定し、中心 `(u,v)` を pinhole で `x,y` に戻して `root[x,y,z]` を返す（単純近似）。
  - **`pseudo3d_pose_from_keypoints(kpts_xy, kpts_conf, fx,fy,cx,cy, z, kp_conf_thresh)`**
    - 2D keypoints を固定深度 `z` 上に投影して `pose_flat[J*3]` と `conf[J]` を返す。

#### `src/preprocess/ego.py`
- **役割**: 人物領域を避けた背景特徴点の追跡から ego motion（画素移動）を推定。
- **主要クラス**
  - **`EgoMotionTracker.update(frame, exclude_boxes)`**: `(ego_vx, ego_vy)` を返す。

#### `src/preprocess/labels.py`
- **役割**: 未来 `H` フレームの人物 `x,z` から soft label `y[K]` を生成。
- **主要関数**
  - **`bin_index(x, x_min, x_max, K)`**: `x` をbinへ量子化。
  - **`compute_soft_label_from_future_xz(future_people_xz, K, x_min, x_max, alpha_depth, gamma_time, beta_risk)`**
    - `R[k]`（リスク蓄積）→ `S=exp(-beta*R)` → `q = S/sum(S)`。

#### `src/preprocess/build_npz.py`
- **役割**: 前処理バッファ（フレーム毎の人物状態）から学習用 `X/M/y` を構築して `.npz` 保存。
- **主要関数**
  - **`build_npz_from_video_buffers(frames_state, frames_ego, fps, width, height, cfg, out_npz_path, video_path)`**
    - **入力**:
      - `frames_state`: 各フレーム `tid -> {"root","pose","conf"}`
      - `frames_ego`: 各フレーム `(ego_vx, ego_vy)`（pixel/frame）
    - **出力**: `.npz` に `X/M/y` とメタ（`video/fps/cfg`）を保存
    - **重要（cfg）**: 距離閾値は `D`

#### `src/preprocess/depth_anything_v2.py`
- **役割**: DepthAnythingV2 を本プロジェクトに接続するアダプタ。
- **主要クラス**
  - **`DepthAnythingV2DepthEstimator(cfg)`**
    - **主な設定（cfg_getで任意）**:
      - `depth_anything_encoder`（vits/vitb/vitl/vitg）
      - `depth_anything_ckpt`（checkpoint path）
      - `depth_anything_calibrate_to_meters`（相対深度→mスケール合わせ）
    - **主要メソッド**:
      - `infer_depth_map(frame_bgr) -> depth_map[H,W]`
      - `infer_and_calibrate(...) -> depth_map`（bbox由来zでスケールをEMA更新）
      - `root_xyz_from_bbox(...) -> root[3]`（depth優先、失敗時はbbox推定へ）
- **注記**: DepthAnything 本体は `Depth-Anything-V2/` 同梱コードに依存します。

---

### `src/infer/`（分割版・推奨）
#### `src/infer/cli.py`
- **役割**: 推論CLI（動画 + checkpoint → overlay動画/CSV）。
- **主な引数**: `--video`, `--ckpt`, `--config`, `--out-video`, `--out-csv`, `--show`, `--max-frames`
- **内部呼び出し**:
  - `model_io.load_safetynet()` でモデル/次元/K/deviceを復元
  - `pipeline.run_inference()` で推論ループ

#### `src/infer/model_io.py`
- **役割**: checkpoint のロードを分離。
- **主要関数**
  - **`load_safetynet(ckpt_path, cfg) -> (model, in_dim, K, device)`**
    - **復元ルール**:
      - checkpoint に `model_kwargs` があればそれを使用（最優先）
      - 無ければ `SafetyConfig` から `model_kwargs` を生成してフォールバック

#### `src/infer/features.py`
- **役割**: 推論時に rolling buffer（長さ `T`）から `X/M` を組み立てる（前処理と仕様一致が必須）。
- **主要関数**
  - **`build_feature_tensor(frames_state, frames_ego, fps, width, height, cfg, in_dim_expected)`**
    - **出力**: `X[1,T,Nmax,F]`, `M[1,T,Nmax]`
    - **注意**: `F != in_dim_expected` の場合は例外（cfg不一致）。

#### `src/infer/pipeline.py`
- **役割**: 推論ループ本体（YOLO + ego + depth + 特徴量→SafetyNet→出力）。
- **主要関数**
  - **`run_inference(video_path, ckpt_in_dim, K, cfg, device, safety_model, out_video="", out_csv="", show=False, max_frames=0)`**
    - CLIが直接呼ぶエントリポイント（モデルは外でロード済み）
  - **`run_infer(video_path, ckpt_path, cfg, ...)`**
    - 互換用（この関数内で `load_safetynet()` を呼んでから `run_inference` を呼ぶ）
    - **注意**: モデル構築ロジックは `infer/model_io.py` に一本化されている（GRU/Transformer切替を含む）

#### `src/infer/viz.py`
- **役割**: 予測確率の簡易可視化（バー表示）。
- **主要関数**
  - `draw_prob_bar(frame, p, ...)`
  - `overlay_prediction(frame, p)`

#### `src/infer/outputs.py`
- **役割**: 出力（動画writer / CSV）を開閉する小物ユーティリティ。
- **主要クラス/関数**
  - `OutputWriters(writer, csv_f).close(show)`
  - `open_video_writer(out_video, fps, size_wh)`
  - `open_csv(out_csv, K)`

---


