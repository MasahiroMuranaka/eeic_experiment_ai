# Spec: mv4 (mv4.py) — カメラ（Ego）速度推定付きトラッキングパイプライン

この仕様書は `yolo_exp/sample/mv4.py`（以下「mv4」）の関数・入出力・内部処理・注意点をまとめたものです。

目標: 動画から人物を検出・追跡し、各検出に対して被写体の特徴量（位置・速度等）に加えてカメラ（背景）移動（ego motion）を付与してCSV保存する。

前提
- 実行ディレクトリ: `yolo_exp/sample` からの実行を想定。スクリプト内部で `resolve_path()` によるパス解決を行う。
- 必要ライブラリ: `ultralytics`, `opencv-python`, `numpy`, `pandas`（軽量依存のみ）。

--------------------------------------------------------------------------------
関数一覧（要点）

1) `resolve_path(path: str) -> str`
   - 入力: 相対または絶対のパス文字列。
   - 出力: スクリプトファイル（`mv4.py`）を基準に解決された絶対パス文字列。
   - 内部処理: `os.path.dirname(os.path.abspath(__file__))` を基準に `os.path.join` で結合して `os.path.abspath` を返す。
   - 注意: 実行カレントディレクトリに依存する問題を避けるため必ず I/O に使用すること。

2) `class EgoMotionTracker`
   - 概要: 背景の特徴点追跡からカメラ移動を推定する。主に `update()` を呼び出す。

   - 属性:
     - `feature_params`: `goodFeaturesToTrack` 用パラメータ（maxCorners, qualityLevel, minDistance, blockSize）
     - `lk_params`: Lucas-Kanade (`calcOpticalFlowPyrLK`) のパラメータ（winSize, maxLevel, criteria）
     - `prev_gray`: 前フレームのグレースケール画像（None 初期）
     - `prev_pts`: 前フレームの追跡点集合（None 初期、shape: (N,1,2)）

   - メソッド: `update(frame: np.ndarray, exclude_boxes: List[List[int]]) -> (float, float)`
     - 入力:
       - `frame`: BGR カラー画像（np.ndarray）
       - `exclude_boxes`: 現在フレームで検出された人物領域のリスト、各要素は `[x1,y1,x2,y2]`
     - 出力: `(ego_vx, ego_vy)` — 背景の平均的な移動ベクトル（ピクセル／フレーム、float）。
     - 内部処理（要点）:
       1. 入力フレームをグレースケール化。
       2. `prev_gray` が存在すれば、`prev_pts` の数が少なければ `goodFeaturesToTrack` で補充。
       3. `prev_pts` から人物領域に入っている点を除外して `good_prev_pts` を作成。
       4. `calcOpticalFlowPyrLK(prev_gray, gray, good_prev_pts)` で forward フローを計算。
       5. forward-backward チェック: `calcOpticalFlowPyrLK(gray, prev_gray, next_pt)` を使って再投影誤差を計算し、閾値以内の点のみを残す。
       6. 残った点群で `estimateAffinePartial2D(pts_prev, pts_next, method=RANSAC)` を実行。成功すれば `M[0,2], M[1,2]` を並進（ego_vx, ego_vy）として採用。
       7. アフィン推定失敗または点が少ない場合は `pts_next - pts_prev` の中央値を用いる。
       8. `prev_gray` と `prev_pts` を更新して次フレームに備える。
     - 例外/エッジケース:
       - 特徴点が完全に除去される（人物に占有された等）場合は None 相当の状態になり、次フレームで再補充される。出力は `(0.0,0.0)` にフォールバックする実装でもよい。
     - 注意事項:
       - FB閾値（現在 `fb_thresh=1.5`）や RANSAC の `ransacReprojThreshold` は動画特性に合わせて調整必須。
       - アフィン行列は回転・スケール成分も持つが、本実装では並進成分のみを ego として扱う。

3) `extract_tracking_data(results) -> List[Dict]`
   - 入力: `results` — Ultralytics YOLO の `model.track()` が返すオブジェクト（バッチ配列）
   - 出力: 検出オブジェクトのリスト。各要素は辞書で少なくとも以下を含む:
     - `id` (int), `class_id` (int), `box` ([x1,y1,x2,y2]), `conf` (float)
   - 内部処理: `results[0].boxes` の `xyxy`, `id`, `conf`, `cls` を取り出して Python ネイティブ型に変換する。
   - 注意: `boxes.id` が None の場合は空リストを返す。

4) `compute_features(tracked_objects, frame_idx, fps, prev_state, frame_height, ego_motion) -> List[Dict]`
   - 入力:
     - `tracked_objects`: `extract_tracking_data` の出力リスト
     - `frame_idx`: 現在フレーム番号（int）
     - `fps`: 動画のフレームレート（float）
     - `prev_state`: dict で `track_id -> {'cx','cy','frame'}` を保持（速度計算に使用）
     - `frame_height`: フレームの高さ（ピクセル）
     - `ego_motion`: `(ego_vx, ego_vy)` — EgoMotionTracker の出力
   - 出力: CSV書き込み用の行リスト。各行は辞書で以下のキーを持つ（例）:
     - `frame, time, id, class_id, conf, x1, y1, x2, y2, cx, cy, w, h, vx, vy, speed, est_depth, ego_vx, ego_vy`
   - 内部処理:
     - bbox から `w,h,cx,cy,area,aspect_ratio` を計算。
     - `prev_state` があればフレーム差分と `fps` を用いて `vx,vy` を計算（ピクセル／フレーム）。
     - `est_depth` は MVP 用の簡易指標として `frame_height / h` を使用。
     - 行を作成し `prev_state` を更新する。
   - 注意:
     - `vx,vy` は「被写体の画面座標速度」。本実装では補正前の値に `ego` を追加で出力する形（下流で差し引く想定）。

5) `setup_video_io(config) -> (cv2.VideoCapture, cv2.VideoWriter, width, height, fps)`
   - 入力: `config` 辞書（`input_path`, `output_path` など）
   - 出力: OpenCV の `VideoCapture`、`VideoWriter` と幅・高さ・fps
   - 内部処理: `resolve_path` でパス解決、`cv2.VideoCapture` の生成、`VideoWriter` の fourcc と設定
   - 例外: 動画が開けない場合は `FileNotFoundError` を投げる

6) `write_csv_header(csv_path: str)`
   - 入力: 出力CSVのパス
   - 処理: ディレクトリを作成し、CSVヘッダ（関数で定義された列順）を書き出す。

7) `append_rows_to_csv(csv_path: str, rows: List[Dict])`
   - 入力: CSVパスと行リスト
   - 処理: 追記モードで `csv.DictWriter.writerows(rows)` を実行
   - 注意: `rows` が空の場合は何もしない

8) `draw_tracking_info(frame, tracked_objects, id_color_map, ego_motion) -> frame`
   - 入力: 描画するフレーム、追跡オブジェクトリスト、色マップ、ego ベクトル
   - 出力: 描画済みフレーム
   - 処理: 各 bbox を矩形で描き、IDラベルを付与。左上に ego ベクトルの矢印を描画して可視化。

9) `main()`
   - 入力: なし（`CONFIG` を参照）
   - 流れ:
     1. YOLO モデルのロード
     2. `setup_video_io` で入出力初期化
     3. `EgoMotionTracker` の初期化
     4. 各フレームで `model.track()` → `extract_tracking_data()` → `ego_tracker.update()` → `compute_features()` → `append_rows_to_csv()` → `draw_tracking_info()` → 書き出し
     5. 終了時に I/O 解放
   - 副作用: 指定の出力動画とCSVが更新される

--------------------------------------------------------------------------------
運用上の注意（短く）
- 単位の明記: CSVに `fps` を含めるか、yolo.md に単位（ピクセル/フレーム）を明記しておく。
- 閾値調整: FBチェック閾値、RANSAC閾値、`goodFeaturesToTrack` の `maxCorners` などは動画特性により最適値が変わる。
- パフォーマンス: 高解像度では LK が重いので、入力を縮小して推定→スケール戻しで高速化可能。
- テスト: 小さなサンプル動画で `ego_vx/ego_vy` を確認し、背景流れと一致するかを可視化で確認すること。

--------------------------------------------------------------------------------
付録: CSV スキーマ（推奨順）
- frame, time, id, class_id, conf, x1, y1, x2, y2, cx, cy, w, h, vx, vy, speed, est_depth, ego_vx, ego_vy

--------------------------------------------------------------------------------
更新履歴
- 2025-12-14: 初版（mv4.py 仕様に基づく）
