# predict_area — 簡易作業メモ (README2)

目的
- 動画に対してブロック（K分割）ごとの安全確率を推論し、可視化（オーバーレイ）で検証するための実験パイプラインの整理。

現在の進捗（要点）
- 新規: `src/infer/viz2.py` を追加 — 確率読み込み、ヒートマップ合成、矩形で「最も安全」セルを描画する関数を実装。
- 新規: `src/infer/run_viz_infer.py` を追加 — 動画 + `.npz`（確率）を受け取ってオーバーレイ動画を生成する簡易ランナー。デフォルト出力先は `predict_area/result/test_result/`、生成NPZは `npz/test_npz/`。
- 既存: `src/infer/create_answer.py` を拡張 — 動画モードを追加し、（存在する）検出JSONを使ってフレームごとの確率（JSON/NPZ）を生成できるようにした。検出JSONが無い場合は一様分布を返す実装。
- 仮想環境: プロジェクト直下に `.venv` を作成し、`opencv-python`, `numpy`, `Pillow` をインストール済み（ローカル開発用）。
- テストデータで確認: `predict_area/data/json_data/*.json` を `.npz` に変換し、`predict_area/viz_from_json.mp4` と `predict_area/viz_from_json_rect.mp4` を生成。最終的にファイルは整理済み（`npz/test_npz/` と `predict_area/result/test_result/`）。

現在のファイル/出力（主なもの）
- スクリプト
  - `predict_area/src/infer/viz2.py` — 可視化ユーティリティ
  - `predict_area/src/infer/run_viz_infer.py` — 動画 + 確率 → オーバーレイ
  - `predict_area/src/infer/create_answer.py` — 検出JSON → フレーム確率（動画対応追加）
- データ（テスト）
  - `npz/test_npz/` — 生成した npz（例: `clark-center-2019-02-28_0.npz`）
  - `predict_area/result/test_result/` — 生成したオーバーレイ動画（例: `viz_from_json_rect.mp4`）

簡単な実行手順（現在のコードでの流れ）
1. （任意）検出JSONがある場合はそのまま `create_answer.py` を使って `.npz` を作成:
   ```powershell
   python predict_area/src/infer/create_answer.py --video predict_area/data/train_data/data_1.mp4 --detections <path_to_detections.json> --out_json predict_area/answers_data_1.json --out_npz npz/test_npz/data_1.npz --num_area 8
   ```
2. 検出JSONが無い場合は自動検出を行う（未実行）か、テスト用の JSON を `predict_area/data/json_data/` に置く。
3. 可視化実行（デフォルトで `npz/test_npz/<video_basename>.npz` と `predict_area/result/test_result/` を使用）:
   ```powershell
   python predict_area/src/infer/run_viz_infer.py --video predict_area/data/train_data/data_1.mp4 --probs npz/test_npz/data_1.npz --out predict_area/result/test_result/my_output.mp4 --num_area 8
   ```

残りタスク（優先順位順）
1. 自動検出パイプラインの追加: `ultralytics`（YOLO）を用いて動画→検出JSON を生成するスクリプトを追加し、`create_answer.py` と連結する（自動化）。
2. `create_answer.py` の検出JSONフォーマット統一: 生成する/受け取る JSON のスキーマを明確化・ドキュメント化する。
3. 可視化オプションの CLI 化: カラーマップ/矩形色/alpha/EMA係数を `run_viz_infer.py` の引数で変更できるようにする。
4. パフォーマンス改善: 大きな動画処理でのバッファ/マルチプロセス分離（デコード→推論→描画）を検討。
5. テスト自動化: 生成物（npz, 動画）をまとめて保存する小さな `make_test_run.sh` / PowerShell スクリプトを作成。

短期の提案（次にやること）
- 今すぐ実行するなら: 自動検出を有効化するために `ultralytics` を `.venv` にインストールし、`src/preprocess/yolo_pose.py` の `yolo_track_pose()` を呼ぶラッパースクリプトを用意します（私が代行で実行します）。
- 先に確認したいなら: `predict_area/data/json_data/` にあるテストJSONでさらに可視化パラメータを調整して見やすくします（色/枠/厚さ）。

備考
- `create_answer.py` は「検出→確率生成」の後半を担当するツールです。自動で動画から検出を行いたい場合は前処理（YOLOなど）を追加する必要があります。これが今回の次の主要作業です。

----
+更新: 2025-12-16 生成・テスト実行の記録を反映
