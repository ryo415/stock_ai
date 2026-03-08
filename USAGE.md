# USAGE

このファイルは、`stock_ai` を実際に使うときの操作手順をまとめたものです。
現在の実装に合わせて、まずは手動実行ベースの運用を前提にしています。

## 前提

- Python 3.11 以上
- このリポジトリ直下で作業する
- 依存関係をインストール済みであること

```bash
pip install -e .
```

EDINET を使う場合だけ、事前に API キーを設定します。

```bash
export EDINET_API_KEY=your_api_key
```

## Web UI を使う

Docker Compose で Web UI を起動する場合:

```bash
docker compose up --build
```

起動後に `http://localhost:8501` を開くと、以下をブラウザから実行できます。

- 最新成果物の確認
- データ取得から dataset 更新
- Logistic Regression / LightGBM の学習
- walk-forward 実行
- モデル比較レポート生成
- 指定日付での推論
- 過去推論の答え合わせ

停止する場合:

```bash
docker compose down
```

### Web UI の画面ごとの使い方

`Overview`

- 最新の dataset、ユニバース、比較レポート、推論結果を確認する画面です
- 直近の状況を把握したいときに最初に開きます

`Data Pipeline`

- 価格取得、market 取得、正規化、ユニバース生成、labels 生成、dataset 生成を行う画面です
- `Start Date` と `End Date` はカレンダーから指定できます
- `財務データを今回更新する` をオンにした場合だけ、EDINET から財務データを取り直します

`Training`

- `baseline_logreg` と `baseline_lightgbm` を学習する画面です
- 最新の dataset を使ってモデルを更新します

`Walk-Forward`

- 時系列の実運用に近い評価を行う画面です
- 再学習を繰り返しながら、各時点の予測性能を検証します

`Compare`

- 2 つのモデルを同条件で比較する画面です
- 学習結果と walk-forward の両方をまとめて確認します

`Inference`

- 指定日時点での推論を行う画面です
- その日から約60営業日後に10%以上上がる確率を銘柄ごとに出します
- あわせて、保存済み prediction の答え合わせもここで実行できます

`Artifacts`

- 生成済みのユニバース、dataset、モデル、比較レポート、推論結果の保存先を確認する画面です

### Web UI でのおすすめ操作順

1. `Data Pipeline` で価格・market・dataset を更新
2. `Training` で 2 モデルを学習
3. `Walk-Forward` で 2 モデルを評価
4. `Compare` で比較レポートを生成
5. `Inference` で良い方のモデルを使って推論
6. 過去推論を検証したい場合は同じ `Inference` 画面で答え合わせ

## まず確認すること

設定を確認したいとき:

```bash
python -m stock_ai config list
python -m stock_ai config show data universe
python -m stock_ai config show train baseline_logreg
python -m stock_ai config show train baseline_lightgbm
python -m stock_ai config show backtest default
```

## ユースケース 1: 初回セットアップ

最初に全体を一通り動かす場合:

```bash
python -m stock_ai data fetch-prices
python -m stock_ai data fetch-macro
python -m stock_ai data normalize-prices
python -m stock_ai data normalize-macro
python -m stock_ai data build-universe
python -m stock_ai features build-labels
python -m stock_ai features build-dataset
python -m stock_ai train run --config baseline_logreg
python -m stock_ai train run --config baseline_lightgbm
python -m stock_ai backtest walk-forward --config default --train-config baseline_logreg
python -m stock_ai backtest walk-forward --config default --train-config baseline_lightgbm
python -m stock_ai report compare-models
```

この流れで、価格取得から比較レポート生成まで一通り確認できます。

## ユースケース 2: 流動性条件でユニバースを更新したい

候補銘柄の価格履歴をもとに、平均売買代金で投資対象を自動生成する場合:

```bash
python -m stock_ai data fetch-prices
python -m stock_ai data normalize-prices
python -m stock_ai data build-universe
```

ユニバース設定は [configs/data/universe.yaml](/home/ryo/Programs/stock_ai/configs/data/universe.yaml) の以下を見ます。

- `universe.liquidity_filter.min_average_daily_value_jpy`
- `universe.liquidity_filter.lookback_days`
- `universe.liquidity_filter.min_observation_days`
- `universe.liquidity_filter.max_tickers`
- `universe.ticker_selection.candidate_tickers`

生成結果は `data/processed/universe/latest.json` と `data/processed/universe/liquidity_universe_*.json` に保存されます。

## ユースケース 3: 最新データで学習用 dataset を更新したい

価格と market 系を更新して dataset を作り直す場合:

```bash
python -m stock_ai data fetch-prices
python -m stock_ai data fetch-macro
python -m stock_ai data normalize-prices
python -m stock_ai data normalize-macro
python -m stock_ai features build-labels
python -m stock_ai features build-dataset
```

出力先:

- `data/interim/prices/`
- `data/interim/market/`
- `data/processed/labels/`
- `data/processed/datasets/`

## ユースケース 4: 財務データも更新したい

EDINET の raw 書類を取得して中間特徴量まで更新する場合:

```bash
export EDINET_API_KEY=your_api_key
python -m stock_ai data fetch-fundamentals --tickers 7203.T 6758.T
python -m stock_ai data normalize-fundamentals
python -m stock_ai features build-dataset
```

注意:

- `fetch-fundamentals` は EDINET API キーが必要です
- 今の実装は CSV ZIP ベースの抽出です
- 実書類によっては項目名の揺れで抽出精度の調整が必要です

## ユースケース 5: Logistic Regression を学習したい

```bash
python -m stock_ai train run --config baseline_logreg
```

入力 dataset を明示したい場合:

```bash
python -m stock_ai train run \
  --config baseline_logreg \
  --dataset-input-path data/processed/datasets/dataset_baseline_v1_YYYYMMDDTHHMMSSZ.csv
```

出力先:

- `models/`
- `reports/tables/train_*.json`
- `data/metadata/train_*.json`

## ユースケース 6: LightGBM を学習したい

```bash
python -m stock_ai train run --config baseline_lightgbm
```

LightGBM は非線形な関係を拾いやすい一方で、過学習しやすいので、単発の test 指標だけでなく walk-forward も併せて見ます。

## ユースケース 7: ある時点の推奨候補を見たい

最新日付で推論する場合:

```bash
python -m stock_ai inference predict --config default --train-config baseline_logreg
```

LightGBM を使う場合:

```bash
python -m stock_ai inference predict --config default --train-config baseline_lightgbm
```

日付を指定したい場合:

```bash
python -m stock_ai inference predict \
  --config default \
  --train-config baseline_lightgbm \
  --prediction-date 2025-10-02
```

出力先:

- `reports/tables/predictions/`
- `data/metadata/`

見ればよい列:

- `ticker`
- `probability`
- `prediction`

## ユースケース 8: 固定モデルで簡易バックテストしたい

1つの学習済みモデルで全期間をスコアして簡易評価する場合:

```bash
python -m stock_ai backtest run --config default --train-config baseline_logreg
```

これは最初の確認用です。実運用に近い評価を見たい場合は、次の walk-forward を優先します。

## ユースケース 9: 実運用に近い walk-forward 評価をしたい

Logistic Regression:

```bash
python -m stock_ai backtest walk-forward --config default --train-config baseline_logreg
```

LightGBM:

```bash
python -m stock_ai backtest walk-forward --config default --train-config baseline_lightgbm
```

walk-forward は、各リバランス時点までの過去データだけで再学習し、その時点の銘柄群を予測します。

見るべき指標:

- `total_return`
- `benchmark_total_return`
- `avg_portfolio_return`
- `win_rate`
- `max_drawdown`
- `trained_window_count`

## ユースケース 10: 2 つのモデルを同条件で比較したい

最新の学習レポートと walk-forward レポートを使う場合:

```bash
python -m stock_ai report compare-models
```

ファイルを明示したい場合:

```bash
python -m stock_ai report compare-models \
  --left-train-report-path reports/tables/train_baseline_logreg_v1_YYYYMMDDTHHMMSSZ.json \
  --right-train-report-path reports/tables/train_baseline_lightgbm_v1_YYYYMMDDTHHMMSSZ.json \
  --left-walk-forward-report-path reports/tables/backtest_top_n_hold_60bd_walk_forward_YYYYMMDDTHHMMSSZ.json \
  --right-walk-forward-report-path reports/tables/backtest_top_n_hold_60bd_walk_forward_YYYYMMDDTHHMMSSZ.json
```

出力先:

- `reports/tables/model_comparison_*.json`
- `reports/tables/model_comparison_*.md`

## ユースケース 10.5: 過去の推論が実際に当たっていたか確認したい

最新の prediction を評価する場合:

```bash
python -m stock_ai report evaluate-prediction
```

ファイルを明示する場合:

```bash
python -m stock_ai report evaluate-prediction \
  --prediction-input-path reports/tables/predictions/predict_default_inference_YYYYMMDDTHHMMSSZ.csv \
  --dataset-input-path data/processed/datasets/dataset_baseline_v1_YYYYMMDDTHHMMSSZ.csv
```

出力先:

- `reports/tables/prediction_evaluation_*.json`
- `reports/tables/prediction_evaluation_*.csv`

見るべき項目:

- `accuracy`
- `predicted_positive_count`
- `realized_positive_count`
- `avg_realized_future_return_top5`
- `positive_rate_top5`

### 過去推論の答え合わせ手順

1. まず過去日付で推論する

```bash
python -m stock_ai inference predict \
  --config default \
  --train-config baseline_lightgbm \
  --prediction-date 2025-03-05
```

2. その prediction ファイルを評価する

```bash
python -m stock_ai report evaluate-prediction
```

3. `prediction_evaluation_*.json` を見て、上位候補が実際にどうだったかを確認する

特に見る項目:

- `avg_realized_future_return_top5`
  - 推論上位 5 銘柄の実際の平均リターン
- `positive_rate_top5`
  - 推論上位 5 銘柄のうち、実際に 10%以上上昇した割合
- `predicted_positive_count`
  - モデルが `prediction=1` と判断した銘柄数
- `realized_positive_count`
  - 実際に条件達成した銘柄数

### Web UI で過去推論を答え合わせする手順

1. `Inference` 画面で `Prediction Date` を指定して `Run Inference`
2. その下の `Evaluate Latest Prediction` を押す
3. `Latest Evaluation Summary` で結果を確認する

補足:

- 過去推論の答え合わせは、単発確認として有効です
- モデル調整の主軸は、引き続き `Walk-Forward` と `Compare` に置くのが安全です

## ユースケース 11: 今のおすすめ標準フロー

通常運用で一番無難なのは次です。

1. `data fetch-prices`
2. `data fetch-macro`
3. `data normalize-prices`
4. `data normalize-macro`
5. `data build-universe`
6. `features build-labels`
7. `features build-dataset`
8. `train run --config baseline_logreg`
9. `train run --config baseline_lightgbm`
10. `backtest walk-forward --train-config baseline_logreg`
11. `backtest walk-forward --train-config baseline_lightgbm`
12. `report compare-models`
13. 良い方のモデルで `inference predict`

この一連を 1 本で回したい場合は、以下を使えます。

```bash
bash scripts/run_workflow.sh
```

既定では比較結果から `baseline_logreg` と `baseline_lightgbm` のどちらを推論に使うか自動選択します。

環境変数:

- `PREDICT_MODEL=best`
- `PREDICT_MODEL=baseline_logreg`
- `PREDICT_MODEL=baseline_lightgbm`
- `RUN_FUNDAMENTALS=1`

例:

```bash
PREDICT_MODEL=baseline_lightgbm bash scripts/run_workflow.sh
RUN_FUNDAMENTALS=1 bash scripts/run_workflow.sh
```

## よくある調整ポイント

ユニバースを調整したい:

- [configs/data/universe.yaml](/home/ryo/Programs/stock_ai/configs/data/universe.yaml)

ラベル条件を変えたい:

- [configs/features/labels.yaml](/home/ryo/Programs/stock_ai/configs/features/labels.yaml)

特徴量セットを変えたい:

- [configs/features/feature_set_baseline.yaml](/home/ryo/Programs/stock_ai/configs/features/feature_set_baseline.yaml)

学習期間やモデル設定を変えたい:

- [configs/train/baseline_logreg.yaml](/home/ryo/Programs/stock_ai/configs/train/baseline_logreg.yaml)
- [configs/train/baseline_lightgbm.yaml](/home/ryo/Programs/stock_ai/configs/train/baseline_lightgbm.yaml)

バックテスト条件を変えたい:

- [configs/backtest/default.yaml](/home/ryo/Programs/stock_ai/configs/backtest/default.yaml)

## 主要パラメータの意味

このプロジェクトでは、設定値は「何を対象にするか」「どの未来を当てにいくか」「どう評価するか」を決めます。
ここでは、現時点の主要パラメータとその意味を簡潔にまとめます。

### ユニバース設定

対象ファイル:

- [configs/data/universe.yaml](/home/ryo/Programs/stock_ai/configs/data/universe.yaml)

`universe.liquidity_filter.lookback_days`

- 直近何営業日で流動性を測るか
- 現在は `20`
- 1か月弱の売買状況で見る設定です
- 短くすると直近の変化に敏感になります
- 長くすると安定しますが、最近の変化は拾いにくくなります

`universe.liquidity_filter.min_observation_days`

- 流動性判定に必要な最低観測日数
- 現在は `15`
- 欠損や上場直後銘柄をある程度除くための条件です
- 下げると候補は増えますが、安定性は下がります

`universe.liquidity_filter.min_average_daily_value_jpy`

- 平均売買代金の下限
- 現在は `3000000000` 円
- 低流動性銘柄を除いて、実運用で売買しやすい銘柄に寄せるための条件です
- 下げると対象銘柄は増えます
- 上げるとより大型・高流動性寄りになります

`universe.liquidity_filter.max_tickers`

- 最終的に採用する銘柄数
- 現在は `30`
- 実運用寄りに、流動性上位だけへ絞る設定です
- 候補母集団とは別で、実際にモデル対象へ残す数です

`universe.ticker_selection.candidate_tickers`

- 流動性フィルタ前の候補母集団
- 現在は `200` 銘柄
- 日本株の中で比較的流動性の高い大型・中大型銘柄を中心に入れています
- 広げるほど上位流動性銘柄を選びやすくなりますが、取得コストは増えます

### ラベル設定

対象ファイル:

- [configs/features/labels.yaml](/home/ryo/Programs/stock_ai/configs/features/labels.yaml)

`label.threshold`

- 何%以上上がれば `label=1` とするか
- 現在は `0.10`
- つまり 10%以上上昇で成功判定です
- 下げると positive が増え、上げると positive は減ります

`label.horizon_business_days`

- 何営業日先を予測対象にするか
- 現在は `60`
- 約3か月先を当てにいく設定です
- 短くすると短期寄り、長くすると中期寄りになります

### 特徴量設定

対象ファイル:

- [configs/features/feature_set_baseline.yaml](/home/ryo/Programs/stock_ai/configs/features/feature_set_baseline.yaml)

`price_features`

- 価格変化率、移動平均乖離、ボラティリティ、出来高変化などを使うかを決めます
- ベースラインでは、まず価格だけで end-to-end を成立させるための基本特徴量です

`relative_strength`

- TOPIX や日経平均に対してどれだけ強いかを見る特徴量です
- 個別銘柄の地合いに対する相対的な強さを測る意図です

`fundamentals`

- PER、PBR、ROE、売上成長率、営業利益率などを使う設定です
- 長めの保有期間では価格だけでなく財務も効く可能性があるため入れています

### 学習期間設定

対象ファイル:

- [configs/train/baseline_logreg.yaml](/home/ryo/Programs/stock_ai/configs/train/baseline_logreg.yaml)
- [configs/train/baseline_lightgbm.yaml](/home/ryo/Programs/stock_ai/configs/train/baseline_lightgbm.yaml)

`dataset.train_start_date`

- 学習データの開始日
- 現在は `2015-01-01`
- 長めの履歴を使って市場局面のばらつきを入れる意図です

`dataset.validation_start_date`

- validation の開始日
- 現在は `2022-01-01`
- 学習データと評価データを時系列で分けるための境目です

`dataset.test_start_date`

- test の開始日
- 現在は `2024-01-01`
- 比較的新しい期間を out-of-sample として残すための設定です

### Logistic Regression の主なパラメータ

対象ファイル:

- [configs/train/baseline_logreg.yaml](/home/ryo/Programs/stock_ai/configs/train/baseline_logreg.yaml)

`model.params.c`

- 正則化の強さ
- 現在は `1.0`
- 小さくすると保守的、大きくすると複雑な当てはまりを許します

`model.params.class_weight`

- クラス不均衡への対応
- 現在は `balanced`
- 上昇する銘柄が少ないときでも学習しやすくするための設定です

`model.params.max_iter`

- 最適化の反復回数
- 現在は `1000`
- 収束不足を避けるために標準より長めにしています

### LightGBM の主なパラメータ

対象ファイル:

- [configs/train/baseline_lightgbm.yaml](/home/ryo/Programs/stock_ai/configs/train/baseline_lightgbm.yaml)

`model.params.learning_rate`

- 1本ごとの木の学習率
- 現在は `0.05`
- やや控えめで安定寄りの設定です

`model.params.num_leaves`

- 木の複雑さ
- 現在は `31`
- ベースラインとして過剰に複雑にしすぎない設定です

`model.params.n_estimators`

- 木の本数
- 現在は `300`
- ある程度の表現力を持たせつつ、学習時間を抑えるための初期値です

`model.params.subsample`

- 各木で使うサンプル比率
- 現在は `0.8`
- 過学習抑制のため、毎回全件を使わない設定です

`model.params.colsample_bytree`

- 各木で使う特徴量比率
- 現在は `0.8`
- 特徴量の使いすぎによる過学習を少し抑えます

`model.params.class_weight`

- クラス不均衡への対応
- 現在は `balanced`

### 推論設定

対象ファイル:

- [configs/inference/default.yaml](/home/ryo/Programs/stock_ai/configs/inference/default.yaml)

`selection.return_top_n`

- 推論結果として何銘柄返すか
- 出力を見やすくするための件数制御です
- backtest の `top_n` とは別です

`selection.sort_by`

- 通常は `probability`
- 確率の高い順に候補を並べます

### バックテスト設定

対象ファイル:

- [configs/backtest/default.yaml](/home/ryo/Programs/stock_ai/configs/backtest/default.yaml)

`backtest.holding_period_business_days`

- 何営業日保有する前提で評価するか
- 現在は `60`
- ラベルと同じ horizon に揃えています

`backtest.rebalance_frequency`

- 何頻度で銘柄を入れ替えるか
- 現在は `monthly`
- 個人運用でも回しやすい月次です

`portfolio.top_n`

- 各時点で何銘柄採用するか
- 現在は `5`
- 30銘柄ユニバースに対して、厳選型に寄せた設定です

`portfolio.weighting`

- 現在は `equal_weight`
- まずは単純で比較しやすい等金額配分にしています

`costs.fee_bps`

- 片道手数料の想定
- 現在は `10bps`

`costs.slippage_bps`

- 片道スリッページの想定
- 現在は `5bps`

`execution.allow_overlap_positions`

- 前のポジション保有中でも次のリバランスを重ねるか
- 現在は `false`
- 単純な非オーバーラップ運用で評価しています

### walk-forward 設定

対象ファイル:

- [configs/backtest/default.yaml](/home/ryo/Programs/stock_ai/configs/backtest/default.yaml)

`walk_forward.prediction_start_date`

- walk-forward をどこから始めるか
- 現在は `2024-01-01`
- 比較的新しい期間で実運用近い評価を見るためです

`walk_forward.training_start_date`

- 各時点の再学習で、どこから履歴を使い始めるか
- 現在は `2015-01-01`

`walk_forward.min_training_rows`

- 再学習を行うための最小学習行数
- 現在は `252`
- 少なすぎるデータで学習しないようにする保険です

## よくある詰まり方

`No files found` が出る:

- 前段のコマンドをまだ実行していないことが多いです
- `data/raw`、`data/interim`、`data/processed` の順で出力があるか確認します

`EDINET API key is missing` が出る:

- `EDINET_API_KEY` を設定してから実行します

`No liquidity-filtered universe available` が出る:

- 先に `python -m stock_ai data build-universe` を実行します

銘柄数が少なすぎる:

- `min_average_daily_value_jpy` を下げます
- `max_tickers` を増やします
- `candidate_tickers` を増やします

モデル比較で差が出ない:

- ユニバースが小さすぎる可能性があります
- `top_n` が候補銘柄数に対して大きすぎる可能性があります

## 補足

このシステムは、現段階では研究・分析・検証用途の個人利用を前提にしています。
まずは `walk-forward` と比較レポートを見ながら、設定とユニバースを調整していく使い方が安全です。
