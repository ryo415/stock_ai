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
