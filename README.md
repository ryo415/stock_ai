# stock_ai

個人で構築する株式予測 AI プロジェクトの初期リポジトリです。
目的は、過去のさまざまな情報から「約 60 営業日後に 10%以上上昇しているか」を分類するモデルを作ることです。

## 現在の状態

現時点では、最小のデータ取得、特徴量生成、学習、推論、バックテストまで通る土台を実装済みです。
初版ドキュメントとして以下を用意しています。

- [要件定義書](./docs/requirements.md)
- [アーキテクチャ設計](./docs/architecture.md)
- [実装タスク一覧](./docs/tasks.md)
- [エージェント運用ルール](./AGENTS.md)

## 目標

- 株価、財務、マクロの過去データを統合する
- 約 60 営業日先の上昇有無を二値分類する
- 時系列リークを避けた評価を行う
- 個人開発で継続改善できる構成にする

## 想定スコープ

初期スコープ:
- データ収集
- 前処理
- 特徴量生成
- ベースラインモデル学習
- 時系列評価
- 簡易バックテスト

初期スコープ外:
- 完全自動売買
- 高頻度取引
- 大規模分散基盤

## 想定ディレクトリ

現在は以下の構成で整備しています。

```text
stock_ai/
├── AGENTS.md
├── README.md
├── docs/
│   ├── architecture.md
│   ├── requirements.md
│   └── tasks.md
├── configs/
│   ├── backtest/
│   ├── data/
│   ├── features/
│   ├── inference/
│   └── train/
├── data/
│   ├── interim/
│   ├── metadata/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── tables/
└── src/
    └── stock_ai/
```

## 最初に決めること

このプロジェクトでは、次の 5 点を早めに確定する必要があります。

1. 日本株の対象ユニバース
2. 利用可能な無料データソースと必要時の有料データソース
3. 推論頻度
4. 流動性フィルタ
5. 初期バックテスト条件

## 設計方針

- まずは説明可能で再現しやすいベースラインを作る
- 未来情報リークを最優先で防ぐ
- 精度だけでなく、投資判断に近い評価指標も見る
- 初期段階では無料データソースを優先する
- 未確定事項は `TBD` として管理する

## 開発セットアップ

最小構成の Python パッケージ設定として `pyproject.toml` を用意しています。

```bash
pip install -e .
```

Docker Compose で Web UI を起動する場合:

```bash
docker compose up --build
```

起動後は `http://localhost:8501` で Web UI を開けます。

設定ファイルは YAML で管理し、`stock_ai.utils.config` から読み込みます。

```python
from stock_ai.utils import load_config

config = load_config("data", "universe")
```

共通エントリポイントは `python -m stock_ai` で利用できます。

```bash
python -m stock_ai config list
python -m stock_ai config list data
python -m stock_ai config show data universe
```

現時点の実処理サブコマンドは、`fetch-prices`、`fetch-fundamentals`、`fetch-macro`、`normalize-prices`、`normalize-fundamentals`、`normalize-macro`、`build-labels`、`build-dataset`、`train run`、`inference predict`、`backtest run`、`backtest walk-forward` です。`fetch-fundamentals` の実データ取得には EDINET API キーが必要です。

```bash
python -m stock_ai data fetch-prices
python -m stock_ai data fetch-fundamentals
python -m stock_ai data fetch-macro
python -m stock_ai data normalize-prices
python -m stock_ai data normalize-fundamentals
python -m stock_ai features build-labels
python -m stock_ai features build-dataset
python -m stock_ai train run --config baseline_logreg
python -m stock_ai inference predict --config default --train-config baseline_logreg
python -m stock_ai backtest run --config default --train-config baseline_logreg
python -m stock_ai backtest walk-forward --config default --train-config baseline_logreg
```

`fetch-prices` は現在 `yfinance` を使い、`configs/data/universe.yaml` の `ticker_selection` を取得対象として使います。今は `candidate_tickers` から価格を取得し、`data build-universe` で平均売買代金条件を満たす銘柄だけを自動選定できます。
必要なら以下のように CLI から上書きできます。

```bash
python -m stock_ai data fetch-prices --tickers 7203.T 6758.T --start-date 2024-01-01 --end-date 2024-03-31
```

流動性条件で投資対象ユニバースを自動生成するには以下を使います。

```bash
python -m stock_ai data build-universe
```

生成結果は `data/processed/universe/liquidity_universe_*.json` に保存され、以後の `fetch-prices` は最新の生成済みユニバースを優先して使います。まだ生成済みファイルがない場合は `candidate_tickers` にフォールバックします。

市場指数と為替の系列は以下で取得できます。

```bash
python -m stock_ai data fetch-macro --start-date 2015-01-01 --end-date 2025-12-31
python -m stock_ai data normalize-macro
```

取得後の価格データは以下で `data/interim/prices/` に正規化できます。

```bash
python -m stock_ai data normalize-prices
```

正規化済み価格データから `future_return_60bd` と `label` を生成するには以下を実行します。

```bash
python -m stock_ai features build-labels
```

価格特徴量とラベルを結合した最小の学習用データセットは以下で生成できます。

```bash
python -m stock_ai features build-dataset
```

現在の `build-dataset` は、価格特徴量、`relative_strength`、市場指数ベースの macro 特徴量、fundamentals 系の一部を実装済みです。
fundamentals では `market_cap`、`per`、`pbr`、`roe`、`revenue_growth_yoy`、`operating_margin` を扱います。
未実装なのは `policy_rate` と `cpi_yoy`、および EDINET raw からの抽出精度改善です。

最小の Logistic Regression 学習は以下で実行できます。

```bash
python -m stock_ai train run --config baseline_logreg
```

LightGBM ベースラインを使う場合は以下です。

```bash
python -m stock_ai train run --config baseline_lightgbm
```

モデルは `models/`、評価結果は `reports/tables/` に保存されます。

学習済みモデルで最新日付の銘柄群を実推論するには以下を実行します。

```bash
python -m stock_ai inference predict --config default --train-config baseline_logreg
```

推論結果は `reports/tables/predictions/`、メタデータは `data/metadata/` に保存されます。

モデル確率に基づく簡易バックテストは以下で実行できます。

```bash
python -m stock_ai backtest run --config default --train-config baseline_logreg
```

バックテスト結果は `reports/tables/` に、サマリ JSON、取引一覧 CSV、エクイティカーブ CSV として保存されます。

walk-forward 評価は、各リバランス時点までの過去データだけで学習し直し、その時点の銘柄群を予測して評価します。

```bash
python -m stock_ai backtest walk-forward --config default --train-config baseline_logreg
```

既定では `configs/backtest/default.yaml` の `walk_forward` 設定を使い、`2024-01-01` 以降を対象に expanding window で再学習します。

2 つのモデルを同条件で比較するレポートは以下で生成できます。

```bash
python -m stock_ai report compare-models
```

比較結果は `reports/tables/model_comparison_*.json` と `reports/tables/model_comparison_*.md` に保存されます。

過去の推論結果が実際に当たっていたかを確認するには以下を使います。

```bash
python -m stock_ai report evaluate-prediction
```

これは prediction ファイルと dataset を照合し、実現した `future_return_60bd` と `label` をまとめます。

`fetch-fundamentals` は現在 EDINET API を使い、実行前に `EDINET_API_KEY` 環境変数が必要です。

```bash
export EDINET_API_KEY=your_api_key
python -m stock_ai data fetch-fundamentals --tickers 7203.T --start-date 2024-04-01 --end-date 2024-04-05
```

fundamentals の正規化は、raw 書類取得後に以下で `data/interim/fundamentals/` へ変換します。

```bash
python -m stock_ai data normalize-fundamentals
```

その後、fundamentals を含む dataset を組む場合は `features build-dataset` が最新の `data/interim/fundamentals/fundamentals_features_*.csv` を自動利用します。

生成先の例:
- `data/metadata/`
- `data/raw/prices/`
- `data/raw/fundamentals/`
- `data/raw/market/`
- `data/interim/prices/`
- `data/interim/fundamentals/`
- `data/interim/market/`
- `data/processed/labels/`
- `data/processed/datasets/`
- `data/processed/`
- `models/`
- `reports/tables/`

## 注意

このリポジトリは、初期段階では自己利用を前提とした分析・研究・検証用途を想定しています。
