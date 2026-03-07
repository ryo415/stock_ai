# Configs

このディレクトリは、`stock_ai` の設定ファイルを用途別に管理する。
初期段階では YAML を採用し、コード内への直書きを避ける。

## 構成

- `configs/data/`
  - データソース
  - 対象期間
  - 対象ユニバース
- `configs/features/`
  - 特徴量定義
  - ラベル定義
  - 欠損処理
- `configs/train/`
  - 学習条件
  - モデル種別
  - 評価分割
- `configs/inference/`
  - 推論対象日
  - 出力先
  - 上位銘柄抽出条件
- `configs/backtest/`
  - 売買ルール
  - コスト
  - ベンチマーク

## 命名方針

- 1 ファイル 1 目的を原則とする
- 環境差分よりも「用途差分」を優先して分割する
- 確定していない値は `TBD` ではなく、候補値か `null` で表現する

## 初期ファイル

- `configs/data/universe.yaml`
- `configs/data/sources.yaml`
- `configs/features/labels.yaml`
- `configs/features/feature_set_baseline.yaml`
- `configs/train/baseline_logreg.yaml`
- `configs/train/baseline_lightgbm.yaml`
- `configs/inference/default.yaml`
- `configs/backtest/default.yaml`
