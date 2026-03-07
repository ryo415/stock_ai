"""Streamlit web UI for stock_ai."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from datetime import date

from stock_ai.backtest.commands import run_walk_forward_command
from stock_ai.data.commands import (
    run_build_universe_command,
    run_fetch_fundamentals_command,
    run_fetch_macro_command,
    run_fetch_prices_command,
    run_normalize_macro_command,
    run_normalize_prices_command,
)
from stock_ai.features.commands import run_build_dataset_command, run_build_labels_command
from stock_ai.inference.commands import run_predict_command
from stock_ai.modeling.commands import run_train_command
from stock_ai.reporting.commands import run_compare_models_command
from stock_ai.utils import ConfigError
from stock_ai.utils.io import project_path


TRAIN_CONFIGS = ("baseline_logreg", "baseline_lightgbm")


def _latest_file(directory: str, pattern: str) -> Path | None:
    root = project_path(directory)
    files = sorted(root.glob(pattern)) if root.exists() else []
    return files[-1] if files else None


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _status_card(title: str, value: str, caption: str = "") -> None:
    st.metric(label=title, value=value)
    if caption:
        st.caption(caption)


def _latest_prediction_path() -> Path | None:
    predictions_dir = project_path("reports/tables/predictions")
    if not predictions_dir.exists():
        return None
    csv_files = sorted(predictions_dir.glob("predict_default_inference_*.csv"))
    parquet_files = sorted(predictions_dir.glob("predict_default_inference_*.parquet"))
    candidates = csv_files + parquet_files
    return sorted(candidates)[-1] if candidates else None


def render_overview() -> None:
    st.subheader("Overview")
    st.write(
        "この画面は全体の状況確認用です。最新の dataset、比較レポート、推論結果、"
        "流動性ユニバースをまとめて見て、直近の状態を素早く把握できます。"
    )
    dataset_path = _latest_file("data/processed/datasets", "dataset_*.csv")
    comparison_path = _latest_file("reports/tables", "model_comparison_*.json")
    prediction_path = _latest_prediction_path()
    universe_path = _latest_file("data/processed/universe", "liquidity_universe_*.json")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _status_card("Latest Dataset", dataset_path.name if dataset_path else "-", "data/processed/datasets")
    with col2:
        _status_card("Latest Comparison", comparison_path.name if comparison_path else "-", "reports/tables")
    with col3:
        _status_card("Latest Prediction", prediction_path.name if prediction_path else "-", "reports/tables/predictions")
    with col4:
        _status_card("Latest Universe", universe_path.name if universe_path else "-", "data/processed/universe")

    comparison = _load_json(comparison_path)
    if comparison:
        st.markdown("**Latest Model Comparison**")
        left = comparison["models"][0]
        right = comparison["models"][1]
        delta = comparison["comparison"]["walk_forward_delta_right_minus_left"]
        st.write(
            {
                "left_model": left["name"],
                "right_model": right["name"],
                "right_minus_left_total_return": delta.get("total_return"),
                "right_minus_left_win_rate": delta.get("win_rate"),
            }
        )

    prediction_df = _load_csv(prediction_path)
    if prediction_df is not None and not prediction_df.empty:
        st.markdown("**Latest Top Predictions**")
        st.dataframe(
            prediction_df[["ticker", "probability", "prediction"]]
            .sort_values("probability", ascending=False)
            .head(10),
            use_container_width=True,
        )


def render_training() -> None:
    st.subheader("Training")
    st.write(
        "この画面はモデル学習用です。最新の dataset を使って "
        "`baseline_logreg` または `baseline_lightgbm` を学習し、"
        "学習済みモデルと評価レポートを生成します。"
    )
    dataset_path = _latest_file("data/processed/datasets", "dataset_*.csv")
    st.text_input("Dataset", value=str(dataset_path) if dataset_path else "", disabled=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Logistic Regression", use_container_width=True):
            try:
                result = run_train_command("baseline_logreg", dataset_input_path=str(dataset_path) if dataset_path else None)
                st.success("baseline_logreg training completed")
                st.json(result)
            except ConfigError as exc:
                st.error(str(exc))
    with col2:
        if st.button("Train LightGBM", use_container_width=True):
            try:
                result = run_train_command("baseline_lightgbm", dataset_input_path=str(dataset_path) if dataset_path else None)
                st.success("baseline_lightgbm training completed")
                st.json(result)
            except ConfigError as exc:
                st.error(str(exc))


def render_data_pipeline() -> None:
    st.subheader("Data Pipeline")
    st.write(
        "この画面はデータセット更新用です。価格取得、market 取得、正規化、"
        "流動性ユニバース生成、labels 生成、dataset 生成までをブラウザから順番に実行できます。"
    )

    latest_raw_prices = _latest_file("data/raw/prices", "prices_yfinance_*.csv")
    latest_raw_macro = _latest_file("data/raw/market", "market_yfinance_*.csv")
    latest_normalized_prices = _latest_file("data/interim/prices", "prices_normalized_*.csv")
    latest_normalized_macro = _latest_file("data/interim/market", "market_normalized_*.csv")
    latest_labels = _latest_file("data/processed/labels", "labels_*.csv")

    st.markdown("**Fetch Options**")
    fetch_start_date_value = st.date_input(
        "Start Date",
        value=date(2015, 1, 1),
        key="data_fetch_start_date",
    )
    use_end_date = st.checkbox("Specify End Date", value=False, key="data_use_end_date")
    fetch_end_date_value = None
    if use_end_date:
        fetch_end_date_value = st.date_input(
            "End Date",
            value=date.today(),
            key="data_fetch_end_date",
        )
    else:
        st.caption("End Date を指定しない場合は最新日まで取得します。")

    fetch_start_date = fetch_start_date_value.isoformat()
    normalized_end_date = fetch_end_date_value.isoformat() if fetch_end_date_value else None

    if normalized_end_date and normalized_end_date < fetch_start_date:
        st.error("End Date must be on or after Start Date.")
        return

    st.caption(
        f"Selected range: {fetch_start_date}"
        + (f" to {normalized_end_date}" if normalized_end_date else " to latest available date")
    )
    run_fundamentals = st.checkbox(
        "財務データを今回更新する",
        value=False,
        key="data_run_fundamentals",
        help="チェックした場合のみ EDINET から財務データを再取得します。チェックしない場合は既存の財務特徴量をそのまま使います。",
    )
    fundamentals_tickers_raw = st.text_input(
        "財務データ取得対象ティッカー（スペース区切り、任意）",
        value="",
        key="data_fundamentals_tickers",
        help="空欄の場合は、現在の候補銘柄全体を対象に財務データを取得します。",
    )
    fundamentals_tickers = [ticker.strip().upper() for ticker in fundamentals_tickers_raw.split() if ticker.strip()]

    st.markdown("**Current Inputs**")
    st.table(
        pd.DataFrame(
            [
                {"artifact": "Raw Prices", "path": str(latest_raw_prices) if latest_raw_prices else "-"},
                {"artifact": "Raw Macro", "path": str(latest_raw_macro) if latest_raw_macro else "-"},
                {"artifact": "Normalized Prices", "path": str(latest_normalized_prices) if latest_normalized_prices else "-"},
                {"artifact": "Normalized Macro", "path": str(latest_normalized_macro) if latest_normalized_macro else "-"},
                {"artifact": "Labels", "path": str(latest_labels) if latest_labels else "-"},
            ]
        )
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch Prices", use_container_width=True):
            try:
                result = run_fetch_prices_command(start_date=fetch_start_date, end_date=normalized_end_date)
                st.success("Price fetch completed")
                st.json(result)
            except ConfigError as exc:
                st.error(str(exc))
        if st.button("Normalize Prices", use_container_width=True):
            try:
                result = run_normalize_prices_command()
                st.success("Price normalization completed")
                st.json(result)
            except ConfigError as exc:
                st.error(str(exc))
        if st.button("Build Labels", use_container_width=True):
            try:
                result = run_build_labels_command()
                st.success("Label generation completed")
                st.json(result)
            except ConfigError as exc:
                st.error(str(exc))
    with col2:
        if st.button("Fetch Macro", use_container_width=True):
            try:
                result = run_fetch_macro_command(start_date=fetch_start_date, end_date=normalized_end_date)
                st.success("Macro fetch completed")
                st.json(result)
            except ConfigError as exc:
                st.error(str(exc))
        if st.button("Normalize Macro", use_container_width=True):
            try:
                result = run_normalize_macro_command()
                st.success("Macro normalization completed")
                st.json(result)
            except ConfigError as exc:
                st.error(str(exc))

    if run_fundamentals:
        st.markdown("**財務データ取得**")
        st.caption(
            "EDINET API キーが必要です。指定した期間に提出・公開された EDINET 書類を取得します。"
            "ティッカー未指定の場合は現在の候補群を対象にします。"
        )
        if st.button("財務データを取得", use_container_width=True):
            try:
                result = run_fetch_fundamentals_command(
                    tickers=fundamentals_tickers or None,
                    start_date=fetch_start_date,
                    end_date=normalized_end_date,
                )
                st.success("財務データ取得が完了しました")
                st.json(result)
            except ConfigError as exc:
                st.error(str(exc))
        if st.button("Build Universe", use_container_width=True):
            try:
                result = run_build_universe_command()
                st.success("Universe build completed")
                st.json(result)
            except ConfigError as exc:
                st.error(str(exc))

    st.markdown("**Dataset Build**")
    st.caption("最新の normalized prices / macro / labels を使って dataset を生成します。")
    if st.button("Build Dataset", use_container_width=True):
        try:
            result = run_build_dataset_command()
            st.success("Dataset build completed")
            st.json(result)
        except ConfigError as exc:
            st.error(str(exc))


def render_walk_forward() -> None:
    st.subheader("Walk-Forward")
    st.write(
        "この画面は時系列の実運用に近い評価を行うためのものです。"
        "各リバランス時点までの過去データだけで再学習し、その時点の銘柄群を予測して"
        "パフォーマンスを検証します。"
    )
    dataset_path = _latest_file("data/processed/datasets", "dataset_*.csv")
    train_config = st.selectbox("Train Config", TRAIN_CONFIGS, key="walk_train_config")
    if st.button("Run Walk-Forward", use_container_width=True):
        try:
            result = run_walk_forward_command(
                "default",
                train_config,
                dataset_input_path=str(dataset_path) if dataset_path else None,
            )
            st.success(f"{train_config} walk-forward completed")
            st.json(result)
        except ConfigError as exc:
            st.error(str(exc))


def render_compare() -> None:
    st.subheader("Model Comparison")
    st.write(
        "この画面は 2 つのモデルを同条件で比較するためのものです。"
        "学習レポートと walk-forward 結果をまとめて、どちらを採用するか判断しやすくします。"
    )
    if st.button("Generate Comparison Report", use_container_width=True):
        try:
            result = run_compare_models_command()
            st.success("Model comparison completed")
            st.json(result)
            markdown_path = Path(result["markdown_output_path"])
            if markdown_path.exists():
                st.markdown(markdown_path.read_text(encoding="utf-8"))
        except ConfigError as exc:
            st.error(str(exc))

    latest_comparison = _latest_file("reports/tables", "model_comparison_*.md")
    if latest_comparison and latest_comparison.exists():
        with st.expander("Latest Comparison Markdown", expanded=False):
            st.markdown(latest_comparison.read_text(encoding="utf-8"))


def render_inference() -> None:
    st.subheader("Inference")
    st.write(
        "この画面は指定日時点での推論実行用です。選んだモデルを使って、その日から"
        "約60営業日後に 10%以上上がる確率を銘柄ごとに計算します。"
    )
    dataset_path = _latest_file("data/processed/datasets", "dataset_*.csv")
    default_date = "2025-12-04"
    train_config = st.selectbox("Model Config", TRAIN_CONFIGS, index=1, key="infer_train_config")
    prediction_date = st.text_input("Prediction Date (YYYY-MM-DD)", value=default_date)

    if st.button("Run Inference", use_container_width=True):
        model_path = _latest_file("models", f"{train_config}*.joblib")
        try:
            result = run_predict_command(
                "default",
                train_config,
                dataset_input_path=str(dataset_path) if dataset_path else None,
                model_input_path=str(model_path) if model_path else None,
                prediction_date=prediction_date,
            )
            st.success("Inference completed")
            st.json(result)
            output_path = Path(result["output_path"])
            if output_path.suffix == ".csv" and output_path.exists():
                frame = pd.read_csv(output_path)
                st.dataframe(
                    frame[["ticker", "probability", "prediction"]].sort_values("probability", ascending=False),
                    use_container_width=True,
                )
        except ConfigError as exc:
            st.error(str(exc))


def render_files() -> None:
    st.subheader("Artifacts")
    st.write(
        "この画面は生成済み成果物の確認用です。最新のユニバース、dataset、モデル、"
        "比較レポート、推論結果の保存先を一覧で確認できます。"
    )
    sections = {
        "Universe": _latest_file("data/processed/universe", "liquidity_universe_*.json"),
        "Dataset": _latest_file("data/processed/datasets", "dataset_*.csv"),
        "LogReg Model": _latest_file("models", "baseline_logreg*.joblib"),
        "LightGBM Model": _latest_file("models", "baseline_lightgbm*.joblib"),
        "Comparison": _latest_file("reports/tables", "model_comparison_*.json"),
        "Prediction": _latest_prediction_path(),
    }
    st.table(
        pd.DataFrame(
            [{"artifact": name, "path": str(path) if path else "-"} for name, path in sections.items()]
        )
    )


def main() -> None:
    st.set_page_config(page_title="stock_ai", layout="wide")
    st.title("stock_ai Web UI")
    st.caption("CLI ベースの学習・推論・比較処理を Web UI から呼び出す最小運用画面です。")
    st.sidebar.caption(
        "各メニューは、状況確認、データ更新、学習、時系列評価、モデル比較、推論、成果物確認の"
        "順に使う想定です。"
    )

    page = st.sidebar.radio(
        "Menu",
        ("Overview", "Data Pipeline", "Training", "Walk-Forward", "Compare", "Inference", "Artifacts"),
    )

    if page == "Overview":
        render_overview()
    elif page == "Data Pipeline":
        render_data_pipeline()
    elif page == "Training":
        render_training()
    elif page == "Walk-Forward":
        render_walk_forward()
    elif page == "Compare":
        render_compare()
    elif page == "Inference":
        render_inference()
    else:
        render_files()


if __name__ == "__main__":
    main()
