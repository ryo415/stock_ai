#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREDICT_MODEL="${PREDICT_MODEL:-best}"
RUN_FUNDAMENTALS="${RUN_FUNDAMENTALS:-0}"

log() {
  printf '[stock_ai] %s\n' "$1"
}

latest_file() {
  local pattern="$1"
  python3 - "$pattern" <<'PY'
from pathlib import Path
import sys

pattern = sys.argv[1]
matches = sorted(Path(".").glob(pattern))
if not matches:
    raise SystemExit(1)
print(matches[-1])
PY
}

select_predict_model() {
  local comparison_json="$1"
  python3 - "$comparison_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
walk = payload["comparison"]["walk_forward_delta_right_minus_left"]
test = payload["comparison"]["test_metrics_delta_right_minus_left"]

if (walk.get("total_return") or 0) > 0:
    print("baseline_lightgbm")
elif (walk.get("total_return") or 0) < 0:
    print("baseline_logreg")
elif (test.get("pr_auc") or 0) > 0:
    print("baseline_lightgbm")
else:
    print("baseline_logreg")
PY
}

log "Refreshing prices and market series"
python3 -m stock_ai data fetch-prices
python3 -m stock_ai data fetch-macro

log "Normalizing fetched data"
python3 -m stock_ai data normalize-prices
python3 -m stock_ai data normalize-macro

if [[ "$RUN_FUNDAMENTALS" == "1" ]]; then
  log "Refreshing fundamentals"
  python3 -m stock_ai data fetch-fundamentals
  python3 -m stock_ai data normalize-fundamentals
fi

log "Building liquidity-filtered universe"
python3 -m stock_ai data build-universe

log "Building labels and dataset"
python3 -m stock_ai features build-labels
python3 -m stock_ai features build-dataset

DATASET_PATH="$(latest_file 'data/processed/datasets/dataset_*.csv')"

log "Training baseline_logreg"
python3 -m stock_ai train run --config baseline_logreg --dataset-input-path "$DATASET_PATH"

log "Training baseline_lightgbm"
python3 -m stock_ai train run --config baseline_lightgbm --dataset-input-path "$DATASET_PATH"

LOGREG_TRAIN_REPORT="$(latest_file 'reports/tables/train_baseline_logreg*.json')"
LIGHTGBM_TRAIN_REPORT="$(latest_file 'reports/tables/train_baseline_lightgbm*.json')"

log "Running walk-forward for baseline_logreg"
python3 -m stock_ai backtest walk-forward --config default --train-config baseline_logreg --dataset-input-path "$DATASET_PATH"
LOGREG_WALK_REPORT="$(latest_file 'reports/tables/backtest_*walk_forward*.json')"
python3 - "$LOGREG_WALK_REPORT" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("train_config_name") != "baseline_logreg":
    raise SystemExit(1)
PY

log "Running walk-forward for baseline_lightgbm"
python3 -m stock_ai backtest walk-forward --config default --train-config baseline_lightgbm --dataset-input-path "$DATASET_PATH"
LIGHTGBM_WALK_REPORT="$(latest_file 'reports/tables/backtest_*walk_forward*.json')"
python3 - "$LIGHTGBM_WALK_REPORT" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("train_config_name") != "baseline_lightgbm":
    raise SystemExit(1)
PY

log "Building comparison report"
python3 -m stock_ai report compare-models \
  --left-train-report-path "$LOGREG_TRAIN_REPORT" \
  --right-train-report-path "$LIGHTGBM_TRAIN_REPORT" \
  --left-walk-forward-report-path "$LOGREG_WALK_REPORT" \
  --right-walk-forward-report-path "$LIGHTGBM_WALK_REPORT"

COMPARISON_JSON="$(latest_file 'reports/tables/model_comparison_*.json')"

if [[ "$PREDICT_MODEL" == "best" ]]; then
  PREDICT_MODEL="$(select_predict_model "$COMPARISON_JSON")"
fi

case "$PREDICT_MODEL" in
  baseline_logreg)
    MODEL_PATH="$(latest_file 'models/baseline_logreg*.joblib')"
    ;;
  baseline_lightgbm)
    MODEL_PATH="$(latest_file 'models/baseline_lightgbm*.joblib')"
    ;;
  *)
    printf 'Unsupported PREDICT_MODEL: %s\n' "$PREDICT_MODEL" >&2
    exit 1
    ;;
esac

log "Running inference with $PREDICT_MODEL"
python3 -m stock_ai inference predict \
  --config default \
  --train-config "$PREDICT_MODEL" \
  --dataset-input-path "$DATASET_PATH" \
  --model-input-path "$MODEL_PATH"

PREDICTION_PATH="$(latest_file 'reports/tables/predictions/predict_default_inference_*')"
log "Workflow completed"
log "Dataset: $DATASET_PATH"
log "Comparison: $COMPARISON_JSON"
log "Prediction: $PREDICTION_PATH"
