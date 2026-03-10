# ============================================================
# src/modeling/sarimax_walkforward.py
#
# SARIMAX Walk-Forward — langsung pakai order terbaik hasil
# grid search: SARIMAX(1,1,2)×(0,1,1,21)
#
# Skip grid search, langsung ke walk-forward prediction.
# Estimasi waktu: ~5-8 menit.
#
# CARA PAKAI:
#   python src/modeling/sarimax_walkforward.py
# ============================================================

import sys
import logging
import warnings
import json
import numpy as np
import pandas as pd
from pathlib import Path

_root        = Path(__file__).resolve().parents[2]
_modeling_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_modeling_dir))
from config import PROCESSED_DIR
from metrics import evaluate_all

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR        = PROCESSED_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
FORECAST_HORIZON = 21

# ── Order terbaik hasil grid search ───────────────────────
BEST_ORDER         = (1, 1, 2)
BEST_SEASONAL      = (0, 1, 1, 21)
BEST_AIC_GRIDSEARCH = 17000.24


def predict_walkforward(y_train, exog_train, y_test, exog_test,
                        order=BEST_ORDER, seasonal_order=BEST_SEASONAL,
                        horizon=FORECAST_HORIZON):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    log.info(f"  Walk-forward SARIMAX{order}×{seasonal_order} (horizon={horizon})...")
    history_y    = list(y_train.values)
    history_exog = exog_train.values.tolist()
    test_y       = list(y_test.values)
    test_exog    = exog_test.values
    test_dates   = list(y_test.index)

    predictions, actuals, pred_dates = [], [], []
    i = 0
    step = 0

    while i + horizon <= len(test_y):
        step += 1
        log.info(f"  Step {step}: training {len(history_y)} sampel → prediksi hari ke-{i+horizon}...")
        try:
            exog_hist_df = pd.DataFrame(history_exog, columns=exog_train.columns)
            fitted = SARIMAX(
                history_y, exog=exog_hist_df,
                order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=150)

            exog_future = pd.DataFrame(
                test_exog[i:i + horizon], columns=exog_train.columns)
            forecast = fitted.forecast(steps=horizon, exog=exog_future)
            pred_val = float(forecast.iloc[-1]) if hasattr(forecast, 'iloc') \
                       else float(forecast[-1])

            predictions.append(pred_val)
            actuals.append(test_y[i + horizon - 1])
            pred_dates.append(test_dates[i + horizon - 1])

            history_y.extend(test_y[i:i + horizon])
            history_exog.extend(test_exog[i:i + horizon].tolist())
            i += horizon

        except Exception as e:
            log.warning(f"  Step {step} gagal: {e}")
            i += horizon

    log.info(f"  ✓ {len(predictions)} prediksi berhasil")
    return pd.Series(predictions, index=pred_dates), pd.Series(actuals, index=pred_dates)


def main():
    log.info("\n" + "=" * 60)
    log.info("  SARIMAX TUNED — Walk-Forward (order sudah diketahui)")
    log.info(f"  Order    : SARIMAX{BEST_ORDER}×{BEST_SEASONAL}")
    log.info(f"  AIC grid : {BEST_AIC_GRIDSEARCH} (vs original 17113.93)")
    log.info("=" * 60)

    log.info("\nMemuat data...")
    from data_preparation import get_all_splits
    data = get_all_splits()

    y_train    = data["sarimax_y_train"]
    exog_train = data["sarimax_exog_train"]
    y_test     = data["sarimax_y_test"]
    exog_test  = data["sarimax_exog_test"]

    log.info(f"  Train: {len(y_train)} | Test: {len(y_test)} | Exog: {exog_train.shape[1]} vars")
    log.info(f"  Horizon: {FORECAST_HORIZON} hari → {len(y_test)//FORECAST_HORIZON} langkah walk-forward\n")

    preds, actuals = predict_walkforward(
        y_train, exog_train, y_test, exog_test,
        BEST_ORDER, BEST_SEASONAL, FORECAST_HORIZON
    )

    # Evaluasi
    label   = f"SARIMAX Tuned {BEST_ORDER}×{BEST_SEASONAL}"
    metrics = evaluate_all(actuals, preds, model_name=label)

    # Simpan predictions
    pred_df = pd.DataFrame({
        "actual":    actuals,
        "predicted": preds,
        "error":     actuals.values - preds.values,
        "pct_error": (actuals.values - preds.values) / actuals.values * 100,
    })
    pred_df.to_csv(PROCESSED_DIR / "predictions_sarimax_tuned.csv")
    log.info(f"  ✓ Predictions → predictions_sarimax_tuned.csv")

    # Simpan metadata
    meta = {
        "order":          list(BEST_ORDER),
        "seasonal_order": list(BEST_SEASONAL),
        "aic_gridsearch": BEST_AIC_GRIDSEARCH,
        "mae":  metrics.get("MAE"),
        "rmse": metrics.get("RMSE"),
        "mape": metrics.get("MAPE"),
    }
    meta_path = MODEL_DIR / "sarimax_tuned_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  ✓ Metadata → {meta_path}")

    # Perbandingan vs original
    orig_mape = 3.61
    orig_mae  = 94.22
    delta_mape = metrics.get("MAPE", 0) - orig_mape
    delta_mae  = metrics.get("MAE",  0) - orig_mae
    arrow_mape = "↓" if delta_mape < 0 else "↑"
    arrow_mae  = "↓" if delta_mae  < 0 else "↑"

    log.info(f"\n{'='*55}")
    log.info(f"  PERBANDINGAN SARIMAX: Original vs Tuned")
    log.info(f"{'='*55}")
    log.info(f"  {'':12s} {'Original':>10s} {'Tuned':>10s} {'Delta':>10s}")
    log.info(f"  {'MAPE (%)':12s} {orig_mape:>10.2f} {metrics.get('MAPE',0):>10.2f} "
             f"  {arrow_mape}{abs(delta_mape):.2f}%")
    log.info(f"  {'MAE ($)':12s} {orig_mae:>10.2f} {metrics.get('MAE',0):>10.2f} "
             f"  {arrow_mae}${abs(delta_mae):.2f}")
    log.info(f"{'='*55}")

    if delta_mape < 0:
        log.info(f"  🏆 SARIMAX Tuned lebih baik! MAPE turun {abs(delta_mape):.2f}%")
    else:
        log.info(f"  → SARIMAX original lebih baik (MAPE {orig_mape}%)")


if __name__ == "__main__":
    main()
