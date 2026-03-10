# ============================================================
# src/modeling/sarimax_tuned.py
#
# SARIMAX Tuned — coba seasonal period berbeda
#
# Perubahan dari sarimax_model.py:
#   - Grid search seasonal period: s = 5, 10, 21, 63
#   - s=5  : weekly seasonality (5 hari trading)
#   - s=10 : bi-weekly
#   - s=21 : monthly (versi lama)
#   - s=63 : quarterly
#   - Pilih s terbaik berdasarkan AIC
#   - Juga coba P,D,Q yang lebih bervariasi
# ============================================================

import sys
import logging
import warnings
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_modeling_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_modeling_dir))
from config import PROCESSED_DIR
from metrics import evaluate_all

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

MODEL_DIR        = PROCESSED_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
FORECAST_HORIZON = 21

# Seasonal periods yang akan dicoba
SEASONAL_PERIODS = [5, 10, 21, 63]


def find_best_sarimax_order(y: pd.Series, exog: pd.DataFrame):
    """
    Strategi 2-fase untuk SARIMAX yang cepat (~5-10 menit):

    FASE 1 — Pilih seasonal period terbaik:
      Pakai order sederhana (1,d,1) sebagai proxy
      Coba s = 5, 10, 21, 63 → pilih AIC terkecil

    FASE 2 — Fine-tune order (p,q) pada period terbaik:
      Grid search p,q 0-2 pada seasonal period terpilih
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(y.dropna())
    d = 0 if result[1] < 0.05 else 1
    log.info(f"  ADF p-value: {result[1]:.4f} → d={d}")

    # ── FASE 1: Cari seasonal period terbaik ──────────────
    log.info("\n  FASE 1: Menentukan seasonal period terbaik (s=5,10,21,63)...")
    proxy_order   = (1, d, 1)
    period_results = []

    for s in [5, 10, 21, 63]:
        # Coba 2 konfigurasi per s (cepat)
        for seas_ord in [(1,0,1,s), (0,1,1,s)]:
            try:
                res = SARIMAX(
                    y, exog=exog,
                    order=proxy_order,
                    seasonal_order=seas_ord,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=80)
                period_results.append({"s": s, "seas": seas_ord, "aic": res.aic})
                log.info(f"    s={s:2d} {seas_ord} → AIC: {res.aic:.2f}")
            except Exception as e:
                log.debug(f"    s={s} {seas_ord} gagal: {e}")
                continue

    if not period_results:
        log.warning("  Semua konfigurasi gagal, fallback ke s=21")
        best_s          = 21
        best_seas_proxy = (1, 0, 1, 21)
    else:
        period_results.sort(key=lambda x: x["aic"])
        best_s          = period_results[0]["s"]
        best_seas_proxy = period_results[0]["seas"]
        log.info(f"\n  → Seasonal period terbaik: s={best_s} | AIC: {period_results[0]['aic']:.2f}")

    # ── FASE 2: Fine-tune order pada s terbaik ────────────
    log.info(f"\n  FASE 2: Fine-tune order pada s={best_s}...")

    # Seasonal configs pada period terpilih
    seasonal_variants = [
        (1, 0, 1, best_s),
        (0, 1, 1, best_s),
        (1, 1, 0, best_s),
        (1, 1, 1, best_s),
    ]

    best_aic      = np.inf
    best_order    = (1, d, 1)
    best_seasonal = (1, 0, 1, best_s)
    results_log   = []

    for p in range(0, 3):
        for q in range(0, 3):
            base_ord = (p, d, q)
            for seas_ord in seasonal_variants[:2]:  # Batasi 2 seasonal variant
                try:
                    res = SARIMAX(
                        y, exog=exog,
                        order=base_ord,
                        seasonal_order=seas_ord,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=100)
                    results_log.append({
                        "order": base_ord, "seasonal": seas_ord, "aic": res.aic
                    })
                    if res.aic < best_aic:
                        best_aic      = res.aic
                        best_order    = base_ord
                        best_seasonal = seas_ord
                except Exception:
                    continue

    results_log.sort(key=lambda x: x["aic"])
    log.info("\n  Top 5 SARIMAX configurations:")
    for r in results_log[:5]:
        marker = " ← BEST" if r["aic"] == best_aic else ""
        log.info(f"    SARIMAX{r['order']}×{r['seasonal']} | AIC: {r['aic']:.2f}{marker}")

    return best_order, best_seasonal, best_aic


def train_sarimax_tuned(y_train: pd.Series, exog_train: pd.DataFrame):
    """Train SARIMAX dengan order dan seasonal terbaik."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    order, seasonal_order, best_aic = find_best_sarimax_order(y_train, exog_train)

    log.info(f"\n  Training SARIMAX{order}×{seasonal_order}...")
    model = SARIMAX(
        y_train, exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=200)
    log.info(f"  ✓ SARIMAX Tuned trained | AIC: {fitted.aic:.2f}")
    return fitted, order, seasonal_order


def predict_walkforward(y_train, exog_train, y_test, exog_test,
                         order, seasonal_order, horizon=FORECAST_HORIZON):
    """Walk-forward SARIMAX."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    log.info(f"  Walk-forward prediction (horizon={horizon})...")
    history_y    = list(y_train.values)
    history_exog = exog_train.values.tolist()
    test_y       = list(y_test.values)
    test_exog    = exog_test.values
    test_dates   = list(y_test.index)

    predictions, actuals, pred_dates = [], [], []
    i = 0

    while i + horizon <= len(test_y):
        try:
            exog_hist_df = pd.DataFrame(history_exog, columns=exog_train.columns)
            fitted = SARIMAX(
                history_y, exog=exog_hist_df,
                order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)

            exog_future = pd.DataFrame(
                test_exog[i:i + horizon], columns=exog_train.columns)
            forecast  = fitted.forecast(steps=horizon, exog=exog_future)
            pred_val  = float(forecast.iloc[-1]) if hasattr(forecast, 'iloc') \
                        else float(forecast[-1])

            predictions.append(pred_val)
            actuals.append(test_y[i + horizon - 1])
            pred_dates.append(test_dates[i + horizon - 1])

            history_y.extend(test_y[i:i + horizon])
            history_exog.extend(test_exog[i:i + horizon].tolist())
            i += horizon

        except Exception as e:
            log.warning(f"  Step gagal: {e}")
            i += horizon

    log.info(f"  ✓ {len(predictions)} prediksi berhasil")
    return pd.Series(predictions, index=pred_dates), pd.Series(actuals, index=pred_dates)


def run_sarimax_tuned(data: dict) -> dict:
    log.info("\n" + "=" * 55)
    log.info("MODEL 2 TUNED: SARIMAX (multi-seasonal search)")
    log.info("=" * 55)

    y_train    = data["sarimax_y_train"]
    exog_train = data["sarimax_exog_train"]
    y_test     = data["sarimax_y_test"]
    exog_test  = data["sarimax_exog_test"]

    fitted, order, seasonal_order = train_sarimax_tuned(y_train, exog_train)

    model_path = MODEL_DIR / "sarimax_tuned_model.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump({
                "fitted": fitted, "order": order,
                "seasonal_order": seasonal_order,
                "exog_cols": list(exog_train.columns),
            }, f)
        log.info(f"  ✓ Model disimpan: {model_path}")
    except (MemoryError, Exception) as e:
        log.warning(f"  ⚠ Tidak bisa simpan pkl ({e.__class__.__name__}) — lanjut ke walk-forward")
        import json
        meta = {
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "exog_cols": list(exog_train.columns),
            "aic": float(fitted.aic),
        }
        meta_path = MODEL_DIR / "sarimax_tuned_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"  ✓ Metadata order disimpan: {meta_path}")

    preds, actuals = predict_walkforward(
        y_train, exog_train, y_test, exog_test,
        order, seasonal_order, FORECAST_HORIZON
    )

    label = f"SARIMAX Tuned {order}×{seasonal_order}"
    metrics = evaluate_all(actuals, preds, model_name=label)

    pred_df = pd.DataFrame({
        "actual": actuals, "predicted": preds,
        "error": actuals - preds,
        "pct_error": (actuals - preds) / actuals * 100,
    })
    pred_df.to_csv(PROCESSED_DIR / "predictions_sarimax_tuned.csv")
    log.info(f"  ✓ Predictions disimpan")

    return {"model": fitted, "order": order, "seasonal_order": seasonal_order,
            "metrics": metrics, "preds": preds, "actuals": actuals}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from data_preparation import get_all_splits
    data = get_all_splits()
    run_sarimax_tuned(data)
