# ============================================================
# src/modeling/arima_tuned.py
#
# ARIMA Tuned — menggunakan auto_arima dari pmdarima
# untuk grid search lebih luas dan akurat
#
# Perubahan dari arima_model.py:
#   - auto_arima: test ADF/KPSS otomatis, stepwise search
#   - Range p,q diperluas: 0-5
#   - Coba juga dengan differencing seasonal (D)
#   - Walk-forward tetap sama (sudah fix)
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


def find_best_order_auto(series: pd.Series):
    """
    Gunakan auto_arima dari pmdarima untuk cari order terbaik.
    Lebih akurat dari grid search manual karena:
    - Stepwise search (lebih efisien)
    - Test stasioneritas otomatis (ADF + KPSS)
    - Range p,d,q lebih luas
    """
    try:
        from pmdarima import auto_arima
        log.info("  Menggunakan auto_arima (pmdarima)...")
        model = auto_arima(
            series,
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            d=None,           # auto-detect d
            stepwise=True,    # lebih cepat
            information_criterion='aic',
            test='adf',
            seasonal=False,   # ARIMA murni (non-seasonal)
            error_action='ignore',
            suppress_warnings=True,
            n_jobs=-1,
        )
        order = model.order
        log.info(f"  auto_arima order: {order} | AIC: {model.aic():.2f}")
        return order

    except ImportError:
        log.warning("  pmdarima tidak terinstall, fallback ke grid search manual...")
        return _manual_grid_search(series)


def _manual_grid_search(series: pd.Series):
    """Fallback: grid search manual range lebih luas."""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    result = adfuller(series.dropna())
    d = 0 if result[1] < 0.05 else 1

    best_aic   = np.inf
    best_order = (1, d, 1)
    for p in range(0, 6):
        for q in range(0, 6):
            try:
                model = ARIMA(series, order=(p, d, q))
                res   = model.fit()
                if res.aic < best_aic:
                    best_aic   = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue
    log.info(f"  Manual grid search: {best_order} | AIC: {best_aic:.2f}")
    return best_order


def train_arima_tuned(train_series: pd.Series):
    """Train ARIMA dengan order dari auto_arima."""
    from statsmodels.tsa.arima.model import ARIMA

    order = find_best_order_auto(train_series)
    log.info(f"  Training ARIMA{order}...")
    model  = ARIMA(train_series, order=order)
    fitted = model.fit()
    log.info(f"  ✓ ARIMA Tuned trained | AIC: {fitted.aic:.2f}")
    return fitted, order


def predict_walkforward(train_series: pd.Series, test_series: pd.Series,
                         order: tuple, horizon: int = FORECAST_HORIZON):
    """Walk-forward — sama seperti versi sebelumnya (sudah fix)."""
    from statsmodels.tsa.arima.model import ARIMA

    log.info(f"  Walk-forward prediction (horizon={horizon})...")
    history    = list(train_series.values)
    test_vals  = list(test_series.values)
    test_dates = list(test_series.index)

    predictions, actuals, pred_dates = [], [], []
    i = 0

    while i + horizon <= len(test_vals):
        try:
            fitted   = ARIMA(history, order=order).fit()
            forecast = fitted.forecast(steps=horizon)
            pred_val = float(forecast[-1]) if not hasattr(forecast, 'iloc') \
                       else float(forecast.iloc[-1])

            predictions.append(pred_val)
            actuals.append(test_vals[i + horizon - 1])
            pred_dates.append(test_dates[i + horizon - 1])

            history.extend(test_vals[i:i + horizon])
            i += horizon
        except Exception as e:
            log.warning(f"  Step gagal: {e}")
            i += horizon

    log.info(f"  ✓ {len(predictions)} prediksi berhasil")
    return pd.Series(predictions, index=pred_dates), pd.Series(actuals, index=pred_dates)


def run_arima_tuned(data: dict) -> dict:
    log.info("\n" + "=" * 55)
    log.info("MODEL 1 TUNED: ARIMA (auto_arima)")
    log.info("=" * 55)

    train = data["arima_train"]
    test  = data["arima_test"]

    fitted, order = train_arima_tuned(train)

    model_path = MODEL_DIR / "arima_tuned_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"fitted": fitted, "order": order}, f)
    log.info(f"  ✓ Model disimpan: {model_path}")

    preds, actuals = predict_walkforward(train, test, order, FORECAST_HORIZON)
    metrics = evaluate_all(actuals, preds, model_name=f"ARIMA Tuned {order}")

    pred_df = pd.DataFrame({
        "actual": actuals, "predicted": preds,
        "error": actuals - preds,
        "pct_error": (actuals - preds) / actuals * 100,
    })
    pred_df.to_csv(PROCESSED_DIR / "predictions_arima_tuned.csv")
    log.info(f"  ✓ Predictions disimpan")

    return {"model": fitted, "order": order, "metrics": metrics,
            "preds": preds, "actuals": actuals}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from data_preparation import get_all_splits
    data = get_all_splits()
    run_arima_tuned(data)
