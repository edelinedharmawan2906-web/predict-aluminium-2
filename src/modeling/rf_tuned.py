# ============================================================
# src/modeling/rf_tuned.py
#
# Random Forest Tuned — Walk-Forward Prediction
#
# Perubahan utama dari random_forest_model.py:
#   - Walk-forward: refit RF setiap horizon langkah
#     dengan data historis yang terus bertambah
#   - Ini adil dibandingkan ARIMA/SARIMAX yang juga walk-forward
#   - RF tetap pakai 68 fitur tabular (lag, MA, exog, dll)
#   - Fitur untuk prediksi: pakai nilai dari hari ke-i,
#     target: harga hari ke-(i+21)
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


def get_rf_model(n_estimators=200, max_depth=15, min_samples_leaf=2):
    """Buat RF dengan parameter hasil tuning sebelumnya."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def predict_rf_walkforward(X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            horizon: int = FORECAST_HORIZON,
                            refit_every: int = None) -> tuple:
    """
    Walk-forward untuk RF:
    - Setiap `horizon` langkah, refit RF dengan semua data historis
    - Prediksi menggunakan fitur dari baris test ke-i
    - Target: harga hari ke-(i + horizon - 1)

    Parameter:
        refit_every: refit setiap N sampel baru (default = horizon)
                     Bisa diperbesar (misal 63) agar lebih cepat
    """
    if refit_every is None:
        refit_every = horizon

    log.info(f"  RF Walk-forward (horizon={horizon}, refit_every={refit_every})...")

    # Gabungkan train + test sebagai pool data
    X_all = pd.concat([X_train, X_test], axis=0)
    y_all = pd.concat([y_train, y_test], axis=0)

    n_train  = len(X_train)
    n_test   = len(X_test)

    predictions = []
    actuals     = []
    pred_dates  = []

    # Inisialisasi: train dengan data train penuh
    log.info(f"  Initial fit: {n_train} sampel training...")
    model = get_rf_model()
    model.fit(X_train, y_train)

    i         = 0   # indeks dalam test set
    last_fit  = 0   # jumlah sampel test yang sudah masuk ke history saat terakhir refit

    while i + horizon <= n_test:
        # Prediksi pada baris test ke-i
        x_row = X_test.iloc[i:i+1]
        pred_val   = float(model.predict(x_row)[0])
        actual_val = float(y_test.iloc[i + horizon - 1])

        predictions.append(pred_val)
        actuals.append(actual_val)
        pred_dates.append(X_test.index[i + horizon - 1])

        i += horizon

        # Refit dengan data historis yang bertambah
        if (i - last_fit) >= refit_every:
            n_history = n_train + i
            X_hist = X_all.iloc[:n_history]
            y_hist = y_all.iloc[:n_history]
            model.fit(X_hist, y_hist)
            last_fit = i
            log.info(f"    Refit @ test step {i}: {n_history} sampel historis")

    log.info(f"  ✓ {len(predictions)} prediksi berhasil")
    return (pd.Series(predictions, index=pred_dates),
            pd.Series(actuals, index=pred_dates),
            model)


def get_feature_importance(model, feature_names, top_n=20):
    importance = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    log.info(f"\n  Top 10 Feature Importance:")
    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row["importance"] * 200)
        log.info(f"    {row['feature']:35s} {row['importance']:.4f} {bar}")

    return importance


def run_rf_tuned(data: dict) -> dict:
    log.info("\n" + "=" * 55)
    log.info("MODEL 3 TUNED: RANDOM FOREST (Walk-Forward)")
    log.info("=" * 55)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    log.info(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    log.info(f"  Refit setiap {FORECAST_HORIZON} langkah (non-overlapping walk-forward)")

    preds, actuals, final_model = predict_rf_walkforward(
        X_train, y_train, X_test, y_test,
        horizon=FORECAST_HORIZON,
        refit_every=FORECAST_HORIZON,
    )

    # Feature importance dari model terakhir
    importance_df = get_feature_importance(final_model, list(X_train.columns))
    importance_path = PROCESSED_DIR / "rf_tuned_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    # Simpan model
    model_path = MODEL_DIR / "rf_tuned_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":         final_model,
            "feature_names": list(X_train.columns),
        }, f)
    log.info(f"  ✓ Model disimpan: {model_path}")

    # Evaluasi
    metrics = evaluate_all(actuals, preds, model_name="RF Tuned (Walk-Forward)")

    # Simpan predictions
    pred_df = pd.DataFrame({
        "actual":    actuals,
        "predicted": preds,
        "error":     actuals.values - preds.values,
        "pct_error": (actuals.values - preds.values) / actuals.values * 100,
    })
    pred_df.to_csv(PROCESSED_DIR / "predictions_rf_tuned.csv")
    log.info(f"  ✓ Predictions disimpan")

    return {
        "model":      final_model,
        "metrics":    metrics,
        "preds":      preds,
        "actuals":    actuals,
        "importance": importance_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from data_preparation import get_all_splits
    data = get_all_splits()
    run_rf_tuned(data)
