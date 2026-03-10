# ============================================================
# src/modeling/metrics.py
#
# Fungsi evaluasi model: MAE, RMSE, MAPE
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def evaluate_all(y_true, y_pred, model_name: str = "") -> Dict[str, float]:
    """Hitung semua metrik sekaligus."""
    results = {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
    if model_name:
        print(f"\n{'─'*40}")
        print(f"  Model  : {model_name}")
        print(f"  MAE    : ${results['MAE']:,.2f} per ton")
        print(f"  RMSE   : ${results['RMSE']:,.2f} per ton")
        print(f"  MAPE   : {results['MAPE']:.2f}%")
        print(f"{'─'*40}")
    return results


def compare_models(results_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Buat tabel perbandingan semua model."""
    rows = []
    for model, metrics in results_dict.items():
        rows.append({
            "Model":    model,
            "MAE ($)":  round(metrics.get("MAE", float("nan")), 2),
            "RMSE ($)": round(metrics.get("RMSE", float("nan")), 2),
            "MAPE (%)": round(metrics.get("MAPE", float("nan")), 2),
        })
    df = pd.DataFrame(rows)
    if "MAPE (%)" in df.columns:
        df = df.sort_values("MAPE (%)", na_position="last")
    return df
