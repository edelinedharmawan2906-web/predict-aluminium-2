# ============================================================
# src/modeling/run_new_features.py
#
# Re-run SARIMAX + RF dengan fitur baru (features_daily_v2.csv)
# Bandingkan hasil vs versi sebelumnya
#
# CARA PAKAI:
#   python src/modeling/run_new_features.py
#   python src/modeling/run_new_features.py --model sarimax
#   python src/modeling/run_new_features.py --model rf
# ============================================================

import sys
import argparse
import logging
import time
import pandas as pd
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_modeling_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_modeling_dir))
from config import PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Hasil terbaik sebelumnya
BASELINE = {
    "SARIMAX": {"MAE": 93.07, "RMSE": 118.95, "MAPE": 3.57},
    "RF":      {"MAE": 109.12, "RMSE": 138.47, "MAPE": 4.12},
}

# Kolom eksogen SARIMAX — ditambah fitur baru
EXOG_COLS_V2 = [
    # Original 11 kolom
    "usd_cny", "usd_eur", "usd_jpy",
    "natural_gas", "crude_oil_wti",
    "pmi_manufacturing_usa", "pmi_manufacturing_china",
    "ism_manufacturing_pmi", "industrial_production",
    "capacity_utilization", "copper_price_monthly",
    # Fitur baru
    "lme_inventory",
    "lme_inventory_change",
    "lme_inventory_ma21",
    "alumina_ppi",
    "alumina_ppi_mom",
]

FORECAST_HORIZON = 21
SPLIT_DATE       = "2024-05-01"


def load_data_v2(features_path: str = None):
    """Load features_daily_v2.csv dan siapkan splits."""
    if features_path is None:
        features_path = PROCESSED_DIR / "features_daily_v2.csv"

    if not Path(features_path).exists():
        log.error(f"  ✗ {features_path} tidak ditemukan!")
        log.info("  → Jalankan: python src/data_collection/new_features.py")
        sys.exit(1)

    df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    log.info(f"  Loaded v2: {df.shape[0]} baris x {df.shape[1]} kolom")

    # Clean kolom yang mengandung string format '34.42K', '1.2M' dst
    def parse_suffix(val):
        if isinstance(val, str):
            val = val.strip().replace(',', '')
            try:
                if val.endswith('K'): return float(val[:-1]) * 1_000
                if val.endswith('M'): return float(val[:-1]) * 1_000_000
                if val.endswith('B'): return float(val[:-1]) * 1_000_000_000
                return float(val)
            except:
                return float('nan')
        return val

    for col in df.columns:
        if df[col].dtype == object:
            log.warning(f"  Cleaning string column: {col}")
            df[col] = df[col].map(parse_suffix)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Target: harga 21 hari ke depan
    df["target"] = df["aluminum_price"].shift(-FORECAST_HORIZON)
    df = df.dropna(subset=["target"])

    train = df[df.index < SPLIT_DATE]
    test  = df[df.index >= SPLIT_DATE]
    log.info(f"  Train: {len(train)} | Test: {len(test)}")

    # Filter exog cols yang tersedia
    available_exog = [c for c in EXOG_COLS_V2 if c in df.columns]
    missing_exog   = [c for c in EXOG_COLS_V2 if c not in df.columns]

    if missing_exog:
        log.warning(f"  ⚠ Kolom eksogen tidak tersedia: {missing_exog}")
    log.info(f"  Exog cols tersedia: {len(available_exog)}/{len(EXOG_COLS_V2)}")

    # Feature cols untuk RF (semua kecuali target)
    feat_cols = [c for c in df.columns if c != "target"]

    return {
        # SARIMAX
        "sarimax_y_train":    train["aluminum_price"],
        "sarimax_exog_train": train[available_exog].ffill().bfill(),
        "sarimax_y_test":     test["aluminum_price"],
        "sarimax_exog_test":  test[available_exog].ffill().bfill(),
        # RF
        "X_train": train[feat_cols].ffill().bfill(),
        "y_train": train["target"],
        "X_test":  test[feat_cols].ffill().bfill(),
        "y_test":  test["target"],
        # Info
        "available_exog": available_exog,
    }


def run_sarimax_v2(data: dict):
    """SARIMAX dengan order terbaik + exog baru."""
    from sarimax_walkforward import predict_walkforward, BEST_ORDER, BEST_SEASONAL
    from metrics import evaluate_all

    log.info("\n" + "=" * 55)
    log.info(f"  SARIMAX v2: {BEST_ORDER}×{BEST_SEASONAL}")
    log.info(f"  Exog: {len(data['available_exog'])} kolom")
    log.info("=" * 55)

    preds, actuals = predict_walkforward(
        data["sarimax_y_train"], data["sarimax_exog_train"],
        data["sarimax_y_test"],  data["sarimax_exog_test"],
        BEST_ORDER, BEST_SEASONAL, FORECAST_HORIZON,
    )

    metrics = evaluate_all(actuals, preds, model_name="SARIMAX v2 (new features)")

    pred_df = pd.DataFrame({
        "actual": actuals, "predicted": preds,
        "error": actuals.values - preds.values,
        "pct_error": (actuals.values - preds.values) / actuals.values * 100,
    })
    pred_df.to_csv(PROCESSED_DIR / "predictions_sarimax_v2.csv")
    log.info("  ✓ predictions_sarimax_v2.csv disimpan")
    return metrics


def run_rf_v2(data: dict):
    """RF walk-forward dengan fitur baru."""
    from rf_tuned import predict_rf_walkforward, get_feature_importance
    from metrics import evaluate_all

    log.info("\n" + "=" * 55)
    log.info(f"  RF v2: walk-forward + {data['X_train'].shape[1]} fitur")
    log.info("=" * 55)

    preds, actuals, model = predict_rf_walkforward(
        data["X_train"], data["y_train"],
        data["X_test"],  data["y_test"],
        horizon=FORECAST_HORIZON,
    )

    importance = get_feature_importance(model, list(data["X_train"].columns))
    importance.to_csv(PROCESSED_DIR / "rf_v2_feature_importance.csv", index=False)

    metrics = evaluate_all(actuals, preds, model_name="RF v2 (new features)")

    pred_df = pd.DataFrame({
        "actual": actuals, "predicted": preds,
        "error": actuals.values - preds.values,
        "pct_error": (actuals.values - preds.values) / actuals.values * 100,
    })
    pred_df.to_csv(PROCESSED_DIR / "predictions_rf_v2.csv")
    log.info("  ✓ predictions_rf_v2.csv disimpan")
    return metrics


def print_comparison(new_results: dict):
    """Tabel perbandingan lengkap semua versi."""
    log.info("\n" + "=" * 65)
    log.info("  REKAP PERBANDINGAN — Original → Tuned → v2 (new features)")
    log.info("=" * 65)

    history = {
        "SARIMAX Orig":  {"MAPE": 3.61, "MAE": 94.22},
        "SARIMAX Tuned": {"MAPE": 3.57, "MAE": 93.07},
        "RF Orig":       {"MAPE": 5.94, "MAE": 164.43},
        "RF Tuned":      {"MAPE": 4.12, "MAE": 109.12},
    }
    history.update(new_results)

    rows = []
    for name, m in history.items():
        rows.append({"Model": name, "MAPE (%)": m.get("MAPE"), "MAE ($)": m.get("MAE")})

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # Simpan
    df.to_csv(PROCESSED_DIR / "model_comparison_all_versions.csv", index=False)
    log.info(f"\n  ✓ model_comparison_all_versions.csv disimpan")

    # Best overall
    best_idx  = df["MAPE (%)"].idxmin()
    best_name = df.loc[best_idx, "Model"]
    best_mape = df.loc[best_idx, "MAPE (%)"]
    log.info(f"\n  🏆 Model terbaik: {best_name} | MAPE: {best_mape:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sarimax", "rf", "all"], default="all")
    args = parser.parse_args()

    log.info("\n" + "=" * 60)
    log.info("  MODELING v2 — Dengan Fitur Baru")
    log.info("  LME Inventory + Alumina PPI ditambahkan")
    log.info("=" * 60)

    data = load_data_v2()
    new_results = {}
    total_start = time.time()

    if args.model in ["sarimax", "all"]:
        t = time.time()
        try:
            metrics = run_sarimax_v2(data)
            new_results["SARIMAX v2"] = metrics
            log.info(f"  ✅ SARIMAX v2 selesai | {(time.time()-t)/60:.1f} menit")
        except Exception as e:
            import traceback
            log.error(f"  ❌ SARIMAX v2 gagal: {e}")
            traceback.print_exc()

    if args.model in ["rf", "all"]:
        t = time.time()
        try:
            metrics = run_rf_v2(data)
            new_results["RF v2"] = metrics
            log.info(f"  ✅ RF v2 selesai | {(time.time()-t)/60:.1f} menit")
        except Exception as e:
            import traceback
            log.error(f"  ❌ RF v2 gagal: {e}")
            traceback.print_exc()

    if new_results:
        print_comparison(new_results)

    log.info(f"\n  Total: {(time.time()-total_start)/60:.1f} menit")


if __name__ == "__main__":
    main()
