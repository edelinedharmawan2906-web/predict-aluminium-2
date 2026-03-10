# ============================================================
# src/modeling/run_tuning.py
#
# Runner tuning semua model + perbandingan vs versi original
#
# CARA PAKAI:
#   python src/modeling/run_tuning.py              ← semua
#   python src/modeling/run_tuning.py --model arima
#   python src/modeling/run_tuning.py --model sarimax
#   python src/modeling/run_tuning.py --model rf
#
# ESTIMASI WAKTU:
#   ARIMA tuned  : ~2-5 menit  (auto_arima lebih luas)
#   SARIMAX tuned: ~30-60 menit (grid search 4 seasonal periods)
#   RF tuned     : ~5-10 menit  (walk-forward refit)
#
# Untuk SARIMAX yang sangat lama, jalankan terpisah:
#   python src/modeling/run_tuning.py --model sarimax
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

MODELS = ["arima", "sarimax", "rf"]

# Hasil versi original (dari fase 3)
ORIGINAL_RESULTS = {
    "ARIMA":   {"MAE": 102.29, "RMSE": 135.06, "MAPE": 3.89},
    "SARIMAX": {"MAE": 94.22,  "RMSE": 119.63, "MAPE": 3.61},
    "RF":      {"MAE": 164.43, "RMSE": 219.11, "MAPE": 5.94},
}


def print_banner():
    log.info("\n" + "=" * 60)
    log.info("  TUNING FASE 3 — Optimasi Semua Model")
    log.info("  ARIMA    : auto_arima, range p,q lebih luas")
    log.info("  SARIMAX  : grid search seasonal s=5,10,21,63")
    log.info("  RF       : walk-forward (adil vs ARIMA/SARIMAX)")
    log.info("=" * 60)


def print_comparison(original: dict, tuned: dict):
    """Tampilkan tabel before/after tuning."""
    log.info("\n" + "=" * 65)
    log.info("  HASIL PERBANDINGAN: ORIGINAL vs TUNED")
    log.info("=" * 65)

    rows = []
    for model in ["ARIMA", "SARIMAX", "RF"]:
        orig = original.get(model, {})
        tnd  = tuned.get(model + "_TUNED", {})
        if orig and tnd:
            delta_mape = tnd.get("MAPE", 0) - orig.get("MAPE", 0)
            arrow = "↓" if delta_mape < 0 else "↑"
            rows.append({
                "Model":          model,
                "MAPE Orig (%)":  orig.get("MAPE"),
                "MAPE Tuned (%)": tnd.get("MAPE"),
                "Delta":          f"{arrow} {abs(delta_mape):.2f}%",
                "MAE Orig ($)":   orig.get("MAE"),
                "MAE Tuned ($)":  tnd.get("MAE"),
            })

    if rows:
        df = pd.DataFrame(rows)
        print("\n" + df.to_string(index=False))

        # Simpan perbandingan
        comp_path = PROCESSED_DIR / "model_comparison_tuned.csv"
        df.to_csv(comp_path, index=False)
        log.info(f"\n✓ Perbandingan disimpan: {comp_path}")

        # Tentukan best overall
        all_results = {}
        for model in ["ARIMA", "SARIMAX", "RF"]:
            for suffix in ["", "_TUNED"]:
                key = model + suffix
                src = tuned if suffix else original
                if model in src or key in src:
                    all_results[key] = src.get(model, src.get(key, {}))

        best = min(all_results.items(), key=lambda x: x[1].get("MAPE", 999))
        log.info(f"\n🏆 Model terbaik keseluruhan: {best[0]} (MAPE: {best[1]['MAPE']:.2f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS + ["all"], default="all")
    args = parser.parse_args()

    print_banner()

    log.info("\nMemuat data...")
    from data_preparation import get_all_splits
    data = get_all_splits()

    models_to_run = MODELS if args.model == "all" else [args.model]
    tuned_results = {}
    total_start   = time.time()

    for model_name in models_to_run:
        start = time.time()
        log.info(f"\n{'█'*60}")
        log.info(f"  TUNING: {model_name.upper()}")
        log.info(f"{'█'*60}")

        try:
            if model_name == "arima":
                from arima_tuned import run_arima_tuned
                result = run_arima_tuned(data)
                tuned_results["ARIMA_TUNED"] = result["metrics"]

            elif model_name == "sarimax":
                from sarimax_tuned import run_sarimax_tuned
                result = run_sarimax_tuned(data)
                tuned_results["SARIMAX_TUNED"] = result["metrics"]

            elif model_name == "rf":
                from rf_tuned import run_rf_tuned
                result = run_rf_tuned(data)
                tuned_results["RF_TUNED"] = result["metrics"]

            elapsed = time.time() - start
            log.info(f"\n  ✅ {model_name.upper()} TUNED selesai | Waktu: {elapsed/60:.1f} menit")

        except Exception as e:
            import traceback
            log.error(f"\n  ❌ {model_name.upper()} TUNED GAGAL: {e}")
            traceback.print_exc()

    # Tampilkan perbandingan
    if tuned_results:
        print_comparison(ORIGINAL_RESULTS, tuned_results)

    total = time.time() - total_start
    log.info(f"\n  Total waktu: {total/60:.1f} menit")
    log.info(f"  Output di: data/processed/")
    log.info(f"    - predictions_arima_tuned.csv")
    log.info(f"    - predictions_sarimax_tuned.csv")
    log.info(f"    - predictions_rf_tuned.csv")
    log.info(f"    - model_comparison_tuned.csv")


if __name__ == "__main__":
    main()
