# ============================================================
# src/preprocessing/run_preprocessing.py
#
# FASE 2 — Runner: Jalankan semua langkah preprocessing
#
# CARA PAKAI:
#   python src/preprocessing/run_preprocessing.py
#   python src/preprocessing/run_preprocessing.py --step clean
#   python src/preprocessing/run_preprocessing.py --step merge
#   python src/preprocessing/run_preprocessing.py --step features
# ============================================================

import sys
import argparse
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

STEPS = ["clean", "merge", "features"]


def run_step(step: str):
    start = time.time()
    log.info(f"\n{'█' * 60}")
    log.info(f"LANGKAH: {step.upper()}")
    log.info(f"{'█' * 60}")

    try:
        if step == "clean":
            from src.preprocessing.cleaning import run_cleaning
            run_cleaning()

        elif step == "merge":
            from src.preprocessing.merging import merge_all
            merge_all()

        elif step == "features":
            from src.preprocessing.feature_engineering import run_feature_engineering
            run_feature_engineering()

        elapsed = time.time() - start
        log.info(f"\n  [{step.upper()}] ✅ Selesai | Waktu: {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.time() - start
        log.error(f"\n  [{step.upper()}] ❌ GAGAL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Fase 2: Preprocessing Pipeline")
    parser.add_argument(
        "--step",
        choices=STEPS + ["all"],
        default="all",
        help="Langkah yang ingin dijalankan (default: all)"
    )
    args = parser.parse_args()

    log.info("\n" + "=" * 60)
    log.info("  PIPELINE FASE 2: PREPROCESSING DATA ALUMINIUM")
    log.info("=" * 60)

    steps_to_run = STEPS if args.step == "all" else [args.step]
    results = {}
    total_start = time.time()

    for step in steps_to_run:
        results[step] = run_step(step)

    # Ringkasan
    total_elapsed = time.time() - total_start
    success = sum(results.values())
    total = len(results)

    log.info("\n" + "=" * 60)
    log.info("  FASE 2 SELESAI")
    log.info("=" * 60)
    log.info(f"  Berhasil    : {success}/{total} langkah")
    log.info(f"  Total waktu : {total_elapsed/60:.1f} menit")

    for step, ok in results.items():
        status = "✅" if ok else "❌"
        log.info(f"  {status} {step}")

    log.info("\nOutput di: data/processed/")
    log.info("  - cleaned_aluminum.csv")
    log.info("  - cleaned_exchange_rates.csv")
    log.info("  - cleaned_energy.csv")
    log.info("  - cleaned_macro_monthly.csv")
    log.info("  - cleaned_pmi.csv")
    log.info("  - cleaned_worldbank.csv")
    log.info("  - master_daily.csv       ← hasil merge")
    log.info("  - features_daily.csv     ← siap untuk modeling")


if __name__ == "__main__":
    main()
