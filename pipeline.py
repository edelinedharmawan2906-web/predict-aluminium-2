# ============================================================
# pipeline.py — Orkestrasi Pipeline Fase 1
#
# FUNGSI : Menjalankan semua tahap pengumpulan data secara
#          berurutan dan terkoordinasi. Ini adalah file utama
#          yang kamu jalankan untuk memulai Fase 1.
#
# CARA PAKAI:
#   python pipeline.py                    # jalankan semua
#   python pipeline.py --step price       # hanya harga aluminium
#   python pipeline.py --step exog        # hanya variabel eksogen
#   python pipeline.py --step macro       # hanya makroekonomi
#   python pipeline.py --step sentiment   # hanya sentimen
#   python pipeline.py --step report      # hanya buat laporan
#
# OUTPUT:
#   data/raw/   — semua file CSV mentah
#   data/processed/phase1_report.txt — ringkasan kualitas data
# ============================================================

import sys
import argparse
import time
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    RAW_DATA_DIR, PROCESSED_DIR,
    DATA_START_DATE, DATA_END_DATE,
    validate_config,
)

# Import semua collector
from src.data_collection import (
    aluminum_price      as col_price,
    exogenous_variables as col_exog,
    macro_indicators    as col_macro,
    news_sentiment      as col_sentiment,
    pmi_data            as col_pmi,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).parent / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# LANGKAH-LANGKAH PIPELINE
# ─────────────────────────────────────────────────────────────

def step_price(start: str, end: str) -> dict:
    """Langkah 1: Ambil data harga aluminium."""
    log.info("\n" + "█" * 60)
    log.info("LANGKAH 1/4 — Data Harga Aluminium")
    log.info("█" * 60)
    try:
        result = col_price.run(start=start, end=end)
        log.info("✓ Langkah 1 selesai")
        return {"status": "success", **result}
    except Exception as e:
        log.error(f"✗ Langkah 1 GAGAL: {e}")
        log.debug(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


def step_exogenous(start: str, end: str) -> dict:
    """Langkah 2: Ambil variabel eksogen (kurs, energi, cross-commodity)."""
    log.info("\n" + "█" * 60)
    log.info("LANGKAH 2/4 — Variabel Eksogen (Kurs & Energi)")
    log.info("█" * 60)
    try:
        result = col_exog.run(start=start, end=end)
        log.info("✓ Langkah 2 selesai")
        return {"status": "success", **result}
    except Exception as e:
        log.error(f"✗ Langkah 2 GAGAL: {e}")
        log.debug(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


def step_macro(start: str, end: str) -> dict:
    """Langkah 3: Ambil data makroekonomi global."""
    log.info("\n" + "█" * 60)
    log.info("LANGKAH 3/5 — Data Makroekonomi")
    log.info("█" * 60)
    try:
        result = col_macro.run(start=start, end=end)
        log.info("✓ Langkah 3 selesai")
        return {"status": "success", **result}
    except Exception as e:
        log.error(f"✗ Langkah 3 GAGAL: {e}")
        log.debug(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


def step_pmi(start: str, end: str) -> dict:
    """Langkah 4: Ambil data PMI Manufaktur (otomatis + manual fallback)."""
    log.info("\n" + "█" * 60)
    log.info("LANGKAH 4/5 — Data PMI Manufaktur")
    log.info("█" * 60)
    try:
        df = col_pmi.run(start=start, end=end)
        log.info("✓ Langkah 4 selesai")
        return {"status": "success", "pmi": df}
    except Exception as e:
        log.error(f"✗ Langkah 4 GAGAL: {e}")
        log.debug(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


def step_sentiment(start: str, end: str, sentiment_method: str = "vader") -> dict:
    """Langkah 5: Ambil dan proses data sentimen berita."""
    log.info("\n" + "█" * 60)
    log.info("LANGKAH 5/5 — Sentimen Berita")
    log.info("█" * 60)
    try:
        result = col_sentiment.run(
            start=start, end=end, method=sentiment_method
        )
        log.info("✓ Langkah 4 selesai")
        return {"status": "success", **result}
    except Exception as e:
        log.error(f"✗ Langkah 4 GAGAL: {e}")
        log.debug(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


# ─────────────────────────────────────────────────────────────
# LAPORAN KUALITAS DATA
# ─────────────────────────────────────────────────────────────

def generate_data_quality_report(results: dict) -> str:
    """
    Buat laporan kualitas data setelah semua langkah selesai.

    Laporan mencakup:
    - Jumlah file yang berhasil diunduh
    - Periode dan kelengkapan data per sumber
    - Missing values per dataset
    - Peringatan jika ada data yang mencurigakan
    """
    import pandas as pd
    from pathlib import Path

    log.info("\n" + "█" * 60)
    log.info("MEMBUAT LAPORAN KUALITAS DATA")
    log.info("█" * 60)

    lines = []
    lines.append("=" * 65)
    lines.append("LAPORAN KUALITAS DATA — FASE 1: PENGUMPULAN DATA")
    lines.append(f"Dibuat pada : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Periode data: {DATA_START_DATE} s/d {DATA_END_DATE}")
    lines.append("=" * 65)

    # Cek semua file CSV yang ada
    csv_files = list(RAW_DATA_DIR.glob("*.csv"))
    lines.append(f"\nTotal file CSV di data/raw/ : {len(csv_files)}")
    lines.append("")

    total_missing = 0
    files_ok = 0

    for csv_path in sorted(csv_files):
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            n_rows   = len(df)
            n_cols   = len(df.columns)
            missing  = df.isnull().sum().sum()
            total_missing += missing

            # Tentukan range tanggal
            if "date" in df.columns:
                date_min = df["date"].min()
                date_max = df["date"].max()
                date_range_str = f"{date_min.date()} s/d {date_max.date()}"
            else:
                date_range_str = "N/A"

            status = "✓ OK" if missing == 0 else f"⚠ {missing} NaN"
            files_ok += 1 if missing == 0 else 0

            lines.append(f"  [{status}] {csv_path.name}")
            lines.append(f"           Baris: {n_rows:,} | Kolom: {n_cols} | Periode: {date_range_str}")

            # Detail missing per kolom jika ada
            missing_per_col = df.isnull().sum()
            missing_cols = missing_per_col[missing_per_col > 0]
            if not missing_cols.empty:
                for col, n in missing_cols.items():
                    pct = n / n_rows * 100
                    lines.append(f"           ⚠ {col}: {n} missing ({pct:.1f}%)")

            lines.append("")

        except Exception as e:
            lines.append(f"  [✗ ERROR] {csv_path.name}: {e}")
            lines.append("")

    # Ringkasan
    lines.append("-" * 65)
    lines.append("RINGKASAN:")
    lines.append(f"  File berhasil dibuat    : {len(csv_files)}")
    lines.append(f"  File tanpa missing      : {files_ok}")
    lines.append(f"  Total nilai missing     : {total_missing:,}")
    lines.append("")

    # Rekomendasi
    lines.append("LANGKAH SELANJUTNYA (Fase 2 - Preprocessing):")
    lines.append("  1. Isi template PMI manual dari sumber resmi")
    lines.append("  2. Jalankan src/preprocessing/01_cleaning.py")
    lines.append("  3. Lakukan imputasi missing values (forward-fill untuk weekend)")
    lines.append("  4. Resample semua data ke frekuensi harian")
    lines.append("  5. Merge semua dataset menjadi satu master DataFrame")
    lines.append("=" * 65)

    report_text = "\n".join(lines)

    # Simpan laporan
    report_path = PROCESSED_DIR / "phase1_data_quality_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    log.info(f"Laporan disimpan: {report_path}")
    print("\n" + report_text)
    return report_text


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    steps: list         = None,
    start: str          = DATA_START_DATE,
    end: str            = DATA_END_DATE,
    sentiment_method: str = "vader",
):
    """
    Jalankan pipeline Fase 1 secara lengkap atau sebagian.

    Parameters
    ----------
    steps            : list — langkah yang mau dijalankan
                              None = semua langkah
                              ["price", "exog", "macro", "sentiment", "report"]
    start, end       : str  — periode data
    sentiment_method : str  — "vader" (cepat) atau "finbert" (akurat)
    """
    all_steps = ["price", "exog", "macro", "pmi", "sentiment", "report"]
    if steps is None:
        steps = all_steps

    start_time = time.time()

    log.info("\n" + "=" * 60)
    log.info("  PIPELINE FASE 1: PENGUMPULAN DATA ALUMINIUM")
    log.info("=" * 60)
    log.info(f"  Periode : {start} s/d {end}")
    log.info(f"  Langkah : {steps}")
    log.info(f"  Sentimen: {sentiment_method.upper()}")
    log.info("=" * 60)

    # Validasi konfigurasi sebelum mulai
    validate_config()

    results = {}

    step_map = {
        "price":     lambda: step_price(start, end),
        "exog":      lambda: step_exogenous(start, end),
        "macro":     lambda: step_macro(start, end),
        "pmi":       lambda: step_pmi(start, end),
        "sentiment": lambda: step_sentiment(start, end, sentiment_method),
        "report":    lambda: {"report": generate_data_quality_report(results)},
    }

    for step_name in steps:
        if step_name not in step_map:
            log.warning(f"Langkah tidak dikenal: {step_name}. Skip.")
            continue

        step_start = time.time()
        result = step_map[step_name]()
        step_elapsed = time.time() - step_start

        results[step_name] = result
        status = result.get("status", "unknown")
        log.info(f"\n  [{step_name.upper()}] Status: {status} | Waktu: {step_elapsed:.1f}s")

    # Ringkasan akhir
    total_elapsed = time.time() - start_time
    success_count = sum(1 for r in results.values() if r.get("status") == "success")
    total_steps   = len([s for s in steps if s != "report"])

    log.info("\n" + "=" * 60)
    log.info("  PIPELINE FASE 1 SELESAI")
    log.info("=" * 60)
    log.info(f"  Berhasil    : {success_count}/{total_steps} langkah")
    log.info(f"  Total waktu : {total_elapsed/60:.1f} menit")
    log.info(f"  Output dir  : {RAW_DATA_DIR}")
    log.info("=" * 60)

    return results


# ─────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Fase 1: Pengumpulan Data Prediksi Harga Aluminium",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        choices=["price", "exog", "macro", "pmi", "sentiment", "report", "all"],
        help=(
            "Langkah yang ingin dijalankan:\n"
            "  price     — harga aluminium saja\n"
            "  exog      — kurs, energi, cross-commodity\n"
            "  macro     — indikator makroekonomi\n"
            "  pmi       — PMI manufaktur (otomatis + manual fallback)\n"
            "  sentiment — sentimen berita\n"
            "  report    — laporan kualitas data saja\n"
            "  all       — semua langkah (default)"
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DATA_START_DATE,
        help=f"Tanggal mulai (default: {DATA_START_DATE})",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DATA_END_DATE,
        help=f"Tanggal akhir (default: {DATA_END_DATE})",
    )
    parser.add_argument(
        "--sentiment-method",
        type=str,
        default="vader",
        choices=["vader", "finbert"],
        help="Metode sentiment scoring (default: vader)",
    )

    args = parser.parse_args()

    if args.step is None or args.step == "all":
        steps = None  # Semua langkah
    else:
        steps = [args.step]

    run_pipeline(
        steps=steps,
        start=args.start,
        end=args.end,
        sentiment_method=args.sentiment_method,
    )