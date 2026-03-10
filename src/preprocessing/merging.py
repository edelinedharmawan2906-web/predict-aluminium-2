# ============================================================
# src/preprocessing/merging.py
#
# FASE 2 — LANGKAH 2: Merge Semua Dataset → Master DataFrame
#
# Fungsi:
#   1. Load semua cleaned CSV
#   2. Merge berdasarkan tanggal (date index)
#   3. Align ke date range harga aluminium (target variable)
#   4. Simpan ke data/processed/master_daily.csv
# ============================================================

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_cleaned(name: str) -> pd.DataFrame:
    """Load file cleaned dari data/processed/."""
    path = PROCESSED_DIR / f"cleaned_{name}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


def merge_all() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("FASE 2 — LANGKAH 2: Merge Dataset → Master DataFrame")
    log.info("=" * 60)

    # ── 1. Load semua cleaned data ──
    log.info("\nMemuat data yang sudah dibersihkan...")
    df_al    = load_cleaned("aluminum")
    df_fx    = load_cleaned("exchange_rates")
    df_en    = load_cleaned("energy")
    df_macro = load_cleaned("macro_monthly")
    df_pmi   = load_cleaned("pmi")
    df_wb    = load_cleaned("worldbank")

    log.info(f"  Aluminum    : {len(df_al)} baris")
    log.info(f"  Exchange FX : {len(df_fx)} baris")
    log.info(f"  Energy      : {len(df_en)} baris")
    log.info(f"  Macro       : {len(df_macro)} baris")
    log.info(f"  PMI         : {len(df_pmi)} baris")
    log.info(f"  World Bank  : {len(df_wb)} baris")

    # ── 2. Merge semua ke date range aluminum (sebagai anchor) ──
    log.info("\nMerge semua dataset...")

    # Aluminum sebagai base (hanya hari trading)
    master = df_al.copy()

    # Join semua dataset — gunakan left join agar tetap pada hari trading aluminium
    for name, df in [
        ("exchange_rates", df_fx),
        ("energy",         df_en),
        ("macro_monthly",  df_macro),
        ("pmi",            df_pmi),
        ("worldbank",      df_wb),
    ]:
        # Reindex ke index aluminum untuk alignment yang tepat
        df_aligned = df.reindex(master.index, method="ffill")
        master = master.join(df_aligned, how="left", rsuffix=f"_{name}")
        log.info(f"  ✓ Merge {name}: +{df_aligned.shape[1]} kolom")

    log.info(f"\n  → Master shape: {master.shape}")
    log.info(f"  → Periode: {master.index.min().date()} s/d {master.index.max().date()}")

    # ── 3. Final forward-fill untuk sisa NaN ──
    log.info("\nFinal forward-fill & backward-fill...")
    before_nan = master.isna().sum().sum()
    master = master.ffill().bfill()
    after_nan = master.isna().sum().sum()
    log.info(f"  NaN berkurang: {before_nan} → {after_nan}")

    # ── 4. Report missing per kolom ──
    missing = master.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        log.warning(f"\n  Kolom masih ada NaN ({len(missing)} kolom):")
        for col, n in missing.items():
            pct = n / len(master) * 100
            log.warning(f"    ⚠ {col}: {n} ({pct:.1f}%)")
    else:
        log.info("  ✓ Tidak ada NaN tersisa!")

    # ── 5. Simpan master ──
    out_path = PROCESSED_DIR / "master_daily.csv"
    master.to_csv(out_path)
    log.info(f"\n✅ Master DataFrame disimpan: {out_path}")
    log.info(f"   {master.shape[0]} baris × {master.shape[1]} kolom")

    return master


if __name__ == "__main__":
    merge_all()
