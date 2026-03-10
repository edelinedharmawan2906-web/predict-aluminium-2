# ============================================================
# src/preprocessing/feature_engineering.py
#
# FASE 2 — LANGKAH 3: Feature Engineering
#
# Fitur yang dibuat:
#   A. Lag features        : harga t-1, t-5, t-10, t-21 (1,5,10,21 hari lalu)
#   B. Rolling statistics  : MA-7, MA-21, MA-63, std-21
#   C. Return/momentum     : daily return, weekly return, monthly return
#   D. Volatility          : rolling std 21 hari
#   E. Seasonal features   : bulan, kuartal, day-of-week
#   F. Cross-asset ratio   : harga aluminium / harga tembaga
#   G. PMI momentum        : perubahan PMI bulan ini vs bulan lalu
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


def add_lag_features(df: pd.DataFrame, col: str = "aluminum_price",
                     lags: list = [1, 5, 10, 21, 63]) -> pd.DataFrame:
    """Tambah lag features untuk target variable."""
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    log.info(f"  ✓ Lag features: {[f'lag{l}' for l in lags]}")
    return df


def add_rolling_features(df: pd.DataFrame, col: str = "aluminum_price") -> pd.DataFrame:
    """Tambah rolling statistics."""
    windows = [7, 21, 63]  # 1 minggu, 1 bulan, 1 kuartal
    for w in windows:
        df[f"{col}_ma{w}"]  = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_std{w}"] = df[col].rolling(w, min_periods=1).std()

    # Rasio harga vs moving average (momentum indicator)
    df[f"{col}_vs_ma21"] = df[col] / df[f"{col}_ma21"]
    df[f"{col}_vs_ma63"] = df[col] / df[f"{col}_ma63"]

    log.info(f"  ✓ Rolling features: MA & STD untuk window {windows}")
    return df


def add_return_features(df: pd.DataFrame, col: str = "aluminum_price") -> pd.DataFrame:
    """Tambah return/momentum features."""
    df[f"{col}_return_1d"]  = df[col].pct_change(1)   * 100   # daily return %
    df[f"{col}_return_5d"]  = df[col].pct_change(5)   * 100   # weekly return %
    df[f"{col}_return_21d"] = df[col].pct_change(21)  * 100   # monthly return %
    df[f"{col}_return_63d"] = df[col].pct_change(63)  * 100   # quarterly return %

    log.info(f"  ✓ Return features: 1d, 5d, 21d, 63d")
    return df


def add_volatility_features(df: pd.DataFrame, col: str = "aluminum_price") -> pd.DataFrame:
    """Tambah volatility features."""
    daily_ret = df[col].pct_change()

    # Realized volatility (annualized)
    df[f"{col}_vol_21d"] = daily_ret.rolling(21, min_periods=5).std() * np.sqrt(252) * 100
    df[f"{col}_vol_63d"] = daily_ret.rolling(63, min_periods=10).std() * np.sqrt(252) * 100

    log.info(f"  ✓ Volatility features: vol_21d, vol_63d (annualized %)")
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah fitur seasonal/kalender."""
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["day_of_week"]  = df.index.dayofweek   # 0=Senin, 4=Jumat
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["year"]         = df.index.year

    # One-hot encoding untuk bulan (12 kolom)
    for m in range(1, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)

    log.info(f"  ✓ Seasonal features: month, quarter, dow, week, year + month dummies")
    return df


def add_pmi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah PMI momentum features."""
    pmi_cols = [c for c in df.columns if c.startswith("pmi_manufacturing")]

    for col in pmi_cols:
        # Perubahan PMI bulan ini vs bulan lalu
        df[f"{col}_change"] = df[col].diff(21)  # ~1 bulan trading days
        # PMI di atas/bawah 50 (ekspansi/kontraksi)
        df[f"{col}_above50"] = (df[col] > 50).astype(int)

    if pmi_cols:
        log.info(f"  ✓ PMI features: change & above50 untuk {pmi_cols}")
    return df


def add_fx_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah fitur nilai tukar tambahan."""
    fx_cols = [c for c in df.columns if c.startswith("usd_")]

    for col in fx_cols:
        df[f"{col}_change_5d"]  = df[col].pct_change(5) * 100
        df[f"{col}_change_21d"] = df[col].pct_change(21) * 100

    if fx_cols:
        log.info(f"  ✓ FX change features: 5d & 21d untuk {fx_cols}")
    return df


def add_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah fitur harga energi tambahan."""
    energy_cols = [c for c in df.columns if c in ["natural_gas", "crude_oil_wti"]]

    for col in energy_cols:
        df[f"{col}_ma21"]      = df[col].rolling(21, min_periods=1).mean()
        df[f"{col}_change_21d"] = df[col].pct_change(21) * 100

    if energy_cols:
        log.info(f"  ✓ Energy features: MA21 & change_21d untuk {energy_cols}")
    return df


def run_feature_engineering() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("FASE 2 — LANGKAH 3: Feature Engineering")
    log.info("=" * 60)

    # Load master
    master_path = PROCESSED_DIR / "master_daily.csv"
    df = pd.read_csv(master_path, index_col=0, parse_dates=True)
    df.index.name = "date"
    log.info(f"\nLoaded master: {df.shape[0]} baris × {df.shape[1]} kolom")

    # ── Tambah semua fitur ──
    log.info("\nMenambah fitur...")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_seasonal_features(df)
    df = add_pmi_features(df)
    df = add_fx_features(df)
    df = add_energy_features(df)

    log.info(f"\n  → Total fitur: {df.shape[1]} kolom")

    # ── Drop baris awal yang NaN karena lag ──
    # Lag terpanjang = 63 hari, jadi drop 63 baris pertama
    n_before = len(df)
    df = df.iloc[63:]  # Skip 63 hari pertama
    log.info(f"  → Drop {n_before - len(df)} baris awal (lag warmup)")
    log.info(f"  → Final: {len(df)} baris × {df.shape[1]} kolom")
    log.info(f"  → Periode: {df.index.min().date()} s/d {df.index.max().date()}")

    # ── Handle NaN sisa ──
    before_nan = df.isna().sum().sum()
    df = df.ffill().bfill()
    after_nan = df.isna().sum().sum()
    log.info(f"  → NaN setelah fillna: {after_nan} (dari {before_nan})")

    # ── Simpan ──
    out_path = PROCESSED_DIR / "features_daily.csv"
    df.to_csv(out_path)
    log.info(f"\n✅ Features disimpan: {out_path}")
    log.info(f"   {df.shape[0]} baris × {df.shape[1]} kolom")

    # ── Ringkasan kolom ──
    log.info("\n=== Daftar Fitur ===")
    categories = {
        "Target"     : [c for c in df.columns if c == "aluminum_price"],
        "Lag"        : [c for c in df.columns if "_lag" in c],
        "Rolling MA" : [c for c in df.columns if "_ma" in c],
        "Rolling STD": [c for c in df.columns if "_std" in c],
        "Return"     : [c for c in df.columns if "_return_" in c],
        "Volatility" : [c for c in df.columns if "_vol_" in c],
        "FX"         : [c for c in df.columns if c.startswith("usd_")],
        "Energy"     : [c for c in df.columns if c in ["natural_gas", "crude_oil_wti"] or "natural_gas_" in c or "crude_oil_" in c],
        "PMI"        : [c for c in df.columns if "pmi_" in c],
        "Macro"      : [c for c in df.columns if any(x in c for x in ["industrial", "capacity", "construction", "retail", "copper_price", "coal_price", "gdp"])],
        "Seasonal"   : [c for c in df.columns if c in ["month", "quarter", "day_of_week", "week_of_year", "year"] or c.startswith("month_")],
    }
    for cat, cols in categories.items():
        if cols:
            log.info(f"  {cat:12s}: {len(cols):3d} fitur")

    return df


if __name__ == "__main__":
    run_feature_engineering()
