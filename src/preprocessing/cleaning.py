# ============================================================
# src/preprocessing/cleaning.py
#
# FASE 2 — LANGKAH 1: Pembersihan & Standardisasi Data
#
# Fungsi:
#   1. Load semua CSV dari data/raw/
#   2. Standarisasi format tanggal & kolom
#   3. Handle missing values (forward-fill weekend, backward-fill PMI awal)
#   4. Drop kolom yang 100% kosong (PMI EU, PMI Global)
#   5. Simpan ke data/processed/cleaned_*.csv
# ============================================================

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_DIR, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────

def load_aluminum_daily() -> pd.DataFrame:
    """Load harga aluminium harian dari Investing.com."""
    path = RAW_DATA_DIR / "aluminum_price_daily.csv"
    df = pd.read_csv(path)

    # Cari kolom tanggal
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime"]), None)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})
    df = df.set_index("date").sort_index()

    # Ambil kolom harga utama
    price_col = next((c for c in df.columns if "price" in c.lower() or "close" in c.lower()), df.columns[0])
    df = df.rename(columns={price_col: "aluminum_price"})

    # Pastikan kolom yang dibutuhkan ada
    keep = ["aluminum_price"]
    for col in ["open", "high", "low", "volume"]:
        match = next((c for c in df.columns if col in c.lower()), None)
        if match:
            df = df.rename(columns={match: col})
            keep.append(col)

    log.info(f"  ✓ Aluminum daily: {len(df)} baris | {df.index.min().date()} s/d {df.index.max().date()}")
    return df[keep]


def load_exchange_rates() -> pd.DataFrame:
    """Load data nilai tukar harian dari FRED."""
    path = RAW_DATA_DIR / "exchange_rates_daily.csv"
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime"]), None)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()

    # Ambil hanya kolom nilai tukar utama (bukan pct_change dulu)
    cols = [c for c in df.columns if c in ["usd_cny", "usd_eur", "usd_jpy"]]
    log.info(f"  ✓ Exchange rates: {len(df)} baris | kolom: {cols}")
    return df[cols]


def load_energy_prices() -> pd.DataFrame:
    """Load data harga energi harian dari FRED."""
    path = RAW_DATA_DIR / "energy_prices_daily.csv"
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime"]), None)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()

    cols = [c for c in df.columns if c in ["natural_gas", "crude_oil_wti"]]
    log.info(f"  ✓ Energy prices: {len(df)} baris | kolom: {cols}")
    return df[cols]


def load_macro_monthly() -> pd.DataFrame:
    """Load indikator makro bulanan dari FRED."""
    path = RAW_DATA_DIR / "macro_indicators_monthly.csv"
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime"]), None)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()

    log.info(f"  ✓ Macro monthly: {len(df)} baris | kolom: {list(df.columns)}")
    return df


def load_macro_quarterly() -> pd.DataFrame:
    """Load GDP data kuartalan dari FRED."""
    path = RAW_DATA_DIR / "macro_indicators_quarterly.csv"
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime"]), None)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()

    log.info(f"  ✓ Macro quarterly: {len(df)} baris | kolom: {list(df.columns)}")
    return df


def load_pmi() -> pd.DataFrame:
    """Load data PMI manufaktur."""
    path = RAW_DATA_DIR / "pmi_data.csv"
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime"]), None)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()

    # Drop kolom yang 100% NaN (PMI EU, PMI Global)
    before = df.shape[1]
    df = df.dropna(axis=1, how="all")
    after = df.shape[1]
    if before != after:
        log.info(f"  → Drop {before - after} kolom 100% NaN (PMI EU/Global)")

    log.info(f"  ✓ PMI: {len(df)} baris | kolom: {list(df.columns)}")
    return df


def load_worldbank() -> pd.DataFrame:
    """Load data GDP tahunan World Bank."""
    path = RAW_DATA_DIR / "macro_worldbank_annual.csv"
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ["date", "datetime"]), None)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()

    log.info(f"  ✓ World Bank annual: {len(df)} baris | kolom: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. RESAMPLE KE FREKUENSI HARIAN
# ─────────────────────────────────────────────────────────────

def resample_to_daily(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """
    Resample DataFrame ke frekuensi harian.
    method: 'ffill' untuk forward-fill (kurs, energi, PMI)
            'interpolate' untuk interpolasi linear (GDP)
    """
    # Buat date range harian
    date_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df_daily = df.reindex(date_range)

    if method == "ffill":
        df_daily = df_daily.ffill()
    elif method == "interpolate":
        df_daily = df_daily.interpolate(method="time")
    elif method == "bfill":
        df_daily = df_daily.bfill()

    df_daily.index.name = "date"
    return df_daily


# ─────────────────────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────────────────────────

def clean_aluminum(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data harga aluminium."""
    # Forward-fill harga untuk weekend/holiday
    df["aluminum_price"] = df["aluminum_price"].ffill()

    # Volume: isi dengan 0 jika NaN (volume tidak tersedia = tidak ada trading)
    if "volume" in df.columns:
        df["volume"] = df["volume"].fillna(0)

    # Drop baris yang masih NaN di harga (seharusnya tidak ada)
    before = len(df)
    df = df.dropna(subset=["aluminum_price"])
    if len(df) < before:
        log.warning(f"  Drop {before - len(df)} baris karena harga aluminium NaN")

    return df


def clean_exchange_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data nilai tukar - forward-fill weekend."""
    df = df.ffill().bfill()
    return df


def clean_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data energi - forward-fill weekend."""
    df = df.ffill().bfill()
    return df


def clean_macro_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data makro bulanan - forward-fill 1-2 data missing."""
    df = df.ffill().bfill()
    return df


def clean_pmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan data PMI:
    - Backward-fill untuk Jan-Oct 2017 (isi dengan nilai Nov 2017)
    - Forward-fill untuk bulan-bulan berikutnya
    """
    # Backward-fill dulu untuk mengisi awal periode
    df = df.bfill()
    # Forward-fill untuk sisanya
    df = df.ffill()
    return df


def clean_worldbank(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan data World Bank tahunan - interpolasi linear."""
    df = df.interpolate(method="time").ffill().bfill()
    return df


# ─────────────────────────────────────────────────────────────
# 4. MAIN: JALANKAN SEMUA CLEANING
# ─────────────────────────────────────────────────────────────

def run_cleaning():
    log.info("=" * 60)
    log.info("FASE 2 — LANGKAH 1: Pembersihan & Standardisasi Data")
    log.info("=" * 60)

    cleaned = {}

    # ── 1. Aluminum Price (harian) ──
    log.info("\n[1/6] Harga Aluminium...")
    df_al = load_aluminum_daily()
    df_al = clean_aluminum(df_al)
    cleaned["aluminum"] = df_al
    log.info(f"  → Setelah cleaning: {len(df_al)} baris, {df_al.isna().sum().sum()} NaN")

    # ── 2. Exchange Rates (harian → resample harian) ──
    log.info("\n[2/6] Nilai Tukar...")
    df_fx = load_exchange_rates()
    df_fx_daily = resample_to_daily(df_fx, method="ffill")
    df_fx_daily = clean_exchange_rates(df_fx_daily)
    cleaned["exchange_rates"] = df_fx_daily
    log.info(f"  → Setelah cleaning: {len(df_fx_daily)} baris, {df_fx_daily.isna().sum().sum()} NaN")

    # ── 3. Energy Prices (harian → resample harian) ──
    log.info("\n[3/6] Harga Energi...")
    df_en = load_energy_prices()
    df_en_daily = resample_to_daily(df_en, method="ffill")
    df_en_daily = clean_energy(df_en_daily)
    cleaned["energy"] = df_en_daily
    log.info(f"  → Setelah cleaning: {len(df_en_daily)} baris, {df_en_daily.isna().sum().sum()} NaN")

    # ── 4. Macro Monthly (bulanan → resample harian) ──
    log.info("\n[4/6] Makroekonomi Bulanan...")
    df_macro = load_macro_monthly()
    df_macro_daily = resample_to_daily(df_macro, method="ffill")
    df_macro_daily = clean_macro_monthly(df_macro_daily)
    cleaned["macro_monthly"] = df_macro_daily
    log.info(f"  → Setelah cleaning: {len(df_macro_daily)} baris, {df_macro_daily.isna().sum().sum()} NaN")

    # ── 5. PMI (bulanan → resample harian) ──
    log.info("\n[5/6] PMI Manufaktur...")
    df_pmi = load_pmi()
    df_pmi_daily = resample_to_daily(df_pmi, method="ffill")
    df_pmi_daily = clean_pmi(df_pmi_daily)
    cleaned["pmi"] = df_pmi_daily
    log.info(f"  → Setelah cleaning: {len(df_pmi_daily)} baris, {df_pmi_daily.isna().sum().sum()} NaN")

    # ── 6. World Bank Annual (tahunan → resample harian) ──
    log.info("\n[6/6] World Bank GDP...")
    df_wb = load_worldbank()
    df_wb_daily = resample_to_daily(df_wb, method="interpolate")
    df_wb_daily = clean_worldbank(df_wb_daily)
    cleaned["worldbank"] = df_wb_daily
    log.info(f"  → Setelah cleaning: {len(df_wb_daily)} baris, {df_wb_daily.isna().sum().sum()} NaN")

    # ── Simpan semua ──
    log.info("\n=== Menyimpan data yang sudah dibersihkan ===")
    for name, df in cleaned.items():
        out_path = PROCESSED_DIR / f"cleaned_{name}.csv"
        df.to_csv(out_path)
        log.info(f"  ✓ {out_path.name}: {len(df)} baris, {df.shape[1]} kolom")

    log.info("\n✅ Cleaning selesai!")
    return cleaned


if __name__ == "__main__":
    run_cleaning()
