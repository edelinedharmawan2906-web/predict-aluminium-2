# ============================================================
# src/modeling/data_preparation.py
#
# FASE 3 — Persiapan Data untuk Modeling
#
# Fungsi:
#   1. Load features_daily.csv
#   2. Train/test split berdasarkan waktu (80/20)
#   3. Buat target variable: harga 21 hari ke depan
#   4. Sediakan data khusus untuk ARIMA/SARIMAX (univariate/multivariate)
#   5. Sediakan data khusus untuk Random Forest (tabular)
# ============================================================

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR

log = logging.getLogger(__name__)

# ── Konstanta ──────────────────────────────────────────────
FORECAST_HORIZON = 21       # 21 hari trading = ~1 bulan
TEST_RATIO       = 0.20     # 20% data untuk testing
TARGET_COL       = "aluminum_price"

# Fitur eksogen untuk SARIMAX & RF
EXOG_COLS = [
    "usd_cny", "usd_eur", "usd_jpy",
    "natural_gas", "crude_oil_wti",
    "pmi_manufacturing_usa", "pmi_manufacturing_china",
    "ism_manufacturing_pmi", "industrial_production",
    "capacity_utilization", "copper_price_monthly",
]


def load_features() -> pd.DataFrame:
    """Load features_daily.csv."""
    path = PROCESSED_DIR / "features_daily.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    df = df.sort_index()
    log.info(f"Loaded: {df.shape[0]} baris × {df.shape[1]} kolom")
    log.info(f"Periode: {df.index.min().date()} s/d {df.index.max().date()}")
    return df


def create_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    Buat target variable: harga aluminium H hari ke depan.
    Target = aluminum_price shifted -H (harga masa depan).
    """
    df = df.copy()
    df["target"] = df[TARGET_COL].shift(-horizon)

    # Drop baris terakhir yang targetnya NaN (horizon hari terakhir)
    df = df.dropna(subset=["target"])
    log.info(f"Target: harga {horizon} hari ke depan | {len(df)} baris valid")
    return df


def train_test_split_temporal(df: pd.DataFrame, test_ratio: float = TEST_RATIO):
    """
    Split berdasarkan waktu — PENTING untuk time series!
    Tidak boleh random split karena akan menyebabkan data leakage.
    """
    n = len(df)
    split_idx = int(n * (1 - test_ratio))
    split_date = df.index[split_idx]

    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    log.info(f"Train: {len(train)} baris | {train.index.min().date()} s/d {train.index.max().date()}")
    log.info(f"Test : {len(test)} baris  | {test.index.min().date()} s/d {test.index.max().date()}")
    log.info(f"Split date: {split_date.date()}")

    return train, test


def prepare_arima_data(df: pd.DataFrame):
    """
    Persiapkan data untuk ARIMA (univariate).
    ARIMA hanya butuh series harga aluminium.
    """
    series = df[TARGET_COL].copy()
    return series


def prepare_sarimax_data(df: pd.DataFrame):
    """
    Persiapkan data untuk SARIMAX (multivariate).
    SARIMAX butuh target series + matriks eksogen.
    """
    # Filter hanya kolom eksogen yang tersedia
    available_exog = [c for c in EXOG_COLS if c in df.columns]
    missing_exog = [c for c in EXOG_COLS if c not in df.columns]
    if missing_exog:
        log.warning(f"Kolom eksogen tidak ditemukan: {missing_exog}")

    y    = df[TARGET_COL].copy()
    exog = df[available_exog].copy()

    log.info(f"SARIMAX: target='{TARGET_COL}' | {len(available_exog)} variabel eksogen")
    return y, exog


def prepare_rf_data(df: pd.DataFrame):
    """
    Persiapkan data untuk Random Forest (tabular).
    RF butuh feature matrix X dan target vector y.

    Exclude:
    - Kolom target asli (bukan target shifted)
    - Kolom yang akan menyebabkan leakage
    """
    # Kolom yang di-exclude dari fitur
    exclude = [
        "target",           # target variable
        "aluminum_price",   # harga hari ini (ada di lag1 sudah cukup)
        "open", "high", "low", "volume",  # OHLCV — data hari ini (leakage)
    ]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    y = df["target"].copy()

    log.info(f"RF: {X.shape[1]} fitur | {len(y)} sampel")
    return X, y


def get_all_splits():
    """
    Helper utama: load data, buat target, split.
    Return dict berisi semua data yang dibutuhkan setiap model.
    """
    log.info("=" * 55)
    log.info("FASE 3 — Persiapan Data")
    log.info("=" * 55)

    # Load & buat target
    df = load_features()
    df = create_target(df, horizon=FORECAST_HORIZON)

    # Split temporal
    log.info("\nTrain/Test Split (temporal):")
    train, test = train_test_split_temporal(df)

    # Simpan split dates untuk referensi
    split_info = {
        "train_start": train.index.min(),
        "train_end":   train.index.max(),
        "test_start":  test.index.min(),
        "test_end":    test.index.max(),
        "n_train":     len(train),
        "n_test":      len(test),
        "horizon":     FORECAST_HORIZON,
    }

    # ARIMA data
    log.info("\nARIMA data:")
    arima_train = prepare_arima_data(train)
    arima_test  = prepare_arima_data(test)

    # SARIMAX data
    log.info("\nSARIMAX data:")
    sarimax_y_train, sarimax_exog_train = prepare_sarimax_data(train)
    sarimax_y_test,  sarimax_exog_test  = prepare_sarimax_data(test)

    # RF data
    log.info("\nRandom Forest data:")
    X_train, y_train = prepare_rf_data(train)
    X_test,  y_test  = prepare_rf_data(test)

    return {
        "split_info":         split_info,
        "arima_train":        arima_train,
        "arima_test":         arima_test,
        "sarimax_y_train":    sarimax_y_train,
        "sarimax_exog_train": sarimax_exog_train,
        "sarimax_y_test":     sarimax_y_test,
        "sarimax_exog_test":  sarimax_exog_test,
        "X_train":            X_train,
        "y_train":            y_train,
        "X_test":             X_test,
        "y_test":             y_test,
        "df_full":            df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    data = get_all_splits()
    print("\n✅ Data preparation selesai")
    print(f"   Train: {data['split_info']['n_train']} baris")
    print(f"   Test : {data['split_info']['n_test']} baris")
    print(f"   Fitur RF: {data['X_train'].shape[1]} kolom")
