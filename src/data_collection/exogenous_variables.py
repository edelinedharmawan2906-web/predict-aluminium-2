# ============================================================
# src/data_collection/02_exogenous_variables.py
#
# FUNGSI : Mengambil variabel eksogen (exogenous variables):
#          1. Nilai Tukar (Kurs)       — via FRED & Yahoo Finance
#          2. Harga Energi             — via FRED & EIA
#          3. Cross-Commodity          — via Yahoo Finance
#
# OUTPUT : data/raw/exchange_rates_daily.csv
#          data/raw/energy_prices_daily.csv
#          data/raw/cross_commodity_daily.csv
# ============================================================

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    RAW_DATA_DIR, DATA_START_DATE, DATA_END_DATE,
    FRED_API_KEY, FRED_SERIES, YAHOO_TICKERS
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def download_yahoo_batch(
    tickers: dict,
    start: str,
    end: str,
    group_label: str = "data",
) -> pd.DataFrame:
    """
    Download beberapa ticker Yahoo Finance sekaligus dan
    kembalikan sebagai DataFrame dengan kolom per ticker.

    Parameters
    ----------
    tickers     : dict  — {nama_kolom: ticker_symbol}
    start, end  : str   — periode data
    group_label : str   — label untuk log

    Returns
    -------
    pd.DataFrame dengan kolom: date + satu kolom per ticker (harga Close)
    """
    log.info(f"Mengambil {group_label} dari Yahoo Finance: {list(tickers.keys())}")

    ticker_symbols = list(tickers.values())
    col_mapping    = {v: k for k, v in tickers.items()}  # ticker -> nama kolom

    try:
        raw = yf.download(
            ticker_symbols,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            log.error(f"Data kosong untuk {group_label}")
            return pd.DataFrame()

        # Ambil hanya kolom Close
        if isinstance(raw.columns, pd.MultiIndex):
            close_df = raw["Close"].copy()
        else:
            close_df = raw[["Close"]].copy()
            close_df.columns = [ticker_symbols[0]]

        # Reset index
        close_df = close_df.reset_index()
        close_df.columns.name = None

        # Rename kolom ticker -> nama yang lebih deskriptif
        close_df = close_df.rename(columns={"Date": "date", **col_mapping})
        close_df["date"] = pd.to_datetime(close_df["date"])

        # Convert semua kolom harga ke numerik
        price_cols = [c for c in close_df.columns if c != "date"]
        for col in price_cols:
            close_df[col] = pd.to_numeric(close_df[col], errors="coerce")

        close_df = close_df.sort_values("date").reset_index(drop=True)

        # Laporan missing values per kolom
        for col in price_cols:
            n_missing = close_df[col].isna().sum()
            if n_missing > 0:
                log.info(f"  → {col}: {n_missing} missing values")

        log.info(
            f"  → {group_label}: {len(close_df)} baris | "
            f"{close_df['date'].min().date()} s/d {close_df['date'].max().date()}"
        )
        return close_df

    except Exception as e:
        log.error(f"Gagal mengambil {group_label}: {e}")
        return pd.DataFrame()


def fetch_fred_series_batch(
    series_dict: dict,
    start: str,
    end: str,
    group_label: str = "FRED data",
) -> pd.DataFrame:
    """
    Ambil beberapa FRED series sekaligus dan gabungkan
    menjadi satu DataFrame.

    Parameters
    ----------
    series_dict : dict — {nama_kolom: fred_series_id}
    start, end  : str  — periode data
    group_label : str  — label untuk log

    Returns
    -------
    pd.DataFrame dengan kolom: date + satu kolom per series
    """
    if not FRED_API_KEY:
        log.warning(f"FRED_API_KEY tidak diset. Skip {group_label}.")
        return pd.DataFrame()

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        log.info(f"Mengambil {group_label} dari FRED: {list(series_dict.keys())}")

        dfs = []
        for col_name, series_id in series_dict.items():
            try:
                s = fred.get_series(
                    series_id,
                    observation_start=start,
                    observation_end=end,
                )
                df_s = s.reset_index()
                df_s.columns = ["date", col_name]
                df_s["date"] = pd.to_datetime(df_s["date"])
                df_s[col_name] = pd.to_numeric(df_s[col_name], errors="coerce")
                dfs.append(df_s.set_index("date"))
                log.info(f"  ✓ {col_name} ({series_id}): {len(df_s)} observasi")
            except Exception as e:
                log.warning(f"  ✗ {col_name} ({series_id}): {e}")

        if not dfs:
            return pd.DataFrame()

        # Gabungkan semua series pada satu DataFrame
        combined = pd.concat(dfs, axis=1).reset_index()
        combined = combined.sort_values("date").reset_index(drop=True)

        log.info(
            f"  → {group_label}: {len(combined)} baris | "
            f"{combined['date'].min().date()} s/d {combined['date'].max().date()}"
        )
        return combined

    except ImportError:
        log.error("fredapi belum terinstall. Jalankan: pip install fredapi")
        return pd.DataFrame()
    except Exception as e:
        log.error(f"Gagal mengambil {group_label} dari FRED: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 1. NILAI TUKAR (EXCHANGE RATES)
# ─────────────────────────────────────────────────────────────

def fetch_exchange_rates(
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
) -> pd.DataFrame:
    """
    Ambil data nilai tukar harian.

    Strategi:
    - FRED   : USD/CNY, USD/EUR, USD/JPY (akurat, gratis)
    - Yahoo  : DXY Dollar Index (DX-Y.NYB) sebagai tambahan

    Returns
    -------
    pd.DataFrame kolom: date, usd_cny, usd_eur, usd_jpy, dollar_index
    """
    log.info("=== Pengambilan Data Nilai Tukar ===")

    # -- Dari FRED --
    fred_fx = {
        "usd_cny": FRED_SERIES["usd_cny"],   # USD per 1 CNY
        "usd_eur": FRED_SERIES["usd_eur"],   # USD per 1 EUR
        "usd_jpy": FRED_SERIES["usd_jpy"],   # USD per 100 JPY
    }
    df_fred_fx = fetch_fred_series_batch(
        fred_fx, start, end, group_label="Exchange Rates (FRED)"
    )

    # -- Dollar Index dari Yahoo Finance --
    dxy_ticker = {"dollar_index": YAHOO_TICKERS["dollar_index"]}
    df_dxy = download_yahoo_batch(
        dxy_ticker, start, end, group_label="Dollar Index (Yahoo)"
    )

    # -- Gabungkan --
    if not df_fred_fx.empty and not df_dxy.empty:
        df = pd.merge(df_fred_fx, df_dxy, on="date", how="outer")
    elif not df_fred_fx.empty:
        df = df_fred_fx
    elif not df_dxy.empty:
        df = df_dxy
    else:
        log.error("Semua sumber nilai tukar gagal!")
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)
    df["source"] = "fred+yahoo"

    # Hitung perubahan persen harian untuk setiap kurs
    fx_cols = [c for c in df.columns if c not in ["date", "source"]]
    for col in fx_cols:
        df[f"{col}_pct_change"] = df[col].pct_change(fill_method=None) * 100

    log.info(f"  → Total: {len(df)} baris, {len(fx_cols)} variabel kurs")
    return df


# ─────────────────────────────────────────────────────────────
# 2. HARGA ENERGI
# ─────────────────────────────────────────────────────────────

def fetch_energy_prices(
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
) -> pd.DataFrame:
    """
    Ambil harga energi harian.

    Sumber:
    - FRED   : Natural Gas (Henry Hub), WTI Crude Oil
    - Yahoo  : Crude Oil Futures (CL=F) sebagai backup/supplement

    Catatan: Harga batubara dan listrik tersedia bulanan, bukan harian.
    Akan di-resample ke harian dengan forward-fill di tahap preprocessing.

    Returns
    -------
    pd.DataFrame kolom: date, natural_gas, crude_oil_wti, crude_oil_futures
    """
    log.info("=== Pengambilan Data Harga Energi ===")

    # -- Dari FRED --
    fred_energy = {
        "natural_gas":   FRED_SERIES["natural_gas"],    # USD/MMBtu, harian
        "crude_oil_wti": FRED_SERIES["crude_oil_wti"],  # USD/barrel, harian
    }
    df_fred_energy = fetch_fred_series_batch(
        fred_energy, start, end, group_label="Energy Prices (FRED)"
    )

    # -- Crude Oil Futures dari Yahoo sebagai backup --
    oil_yahoo = {"crude_oil_futures": YAHOO_TICKERS["crude_oil"]}
    df_oil_yahoo = download_yahoo_batch(
        oil_yahoo, start, end, group_label="Crude Oil Futures (Yahoo)"
    )

    # -- Gabungkan --
    if not df_fred_energy.empty and not df_oil_yahoo.empty:
        df = pd.merge(df_fred_energy, df_oil_yahoo, on="date", how="outer")
    elif not df_fred_energy.empty:
        df = df_fred_energy
    else:
        df = df_oil_yahoo

    if df.empty:
        log.error("Semua sumber harga energi gagal!")
        return pd.DataFrame()

    df = df.sort_values("date").reset_index(drop=True)
    df["source"] = "fred+yahoo"

    # Hitung perubahan persen harian
    energy_cols = [c for c in df.columns if c not in ["date", "source"]]
    for col in energy_cols:
        df[f"{col}_pct_change"] = df[col].pct_change(fill_method=None) * 100

    log.info(f"  → Total: {len(df)} baris, {len(energy_cols)} variabel energi")
    return df


# ─────────────────────────────────────────────────────────────
# 3. CROSS-COMMODITY (Komoditas Terkait)
# ─────────────────────────────────────────────────────────────

def fetch_cross_commodity(
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
) -> pd.DataFrame:
    """
    Ambil harga komoditas yang berkorelasi dengan aluminium:
    - Tembaga (copper)  : berkorelasi kuat, satu keluarga logam dasar
    - Emas (gold)       : indikator risk sentiment global
    - S&P 500           : risk appetite pasar
    - Shanghai Composite: indikator ekonomi China (produsen utama aluminium)

    Returns
    -------
    pd.DataFrame kolom: date, copper, gold, sp500, shanghai
    """
    log.info("=== Pengambilan Data Cross-Commodity ===")

    cross_tickers = {
        "copper":   YAHOO_TICKERS["copper"],
        "gold":     YAHOO_TICKERS["gold"],
        "sp500":    YAHOO_TICKERS["sp500"],
        "shanghai": YAHOO_TICKERS["shanghai"],
    }

    df = download_yahoo_batch(
        cross_tickers, start, end, group_label="Cross-Commodity"
    )

    if df.empty:
        log.error("Gagal mengambil data cross-commodity!")
        return pd.DataFrame()

    df["source"] = "yahoo_finance"

    # Hitung perubahan persen harian
    price_cols = [c for c in df.columns if c not in ["date", "source"]]
    for col in price_cols:
        df[f"{col}_pct_change"] = df[col].pct_change(fill_method=None) * 100

    log.info(f"  → Total: {len(df)} baris, {len(price_cols)} variabel cross-commodity")
    return df


# ─────────────────────────────────────────────────────────────
# 4. SIMPAN DATA
# ─────────────────────────────────────────────────────────────

def save_exogenous_data(
    df_fx: pd.DataFrame,
    df_energy: pd.DataFrame,
    df_cross: pd.DataFrame,
) -> dict:
    """Simpan semua DataFrame ke CSV."""
    saved = {}

    datasets = {
        "exchange_rates_daily.csv":    df_fx,
        "energy_prices_daily.csv":     df_energy,
        "cross_commodity_daily.csv":   df_cross,
    }

    for filename, df in datasets.items():
        if not df.empty:
            path = RAW_DATA_DIR / filename
            df.to_csv(path, index=False)
            log.info(f"  ✓ Disimpan: {path}")
            saved[filename] = path

    return saved


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

def run(start: str = DATA_START_DATE, end: str = DATA_END_DATE):
    """Entry point untuk pengambilan variabel eksogen."""
    log.info("=" * 60)
    log.info("FASE 1 — Pengumpulan Data: Variabel Eksogen")
    log.info("=" * 60)

    df_fx     = fetch_exchange_rates(start=start, end=end)
    df_energy = fetch_energy_prices(start=start, end=end)
    df_cross  = fetch_cross_commodity(start=start, end=end)

    log.info("\nMenyimpan data variabel eksogen...")
    saved_paths = save_exogenous_data(df_fx, df_energy, df_cross)

    log.info("\n=== RINGKASAN VARIABEL EKSOGEN ===")
    for name, df in [("Exchange Rates", df_fx), ("Energy Prices", df_energy), ("Cross-Commodity", df_cross)]:
        if not df.empty:
            cols = [c for c in df.columns if c not in ["date", "source"] and not c.endswith("_pct_change")]
            log.info(f"  {name}: {len(df)} baris, variabel: {cols}")

    return {
        "exchange_rates":   df_fx,
        "energy_prices":    df_energy,
        "cross_commodity":  df_cross,
        "paths":            saved_paths,
    }


if __name__ == "__main__":
    results = run()