# ============================================================
# src/data_collection/03_macro_indicators.py
#
# FUNGSI : Mengambil data makroekonomi global:
#          - PMI Manufaktur (China, USA, Global)
#          - GDP Growth
#          - Industrial Production Index
#          - China Fixed Asset Investment (proxy demand konstruksi)
#
# CATATAN: Data makro tersedia BULANAN / KUARTALAN, bukan harian.
#          Akan di-resample/interpolasi ke harian di tahap preprocessing.
#
# OUTPUT : data/raw/macro_indicators_monthly.csv
#          data/raw/macro_indicators_quarterly.csv
# ============================================================

import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    RAW_DATA_DIR, DATA_START_DATE, DATA_END_DATE,
    FRED_API_KEY, FRED_SERIES
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. PMI MANUFAKTUR — via FRED
# ─────────────────────────────────────────────────────────────

# FRED Series IDs untuk PMI dan indikator makro
MACRO_FRED_MONTHLY = {
    # PMI Proxy (ISM Manufacturing Index — representasi PMI USA)
    # Nilai > 50 = ekspansi, < 50 = kontraksi
    "ism_manufacturing_pmi":    "MANEMP",      # Manufacturing Employment (FRED, bulanan)
    "industrial_production":    "INDPRO",      # Industrial Production Index USA (bulanan)
    "capacity_utilization":     "TCU",         # Total Capacity Utilization USA (bulanan)

    # Indikator Permintaan
    "us_construction_spending": "TTLCONS",     # Total Construction Spending USA (bulanan)
    "us_retail_sales":          "RSXFS",       # Retail Sales USA (proxy consumer demand)

    # Harga Komoditas Bulanan (supplement)
    "copper_price_monthly":     FRED_SERIES["copper_price"],
    "coal_price_monthly":       FRED_SERIES["coal_price"],
}

MACRO_FRED_QUARTERLY = {
    "us_gdp_growth_rate":       FRED_SERIES["us_gdp_growth"],  # GDP Growth Rate (kuartalan)
    "us_gdp_level":             "GDP",                          # GDP Level nominal (kuartalan)
}

# ─────────────────────────────────────────────────────────────
# PMI China — via World Bank API (GRATIS, tidak perlu API key)
# World Bank indicator code untuk China manufacturing
# ─────────────────────────────────────────────────────────────

WORLDBANK_INDICATORS = {
    # China
    "china_gdp_growth":        ("CN", "NY.GDP.MKTP.KD.ZG"),   # GDP Growth Rate China (tahunan)
    "china_industrial_output": ("CN", "NV.IND.TOTL.KD.ZG"),   # Industrial value added growth

    # Global
    "world_gdp_growth":        ("1W", "NY.GDP.MKTP.KD.ZG"),   # GDP Growth Rate Global

    # Major Economies
    "eu_gdp_growth":           ("EU", "NY.GDP.MKTP.KD.ZG"),   # EU GDP Growth
}


def fetch_fred_macro(
    series_dict: dict,
    start: str,
    end: str,
    group_label: str = "Macro (FRED)",
) -> pd.DataFrame:
    """
    Ambil beberapa FRED series untuk indikator makroekonomi.
    (Fungsi ini mirip dengan di file 02, tapi dipisah untuk modularitas)
    """
    if not FRED_API_KEY:
        log.warning(f"FRED_API_KEY tidak diset. Skip {group_label}.")
        return pd.DataFrame()

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        log.info(f"Mengambil {group_label} dari FRED...")

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
                df_s["date"]    = pd.to_datetime(df_s["date"])
                df_s[col_name]  = pd.to_numeric(df_s[col_name], errors="coerce")
                dfs.append(df_s.set_index("date"))
                log.info(f"  ✓ {col_name} ({series_id}): {len(df_s)} observasi")
            except Exception as e:
                log.warning(f"  ✗ {col_name} ({series_id}): {e}")
                time.sleep(1)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, axis=1).reset_index()
        combined = combined.sort_values("date").reset_index(drop=True)
        log.info(f"  → {group_label}: {len(combined)} baris")
        return combined

    except ImportError:
        log.error("fredapi belum terinstall.")
        return pd.DataFrame()
    except Exception as e:
        log.error(f"Gagal mengambil {group_label}: {e}")
        return pd.DataFrame()


def fetch_worldbank_indicator(
    country_code: str,
    indicator_code: str,
    col_name: str,
    start_year: int,
    end_year: int,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Ambil satu indikator dari World Bank API (gratis, tanpa API key).

    API Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392

    Parameters
    ----------
    country_code   : str — kode negara 2-huruf (misal 'CN' untuk China, '1W' untuk World)
    indicator_code : str — kode indikator World Bank (misal 'NY.GDP.MKTP.KD.ZG')
    col_name       : str — nama kolom output
    start_year     : int — tahun mulai
    end_year       : int — tahun akhir

    Returns
    -------
    pd.DataFrame dengan kolom: date, {col_name}
    """
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}"
        f"/indicator/{indicator_code}"
        f"?format=json&date={start_year}:{end_year}&per_page=100"
    )

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # World Bank API mengembalikan list [metadata, data]
            if not isinstance(data, list) or len(data) < 2:
                log.warning(f"  Format respons tidak terduga untuk {col_name}")
                return pd.DataFrame()

            records = data[1]
            if not records:
                log.warning(f"  Tidak ada data untuk {col_name}")
                return pd.DataFrame()

            rows = []
            for rec in records:
                if rec.get("value") is not None:
                    rows.append({
                        "date":   pd.to_datetime(f"{rec['date']}-01-01"),
                        col_name: float(rec["value"]),
                    })

            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
            log.info(f"  ✓ {col_name}: {len(df)} tahun data")
            return df

        except Exception as e:
            log.warning(f"  Attempt {attempt} gagal untuk {col_name}: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)

    return pd.DataFrame()


def fetch_worldbank_macro(
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
) -> pd.DataFrame:
    """
    Ambil indikator makroekonomi dari World Bank API.

    Catatan: World Bank data adalah TAHUNAN (annual), bukan bulanan/kuartalan.
    Akan di-resample ke bulanan saat preprocessing dengan asumsi nilai konstan
    sepanjang tahun atau interpolasi linear.

    Returns
    -------
    pd.DataFrame dengan kolom: date + satu kolom per indikator
    """
    log.info("Mengambil data makroekonomi dari World Bank API...")

    start_year = int(start[:4])
    end_year   = int(end[:4])

    dfs = []
    for col_name, (country, indicator) in WORLDBANK_INDICATORS.items():
        df = fetch_worldbank_indicator(
            country_code=country,
            indicator_code=indicator,
            col_name=col_name,
            start_year=start_year,
            end_year=end_year,
        )
        if not df.empty:
            dfs.append(df.set_index("date"))
        time.sleep(0.5)  # Rate limit courtesy

    if not dfs:
        log.error("Semua World Bank requests gagal!")
        return pd.DataFrame()

    combined = pd.concat(dfs, axis=1).reset_index()
    combined = combined.sort_values("date").reset_index(drop=True)
    log.info(f"  → World Bank: {len(combined)} baris, {len(combined.columns)-1} indikator")
    return combined


# ─────────────────────────────────────────────────────────────
# 2. PMI MANUAL / TAMBAHAN
# ─────────────────────────────────────────────────────────────

def create_pmi_placeholder() -> pd.DataFrame:
    """
    Buat template PMI placeholder.

    PMI China (Caixin) dan PMI Global (S&P Global/Markit) tidak tersedia
    secara gratis via API. Opsi untuk mengisinya:

    1. Manual: download dari situs resmi dan import ke CSV
       - Caixin PMI: https://www.pmi.spglobal.com/
       - ISM Manufacturing: https://www.ismworld.org/
       - JP Morgan Global PMI: https://www.spglobal.com/

    2. Scraping (perlu cek terms of service)
       - Investing.com memiliki data PMI historis

    3. Berbayar: Bloomberg, Refinitiv Eikon

    Fungsi ini membuat template CSV kosong yang bisa diisi manual.
    """
    log.info("Membuat template PMI untuk diisi manual...")

    # Buat date range bulanan
    date_range = pd.date_range(
        start=DATA_START_DATE,
        end=DATA_END_DATE,
        freq="MS",  # Month Start
    )

    df_template = pd.DataFrame({
        "date":                     date_range,
        "pmi_manufacturing_china":  None,   # Caixin Manufacturing PMI China
        "pmi_manufacturing_usa":    None,   # ISM Manufacturing PMI USA
        "pmi_manufacturing_eu":     None,   # S&P Global Markit Manufacturing PMI EU
        "pmi_manufacturing_global": None,   # JP Morgan Global Manufacturing PMI
        "notes": "MANUAL_REQUIRED",         # Flag untuk isi manual
    })

    path = RAW_DATA_DIR / "pmi_template_manual.csv"
    df_template.to_csv(path, index=False)
    log.info(f"  ✓ Template PMI disimpan: {path}")
    log.info("  → Isi kolom PMI secara manual dari sumber:")
    log.info("    - Caixin China PMI  : https://www.pmi.spglobal.com/")
    log.info("    - ISM USA PMI       : https://www.ismworld.org/")
    log.info("    - S&P Global EU PMI : https://www.spglobal.com/")

    return df_template


# ─────────────────────────────────────────────────────────────
# 3. SIMPAN DATA
# ─────────────────────────────────────────────────────────────

def save_macro_data(
    df_monthly: pd.DataFrame,
    df_quarterly: pd.DataFrame,
    df_worldbank: pd.DataFrame,
) -> dict:
    """Simpan semua DataFrame makroekonomi ke CSV."""
    saved = {}

    datasets = {
        "macro_indicators_monthly.csv":    df_monthly,
        "macro_indicators_quarterly.csv":  df_quarterly,
        "macro_worldbank_annual.csv":      df_worldbank,
    }

    for filename, df in datasets.items():
        if not df.empty:
            path = RAW_DATA_DIR / filename
            df.to_csv(path, index=False)
            log.info(f"  ✓ Disimpan: {path}")
            saved[filename] = path

    return saved


# ─────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────

def run(start: str = DATA_START_DATE, end: str = DATA_END_DATE):
    """Entry point untuk pengambilan data makroekonomi."""
    log.info("=" * 60)
    log.info("FASE 1 — Pengumpulan Data: Indikator Makroekonomi")
    log.info("=" * 60)

    # 1. Data bulanan dari FRED
    df_monthly = fetch_fred_macro(
        MACRO_FRED_MONTHLY, start, end, group_label="Macro Monthly (FRED)"
    )

    # 2. Data kuartalan dari FRED
    df_quarterly = fetch_fred_macro(
        MACRO_FRED_QUARTERLY, start, end, group_label="Macro Quarterly (FRED)"
    )

    # 3. Data tahunan dari World Bank
    df_worldbank = fetch_worldbank_macro(start=start, end=end)

    # 4. Buat template PMI untuk diisi manual
    create_pmi_placeholder()

    # 5. Simpan
    log.info("\nMenyimpan data makroekonomi...")
    saved_paths = save_macro_data(df_monthly, df_quarterly, df_worldbank)

    log.info("\n=== RINGKASAN MAKROEKONOMI ===")
    for name, df in [
        ("Monthly (FRED)", df_monthly),
        ("Quarterly (FRED)", df_quarterly),
        ("Annual (World Bank)", df_worldbank),
    ]:
        if not df.empty:
            cols = [c for c in df.columns if c != "date"]
            log.info(f"  {name}: {len(df)} baris, {len(cols)} variabel")
        else:
            log.info(f"  {name}: tidak ada data (cek API key)")

    return {
        "macro_monthly":    df_monthly,
        "macro_quarterly":  df_quarterly,
        "macro_worldbank":  df_worldbank,
        "paths":            saved_paths,
    }


if __name__ == "__main__":
    results = run()