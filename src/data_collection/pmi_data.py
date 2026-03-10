# ============================================================
# src/data_collection/05_pmi_data.py
#
# FUNGSI : Mengambil data PMI Manufaktur dari berbagai sumber:
#
#   SUMBER GRATIS (dicoba secara otomatis):
#   1. FRED API   — ISM Manufacturing Index USA (resmi, akurat)
#   2. OECD API   — PMI beberapa negara (gratis, JSON)
#   3. stlouisfed — CLI Manufacturing Index sebagai proxy
#
#   SUMBER MANUAL (jika otomatis gagal):
#   4. Baca file CSV yang sudah kamu download manual
#      dari Investing.com atau sumber lain
#
# OUTPUT : data/raw/pmi_data.csv
#          (menggantikan pmi_template_manual.csv yang kosong)
# ============================================================

import sys
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import RAW_DATA_DIR, DATA_START_DATE, FRED_API_KEY

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_END_DATE = date.today().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────
# 1. ISM MANUFACTURING PMI USA — via FRED (Gratis, Akurat)
# ─────────────────────────────────────────────────────────────
# FRED Series: MFNDBISMFGBM
# = ISM Manufacturing: PMI Composite Index
# Nilai > 50 = ekspansi, < 50 = kontraksi

def fetch_pmi_usa_fred(start: str, end: str) -> pd.Series:
    """
    Ambil ISM Manufacturing PMI USA dari FRED.

    Series yang dicoba (berurutan):
    - NAPM    : ISM Manufacturing PMI (series resmi, tersedia panjang)
    - MANEMP  : Manufacturing Employment (proxy jika PMI tidak ada)
    """
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY tidak diset, skip PMI USA dari FRED.")
        return pd.Series(dtype=float)

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        # Coba beberapa series ID ISM PMI USA
        candidates = [
            ("NAPM",           "ISM Manufacturing PMI (NAPM)"),
            ("MFNDBISMFGBM",   "ISM PMI Composite"),
            ("AMFNBISMFGBM",   "ISM PMI alternatif"),
        ]

        for series_id, label in candidates:
            try:
                log.info(f"Mengambil PMI USA dari FRED ({series_id} — {label})...")
                series = fred.get_series(
                    series_id,
                    observation_start=start,
                    observation_end=end,
                )
                if not series.empty:
                    series.name = "pmi_manufacturing_usa"
                    log.info(f"  ✓ PMI USA ({series_id}): {len(series)} bulan | "
                             f"{series.index.min().date()} s/d {series.index.max().date()}")
                    log.info(f"  Range: {series.min():.1f} – {series.max():.1f}")
                    return series
            except Exception as e:
                log.debug(f"  {series_id} gagal: {e}")
                continue

        log.warning("  Semua series PMI USA di FRED gagal.")
        return pd.Series(dtype=float)

    except Exception as e:
        log.warning(f"  FRED PMI USA gagal: {e}")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────
# 2. PMI CHINA (NBS Official) — via FRED
# ─────────────────────────────────────────────────────────────
# Series: CHNPMICNMFGM
# = China NBS Manufacturing PMI (resmi dari National Bureau of Statistics)

def fetch_pmi_china_fred(start: str, end: str) -> pd.Series:
    """
    Ambil PMI Manufaktur China (NBS) dari FRED.
    Series: CHNGBMFMIDXM

    Catatan: Ini PMI versi NBS (pemerintah China), bukan Caixin.
    NBS lebih fokus ke perusahaan besar, Caixin ke UKM.
    Keduanya berguna dan saling melengkapi.
    """
    if not FRED_API_KEY:
        return pd.Series(dtype=float)

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        # Coba beberapa series ID China PMI di FRED
        series_ids = [
            "CHNGBMFMIDXM",   # China NBS Manufacturing PMI
            "CHNPMICNMFGM",   # alternatif
        ]

        for sid in series_ids:
            try:
                log.info(f"Mengambil PMI China dari FRED ({sid})...")
                series = fred.get_series(
                    sid,
                    observation_start=start,
                    observation_end=end,
                )
                if not series.empty:
                    series.name = "pmi_manufacturing_china"
                    log.info(f"  ✓ PMI China: {len(series)} bulan | "
                             f"{series.index.min().date()} s/d {series.index.max().date()}")
                    return series
            except Exception:
                continue

        log.warning("  PMI China tidak ditemukan di FRED. Perlu sumber alternatif.")
        return pd.Series(dtype=float)

    except Exception as e:
        log.warning(f"  FRED PMI China gagal: {e}")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────
# 3. PMI GLOBAL & EU — via OECD API (Gratis, Tanpa API Key)
# ─────────────────────────────────────────────────────────────

def fetch_pmi_oecd(country_code: str, col_name: str,
                   start: str, end: str) -> pd.Series:
    """
    Ambil CLI (Composite Leading Indicator) dari OECD API.
    Mencoba beberapa format URL karena OECD sering update endpoint-nya.
    """
    start_period = start[:7]   # "2017-01"
    end_period   = end[:7]     # "2024-12"

    # OECD punya beberapa versi endpoint — coba satu per satu
    urls_to_try = [
        # Format baru (2024+)
        (
            f"https://sdmx.oecd.org/public/rest/data/"
            f"OECD.SDD.STES,DSD_STES@DF_CLI,4.0/"
            f"{country_code}.M.LI.AA.STSA"
            f"?startPeriod={start_period}&endPeriod={end_period}"
            f"&format=jsondata&detail=dataonly"
        ),
        # Format alternatif dengan dimensi berbeda
        (
            f"https://sdmx.oecd.org/public/rest/data/"
            f"OECD.SDD.STES,DSD_STES@DF_CLI,4.0/"
            f"{country_code}.M.LI.AA.AA"
            f"?startPeriod={start_period}&endPeriod={end_period}"
            f"&format=jsondata"
        ),
        # Format lama (sebelum 2024)
        (
            f"https://stats.oecd.org/SDMX-JSON/data/MEI_CLI/"
            f"LOLITOAA.{country_code}.M"
            f"/all?startTime={start_period}&endTime={end_period}"
            f"&contentType=json"
        ),
    ]

    for url in urls_to_try:
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                continue

            data = resp.json()

            # Coba parse format SDMX-JSON baru
            try:
                series_data = data["data"]["dataSets"][0]["series"]
                times = data["data"]["structure"]["dimensions"]["observation"][0]["values"]
                first_key = list(series_data.keys())[0]
                obs = series_data[first_key]["observations"]

                records = []
                for idx_str, values in obs.items():
                    idx = int(idx_str)
                    period = times[idx]["id"]
                    value  = values[0]
                    if value is not None:
                        records.append({
                            "date":   pd.to_datetime(period + "-01"),
                            col_name: float(value),
                        })

                if records:
                    df = pd.DataFrame(records).set_index("date")[col_name]
                    log.info(f"  ✓ OECD CLI {country_code}: {len(df)} bulan")
                    return df

            except (KeyError, IndexError):
                # Coba format lama
                try:
                    ds = data["dataSets"][0]["series"]
                    time_periods = data["structure"]["dimensions"]["observation"][0]["values"]
                    key = list(ds.keys())[0]
                    observations = ds[key]["observations"]

                    records = []
                    for idx_str, val_list in observations.items():
                        idx    = int(idx_str)
                        period = time_periods[idx]["id"]
                        value  = val_list[0]
                        if value is not None:
                            records.append({
                                "date":   pd.to_datetime(period + "-01"),
                                col_name: float(value),
                            })

                    if records:
                        df = pd.DataFrame(records).set_index("date")[col_name]
                        log.info(f"  ✓ OECD CLI {country_code} (format lama): {len(df)} bulan")
                        return df
                except Exception:
                    continue

        except Exception as e:
            log.debug(f"  URL gagal untuk {country_code}: {e}")
            continue

    log.warning(f"  OECD CLI {country_code}: semua endpoint gagal.")
    return pd.Series(dtype=float)


def fetch_pmi_from_dataportal(start: str, end: str) -> dict:
    """
    Ambil CLI dari OECD Data Portal API (lebih stabil dari SDMX).
    Endpoint: https://data.oecd.org/api/sdmx-json/

    Mengembalikan dict: {col_name: pd.Series}
    """
    log.info("Mencoba OECD Data Portal API (alternatif)...")

    targets = {
        "cli_usa":    "USA",
        "cli_china":  "CHN",
        "cli_eu":     "EU28",
        "cli_global": "OECD",
    }

    results = {}
    base = "https://data.oecd.org/api/sdmx-json/data/MEI_CLI/LOLITOAA.{loc}.M/all"
    params = {
        "startTime":   start[:7],
        "endTime":     end[:7],
        "contentType": "json",
    }

    for col_name, loc in targets.items():
        try:
            url  = base.format(loc=loc)
            resp = requests.get(url, params=params, timeout=20)

            if resp.status_code != 200:
                log.debug(f"  Data portal {loc}: HTTP {resp.status_code}")
                continue

            data = resp.json()
            structure  = data["structure"]
            time_vals  = structure["dimensions"]["observation"][0]["values"]
            series_map = data["dataSets"][0]["series"]

            key = list(series_map.keys())[0]
            obs = series_map[key]["observations"]

            records = []
            for idx_str, val_list in obs.items():
                idx    = int(idx_str)
                period = time_vals[idx]["id"]
                value  = val_list[0]
                if value is not None:
                    records.append({
                        "date":   pd.to_datetime(period + "-01"),
                        col_name: float(value),
                    })

            if records:
                s = pd.DataFrame(records).set_index("date")[col_name]
                results[col_name] = s
                log.info(f"  ✓ OECD DataPortal {loc}: {len(s)} bulan")

            time.sleep(0.3)

        except Exception as e:
            log.debug(f"  Data portal {loc} error: {e}")

    return results




def load_pmi_manual_files() -> dict:
    """
    Baca file CSV PMI yang sudah di-download manual dari Investing.com.

    Format yang diterima (format standar Investing.com):
    - Kolom: "Date", "Price" (nilai PMI), "Open", "High", "Low", "Change %"
    - Atau: "Release Date", "Actual", "Forecast", "Previous"

    Letakkan file di data/raw/ dengan nama:
    - pmi_usa_manual.csv
    - pmi_china_manual.csv
    - pmi_eu_manual.csv
    - pmi_global_manual.csv
    """
    manual_files = {
        "pmi_manufacturing_usa":    RAW_DATA_DIR / "pmi_usa_manual.csv",
        "pmi_manufacturing_china":  RAW_DATA_DIR / "pmi_china_manual.csv",
        "pmi_manufacturing_eu":     RAW_DATA_DIR / "pmi_eu_manual.csv",
        "pmi_manufacturing_global": RAW_DATA_DIR / "pmi_global_manual.csv",
    }

    loaded = {}

    for col_name, filepath in manual_files.items():
        if not filepath.exists():
            continue

        try:
            df = pd.read_csv(filepath)
            df.columns = [c.strip() for c in df.columns]

            # --- Format 1: Investing.com download ---
            # Kolom: "Date", "Price"
            if "Date" in df.columns and "Price" in df.columns:
                df["date"]    = pd.to_datetime(df["Date"], errors="coerce")
                df[col_name]  = pd.to_numeric(
                    df["Price"].astype(str).str.replace(",", ""),
                    errors="coerce"
                )

            # --- Format 2: Economic Calendar Investing.com ---
            # Kolom: "Release Date", "Actual"
            elif "Release Date" in df.columns and "Actual" in df.columns:
                df["date"]   = pd.to_datetime(df["Release Date"], errors="coerce")
                df[col_name] = pd.to_numeric(
                    df["Actual"].astype(str).str.replace(",", ""),
                    errors="coerce"
                )

            # --- Format 3: Sudah benar (kolom date + nilai) ---
            elif "date" in df.columns:
                df["date"]   = pd.to_datetime(df["date"], errors="coerce")
                numeric_cols = [c for c in df.columns
                                if c != "date" and df[c].dtype in ["float64", "int64"]]
                if numeric_cols:
                    df[col_name] = pd.to_numeric(df[numeric_cols[0]], errors="coerce")
            else:
                log.warning(f"  Format tidak dikenal untuk {filepath.name}")
                continue

            series = df.dropna(subset=["date", col_name]).set_index("date")[col_name]
            series.index = pd.DatetimeIndex(series.index)
            series = series.sort_index()

            loaded[col_name] = series
            log.info(f"  ✓ Manual {col_name}: {len(series)} bulan dari {filepath.name}")

        except Exception as e:
            log.warning(f"  Gagal baca {filepath.name}: {e}")

    return loaded


# ─────────────────────────────────────────────────────────────
# 5. GABUNGKAN DAN SIMPAN
# ─────────────────────────────────────────────────────────────

def build_pmi_dataframe(
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
) -> pd.DataFrame:
    """
    Kumpulkan semua PMI dari semua sumber dan buat satu DataFrame bulanan.

    Prioritas per variabel:
    - PMI USA    : FRED (ISM) → Manual file → OECD CLI sebagai proxy
    - PMI China  : FRED (NBS) → Manual file → OECD CLI sebagai proxy
    - PMI EU     : Manual file → OECD CLI sebagai proxy
    - PMI Global : Manual file → OECD CLI sebagai proxy
    """
    log.info("=== Membangun Dataset PMI Manufaktur ===")

    # Buat date range bulanan sebagai backbone
    date_range = pd.date_range(start=start, end=end, freq="MS")
    df = pd.DataFrame({"date": date_range})
    df = df.set_index("date")

    # ── Ambil dari semua sumber ──
    pmi_usa_fred    = fetch_pmi_usa_fred(start, end)
    pmi_china_fred  = fetch_pmi_china_fred(start, end)
    manual_data     = load_pmi_manual_files()

    # OECD CLI: coba SDMX endpoint dulu, lalu Data Portal sebagai backup
    log.info("Mengambil OECD CLI...")
    cli_sdmx = {
        "cli_usa":    fetch_pmi_oecd("USA",       "cli_usa",    start, end),
        "cli_china":  fetch_pmi_oecd("CHN",       "cli_china",  start, end),
        "cli_eu":     fetch_pmi_oecd("EU27_2020", "cli_eu",     start, end),
        "cli_global": fetch_pmi_oecd("OECD",      "cli_global", start, end),
    }

    # Cek apakah SDMX berhasil, kalau tidak coba Data Portal
    sdmx_success = any(not s.empty for s in cli_sdmx.values())
    if not sdmx_success:
        log.info("SDMX gagal semua, mencoba OECD Data Portal...")
        cli_portal = fetch_pmi_from_dataportal(start, end)
    else:
        cli_portal = {}

    # Gabungkan CLI dari kedua sumber
    def get_cli(key):
        s = cli_sdmx.get(key, pd.Series(dtype=float))
        if s.empty:
            s = cli_portal.get(key, pd.Series(dtype=float))
        return s

    cli_usa    = get_cli("cli_usa")
    cli_china  = get_cli("cli_china")
    cli_eu     = get_cli("cli_eu")
    cli_global = get_cli("cli_global")

    # ── Isi setiap kolom PMI dengan prioritas ──

    # PMI USA: FRED > Manual > OECD CLI
    if not pmi_usa_fred.empty:
        df["pmi_manufacturing_usa"] = pmi_usa_fred.reindex(df.index)
        log.info("  PMI USA → sumber: FRED (ISM)")
    elif "pmi_manufacturing_usa" in manual_data:
        df["pmi_manufacturing_usa"] = manual_data["pmi_manufacturing_usa"].reindex(df.index)
        log.info("  PMI USA → sumber: manual file")
    elif not cli_usa.empty:
        df["pmi_manufacturing_usa"] = cli_usa.reindex(df.index)
        log.info("  PMI USA → sumber: OECD CLI (proxy, bukan PMI asli)")
    else:
        df["pmi_manufacturing_usa"] = None
        log.warning("  PMI USA → tidak ada data!")

    # PMI China: FRED > Manual > OECD CLI
    if not pmi_china_fred.empty:
        df["pmi_manufacturing_china"] = pmi_china_fred.reindex(df.index)
        log.info("  PMI China → sumber: FRED (NBS)")
    elif "pmi_manufacturing_china" in manual_data:
        df["pmi_manufacturing_china"] = manual_data["pmi_manufacturing_china"].reindex(df.index)
        log.info("  PMI China → sumber: manual file")
    elif not cli_china.empty:
        df["pmi_manufacturing_china"] = cli_china.reindex(df.index)
        log.info("  PMI China → sumber: OECD CLI (proxy)")
    else:
        df["pmi_manufacturing_china"] = None
        log.warning("  PMI China → tidak ada data!")

    # PMI EU: Manual > OECD CLI
    if "pmi_manufacturing_eu" in manual_data:
        df["pmi_manufacturing_eu"] = manual_data["pmi_manufacturing_eu"].reindex(df.index)
        log.info("  PMI EU → sumber: manual file")
    elif not cli_eu.empty:
        df["pmi_manufacturing_eu"] = cli_eu.reindex(df.index)
        log.info("  PMI EU → sumber: OECD CLI (proxy)")
    else:
        df["pmi_manufacturing_eu"] = None
        log.warning("  PMI EU → tidak ada data! Download manual dari investing.com")

    # PMI Global: Manual > OECD CLI
    if "pmi_manufacturing_global" in manual_data:
        df["pmi_manufacturing_global"] = manual_data["pmi_manufacturing_global"].reindex(df.index)
        log.info("  PMI Global → sumber: manual file")
    elif not cli_global.empty:
        df["pmi_manufacturing_global"] = cli_global.reindex(df.index)
        log.info("  PMI Global → sumber: OECD CLI (proxy)")
    else:
        df["pmi_manufacturing_global"] = None
        log.warning("  PMI Global → tidak ada data! Download manual dari investing.com")

    df = df.reset_index()

    # ── Ringkasan kelengkapan ──
    log.info("\n=== Kelengkapan Data PMI ===")
    pmi_cols = [c for c in df.columns if c.startswith("pmi_") or c.startswith("cli_")]
    for col in pmi_cols:
        if col in df.columns:
            filled  = df[col].notna().sum()
            total   = len(df)
            pct     = filled / total * 100
            status  = "✓" if pct > 90 else ("⚠" if pct > 50 else "✗")
            log.info(f"  [{status}] {col}: {filled}/{total} bulan ({pct:.0f}%)")

    return df


def run(start: str = DATA_START_DATE, end: str = DATA_END_DATE):
    """Entry point utama."""
    log.info("=" * 60)
    log.info("PENGUMPULAN DATA PMI MANUFAKTUR")
    log.info("=" * 60)

    df = build_pmi_dataframe(start=start, end=end)

    if not df.empty:
        # Simpan — menggantikan pmi_template_manual.csv yang kosong
        path = RAW_DATA_DIR / "pmi_data.csv"
        df.to_csv(path, index=False)
        log.info(f"\n✓ PMI data disimpan: {path}")
        log.info(f"  {len(df)} bulan | kolom: {list(df.columns)}")

        # Hapus template lama yang kosong jika ada
        old_template = RAW_DATA_DIR / "pmi_template_manual.csv"
        if old_template.exists():
            old_template.unlink()
            log.info(f"  Template lama dihapus: {old_template.name}")

    return df


if __name__ == "__main__":
    df = run()

    # Tampilkan 5 baris pertama dan terakhir
    if not df.empty:
        print("\n--- 5 Baris Pertama ---")
        print(df.head().to_string())
        print("\n--- 5 Baris Terakhir ---")
        print(df.tail().to_string())