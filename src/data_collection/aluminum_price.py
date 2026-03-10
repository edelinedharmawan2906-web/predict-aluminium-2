# ============================================================
# src/data_collection/aluminum_price.py
#
# FUNGSI : Mengambil data harga aluminium harian dari:
#          1. Yahoo Finance (JJUA ETN)  — sumber utama, GRATIS
#          2. Investing.com (scraping)  — fallback jika Yahoo gagal
#          3. FRED (PALUMUSDM)          — bulanan, sebagai validasi
#
# CATATAN: ALI=F (COMEX Aluminum Futures) sudah DELISTED sejak 2023.
#          Gunakan JJUA (iPath Bloomberg Aluminum ETN) sebagai pengganti.
#          Harga JJUA mengikuti indeks aluminium LME secara ketat.
#
# OUTPUT : data/raw/aluminum_price_daily.csv
#          data/raw/aluminum_price_monthly_fred.csv
# ============================================================

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# Tambahkan root proyek ke path agar bisa import config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    RAW_DATA_DIR, DATA_START_DATE, DATA_END_DATE,
    FRED_API_KEY, ALUMINUM_TICKERS, FRED_SERIES
)

# ── Logger sederhana ──────────────────────────────────────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. YAHOO FINANCE — Data Harian
# ─────────────────────────────────────────────────────────────

def fetch_aluminum_investing_com(
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
) -> pd.DataFrame:
    """
    Ambil harga aluminium historis dari Investing.com menggunakan
    library investpy atau requests + BeautifulSoup sebagai fallback.

    Investing.com menyimpan data harga LME Aluminum Futures historis
    yang cukup lengkap dan gratis diakses.

    CATATAN: Scraping bisa sewaktu-waktu break jika website berubah.
    Selalu validasi hasil dengan sumber lain.

    Returns
    -------
    pd.DataFrame dengan kolom: date, open, high, low, close, volume
    """
    log.info("Mencoba mengambil data dari Investing.com (scraping)...")

    try:
        import investpy
        # investpy menggunakan Investing.com sebagai backend
        df = investpy.commodities.get_commodity_historical_data(
            commodity="aluminum",
            from_date=pd.Timestamp(start).strftime("%d/%m/%Y"),
            to_date=pd.Timestamp(end).strftime("%d/%m/%Y"),
        )
        if not df.empty:
            df = df.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"date": "date", "price": "close"})
            df["date"]   = pd.to_datetime(df["date"])
            df["ticker"] = "INVESTING_LME_AL"
            df["source"] = "investing.com"
            log.info(f"  → investpy: {len(df)} baris")
            return df
    except ImportError:
        log.info("  investpy tidak terinstall, skip.")
    except Exception as e:
        log.warning(f"  investpy gagal: {e}")

    # Fallback: beri instruksi manual download
    log.warning("  Investing.com scraping gagal.")
    log.info("  → SOLUSI MANUAL: Download CSV dari:")
    log.info("    https://www.investing.com/commodities/aluminum-historical-data")
    log.info("    Simpan sebagai: data/raw/aluminum_investing_manual.csv")
    log.info("    Kolom yang dibutuhkan: Date, Price, Open, High, Low, Vol., Change%")

    # Cek apakah file manual sudah ada
    manual_path = RAW_DATA_DIR / "aluminum_investing_manual.csv"
    if manual_path.exists():
        log.info(f"  ✓ File manual ditemukan: {manual_path}")
        try:
            df = pd.read_csv(manual_path)
            # Investing.com menggunakan format: "Date","Price","Open","High","Low","Vol.","Change %"
            col_map = {
                "Date":     "date",
                "Price":    "close",
                "Open":     "open",
                "High":     "high",
                "Low":      "low",
                "Vol.":     "volume",
            }
            df = df.rename(columns=col_map)
            df["date"]   = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
            df["close"]  = pd.to_numeric(df["close"].astype(str).str.replace(",", ""), errors="coerce")
            df["open"]   = pd.to_numeric(df["open"].astype(str).str.replace(",", ""),  errors="coerce")
            df["high"]   = pd.to_numeric(df["high"].astype(str).str.replace(",", ""),  errors="coerce")
            df["low"]    = pd.to_numeric(df["low"].astype(str).str.replace(",", ""),   errors="coerce")
            df           = df.dropna(subset=["date", "close"])
            df           = df.sort_values("date").reset_index(drop=True)
            df["ticker"] = "LME_AL_INVESTING"
            df["source"] = "investing.com_manual"
            log.info(f"  ✓ Manual file loaded: {len(df)} baris")
            return df
        except Exception as e:
            log.error(f"  Gagal membaca file manual: {e}")

    return pd.DataFrame()


def fetch_aluminum_yahoo(
    ticker: str = "JJUA",
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Ambil harga aluminium harian dari Yahoo Finance.

    Parameters
    ----------
    ticker  : str  — Ticker symbol, default 'JJUA' (Bloomberg Aluminum ETN)
    start   : str  — Tanggal mulai format 'YYYY-MM-DD'
    end     : str  — Tanggal akhir format 'YYYY-MM-DD'
    retries : int  — Jumlah retry jika gagal

    Returns
    -------
    pd.DataFrame dengan kolom: Date, Open, High, Low, Close, Volume, Ticker
    """
    log.info(f"Mengambil data harga aluminium dari Yahoo Finance | ticker={ticker}")

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,   # Adjust untuk split & dividen
                progress=False,
            )

            if df.empty:
                log.warning(f"Data kosong untuk ticker {ticker} (attempt {attempt})")
                time.sleep(2 * attempt)
                continue

            # Flatten MultiIndex kolom jika ada
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Reset index agar Date jadi kolom biasa
            df = df.reset_index()
            df.columns.name = None

            # Rename kolom ke format standar proyek
            df = df.rename(columns={
                "Date":   "date",
                "Open":   "open",
                "High":   "high",
                "Low":    "low",
                "Close":  "close",
                "Volume": "volume",
            })

            # Tambahkan kolom metadata
            df["ticker"] = ticker
            df["source"] = "yahoo_finance"

            # Pastikan tipe data benar
            df["date"]   = pd.to_datetime(df["date"])
            df["close"]  = pd.to_numeric(df["close"],  errors="coerce")
            df["open"]   = pd.to_numeric(df["open"],   errors="coerce")
            df["high"]   = pd.to_numeric(df["high"],   errors="coerce")
            df["low"]    = pd.to_numeric(df["low"],    errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

            # Filter hanya hari kerja (hapus baris close = NaN)
            before = len(df)
            df = df.dropna(subset=["close"])
            dropped = before - len(df)
            if dropped > 0:
                log.info(f"  → Removed {dropped} baris dengan close = NaN")

            # Urutkan berdasarkan tanggal
            df = df.sort_values("date").reset_index(drop=True)

            log.info(
                f"  → Berhasil: {len(df)} baris | "
                f"{df['date'].min().date()} s/d {df['date'].max().date()}"
            )
            return df

        except Exception as e:
            log.error(f"  Attempt {attempt} gagal: {e}")
            if attempt < retries:
                time.sleep(3 * attempt)

    log.error(f"Semua {retries} attempt gagal untuk ticker {ticker}")
    return pd.DataFrame()


def fetch_aluminum_with_fallback(
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
) -> pd.DataFrame:
    """
    Coba ambil data harga aluminium dengan urutan prioritas:
    1. Yahoo Finance (JJUA ETN) — mendekati harga LME
    2. Yahoo Finance (AA/Alcoa) — proxy saham aluminium
    3. Investing.com scraping   — harga LME langsung
    4. Informasikan manual download jika semua gagal
    """
    primary = ALUMINUM_TICKERS.get("primary", "JJUA")
    backup  = ALUMINUM_TICKERS.get("backup_1", "AA")

    log.info("=== Pengambilan Data Harga Aluminium (dengan fallback) ===")

    # 1. Coba primary (JJUA)
    df = fetch_aluminum_yahoo(ticker=primary, start=start, end=end)

    # 2. Coba backup Yahoo (AA - Alcoa)
    if df.empty:
        log.warning(f"Primary ticker {primary} gagal. Mencoba backup Yahoo: {backup}")
        df = fetch_aluminum_yahoo(ticker=backup, start=start, end=end)
        if not df.empty:
            log.warning(
                f"  ⚠ Menggunakan harga saham Alcoa (AA) sebagai PROXY aluminium.\n"
                f"  ⚠ Ini bukan harga aluminium langsung — gunakan untuk development saja.\n"
                f"  ⚠ Untuk produksi, gunakan data LME dari Investing.com."
            )

    # 3. Coba Investing.com
    if df.empty:
        log.warning("Yahoo Finance gagal semua. Mencoba Investing.com...")
        df = fetch_aluminum_investing_com(start=start, end=end)

    if df.empty:
        log.error(
            "SEMUA sumber harga aluminium harian gagal!\n"
            "→ Gunakan data FRED bulanan (PALUMUSDM) yang sudah berhasil\n"
            "  sebagai sumber sementara, lalu isi harian dari Investing.com manual.\n"
            "→ Download dari: https://www.investing.com/commodities/aluminum-historical-data\n"
            "→ Simpan sebagai: data/raw/aluminum_investing_manual.csv"
        )

    return df



# ─────────────────────────────────────────────────────────────
# 2. FRED — Data Bulanan (Validasi & Supplement)
# ─────────────────────────────────────────────────────────────

def fetch_aluminum_fred(
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
) -> pd.DataFrame:
    """
    Ambil harga aluminium bulanan dari FRED (World Bank commodity price).
    Seri: PALUMUSDM — Aluminum, 99.5% pure, LME spot price, USD per metric ton.

    Digunakan sebagai:
    - Cross-validasi terhadap data harian Yahoo Finance
    - Supplement untuk analisis tren jangka panjang

    Returns
    -------
    pd.DataFrame dengan kolom: date, price_usd_per_ton, source
    """
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY tidak diset. Skip pengambilan data FRED.")
        return pd.DataFrame()

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        log.info("Mengambil data harga aluminium bulanan dari FRED (PALUMUSDM)...")

        series = fred.get_series(
            FRED_SERIES["aluminum_price_monthly"],
            observation_start=start,
            observation_end=end,
        )

        df = series.reset_index()
        df.columns = ["date", "price_usd_per_ton"]
        df["date"]             = pd.to_datetime(df["date"])
        df["price_usd_per_ton"] = pd.to_numeric(df["price_usd_per_ton"], errors="coerce")
        df["source"]           = "fred_monthly"

        # Hapus NaN
        df = df.dropna(subset=["price_usd_per_ton"])
        df = df.sort_values("date").reset_index(drop=True)

        log.info(
            f"  → Berhasil: {len(df)} bulan | "
            f"{df['date'].min().date()} s/d {df['date'].max().date()}"
        )
        log.info(
            f"  → Range harga: ${df['price_usd_per_ton'].min():,.0f} – "
            f"${df['price_usd_per_ton'].max():,.0f} per ton"
        )
        return df

    except ImportError:
        log.error("fredapi belum terinstall. Jalankan: pip install fredapi")
        return pd.DataFrame()
    except Exception as e:
        log.error(f"Gagal mengambil data FRED: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# 3. VALIDASI CROSS-SOURCE
# ─────────────────────────────────────────────────────────────

def validate_cross_source(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
) -> None:
    """
    Bandingkan rata-rata harga bulanan dari Yahoo Finance
    vs data bulanan FRED untuk deteksi inkonsistensi data.
    """
    if df_daily.empty or df_monthly.empty:
        log.warning("Salah satu DataFrame kosong, skip validasi cross-source.")
        return

    log.info("=== Validasi Cross-Source: Yahoo Finance vs FRED ===")

    # Resample data harian ke bulanan (rata-rata)
    df_yahoo_monthly = (
        df_daily
        .set_index("date")["close"]
        .resample("MS")   # Month Start
        .mean()
        .reset_index()
        .rename(columns={"close": "yahoo_avg_close", "date": "month"})
    )

    # Merge dengan data FRED
    df_fred = df_monthly.copy()
    df_fred["month"] = df_fred["date"].dt.to_period("M").dt.to_timestamp()

    merged = pd.merge(
        df_yahoo_monthly,
        df_fred[["month", "price_usd_per_ton"]].rename(
            columns={"price_usd_per_ton": "fred_price"}
        ),
        on="month",
        how="inner",
    )

    if merged.empty:
        log.warning("Tidak ada bulan yang overlap antara Yahoo Finance dan FRED.")
        return

    # Hitung perbedaan relatif
    merged["diff_pct"] = (
        (merged["yahoo_avg_close"] - merged["fred_price"]).abs()
        / merged["fred_price"] * 100
    )

    avg_diff = merged["diff_pct"].mean()
    max_diff = merged["diff_pct"].max()
    max_month = merged.loc[merged["diff_pct"].idxmax(), "month"]

    log.info(f"  Jumlah bulan overlap : {len(merged)}")
    log.info(f"  Rata-rata perbedaan  : {avg_diff:.2f}%")
    log.info(f"  Perbedaan terbesar   : {max_diff:.2f}% pada {max_month.strftime('%Y-%m')}")

    if avg_diff > 10:
        log.warning(
            f"  ⚠ Rata-rata perbedaan {avg_diff:.1f}% cukup besar. "
            "Perlu dicek apakah unit harga sama (Yahoo: cents/lb vs FRED: USD/ton)."
        )
    else:
        log.info(f"  ✓ Data konsisten antara dua sumber (diff avg < 10%)")


# ─────────────────────────────────────────────────────────────
# 4. SIMPAN DATA
# ─────────────────────────────────────────────────────────────

def save_aluminum_data(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
) -> dict:
    """
    Simpan DataFrame ke CSV di direktori data/raw.

    Returns
    -------
    dict : path file yang disimpan
    """
    saved = {}

    if not df_daily.empty:
        path_daily = RAW_DATA_DIR / "aluminum_price_daily.csv"
        df_daily.to_csv(path_daily, index=False)
        log.info(f"  ✓ Disimpan: {path_daily}")
        saved["daily"] = path_daily

    if not df_monthly.empty:
        path_monthly = RAW_DATA_DIR / "aluminum_price_monthly_fred.csv"
        df_monthly.to_csv(path_monthly, index=False)
        log.info(f"  ✓ Disimpan: {path_monthly}")
        saved["monthly"] = path_monthly

    return saved


# ─────────────────────────────────────────────────────────────
# 5. MAIN — Jalankan semua langkah
# ─────────────────────────────────────────────────────────────

def run(start: str = DATA_START_DATE, end: str = DATA_END_DATE):
    """
    Entry point utama untuk pengambilan data harga aluminium.
    Dipanggil oleh pipeline.py atau bisa dijalankan standalone.
    """
    log.info("=" * 60)
    log.info("FASE 1 — Pengumpulan Data: Harga Aluminium")
    log.info("=" * 60)

    # 1. Ambil data harian (Yahoo Finance)
    df_daily = fetch_aluminum_with_fallback(start=start, end=end)

    # 2. Ambil data bulanan (FRED)
    df_monthly = fetch_aluminum_fred(start=start, end=end)

    # 3. Validasi cross-source
    validate_cross_source(df_daily, df_monthly)

    # 4. Simpan ke CSV
    log.info("Menyimpan data...")
    saved_paths = save_aluminum_data(df_daily, df_monthly)

    # 5. Tampilkan ringkasan
    log.info("\n=== RINGKASAN ===")
    if not df_daily.empty:
        log.info(f"  Data harian   : {len(df_daily):,} baris")
        log.info(f"  Periode       : {df_daily['date'].min().date()} s/d {df_daily['date'].max().date()}")
        log.info(f"  Harga min/max : ${df_daily['close'].min():,.2f} / ${df_daily['close'].max():,.2f}")
        log.info(f"  Missing values: {df_daily['close'].isna().sum()}")

    return {
        "daily":   df_daily,
        "monthly": df_monthly,
        "paths":   saved_paths,
    }


if __name__ == "__main__":
    results = run()