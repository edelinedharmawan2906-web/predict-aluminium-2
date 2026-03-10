# ============================================================
# config.py — Konfigurasi Global Proyek
# ============================================================

import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import date

# Load environment variables dari file .env
load_dotenv()

# ── Direktori Proyek ──────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
RAW_DATA_DIR    = BASE_DIR / os.getenv("RAW_DATA_PATH", "data/raw")
PROCESSED_DIR   = BASE_DIR / os.getenv("PROCESSED_DATA_PATH", "data/processed")
LOG_DIR         = BASE_DIR / os.getenv("LOG_PATH", "logs")

# Buat direktori jika belum ada
for d in [RAW_DATA_DIR, PROCESSED_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")
NEWSAPI_KEY       = os.getenv("NEWSAPI_KEY", "")
ALPHAVANTAGE_KEY  = os.getenv("ALPHAVANTAGE_KEY", "")

# ── Periode Data ──────────────────────────────────────────────
# START: 2017 untuk menangkap siklus komoditas 7+ tahun
# END  : Selalu hari ini secara otomatis — tidak perlu update manual
#        Override dengan DATA_END_DATE di .env jika butuh tanggal spesifik
DATA_START_DATE = os.getenv("DATA_START_DATE", "2017-01-01")
DATA_END_DATE   = os.getenv("DATA_END_DATE",   date.today().strftime("%Y-%m-%d"))

# ── Konfigurasi Sumber Data ───────────────────────────────────

# Ticker harga aluminium di Yahoo Finance
# CATATAN: ALI=F (COMEX Aluminum Futures) sudah delisted sejak 2023.
# Alternatif yang masih aktif:
# - JJUA  : iPath Bloomberg Aluminum ETN — paling mendekati harga aluminium LME
# - AA    : Alcoa Corp — proxy, bukan harga aluminium langsung
# Untuk harga LME akurat, gunakan sumber FRED (PALUMUSDM) yang sudah berhasil.
ALUMINUM_TICKERS = {
    "primary":  "JJUA",     # iPath Bloomberg Aluminum Subindex ETN
    "backup_1": "AA",       # Alcoa Corp saham — proxy harga aluminium
}

# FRED Series IDs yang akan diambil
# Referensi: https://fred.stlouisfed.org
FRED_SERIES = {
    # Harga Komoditas
    "aluminum_price_monthly": "PALUMUSDM",     # Harga aluminium (USD/metric ton, bulanan)

    # Nilai Tukar
    "usd_cny":  "DEXCHUS",   # USD/CNY harian
    "usd_eur":  "DEXUSEU",   # USD/EUR harian
    "usd_jpy":  "DEXJPUS",   # USD/JPY harian

    # Harga Energi
    "natural_gas":  "DHHNGSP",   # Henry Hub Natural Gas (USD/MMBtu, harian)
    "crude_oil_wti":"DCOILWTICO", # WTI Crude Oil (USD/barrel, harian)

    # Makroekonomi AS
    "pmi_manufacturing_us": "MANEMP",    # Manufacturing Employment (proxy PMI)
    "industrial_production": "INDPRO",   # Industrial Production Index (bulanan)
    "us_gdp_growth":        "A191RL1Q225SBEA",  # GDP Growth Rate (kuartalan)

    # Harga Komoditas Lain (cross-commodity)
    "copper_price":  "PCOPPUSDM",   # Harga tembaga (bulanan)
    "coal_price":    "PCOALAUUSDM",  # Harga batubara Australia (bulanan)
}

# Ticker Yahoo Finance untuk variabel tambahan
# CATATAN: Beberapa futures ticker Yahoo berubah format.
# Format terbaru: gunakan ticker tanpa '=F' untuk beberapa komoditas.
YAHOO_TICKERS = {
    # Komoditas Cross-Reference (ticker aktif per 2024)
    "copper":      "HG=F",      # Copper Futures (aktif)
    "gold":        "GC=F",      # Gold Futures (aktif)
    "crude_oil":   "CL=F",      # Crude Oil WTI Futures (aktif)

    # Indeks Pasar
    "sp500":       "^GSPC",     # S&P 500 (aktif)
    "shanghai":    "000001.SS", # Shanghai Composite (aktif)
    "dollar_index":"DX=F",      # US Dollar Index — DX=F lebih stabil dari DX-Y.NYB

    # ETF Aluminium
    "jjua":        "JJUA",      # iPath Bloomberg Aluminum ETN
}

# Keywords untuk NewsAPI (sentimen berita)
NEWS_KEYWORDS = [
    "aluminum price",
    "aluminium price",
    "LME aluminum",
    "aluminum futures",
    "aluminum supply",
    "bauxite",
    "alumina price",
    "China aluminum production",
    "aluminum tariff",
    "aluminum demand",
]

# Validasi konfigurasi
def validate_config():
    """Cek apakah API keys sudah diisi."""
    warnings = []
    if not FRED_API_KEY:
        warnings.append("FRED_API_KEY belum diset — data FRED tidak akan bisa diambil.")
    if not NEWSAPI_KEY:
        warnings.append("NEWSAPI_KEY belum diset — data sentimen berita tidak akan bisa diambil.")
    if warnings:
        print("\n[CONFIG WARNING]")
        for w in warnings:
            print(f"  ⚠ {w}")
        print("  → Salin .env.example ke .env dan isi API keys kamu.\n")
    return len(warnings) == 0

if __name__ == "__main__":
    validate_config()
    print(f"BASE_DIR     : {BASE_DIR}")
    print(f"RAW_DATA_DIR : {RAW_DATA_DIR}")
    print(f"FRED Series  : {list(FRED_SERIES.keys())}")
    print(f"Yahoo Tickers: {list(YAHOO_TICKERS.keys())}")