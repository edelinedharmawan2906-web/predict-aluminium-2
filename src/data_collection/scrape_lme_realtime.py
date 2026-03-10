"""
scrape_lme_realtime.py
======================
Scrape LME Aluminium Inventory dari westmetall.com secara realtime.
Berdasarkan inspeksi HTML: data ada di tag <table><tbody><tr> dalam div.section

Usage:
    pip install requests beautifulsoup4
    python src/data_collection/scrape_lme_realtime.py

Output:
    data/raw/lme_inventory_raw.csv   -- data mentah per baris scraping
    data/raw/lme_inventory_daily.csv -- data daily forward-filled + derived features
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime, date

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR  = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.westmetall.com/en/markdaten.php"
PARAMS_BASE = {
    "action": "table",
    "field":  "LME_Al_stock",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.westmetall.com/en/markdaten.php",
}

MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
}

def parse_date(s: str):
    """Parse '06. March 2026' -> datetime"""
    s = s.strip()
    parts = s.replace(".", "").split()
    if len(parts) != 3:
        return None
    try:
        day   = int(parts[0])
        month = MONTH_MAP.get(parts[1].lower())
        year  = int(parts[2])
        if not month:
            return None
        return datetime(year, month, day)
    except Exception:
        return None

def parse_number(s: str):
    """Parse '1,234,567' or '1.234.567' -> float"""
    if not s or s.strip() == "-":
        return None
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return None

def scrape_year(session: requests.Session, year: int) -> list[dict]:
    """Scrape one year page and return list of {date, lme_inventory} dicts."""
    url = BASE_URL
    params = {**PARAMS_BASE, "year": str(year)}
    
    try:
        resp = session.get(url, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"  Failed to fetch year {year}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find all tables on the page - each year section has its own <table>
    records = []
    tables = soup.find_all("table")
    
    if not tables:
        log.warning(f"  No <table> found for year {year}")
        return []

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            
            date_text = cells[0].get_text(strip=True)
            inv_text  = cells[1].get_text(strip=True)
            
            parsed_date = parse_date(date_text)
            parsed_inv  = parse_number(inv_text)
            
            if parsed_date and parsed_inv and parsed_inv > 50_000:
                # Sanity check: only keep target year
                if parsed_date.year == year:
                    records.append({
                        "date": parsed_date,
                        "lme_inventory": parsed_inv,
                    })

    log.info(f"  Year {year}: {len(records)} records scraped")
    return records


def build_daily(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill to daily and compute derived features."""
    df = (
        df_raw
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )

    # Reindex to daily
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx).rename_axis("date")
    df["lme_inventory"] = df["lme_inventory"].ffill().bfill()

    # Derived features
    df["lme_inventory_change"] = df["lme_inventory"].diff()
    df["lme_inventory_ma5"]    = df["lme_inventory"].rolling(5, min_periods=1).mean()
    df["lme_inventory_ma21"]   = df["lme_inventory"].rolling(21, min_periods=1).mean()
    df["lme_inventory_yoy"]    = df["lme_inventory"].pct_change(252)

    return df.reset_index()


def main():
    start_year = 2017
    end_year   = date.today().year

    log.info(f"Scraping LME Aluminium Inventory {start_year}-{end_year}")
    log.info("=" * 60)

    session = requests.Session()
    all_records = []

    for year in range(start_year, end_year + 1):
        log.info(f"Scraping {year}...")
        records = scrape_year(session, year)
        all_records.extend(records)
        time.sleep(1.5)  # polite delay

    if not all_records:
        log.error("No data collected! Check connection or website structure.")
        return

    df_raw = pd.DataFrame(all_records).drop_duplicates("date").sort_values("date")
    raw_path = RAW_DIR / "lme_inventory_raw.csv"
    df_raw.to_csv(raw_path, index=False)
    log.info(f"\nRaw data saved: {raw_path} ({len(df_raw)} rows)")

    # Build daily
    df_daily = build_daily(df_raw)
    daily_path = RAW_DIR / "lme_inventory_daily.csv"
    df_daily.to_csv(daily_path, index=False)
    log.info(f"Daily data saved: {daily_path} ({len(df_daily)} rows)")

    log.info("\nSummary:")
    log.info(f"  Date range : {df_daily['date'].min()} → {df_daily['date'].max()}")
    log.info(f"  Total rows : {len(df_daily)}")
    log.info(f"  NaN check  : {df_daily['lme_inventory'].isna().sum()} missing")
    log.info(f"\n{df_daily[['lme_inventory','lme_inventory_change','lme_inventory_ma21']].describe().to_string()}")
    log.info("\nDone! Next: run integrate_lme_from_pdf.py to merge into features_daily_v2.csv")


if __name__ == "__main__":
    main()
