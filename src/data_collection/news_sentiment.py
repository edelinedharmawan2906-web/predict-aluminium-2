# ============================================================
# src/data_collection/04_news_sentiment.py
#
# FUNGSI : Mengambil dan memproses data berita untuk sentimen:
#          1. NewsAPI     — berita aluminium global (gratis, 100 req/hari)
#          2. GDELT       — sentimen berita global (gratis, BigQuery)
#          3. Scoring     — hitung skor sentimen dengan VADER (cepat)
#                          atau FinBERT (akurat, butuh GPU/waktu lebih)
#
# OUTPUT : data/raw/news_raw.csv
#          data/raw/news_sentiment_daily.csv
#
# CATATAN: Free tier NewsAPI hanya bisa akses berita 1 bulan terakhir.
#          Untuk data historis, perlu berlangganan atau gunakan GDELT.
# ============================================================

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    RAW_DATA_DIR, DATA_START_DATE, DATA_END_DATE,
    NEWSAPI_KEY, NEWS_KEYWORDS
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 1. NEWSAPI — Berita Real-Time (1 Bulan Terakhir)
# ─────────────────────────────────────────────────────────────

def fetch_newsapi(
    keywords: list = None,
    days_back: int = 30,
    max_articles: int = 500,
    language: str = "en",
) -> pd.DataFrame:
    """
    Ambil berita terkini dari NewsAPI.org.

    PERUBAHAN: Free tier NewsAPI hanya support endpoint /top-headlines
    (bukan /everything yang error 426). Kita gunakan /top-headlines
    dengan category=business, lalu filter manual by keyword.

    Alternatif gratis jika NewsAPI terlalu terbatas:
    - RSS feed dari Reuters, Bloomberg, Mining.com (lihat fetch_rss_feeds)
    """
    if not NEWSAPI_KEY:
        log.warning("NEWSAPI_KEY tidak diset. Skip NewsAPI.")
        return pd.DataFrame()

    if keywords is None:
        keywords = NEWS_KEYWORDS

    log.info(f"Mengambil berita dari NewsAPI /top-headlines (free tier compatible)...")

    all_articles = []

    # Free tier: gunakan /top-headlines dengan sources atau q sederhana
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "q":        "aluminum OR aluminium OR commodity",
        "language": language,
        "pageSize": 100,
        "apiKey":   NEWSAPI_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if data.get("status") == "ok":
            articles = data.get("articles", [])
            log.info(f"  ✓ /top-headlines: {len(articles)} artikel")
            for art in articles:
                all_articles.append({
                    "published_at": art.get("publishedAt", ""),
                    "title":        art.get("title", ""),
                    "description":  art.get("description", ""),
                    "content":      art.get("content", ""),
                    "source":       art.get("source", {}).get("name", ""),
                    "url":          art.get("url", ""),
                    "keyword":      "top-headlines",
                })
        else:
            log.warning(f"  NewsAPI error: {data.get('message', 'Unknown')}")
            log.info("  → Free tier terbatas. Beralih ke RSS feeds sebagai alternatif gratis.")

    except Exception as e:
        log.error(f"  NewsAPI request gagal: {e}")

    if not all_articles:
        log.warning("NewsAPI tidak menghasilkan artikel. Menggunakan RSS feeds sebagai pengganti.")
        return fetch_rss_feeds()

    df = pd.DataFrame(all_articles)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["published_at"] = df["published_at"].dt.tz_localize(None)
    df = df.drop_duplicates(subset=["url"]).sort_values("published_at").reset_index(drop=True)
    return df


def fetch_rss_feeds() -> pd.DataFrame:
    """
    Ambil berita komoditas dari RSS feeds gratis sebagai alternatif NewsAPI.

    RSS feeds yang digunakan (semua gratis, tidak perlu API key):
    - Mining.com        : berita khusus industri mining & logam
    - Kitco News        : harga logam dan komoditas
    - Reuters Commodity : berita komoditas global
    - Metal Bulletin    : harga logam industri

    Returns
    -------
    pd.DataFrame dengan kolom: published_at, title, description, source, url
    """
    log.info("Mengambil berita dari RSS feeds (gratis, tanpa API key)...")

    RSS_FEEDS = {
        "Mining.com":      "https://www.mining.com/feed/",
        "Kitco News":      "https://www.kitco.com/rss/kitco-news.xml",
        "Reuters Metals":  "https://feeds.reuters.com/reuters/businessNews",
        "LME News":        "https://www.lme.com/en/news-and-events/rss-feeds",
    }

    all_articles = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            resp = requests.get(
                feed_url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0 (research bot)"},
            )
            resp.raise_for_status()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.content, "xml")
            items = soup.find_all("item")

            count = 0
            for item in items:
                title = item.find("title")
                desc  = item.find("description")
                link  = item.find("link")
                pubdt = item.find("pubDate")

                title_text = title.get_text(strip=True) if title else ""
                desc_text  = desc.get_text(strip=True)  if desc  else ""

                # Filter: hanya simpan jika ada keyword aluminium
                combined = (title_text + " " + desc_text).lower()
                if any(kw.lower() in combined for kw in ["aluminum", "aluminium", "lme", "metal", "commodity", "mining"]):
                    all_articles.append({
                        "published_at": pubdt.get_text(strip=True) if pubdt else "",
                        "title":        title_text,
                        "description":  desc_text[:500],  # Potong max 500 char
                        "content":      "",
                        "source":       source_name,
                        "url":          link.get_text(strip=True) if link else "",
                        "keyword":      "rss",
                    })
                    count += 1

            log.info(f"  ✓ {source_name}: {count} artikel relevan")
            time.sleep(0.5)

        except Exception as e:
            log.warning(f"  ✗ {source_name}: {e}")

    if not all_articles:
        log.warning("Semua RSS feeds gagal.")
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["published_at"] = df["published_at"].dt.tz_localize(None)
    df = df.dropna(subset=["published_at"])
    df = df.drop_duplicates(subset=["url"]).sort_values("published_at").reset_index(drop=True)
    log.info(f"  → Total RSS: {len(df)} artikel relevan")
    return df




# ─────────────────────────────────────────────────────────────
# 2. GDELT PROJECT — Sentimen Berita Global (GRATIS)
# ─────────────────────────────────────────────────────────────

def fetch_gdelt_sentiment(
    start: str = DATA_START_DATE,
    end: str   = DATA_END_DATE,
    theme: str = "COMMODITY_ALUMINUM",
) -> pd.DataFrame:
    """
    Ambil data sentimen dari GDELT Project menggunakan GKG (Global Knowledge Graph).

    GDELT adalah database berita global yang diupdate setiap 15 menit.
    API GKG menyediakan data sentimen berita termasuk tone/sentiment score.

    Referensi: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/

    Parameter penting dari GDELT GKG:
    - V2Tone : skor tone dokumen (positif = bullish, negatif = bearish)
      Format: avg_tone, pos_score, neg_score, polarity, arousal, novelty
    - Themes : topik yang diidentifikasi dalam artikel

    Returns
    -------
    pd.DataFrame kolom: date, avg_tone, positive_score, negative_score,
                        article_count, net_sentiment
    """
    log.info("Mengambil data sentimen dari GDELT Project...")
    log.info("  → GDELT GKG API endpoint: http://data.gdeltproject.org/api/v2/")

    # Cara penggunaan GDELT DOC API (gratis, tidak perlu API key)
    # Dokumentasi: https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Query untuk berita tentang aluminium
    # Mode: ArtList = daftar artikel, TimelineVol = volume timeline
    query = "aluminum OR aluminium commodity LME"

    all_data = []

    # GDELT membatasi query per tanggal, iterasi per bulan
    date_range = pd.date_range(start=start, end=end, freq="MS")

    for i, month_start in enumerate(date_range):
        month_end = (month_start + pd.DateOffset(months=1) - pd.DateOffset(days=1))
        month_end = min(month_end, pd.Timestamp(end))

        start_str = month_start.strftime("%Y%m%d%H%M%S")
        end_str   = month_end.strftime("%Y%m%d%H%M%S")

        params = {
            "query":      query,
            "mode":       "timelinetone",   # Minta data tone/sentimen per waktu
            "format":     "json",
            "startdatetime": start_str,
            "enddatetime":   end_str,
            "maxrecords":    250,
        }

        try:
            resp = requests.get(base_url, params=params, timeout=20)

            if resp.status_code == 200:
                data = resp.json()

                # Parse GDELT timeline response
                if "timeline" in data:
                    for entry in data["timeline"]:
                        if "data" in entry:
                            for point in entry["data"]:
                                all_data.append({
                                    "date":     pd.to_datetime(point.get("date", ""), errors="coerce"),
                                    "avg_tone": float(point.get("value", 0)),
                                    "source":   "gdelt",
                                })
            else:
                log.debug(f"  GDELT {month_start.strftime('%Y-%m')}: status {resp.status_code}")

            time.sleep(1.0)  # GDELT rate limit: 1 request/detik

        except Exception as e:
            log.debug(f"  GDELT error {month_start.strftime('%Y-%m')}: {e}")
            time.sleep(2)

        # Progress indicator
        if (i + 1) % 12 == 0:
            log.info(f"  Progress: {i+1}/{len(date_range)} bulan diproses...")

    if not all_data:
        log.warning("Tidak ada data GDELT yang berhasil diambil.")
        log.info("  → Alternatif: download GDELT data langsung dari http://data.gdeltproject.org/")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.dropna(subset=["date", "avg_tone"])

    # Agregasi ke harian
    df_daily = (
        df.groupby(df["date"].dt.date)
        .agg(
            avg_tone      = ("avg_tone", "mean"),
            article_count = ("avg_tone", "count"),
        )
        .reset_index()
        .rename(columns={"date": "date"})
    )
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily["source"] = "gdelt"

    log.info(f"  → GDELT: {len(df_daily)} hari data sentimen")
    return df_daily


# ─────────────────────────────────────────────────────────────
# 3. SENTIMENT SCORING
# ─────────────────────────────────────────────────────────────

def score_sentiment_vader(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung skor sentimen menggunakan VADER (Valence Aware Dictionary
    and sEntiment Reasoner).

    VADER cocok untuk:
    - Proses cepat (tidak perlu GPU)
    - Berita berbahasa Inggris
    - Data volume besar

    Kekurangan vs FinBERT:
    - Tidak memahami konteks kalimat secara mendalam
    - Kurang akurat untuk teks finansial/teknis

    Score VADER: compound [-1, +1]
    -1 = sangat negatif (bearish)
     0 = netral
    +1 = sangat positif (bullish)
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import nltk
        nltk.download("vader_lexicon", quiet=True)
    except ImportError:
        log.error("nltk belum terinstall. Jalankan: pip install nltk")
        return pd.DataFrame()

    if df_news.empty:
        return pd.DataFrame()

    log.info(f"Menghitung skor sentimen VADER untuk {len(df_news)} artikel...")

    sia = SentimentIntensityAnalyzer()

    # Gabungkan title + description untuk analisis yang lebih baik
    def get_text(row):
        parts = []
        if pd.notna(row.get("title")):        parts.append(str(row["title"]))
        if pd.notna(row.get("description")):  parts.append(str(row["description"]))
        return " ".join(parts)

    df_scored = df_news.copy()
    df_scored["text"] = df_scored.apply(get_text, axis=1)
    df_scored["sentiment_compound"] = df_scored["text"].apply(
        lambda t: sia.polarity_scores(t)["compound"] if t.strip() else 0.0
    )
    df_scored["sentiment_label"] = df_scored["sentiment_compound"].apply(
        lambda s: "positive" if s > 0.05 else ("negative" if s < -0.05 else "neutral")
    )

    # Agregasi ke harian
    df_scored["date"] = pd.to_datetime(df_scored["published_at"]).dt.date

    df_daily = (
        df_scored.groupby("date")
        .agg(
            sentiment_mean    = ("sentiment_compound", "mean"),
            sentiment_median  = ("sentiment_compound", "median"),
            sentiment_std     = ("sentiment_compound", "std"),
            article_count     = ("sentiment_compound", "count"),
            positive_count    = ("sentiment_label",
                                  lambda x: (x == "positive").sum()),
            negative_count    = ("sentiment_label",
                                  lambda x: (x == "negative").sum()),
        )
        .reset_index()
    )
    df_daily["date"]           = pd.to_datetime(df_daily["date"])
    df_daily["sentiment_std"]  = df_daily["sentiment_std"].fillna(0)
    df_daily["pos_neg_ratio"]  = (
        df_daily["positive_count"] /
        (df_daily["negative_count"] + 1)  # +1 untuk hindari divisi nol
    )
    df_daily["scoring_method"] = "vader"

    log.info(f"  → VADER: {len(df_daily)} hari dengan skor sentimen")
    log.info(f"  → Mean sentiment: {df_daily['sentiment_mean'].mean():.4f}")
    return df_daily


def score_sentiment_finbert(df_news: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """
    Hitung skor sentimen menggunakan FinBERT — model NLP khusus teks finansial.

    FinBERT dilatih pada Financial PhraseBank dan corpus keuangan lainnya,
    sehingga lebih akurat dari VADER untuk berita komoditas.

    Model: ProsusAI/finbert (HuggingFace)
    Output: positive, negative, neutral (probabilitas)

    CATATAN: Butuh ~2GB RAM dan akan lebih cepat dengan GPU.
             Untuk CPU, proses ~100 artikel/menit.

    Parameters
    ----------
    batch_size : int — jumlah artikel per batch (turunkan jika RAM terbatas)
    """
    try:
        from transformers import pipeline
    except ImportError:
        log.error("transformers belum terinstall. Jalankan: pip install transformers torch")
        return pd.DataFrame()

    if df_news.empty:
        return pd.DataFrame()

    log.info(f"Menghitung skor sentimen FinBERT untuk {len(df_news)} artikel...")
    log.info("  → Loading model ProsusAI/finbert (pertama kali: ~2GB download)...")

    try:
        # Load pipeline FinBERT
        sentiment_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,  # -1 = CPU, 0 = GPU (jika tersedia)
            truncation=True,
            max_length=512,
        )

        # Gabungkan title + description
        texts = []
        for _, row in df_news.iterrows():
            parts = []
            if pd.notna(row.get("title")):        parts.append(str(row["title"]))
            if pd.notna(row.get("description")):  parts.append(str(row["description"]))
            text = " ".join(parts)[:512]  # Truncate ke 512 karakter
            texts.append(text if text.strip() else "neutral news")

        # Proses dalam batch
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = sentiment_pipeline(batch)
            results.extend(batch_results)
            if (i // batch_size + 1) % 10 == 0:
                log.info(f"  Progress: {i + batch_size}/{len(texts)} artikel...")

        # Map label ke skor numerik
        label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

        df_scored = df_news.copy()
        df_scored["sentiment_label"]  = [r["label"].lower() for r in results]
        df_scored["sentiment_score"]  = [r["score"] for r in results]
        df_scored["sentiment_numeric"] = df_scored["sentiment_label"].map(label_map)
        # Skor final = arah * confidence
        df_scored["sentiment_compound"] = (
            df_scored["sentiment_numeric"] * df_scored["sentiment_score"]
        )

        # Agregasi harian
        df_scored["date"] = pd.to_datetime(df_scored["published_at"]).dt.date

        df_daily = (
            df_scored.groupby("date")
            .agg(
                sentiment_mean    = ("sentiment_compound", "mean"),
                sentiment_median  = ("sentiment_compound", "median"),
                sentiment_std     = ("sentiment_compound", "std"),
                article_count     = ("sentiment_compound", "count"),
                positive_count    = ("sentiment_label",
                                      lambda x: (x == "positive").sum()),
                negative_count    = ("sentiment_label",
                                      lambda x: (x == "negative").sum()),
            )
            .reset_index()
        )
        df_daily["date"]           = pd.to_datetime(df_daily["date"])
        df_daily["sentiment_std"]  = df_daily["sentiment_std"].fillna(0)
        df_daily["scoring_method"] = "finbert"

        log.info(f"  → FinBERT: {len(df_daily)} hari dengan skor sentimen")
        return df_daily

    except Exception as e:
        log.error(f"FinBERT scoring gagal: {e}")
        log.info("  → Fallback ke VADER scoring...")
        return score_sentiment_vader(df_news)


# ─────────────────────────────────────────────────────────────
# 4. SIMPAN DATA
# ─────────────────────────────────────────────────────────────

def save_news_data(
    df_raw: pd.DataFrame,
    df_sentiment: pd.DataFrame,
) -> dict:
    """Simpan data berita dan skor sentimen ke CSV."""
    saved = {}

    if not df_raw.empty:
        path = RAW_DATA_DIR / "news_raw.csv"
        df_raw.to_csv(path, index=False)
        log.info(f"  ✓ Disimpan: {path}")
        saved["news_raw"] = path

    if not df_sentiment.empty:
        path = RAW_DATA_DIR / "news_sentiment_daily.csv"
        df_sentiment.to_csv(path, index=False)
        log.info(f"  ✓ Disimpan: {path}")
        saved["sentiment"] = path

    return saved


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────

def run(
    start: str   = DATA_START_DATE,
    end: str     = DATA_END_DATE,
    method: str  = "vader",    # "vader" (cepat) atau "finbert" (akurat)
):
    """
    Entry point untuk pengumpulan data sentimen berita.

    Parameters
    ----------
    method : str — metode scoring: 'vader' (cepat) atau 'finbert' (akurat)
    """
    log.info("=" * 60)
    log.info("FASE 1 — Pengumpulan Data: Sentimen Berita")
    log.info("=" * 60)

    df_raw_news = pd.DataFrame()

    # 1. Ambil berita dari NewsAPI (untuk berita terbaru)
    if NEWSAPI_KEY:
        df_raw_news = fetch_newsapi(
            keywords=NEWS_KEYWORDS,
            days_back=30,  # Free tier: max 30 hari
        )
    else:
        log.warning("NewsAPI key tidak ada. Hanya menggunakan GDELT untuk sentimen historis.")

    # 2. Ambil sentimen historis dari GDELT
    df_gdelt = fetch_gdelt_sentiment(start=start, end=end)

    # 3. Scoring sentimen untuk berita NewsAPI (jika ada)
    df_sentiment_newsapi = pd.DataFrame()
    if not df_raw_news.empty:
        log.info(f"\nMenghitung skor sentimen dengan metode: {method.upper()}")
        if method == "finbert":
            df_sentiment_newsapi = score_sentiment_finbert(df_raw_news)
        else:
            df_sentiment_newsapi = score_sentiment_vader(df_raw_news)

    # 4. Gabungkan sentimen dari NewsAPI dan GDELT
    dfs_to_combine = []
    if not df_sentiment_newsapi.empty:
        dfs_to_combine.append(df_sentiment_newsapi[["date", "sentiment_mean", "article_count", "scoring_method"]])
    if not df_gdelt.empty:
        gdelt_renamed = df_gdelt.rename(columns={"avg_tone": "sentiment_mean"})
        gdelt_renamed["scoring_method"] = "gdelt"
        dfs_to_combine.append(gdelt_renamed[["date", "sentiment_mean", "article_count", "scoring_method"]])

    df_sentiment_final = pd.DataFrame()
    if dfs_to_combine:
        df_sentiment_final = pd.concat(dfs_to_combine).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        # Rolling averages — fitur smoothed untuk model
        df_sentiment_final["sentiment_ma3"]  = df_sentiment_final["sentiment_mean"].rolling(3).mean()
        df_sentiment_final["sentiment_ma7"]  = df_sentiment_final["sentiment_mean"].rolling(7).mean()

    # 5. Simpan
    log.info("\nMenyimpan data sentimen...")
    saved_paths = save_news_data(df_raw_news, df_sentiment_final)

    log.info("\n=== RINGKASAN SENTIMEN ===")
    if not df_sentiment_final.empty:
        log.info(f"  Total hari: {len(df_sentiment_final)}")
        log.info(f"  Rata-rata sentimen: {df_sentiment_final['sentiment_mean'].mean():.4f}")
        log.info(f"  Sentimen min/max  : {df_sentiment_final['sentiment_mean'].min():.4f} / {df_sentiment_final['sentiment_mean'].max():.4f}")

    return {
        "news_raw":           df_raw_news,
        "sentiment_newsapi":  df_sentiment_newsapi,
        "sentiment_gdelt":    df_gdelt,
        "sentiment_final":    df_sentiment_final,
        "paths":              saved_paths,
    }


if __name__ == "__main__":
    # Ganti ke "finbert" untuk akurasi lebih tinggi (butuh lebih lama)
    results = run(method="vader")