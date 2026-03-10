"""
integrate_lme_from_pdf.py
=========================
Integrates LME inventory data (parsed from westmetall PDF) into features_daily_v2.csv
Run AFTER generating lme_inventory_daily.csv from parse_westmetall_pdf.py

Usage:
    python src/data_collection/integrate_lme_from_pdf.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]  # src/data_collection/ -> src/ -> project root
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

def load_lme_inventory(path):
    """Load LME inventory daily CSV (generated from PDF parsing)."""
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    log.info(f"LME inventory loaded: {len(df)} rows, {df.index.min()} to {df.index.max()}")
    return df

def integrate(features_path, lme_path, output_path):
    """Merge LME features into existing features_daily.csv."""
    
    # Load base features
    feat = pd.read_csv(features_path, parse_dates=['date'])
    feat = feat.set_index('date').sort_index()
    log.info(f"Base features: {feat.shape}")

    # Load LME daily
    lme = load_lme_inventory(lme_path)

    # Select only the columns we want to add
    lme_cols = ['lme_inventory', 'lme_inventory_change', 'lme_inventory_ma5',
                'lme_inventory_ma21', 'lme_inventory_yoy']
    lme_sub = lme[lme_cols]

    # Align to features index (trading days)
    lme_aligned = lme_sub.reindex(feat.index, method='ffill')

    # Drop if already present (re-run safe)
    for col in lme_cols:
        if col in feat.columns:
            feat = feat.drop(columns=[col])

    feat = feat.join(lme_aligned, how='left')

    # Fill any remaining NaN with forward fill then median
    for col in lme_cols:
        if feat[col].isna().sum() > 0:
            n_before = feat[col].isna().sum()
            feat[col] = feat[col].ffill().bfill()
            feat[col] = feat[col].fillna(feat[col].median())
            log.warning(f"{col}: filled {n_before} NaN values")

    log.info(f"Final features: {feat.shape}")
    log.info(f"NaN check:\n{feat[lme_cols].isna().sum()}")

    feat.reset_index().to_csv(output_path, index=False)
    log.info(f"Saved: {output_path}")

    # Quick stats
    log.info("\nLME Inventory stats:")
    log.info(f"\n{feat[lme_cols].describe().to_string()}")

    return feat


if __name__ == "__main__":
    features_path = PROCESSED_DIR / "features_daily.csv"
    lme_path = RAW_DIR / "lme_inventory_daily.csv"
    output_path = PROCESSED_DIR / "features_daily_v2.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Base features not found: {features_path}")
    if not lme_path.exists():
        raise FileNotFoundError(
            f"LME daily CSV not found: {lme_path}\n"
            "Run parse_westmetall_pdf.py first to generate this file."
        )

    integrate(features_path, lme_path, output_path)
    log.info("Done! features_daily_v2.csv ready.")
