# ============================================================
# src/visualization/generate_charts.py
#
# Generate chart dinamis — auto-detect semua versi model
# dari CSV yang tersedia di data/processed/
#
# CARA PAKAI:
#   python src/visualization/generate_charts.py
# ============================================================

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR

CHART_DIR = PROCESSED_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

# Warna per model key
PALETTE = {
    "ARIMA":       "#4fc3f7",
    "SARIMAX":     "#69f0ae",
    "RF":          "#ffb74d",
    "ARIMA_TUNED":   "#29b6f6",
    "SARIMAX_TUNED": "#00e676",
    "RF_TUNED":      "#ffa726",
    "SARIMAX_V2":  "#b9f6ca",
    "RF_V2":       "#ffe082",
    "actual":      "#e0e0e0",
}

def fmt_dollar(x, pos): return f"${x:,.0f}"

# ── Auto-detect semua file prediksi ─────────────────────────
def load_all_predictions():
    """
    Scan PROCESSED_DIR untuk semua predictions_*.csv
    Return dict: { display_label: df }
    """
    # Urutan prioritas tampilan
    KNOWN = [
        ("predictions_arima.csv",        "ARIMA"),
        ("predictions_arima_tuned.csv",  "ARIMA Tuned"),
        ("predictions_sarimax.csv",      "SARIMAX"),
        ("predictions_sarimax_tuned.csv","SARIMAX Tuned ★"),
        ("predictions_sarimax_v2.csv",   "SARIMAX v2"),
        ("predictions_rf.csv",           "RF"),
        ("predictions_rf_tuned.csv",     "RF Tuned"),
        ("predictions_rf_v2.csv",        "RF v2"),
    ]
    dfs = {}
    for fname, label in KNOWN:
        p = PROCESSED_DIR / fname
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            dfs[label] = df
            print(f"  ✓ {label}: {len(df)} prediksi")
    return dfs

def load_comparison():
    """Load versi terbaru model_comparison — cari all_versions dulu."""
    for fname in ["model_comparison_all_versions.csv",
                  "model_comparison_tuned.csv",
                  "model_comparison.csv"]:
        p = PROCESSED_DIR / fname
        if p.exists():
            print(f"  ✓ Comparison: {fname}")
            return pd.read_csv(p)
    return None

def load_feature_importance():
    """Load feature importance terbaru (v2 > tuned > original)."""
    for fname in ["rf_v2_feature_importance.csv",
                  "rf_tuned_feature_importance.csv",
                  "rf_feature_importance.csv"]:
        p = PROCESSED_DIR / fname
        if p.exists():
            print(f"  ✓ Feature importance: {fname}")
            return pd.read_csv(p)
    return None

def get_color(label):
    key = label.upper().replace(" ","_").replace("★","").strip("_")
    return PALETTE.get(key, "#90a4ae")

# ── Chart 1: Actual vs Predicted per model ──────────────────
def chart_actual_vs_predicted(dfs):
    n = len(dfs)
    if n == 0: return
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5*rows), facecolor="#0d0f14")
    fig.suptitle("Prediksi Harga Aluminium vs Aktual",
                 fontsize=14, fontweight="bold", color="white", y=1.01)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, (label, df) in enumerate(dfs.items()):
        ax = axes_flat[i]
        ax.set_facecolor("#161920")
        color = get_color(label)
        mape = df["pct_error"].abs().mean() if "pct_error" in df.columns else None

        ax.plot(df.index, df["actual"], color=PALETTE["actual"],
                lw=1.5, label="Aktual", zorder=3)
        ax.plot(df.index, df["predicted"], color=color,
                lw=1.5, linestyle="--", label=f"Prediksi", zorder=2)
        ax.fill_between(df.index, df["actual"], df["predicted"],
                        alpha=0.1, color=color)

        title = f"{label}"
        if mape: title += f"  |  MAPE: {mape:.2f}%"
        ax.set_title(title, fontsize=10, color="white", pad=6)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_dollar))
        ax.tick_params(colors="#78909c", labelsize=8)
        ax.legend(fontsize=8, framealpha=0.4)
        ax.grid(True, alpha=0.2, linestyle="--")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2f3e")

    # Hide empty subplots
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out = CHART_DIR / "actual_vs_predicted_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0f14")
    plt.close()
    print(f"  ✓ Saved: {out.name}")

# ── Chart 2: Model Comparison Bar ───────────────────────────
def chart_model_comparison(comp_df):
    if comp_df is None: return

    # Normalize kolom nama
    comp_df.columns = [c.strip() for c in comp_df.columns]
    col_mape = next((c for c in comp_df.columns if "MAPE" in c.upper()), None)
    col_mae  = next((c for c in comp_df.columns if "MAE"  in c.upper() and "RMSE" not in c.upper()), None)
    col_rmse = next((c for c in comp_df.columns if "RMSE" in c.upper()), None)
    col_model = next((c for c in comp_df.columns if "Model" in c or "model" in c), comp_df.columns[0])

    models = comp_df[col_model].tolist()
    colors = [get_color(m) for m in models]

    metrics_to_plot = []
    if col_mae:  metrics_to_plot.append((col_mae,  "MAE ($/ton)"))
    if col_rmse: metrics_to_plot.append((col_rmse, "RMSE ($/ton)"))
    if col_mape: metrics_to_plot.append((col_mape, "MAPE (%)"))

    if not metrics_to_plot: return

    fig, axes = plt.subplots(1, len(metrics_to_plot),
                             figsize=(5*len(metrics_to_plot), 5.5), facecolor="#0d0f14")
    fig.suptitle("Perbandingan Performa Semua Versi Model",
                 fontsize=13, fontweight="bold", color="white")
    if len(metrics_to_plot) == 1: axes = [axes]

    for ax, (col, title) in zip(axes, metrics_to_plot):
        vals = comp_df[col].tolist()
        bars = ax.bar(models, vals, color=colors, width=0.6, edgecolor="#2a2f3e", linewidth=1)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold")

        # Highlight terbaik
        min_idx = vals.index(min(vals))
        bars[min_idx].set_edgecolor("#ffd54f")
        bars[min_idx].set_linewidth(2.5)

        ax.set_title(title, fontsize=10, color="white", pad=8)
        ax.set_facecolor("#161920")
        ax.tick_params(colors="#78909c", labelsize=8, axis='y')
        ax.tick_params(colors="white", labelsize=8, axis='x', rotation=30)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_ylim(0, max(vals)*1.28)
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2f3e")

    plt.tight_layout()
    out = CHART_DIR / "model_comparison_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0f14")
    plt.close()
    print(f"  ✓ Saved: {out.name}")

# ── Chart 3: MAPE Evolution (semua versi berurutan) ──────────
def chart_mape_evolution(comp_df):
    if comp_df is None: return

    col_model = next((c for c in comp_df.columns if "Model" in c or "model" in c), comp_df.columns[0])
    col_mape  = next((c for c in comp_df.columns if "MAPE" in c.upper()), None)
    if not col_mape: return

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d0f14")
    ax.set_facecolor("#161920")

    models = comp_df[col_model].tolist()
    mapes  = comp_df[col_mape].tolist()
    colors = [get_color(m) for m in models]

    bars = ax.barh(models, mapes, color=colors, height=0.55, edgecolor="#2a2f3e")
    for bar, val in zip(bars, mapes):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}%", va="center", fontsize=9, color="white")

    # Highlight terbaik
    min_idx = mapes.index(min(mapes))
    bars[min_idx].set_edgecolor("#ffd54f")
    bars[min_idx].set_linewidth(2.5)

    ax.set_xlabel("MAPE (%)", color="#78909c", fontsize=10)
    ax.set_title("Evolusi MAPE — Semua Versi Model\n(lebih kecil = lebih baik)",
                 fontsize=12, fontweight="bold", color="white", pad=10)
    ax.tick_params(colors="white", labelsize=9)
    ax.grid(True, axis="x", alpha=0.2)
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2f3e")

    plt.tight_layout()
    out = CHART_DIR / "mape_evolution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0f14")
    plt.close()
    print(f"  ✓ Saved: {out.name}")

# ── Chart 4: Feature Importance ─────────────────────────────
def chart_feature_importance(fi_df):
    if fi_df is None: return
    top = fi_df.head(20).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d0f14")
    ax.set_facecolor("#161920")

    cmap = plt.cm.YlGn(np.linspace(0.4, 0.9, len(top)))
    bars = ax.barh(top["feature"], top["importance"], color=cmap,
                   edgecolor="#2a2f3e", linewidth=0.5)

    for bar, val in zip(bars, top["importance"]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8, color="white")

    ax.set_xlabel("Importance Score", color="#78909c", fontsize=10)
    ax.set_title("Top 20 Feature Importance — RF (versi terbaru)",
                 fontsize=11, fontweight="bold", color="white", pad=12)
    ax.tick_params(colors="white", labelsize=9)
    ax.grid(True, axis="x", alpha=0.2)
    for spine in ax.spines.values(): spine.set_edgecolor("#2a2f3e")

    plt.tight_layout()
    out = CHART_DIR / "rf_feature_importance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0f14")
    plt.close()
    print(f"  ✓ Saved: {out.name}")

# ── Chart 5: Residual Distribution ──────────────────────────
def chart_residuals(dfs):
    if not dfs: return
    n = len(dfs)
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5), facecolor="#0d0f14")
    fig.suptitle("Distribusi Error (Residual) per Model",
                 fontsize=12, fontweight="bold", color="white")
    if n == 1: axes = [axes]

    for ax, (label, df) in zip(axes, dfs.items()):
        if "error" not in df.columns: continue
        errors = df["error"].dropna()
        color  = get_color(label)
        ax.set_facecolor("#161920")
        ax.hist(errors, bins=20, color=color, alpha=0.75, edgecolor="#2a2f3e")
        ax.axvline(0, color="white", lw=1.5, linestyle="--")
        ax.axvline(errors.mean(), color="#ef5350", lw=1.5,
                   label=f"Mean: ${errors.mean():.1f}")
        ax.set_title(f"{label}\nStd: ${errors.std():.1f}",
                     fontsize=9, color="white")
        ax.tick_params(colors="#78909c", labelsize=8)
        ax.legend(fontsize=8, framealpha=0.4)
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values(): spine.set_edgecolor("#2a2f3e")

    plt.tight_layout()
    out = CHART_DIR / "residuals.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d0f14")
    plt.close()
    print(f"  ✓ Saved: {out.name}")

# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  GENERATE CHARTS — Dinamis (auto-detect semua model)")
    print("=" * 55)

    print("\nMemuat prediksi...")
    dfs = load_all_predictions()
    print(f"  Total model ditemukan: {len(dfs)}")

    print("\nMemuat comparison...")
    comp = load_comparison()

    print("\nMemuat feature importance...")
    fi = load_feature_importance()

    print("\nGenerating charts...")
    chart_actual_vs_predicted(dfs)
    chart_model_comparison(comp)
    chart_mape_evolution(comp)
    chart_feature_importance(fi)
    chart_residuals(dfs)

    print(f"\n✅ Semua chart tersimpan di: {CHART_DIR}")

if __name__ == "__main__":
    main()
