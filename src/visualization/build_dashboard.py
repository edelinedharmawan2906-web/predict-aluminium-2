# ============================================================
# src/visualization/build_dashboard.py
#
# Build dashboard HTML dinamis — auto-detect semua model
# dari CSV di data/processed/
#
# CARA PAKAI:
#   python src/visualization/build_dashboard.py
# ============================================================

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR

ROOT = Path(__file__).resolve().parents[2]

# ── File priority maps ───────────────────────────────────────
PRED_FILES = [
    ("predictions_arima.csv",         "ARIMA",        "#4fc3f7"),
    ("predictions_arima_tuned.csv",   "ARIMA Tuned",  "#29b6f6"),
    ("predictions_sarimax.csv",       "SARIMAX",      "#69f0ae"),
    ("predictions_sarimax_tuned.csv", "SARIMAX Tuned","#00e676"),
    ("predictions_sarimax_v2.csv",    "SARIMAX v2",   "#b9f6ca"),
    ("predictions_rf.csv",            "RF",           "#ffb74d"),
    ("predictions_rf_tuned.csv",      "RF Tuned",     "#ffa726"),
    ("predictions_rf_v2.csv",         "RF v2",        "#ffe082"),
]

COMP_FILES = [
    "model_comparison_all_versions.csv",
    "model_comparison_tuned.csv",
    "model_comparison.csv",
]

FI_FILES = [
    "rf_v2_feature_importance.csv",
    "rf_tuned_feature_importance.csv",
    "rf_feature_importance.csv",
]

# ── Load data ────────────────────────────────────────────────
def load_data():
    print("Memuat data...")

    # Predictions
    preds = {}
    colors = {}
    for fname, label, color in PRED_FILES:
        p = PROCESSED_DIR / fname
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            df.index = df.index.strftime("%Y-%m-%d")
            preds[label] = df
            colors[label] = color
            print(f"  ✓ {label}: {len(df)} baris")

    # Comparison
    comp = None
    for fname in COMP_FILES:
        p = PROCESSED_DIR / fname
        if p.exists():
            comp = pd.read_csv(p)
            print(f"  ✓ Comparison: {fname} ({len(comp)} rows)")
            break

    # Feature importance
    fi = None
    for fname in FI_FILES:
        p = PROCESSED_DIR / fname
        if p.exists():
            fi = pd.read_csv(p)
            print(f"  ✓ Feature importance: {fname}")
            break

    # Master price series
    master = None
    for fname in ["features_daily_v2.csv", "features_daily.csv", "master_daily.csv"]:
        p = PROCESSED_DIR / fname
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if "aluminum_price" in df.columns:
                master = df[["aluminum_price"]].copy()
                master.index = master.index.strftime("%Y-%m-%d")
                print(f"  ✓ Master price: {fname} ({len(master)} baris)")
                break

    return preds, colors, comp, fi, master


def safe_json(obj):
    """Convert NaN/inf to None for JSON serialization."""
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

def df_to_json(df, cols=None):
    if df is None: return {}
    d = {"dates": df.index.tolist()}
    for c in (cols or df.columns.tolist()):
        if c in df.columns:
            d[c] = [safe_json(round(v, 2)) if isinstance(v, float) else v
                    for v in df[c].tolist()]
    return d

def find_best_model(comp):
    """Return label of best model by MAPE."""
    if comp is None: return None
    col_mape  = next((c for c in comp.columns if "MAPE" in c.upper()), None)
    col_model = next((c for c in comp.columns if "model" in c.lower()), comp.columns[0])
    if col_mape is None: return None
    idx = comp[col_mape].idxmin()
    return comp.loc[idx, col_model]

# ── Build HTML ───────────────────────────────────────────────
def build_html(preds, colors, comp, fi, master):
    print("Membangun HTML...")

    best_label = find_best_model(comp)
    generated_at = datetime.now().strftime("%d %B %Y %H:%M")
    n_models = len(preds)

    # Metrics summary dari comp
    metrics_summary = {}
    if comp is not None:
        col_model = next((c for c in comp.columns if "model" in c.lower()), comp.columns[0])
        col_mape  = next((c for c in comp.columns if "MAPE" in c.upper()), None)
        col_mae   = next((c for c in comp.columns if "MAE"  in c.upper() and "RMSE" not in c.upper()), None)
        col_rmse  = next((c for c in comp.columns if "RMSE" in c.upper()), None)
        for _, row in comp.iterrows():
            m = row[col_model]
            metrics_summary[m] = {
                "mape": round(float(row[col_mape]), 4) if col_mape else None,
                "mae":  round(float(row[col_mae]),  2)  if col_mae  else None,
                "rmse": round(float(row[col_rmse]), 2)  if col_rmse else None,
                "color": colors.get(m, "#90a4ae"),
            }

    # Serialize predictions
    preds_json = {label: df_to_json(df, ["actual","predicted","pct_error","error"])
                  for label, df in preds.items()}

    # Serialize comparison
    comp_json = []
    if comp is not None:
        for _, row in comp.iterrows():
            r = {}
            for c in comp.columns:
                v = row[c]
                r[c] = safe_json(round(float(v), 4)) if isinstance(v, float) else v
            comp_json.append(r)

    # Feature importance
    fi_json = fi.head(20).to_dict(orient="records") if fi is not None else []

    # Master
    master_json = df_to_json(master) if master is not None else {}

    data_js = f"""
const DATA = {{
  models:    {json.dumps(list(preds.keys()))},
  colors:    {json.dumps(colors)},
  preds:     {json.dumps(preds_json)},
  metrics:   {json.dumps(metrics_summary)},
  comparison:{json.dumps(comp_json)},
  fi:        {json.dumps(fi_json)},
  master:    {json.dumps(master_json)},
  bestModel: {json.dumps(best_label)},
  generatedAt: {json.dumps(generated_at)},
}};
"""

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Aluminum Price Prediction — Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
:root{{
  --bg:#0d0f14;--surface:#161920;--card:#1e2230;--border:#2a2f3e;
  --text:#e8eaf6;--muted:#78909c;--gold:#ffd54f;--danger:#ef5350;
  --font:'IBM Plex Sans',sans-serif;--mono:'IBM Plex Mono',monospace;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh}}
header{{background:linear-gradient(135deg,#0d1b2a,#1a237e 50%,#0d1b2a);
  border-bottom:1px solid var(--border);padding:28px 40px}}
.htag{{font-family:var(--mono);font-size:10px;color:#69f0ae;letter-spacing:3px;margin-bottom:8px}}
h1{{font-size:clamp(20px,3vw,30px);font-weight:700}}
h1 span{{color:#69f0ae}}
.hmeta{{display:flex;gap:20px;margin-top:14px;flex-wrap:wrap}}
.hmeta-item{{font-family:var(--mono);font-size:10px;color:var(--muted);
  border-left:2px solid var(--border);padding-left:10px}}
.hmeta-item strong{{color:var(--text);display:block;font-size:12px}}
.container{{max-width:1400px;margin:0 auto;padding:28px}}
.section-title{{font-family:var(--mono);font-size:10px;letter-spacing:3px;
  text-transform:uppercase;color:var(--muted);margin-bottom:14px;
  padding-bottom:6px;border-bottom:1px solid var(--border)}}
/* Metric cards */
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin-bottom:36px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:18px;
  position:relative;transition:transform .2s,border-color .2s;cursor:default}}
.card:hover{{transform:translateY(-2px);border-color:#3a3f52}}
.card-best{{border-color:var(--gold)!important}}
.best-tag{{position:absolute;top:8px;right:10px;font-family:var(--mono);font-size:9px;
  color:var(--gold);letter-spacing:1px}}
.card-label{{font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:2px;margin-bottom:10px}}
.card-mape{{font-size:26px;font-weight:700;line-height:1}}
.card-mape-name{{font-size:10px;color:var(--muted);font-family:var(--mono);margin-top:3px}}
.card-sub{{margin-top:10px;display:flex;gap:14px}}
.card-sub-item{{font-size:10px;color:var(--muted)}}
.card-sub-item strong{{color:var(--text);display:block;font-size:12px}}
.mbar{{margin-top:8px;background:var(--bg);border-radius:3px;height:5px;overflow:hidden}}
.mbar-fill{{height:100%;border-radius:3px}}
/* Chart cards */
.chart-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:20px;margin-bottom:20px}}
.chart-card h3{{font-size:13px;font-weight:600;margin-bottom:3px;color:white}}
.chart-desc{{font-size:10px;color:var(--muted);margin-bottom:16px;font-family:var(--mono)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
@media(max-width:860px){{.grid2{{grid-template-columns:1fr}}}}
/* Tabs */
.tabs{{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}}
.tab{{padding:5px 14px;border-radius:6px;border:1px solid var(--border);
  background:transparent;color:var(--muted);font-family:var(--mono);
  font-size:10px;cursor:pointer;transition:all .2s}}
.tab:hover,.tab.active{{border-color:#69f0ae;color:#69f0ae;background:rgba(105,240,174,.07)}}
/* Table */
.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.tbl th{{text-align:left;padding:8px 12px;font-family:var(--mono);font-size:9px;
  letter-spacing:1px;color:var(--muted);border-bottom:1px solid var(--border)}}
.tbl td{{padding:6px 12px;border-bottom:1px solid #1e2230}}
.tbl tr:hover td{{background:#1e2230}}
.badge{{display:inline-block;padding:2px 8px;border-radius:3px;
  font-family:var(--mono);font-size:9px}}
.rank{{display:inline-flex;align-items:center;justify-content:center;
  width:20px;height:20px;border-radius:50%;background:var(--border);
  font-size:10px;font-family:var(--mono)}}
.rank.top3{{background:var(--gold);color:#000}}
.fbar-bg{{background:var(--bg);border-radius:3px;height:6px;width:120px}}
.fbar{{height:100%;border-radius:3px}}
/* Comparison table */
.comp-best td{{background:rgba(255,213,79,0.06)!important}}
.comp-best td:first-child{{border-left:3px solid var(--gold)}}
</style>
</head>
<body>
<header>
  <div class="htag">ALUMINUM PRICE PREDICTION SYSTEM</div>
  <h1>Dashboard <span>Prediksi Harga Aluminium</span></h1>
  <div class="hmeta">
    <div class="hmeta-item"><strong id="hModels">{n_models}</strong>Model Dilatih</div>
    <div class="hmeta-item"><strong id="hBest">{best_label or '–'}</strong>Model Terbaik</div>
    <div class="hmeta-item"><strong>21 Hari</strong>Horizon Prediksi</div>
    <div class="hmeta-item"><strong>{generated_at}</strong>Generated At</div>
  </div>
</header>

<div class="container">

  <!-- METRIC CARDS -->
  <div class="section-title">Ringkasan Performa Model</div>
  <div class="cards" id="metricCards"></div>

  <!-- ACTUAL VS PREDICTED -->
  <div class="section-title">Aktual vs Prediksi</div>
  <div class="tabs" id="predTabs"></div>
  <div class="chart-card">
    <h3 id="predTitle">Aktual vs Prediksi</h3>
    <p class="chart-desc">Horizon 21 hari | Test period: Mei 2024 – Feb 2026</p>
    <canvas id="chartPred" height="90"></canvas>
  </div>

  <!-- ERROR OVER TIME -->
  <div class="section-title">Analisis Error</div>
  <div class="tabs" id="errorTabs"></div>
  <div class="grid2">
    <div class="chart-card">
      <h3>Persentase Error dari Waktu ke Waktu</h3>
      <p class="chart-desc" id="errorDesc">–</p>
      <canvas id="chartError" height="140"></canvas>
    </div>
    <div class="chart-card">
      <h3>Distribusi Error (Residual)</h3>
      <p class="chart-desc" id="residDesc">–</p>
      <canvas id="chartResid" height="140"></canvas>
    </div>
  </div>

  <!-- MODEL COMPARISON -->
  <div class="section-title">Perbandingan Semua Versi Model</div>
  <div class="grid2">
    <div class="chart-card">
      <h3>MAPE Semua Model</h3>
      <p class="chart-desc">Lebih kecil = lebih baik | ★ = terbaik</p>
      <canvas id="chartMape" height="160"></canvas>
    </div>
    <div class="chart-card">
      <h3>MAE & RMSE Semua Model</h3>
      <p class="chart-desc">Error absolut dalam $/ton</p>
      <canvas id="chartMaeRmse" height="160"></canvas>
    </div>
  </div>

  <!-- COMPARISON TABLE -->
  <div class="chart-card">
    <h3>Tabel Perbandingan Lengkap</h3>
    <p class="chart-desc">Semua iterasi model dari original → tuned → v2</p>
    <table class="tbl">
      <thead><tr id="compHead"></tr></thead>
      <tbody id="compBody"></tbody>
    </table>
  </div>

  <!-- FEATURE IMPORTANCE -->
  <div class="section-title">Feature Importance — RF (versi terbaru)</div>
  <div class="grid2">
    <div class="chart-card">
      <h3>Top 15 Features</h3>
      <p class="chart-desc">Fitur yang paling mempengaruhi prediksi RF</p>
      <canvas id="chartFI" height="220"></canvas>
    </div>
    <div class="chart-card">
      <h3>Tabel Feature Importance</h3>
      <p class="chart-desc">Top 20 fitur</p>
      <table class="tbl">
        <thead><tr>
          <th>#</th><th>Feature</th><th>Kategori</th>
          <th>Score</th><th>Bar</th>
        </tr></thead>
        <tbody id="fiBody"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
{data_js}

// ── Helpers ────────────────────────────────────────────────
const C = DATA.colors;
const tickStyle = {{color:'#78909c',font:{{family:'IBM Plex Mono',size:9}}}};
const gridStyle = {{color:'#2a2f3e'}};
const legendStyle = {{labels:{{color:'#b0bec5',font:{{family:'IBM Plex Mono',size:10}}}}}};

// ── Metric Cards ───────────────────────────────────────────
const maxMape = Math.max(...Object.values(DATA.metrics).map(m=>m.mape||0));
const cardsEl = document.getElementById('metricCards');
DATA.models.forEach(model => {{
  const m = DATA.metrics[model] || {{}};
  const isBest = model === DATA.bestModel;
  const color = C[model] || '#90a4ae';
  const mapeBar = ((m.mape||0)/maxMape*100).toFixed(0);
  cardsEl.innerHTML += `
    <div class="card ${{isBest?'card-best':''}}">
      ${{isBest?'<div class="best-tag">★ TERBAIK</div>':''}}
      <div class="card-label" style="color:${{color}}">${{model}}</div>
      <div class="card-mape" style="color:${{color}}">${{(m.mape||0).toFixed(2)}}%</div>
      <div class="card-mape-name">MAPE</div>
      <div class="mbar"><div class="mbar-fill" style="width:${{mapeBar}}%;background:${{color}}"></div></div>
      <div class="card-sub">
        <div class="card-sub-item"><strong>$${{(m.mae||0).toFixed(1)}}</strong>MAE</div>
        <div class="card-sub-item"><strong>$${{(m.rmse||0).toFixed(1)}}</strong>RMSE</div>
      </div>
    </div>`;
}});

// ── Pred + Error tabs ──────────────────────────────────────
let predChart=null, errorChart=null, residChart=null;

function buildTabs(containerId, fn) {{
  const el = document.getElementById(containerId);
  DATA.models.forEach((m,i) => {{
    const btn = document.createElement('button');
    btn.className = 'tab' + (i===0?' active':'');
    btn.textContent = m;
    btn.onclick = function() {{
      el.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active');
      fn(m);
    }};
    el.appendChild(btn);
  }});
}}

function showPred(model) {{
  const d = DATA.preds[model] || {{}};
  const color = C[model] || '#90a4ae';
  document.getElementById('predTitle').textContent = `Aktual vs Prediksi — ${{model}}`;
  if (predChart) predChart.destroy();
  predChart = new Chart(document.getElementById('chartPred'), {{
    type:'line',
    data:{{
      labels: d.dates||[],
      datasets:[
        {{label:'Aktual',data:d.actual||[],borderColor:'#e0e0e0',
          borderWidth:1.5,pointRadius:0,tension:0.1}},
        {{label:'Prediksi',data:d.predicted||[],borderColor:color,
          borderWidth:1.5,borderDash:[4,3],pointRadius:0,tension:0.1,
          fill:{{target:0,above:color+'18',below:color+'18'}}}}
      ]
    }},
    options:{{responsive:true,
      plugins:{{legend:legendStyle,
        tooltip:{{callbacks:{{label:ctx=>`${{ctx.dataset.label}}: $${{ctx.parsed.y?.toFixed(2)}}`}}}}}},
      scales:{{
        x:{{ticks:{{...tickStyle,maxTicksLimit:12}},grid:{{color:'#2a2f3e'}}}},
        y:{{ticks:{{...tickStyle,callback:v=>'$'+v.toLocaleString()}},grid:gridStyle}}
      }}
    }}
  }});
}}

function showError(model) {{
  const d = DATA.preds[model] || {{}};
  const color = C[model] || '#90a4ae';
  const pcts = d.pct_error || [];
  const mean = pcts.filter(v=>v!=null).reduce((a,b)=>a+b,0)/pcts.filter(v=>v!=null).length;
  document.getElementById('errorDesc').textContent = `Model: ${{model}} | Mean error: ${{mean.toFixed(2)}}%`;

  if (errorChart) errorChart.destroy();
  errorChart = new Chart(document.getElementById('chartError'), {{
    type:'bar',
    data:{{
      labels: d.dates||[],
      datasets:[{{
        label:'Error %',data:pcts,
        backgroundColor: pcts.map(v=>(v||0)>=0?color+'aa':'#ef535088'),
        borderWidth:0
      }}]
    }},
    options:{{responsive:true,
      plugins:{{legend:{{display:false}}}},
      scales:{{
        x:{{ticks:{{...tickStyle,maxTicksLimit:12}},grid:{{color:'#2a2f3e'}}}},
        y:{{ticks:{{...tickStyle,callback:v=>v+'%'}},grid:gridStyle}}
      }}
    }}
  }});

  // Residual histogram
  const errors = (d.actual||[]).map((a,i)=>a-(d.predicted[i]||0)).filter(v=>v!=null&&!isNaN(v));
  const mn=Math.min(...errors),mx=Math.max(...errors),nB=15,bsz=(mx-mn)/nB;
  const bins=Array.from({{length:nB}},(_,i)=>mn+i*bsz);
  const counts=new Array(nB).fill(0);
  errors.forEach(e=>{{const idx=Math.min(Math.floor((e-mn)/bsz),nB-1);counts[idx]++;}});
  const meanErr=errors.reduce((a,b)=>a+b,0)/errors.length;
  document.getElementById('residDesc').textContent = `${{model}} | Mean: $${{meanErr.toFixed(1)}} | Std: $${{Math.sqrt(errors.map(e=>(e-meanErr)**2).reduce((a,b)=>a+b,0)/errors.length).toFixed(1)}}`;

  if (residChart) residChart.destroy();
  residChart = new Chart(document.getElementById('chartResid'), {{
    type:'bar',
    data:{{
      labels:bins.map(b=>'$'+b.toFixed(0)),
      datasets:[{{label:'Frekuensi',data:counts,
        backgroundColor:color+'99',borderColor:color,borderWidth:1}}]
    }},
    options:{{responsive:true,
      plugins:{{legend:{{display:false}}}},
      scales:{{
        x:{{ticks:{{...tickStyle,maxTicksLimit:10}},grid:{{color:'#2a2f3e'}}}},
        y:{{ticks:tickStyle,grid:gridStyle,beginAtZero:true}}
      }}
    }}
  }});
}}

buildTabs('predTabs', showPred);
buildTabs('errorTabs', showError);
if (DATA.models.length > 0) {{
  showPred(DATA.models[0]);
  showError(DATA.models[0]);
}}

// ── Comparison charts ──────────────────────────────────────
(function() {{
  const models = DATA.comparison.map(r=>r.Model||r.model||Object.values(r)[0]);
  const bgColors = models.map(m=>C[m]||'#90a4ae');
  const colMape  = Object.keys(DATA.comparison[0]||{{}}).find(k=>k.includes('MAPE'));
  const colMae   = Object.keys(DATA.comparison[0]||{{}}).find(k=>k.includes('MAE')&&!k.includes('RMSE'));
  const colRmse  = Object.keys(DATA.comparison[0]||{{}}).find(k=>k.includes('RMSE'));

  if (colMape) {{
    const mapes = DATA.comparison.map(r=>r[colMape]);
    const minM = Math.min(...mapes);
    new Chart(document.getElementById('chartMape'), {{
      type:'bar',
      data:{{
        labels: models.map(m=>m+(m===DATA.bestModel?' ★':'')),
        datasets:[{{
          label:'MAPE (%)', data:mapes,
          backgroundColor: bgColors.map((c,i)=>mapes[i]===minM?c:c+'88'),
          borderColor: bgColors.map((c,i)=>mapes[i]===minM?'#ffd54f':c),
          borderWidth: mapes.map(v=>v===minM?2.5:1)
        }}]
      }},
      options:{{responsive:true,
        plugins:{{legend:{{display:false}},
          tooltip:{{callbacks:{{label:ctx=>ctx.parsed.y.toFixed(3)+'%'}}}}}},
        scales:{{
          x:{{ticks:{{...tickStyle,maxRotation:30}},grid:{{color:'#2a2f3e'}}}},
          y:{{ticks:{{...tickStyle,callback:v=>v+'%'}},grid:gridStyle,beginAtZero:false,
            min: Math.max(0,(Math.min(...mapes)-0.5))}}
        }}
      }}
    }});
  }}

  if (colMae && colRmse) {{
    const maes  = DATA.comparison.map(r=>r[colMae]);
    const rmses = DATA.comparison.map(r=>r[colRmse]);
    new Chart(document.getElementById('chartMaeRmse'), {{
      type:'bar',
      data:{{
        labels: models,
        datasets:[
          {{label:'MAE ($)',data:maes,backgroundColor:bgColors.map(c=>c+'99'),borderColor:bgColors,borderWidth:1}},
          {{label:'RMSE ($)',data:rmses,backgroundColor:bgColors.map(c=>c+'55'),borderColor:bgColors,borderWidth:1,borderDash:[3,2]}}
        ]
      }},
      options:{{responsive:true,
        plugins:{{legend:legendStyle}},
        scales:{{
          x:{{ticks:{{...tickStyle,maxRotation:30}},grid:{{color:'#2a2f3e'}}}},
          y:{{ticks:{{...tickStyle,callback:v=>'$'+v}},grid:gridStyle,beginAtZero:false}}
        }}
      }}
    }});
  }}
}})();

// ── Comparison Table ──────────────────────────────────────
(function() {{
  if (!DATA.comparison.length) return;
  const cols = Object.keys(DATA.comparison[0]);
  const head = document.getElementById('compHead');
  const colMape = cols.find(c=>c.includes('MAPE'));
  cols.forEach(c=>head.innerHTML+=`<th>${{c}}</th>`);

  const body = document.getElementById('compBody');
  const minMape = colMape ? Math.min(...DATA.comparison.map(r=>r[colMape])) : null;
  DATA.comparison.forEach(row => {{
    const isBest = colMape && row[colMape]===minMape;
    const tr = document.createElement('tr');
    if (isBest) tr.className='comp-best';
    cols.forEach((c,i) => {{
      const td = document.createElement('td');
      const v = row[c];
      if (i===0) {{
        const color = C[v]||'#90a4ae';
        td.innerHTML = `<span style="color:${{color}};font-family:var(--mono);font-size:11px">${{v}}</span>${{isBest?' <span class="badge" style="background:#ffd54f22;color:var(--gold)">★</span>':''}}`;
      }} else {{
        td.style.fontFamily='var(--mono)';td.style.fontSize='11px';
        td.textContent = typeof v==='number' ? v.toFixed(2) : v;
        if (isBest) td.style.color='#ffd54f';
      }}
      tr.appendChild(td);
    }});
    body.appendChild(tr);
  }});
}})();

// ── Feature Importance ────────────────────────────────────
(function() {{
  if (!DATA.fi.length) return;
  const top15 = DATA.fi.slice(0,15);
  const maxImp = top15[0]?.importance||1;
  new Chart(document.getElementById('chartFI'), {{
    type:'bar',
    data:{{
      labels: top15.map(f=>f.feature),
      datasets:[{{
        label:'Importance',
        data: top15.map(f=>f.importance),
        backgroundColor: top15.map((_,i)=>`hsl(${{100+i*8}},60%,50%)`),
        borderWidth:0
      }}]
    }},
    options:{{
      indexAxis:'y',responsive:true,
      plugins:{{legend:{{display:false}}}},
      scales:{{
        x:{{ticks:tickStyle,grid:gridStyle}},
        y:{{ticks:{{...tickStyle,font:{{size:9}}}},grid:{{color:'#2a2f3e'}}}}
      }}
    }}
  }});

  const cats = {{
    'Rolling MA':['ma'],'Lag':['lag'],'FX':['usd','cny','eur','jpy'],
    'Energy':['gas','oil','wti'],'PMI':['pmi'],'Macro':['retail','construct','industrial'],
    'LME':['lme'],'Volatility':['vol','std'],'Seasonal':['month','quarter','dow','week']
  }};
  function getcat(name) {{
    name = name.toLowerCase();
    for (const [cat,keys] of Object.entries(cats)) {{
      if (keys.some(k=>name.includes(k))) return cat;
    }}
    return 'Other';
  }}
  const catColors = {{'Rolling MA':'#69f0ae','Lag':'#4fc3f7','FX':'#f48fb1',
    'Energy':'#ff8a65','PMI':'#fff176','Macro':'#ffb74d',
    'LME':'#b9f6ca','Volatility':'#80cbc4','Seasonal':'#b39ddb','Other':'#90a4ae'}};
  const fiBody = document.getElementById('fiBody');
  DATA.fi.slice(0,20).forEach((f,i)=>{{
    const cat=getcat(f.feature), color=catColors[cat]||'#888';
    const pct=(f.importance/maxImp*100).toFixed(0);
    const tr=document.createElement('tr');
    tr.innerHTML=`
      <td><span class="rank ${{i<3?'top3':''}}">${{i+1}}</span></td>
      <td style="font-family:var(--mono);font-size:11px">${{f.feature}}</td>
      <td><span class="badge" style="background:${{color}}22;color:${{color}}">${{cat}}</span></td>
      <td style="font-family:var(--mono);color:${{color}}">${{f.importance.toFixed(4)}}</td>
      <td><div class="fbar-bg"><div class="fbar" style="width:${{pct}}%;background:${{color}}"></div></div></td>`;
    fiBody.appendChild(tr);
  }});
}})();
</script>
</body>
</html>"""
    return html


def main():
    print("=" * 55)
    print("  BUILD DASHBOARD — Dinamis (auto-detect semua model)")
    print("=" * 55)

    preds, colors, comp, fi, master = load_data()
    print(f"\n  Total model: {len(preds)}")
    if comp is not None:
        print(f"  Comparison: {len(comp)} rows")

    html = build_html(preds, colors, comp, fi, master)

    out = ROOT / "dashboard_fase3.html"
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"\n✅ Dashboard: {out}")
    print(f"   Ukuran: {size_kb:.1f} KB")
    print(f"\n→ Buka di browser: {out}")


if __name__ == "__main__":
    main()
