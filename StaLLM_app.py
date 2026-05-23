# StaLLM_UI.py
# Pro UI: Inter/JetBrains Mono fonts, premium hero, glass cards, KPI chips,
# code viewer with line numbers + highlights, strictness guide, Top-K/Positive illustration.
import base64
import json
import re
import time
import os
import tempfile
import zipfile
from html import escape
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

CHART_FONT = "Manrope, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"

alt.renderers.set_embed_options(
    actions=False,
)

from StaLLM_core import (
    run_experiment,
    run_selected_experiments,
    run_selected_models_experiments,  # model compare
    load_ground_truth_spans,          # span-level loader
    detect_language_from_zip,
    load_strategies,
    save_strategies,
    find_exts_for_language,
    detect_gt_capabilities,
    _read_csv_robust,                 # robust CSV reader
)
from StaLLM_models import Session, SmellDetectionResult, init_db, save_run_result
from StaLLM_llm import (
    ChatModel, LLMConfig, available_models,
    load_llm_registry, build_llm_from_slot,
    test_ollama_connectivity, validate_ollama_config, debug_ollama_response
)
from StaLLM_benchmarks import (
    DACOS_PROMPT_TEMPLATE,
    load_argouml_feature_tasks,
    load_dacos_smell_samples,
    load_mlcq_smell_samples,
    run_dacos_smell_benchmark,
)
from StaLLM_tasks import LOCATION_PROMPT_TEMPLATES, list_repo_candidates, paths_match, run_location_task
from StaLLM_autoresearch import (
    PromptCandidate,
    evaluate_code_smell,
    evaluate_feature_location,
    write_autoresearch_outputs,
)


# =========================
# Minimal .env loader
# =========================
def _load_env_fallback(dotenv_path: Path, override: bool = False) -> bool:
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                if key and (override or key not in os.environ):
                    os.environ[key] = val
        return True
    except FileNotFoundError:
        return False

def _load_dotenv_robust(path: Path, override: bool = False) -> None:
    try:
        from dotenv import load_dotenv as _ld  # type: ignore
        _ld(dotenv_path=path, override=override)
    except Exception:
        _load_env_fallback(path, override=override)

# =========================
# Init
# =========================
APP_DIR = Path(__file__).resolve().parent
BRAND_NAME = "StarLLM"
BRAND_LOGO_PATH = APP_DIR / "assets" / "starllm-logo.svg"
BRAND_LOGO_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    BRAND_LOGO_PATH.read_bytes()
).decode("ascii")

st.set_page_config(
    page_title=f"{BRAND_NAME}: Static Analysis meets LLMs",
    page_icon=str(BRAND_LOGO_PATH),
    layout="wide",
)
_load_dotenv_robust(Path(__file__).with_name(".env"), override=False)
init_db()

RESULTS_SCHEMA_VERSION = 2

# =========================
# Styles (UI) — English + Pro look
# =========================
st.markdown( """ <style> /* Fonts */ @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap'); :root{ --text:#0e1220; --muted:#667085; --border:rgba(14,18,32,.12); --card-bg:rgba(255,255,255,.55); --glass:linear-gradient(180deg, rgba(255,255,255,.80), rgba(255,255,255,.60)); --accent:#7c3aed; --accent2:#06b6d4; --accent3:#22c55e; --warn:#f59e0b; --danger:#ef4444; --hl: rgba(252, 211, 77, .35); --ring: 0 8px 22px rgba(35, 38, 47, .10), 0 2px 6px rgba(35, 38, 47, .06); } #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton { visibility:hidden !important; height:0 !important; display:none !important; } html, body, .stApp { background:#f6f7fb !important; color:var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; } .stApp { padding-top: 10px; } @media (prefers-color-scheme: dark){ :root{ --text:#e6e9ef; --muted:#a3afc6; --border:rgba(255,255,255,.09); --card-bg:rgba(20,24,35,.55); --glass:linear-gradient(180deg, rgba(20,24,35,.75), rgba(20,24,35,.55)); --hl: rgba(234, 179, 8, .20); } html, body, .stApp{ background: radial-gradient(1200px 600px at 10% -10%, rgba(124,58,237,.16), transparent), radial-gradient(1000px 500px at 110% -20%, rgba(6,182,212,.12), transparent), #0b1220 !important; } } /* HERO */ .hero{ position:relative; border:1px solid var(--border); border-radius:22px; padding:28px 24px; overflow:hidden; margin-bottom:14px; background: radial-gradient(900px 220px at -10% -20%, rgba(124,58,237,.18), transparent 55%), radial-gradient(700px 220px at 110% -10%, rgba(6,182,212,.15), transparent 55%), var(--glass); box-shadow: var(--ring); } .hero-brand{ display:flex; align-items:center; gap:14px; margin-bottom:8px; } .hero-logo{ width:48px; height:48px; border-radius:13px; box-shadow:0 10px 24px rgba(15,23,42,.18); flex:0 0 auto; } .hero h1{ margin:0; font-weight:800; letter-spacing:.2px; } .hero p{ margin:0; color:var(--muted); font-size:1.0rem; } .badges{ margin-top:8px; } .badge{ display:inline-flex; align-items:center; gap:.45rem; padding:.38rem .70rem; border:1px solid var(--border); border-radius:999px; font-size:.78rem; font-weight:600; background:linear-gradient(90deg, rgba(124,58,237,.10), rgba(6,182,212,.10)); margin-right:.45rem; } /* KPI CARDS */ .kpi{ background:var(--glass); border:1px solid var(--border); border-radius:16px; padding:14px 16px; box-shadow: var(--ring); } .kpi-label{ color:var(--muted); font-size:.85rem; margin-bottom:.25rem;} .kpi-value{ font-size:1.6rem; font-weight:800; letter-spacing:.2px;} .kpi-ok{ box-shadow:0 0 0 1px rgba(34,197,94,.28) inset; } .kpi-warn{ box-shadow:0 0 0 1px rgba(245,158,11,.28) inset; } .kpi-bad{ box-shadow:0 0 0 1px rgba(239,68,68,.28) inset; } /* CODE BOX with line numbers + highlight */ .codebox{ font-family:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12.5px; border:1px solid var(--border); border-radius:14px; overflow:hidden; background: var(--glass); box-shadow: var(--ring); } .codebox table{ border-collapse: collapse; width:100%; } .codebox td{ vertical-align: top; padding: 0; } .codebox .gutter{ width: 3.2em; background: rgba(2,6,23,.05); color: var(--muted); text-align: right; user-select:none; } .codebox .code{ white-space: pre; } .codebox .line{ padding: 2px 10px; border-bottom: 1px solid rgba(2,6,23,.06); } .codebox .ln{ padding: 2px 8px; border-bottom: 1px solid rgba(2,6,23,.08); } .codebox .hl{ background: var(--hl); } .codebox .gt{ background:#fff7ed; box-shadow:inset 3px 0 #f97316; } .codebox .llm{ background:#eff6ff; box-shadow:inset 3px 0 #2563eb; } .codebox .both{ background:#ecfdf5; box-shadow:inset 3px 0 #16a34a; } .code-legend{ display:flex; gap:8px; margin:.4rem 0 .6rem; flex-wrap:wrap; } .legend-chip{ display:inline-flex; align-items:center; border-radius:999px; padding:4px 8px; font-size:12px; font-weight:750; border:1px solid #d9e0ea; background:#fff; color:#344054; } .legend-dot{ width:8px; height:8px; border-radius:999px; margin-right:6px; } .dot-gt{ background:#f97316; } .dot-llm{ background:#2563eb; } .dot-both{ background:#16a34a; } /* SECTION CARD */ .section{ background:var(--glass); border:1px solid var(--border); border-radius:18px; padding:18px 18px; margin:10px 0 16px; box-shadow: var(--ring); } /* BUTTONS */ .stButton > button { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; color: white !important; border-radius: 12px; font-size: 16px; font-weight: 800; padding: 0.70em 1.25em; border: none; letter-spacing:.2px; box-shadow: 0 10px 18px rgba(124,58,237,.20), 0 6px 14px rgba(6,182,212,.12); } .stButton > button:focus { outline: none; box-shadow: 0 0 0 2px rgba(124,58,237,.45); } /* TABLE polish */ .stDataFrame thead tr th { background:#eef2ff !important; color:#0f172a !important; font-weight:700; text-align:center; } .stDataFrame tbody tr:nth-child(even) { background-color: rgba(2, 6, 23, .03); } .pro-table-wrap{ border:1px solid rgba(15,23,42,.10); border-radius:12px; overflow:auto; background:#fff; box-shadow:0 10px 24px rgba(15,23,42,.06); margin:.6rem 0 1rem; } .pro-table{ width:100%; border-collapse:separate; border-spacing:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size:13px; line-height:1.35; color:#111827; } .pro-table th{ position:sticky; top:0; z-index:1; background:#f8fafc; color:#475569; text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.05em; font-weight:800; padding:11px 12px; border-bottom:1px solid #e5e7eb; white-space:nowrap; } .pro-table td{ padding:10px 12px; border-bottom:1px solid #eef2f7; vertical-align:middle; } .pro-table tr:hover td{ background:#f8fbff; } .pro-table td.num{ text-align:right; font-variant-numeric:tabular-nums; color:#0f172a; font-weight:650; } .pro-table .file-cell{ font-family:"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:12px; color:#1f2937; max-width:620px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .verdict-badge{ display:inline-flex; align-items:center; border-radius:999px; padding:4px 9px; font-size:12px; font-weight:800; border:1px solid transparent; white-space:nowrap; } .verdict-matched{ color:#166534; background:#dcfce7; border-color:#bbf7d0; } .verdict-partial,.verdict-mismatch{ color:#92400e; background:#fef3c7; border-color:#fde68a; } .verdict-missed{ color:#991b1b; background:#fee2e2; border-color:#fecaca; } .verdict-extra{ color:#6d28d9; background:#ede9fe; border-color:#ddd6fe; } .verdict-true_negative{ color:#334155; background:#e2e8f0; border-color:#cbd5e1; } .issue-pill{ display:inline-flex; border-radius:7px; padding:4px 7px; background:#f1f5f9; color:#334155; font-weight:650; white-space:nowrap; } .metric-bar{ min-width:92px; } .metric-bar-track{ height:7px; border-radius:999px; background:#e5e7eb; overflow:hidden; } .metric-bar-fill{ height:100%; border-radius:999px; background:linear-gradient(90deg,#22c55e,#06b6d4); } .metric-bar-label{ margin-top:3px; color:#475569; font-size:11px; font-variant-numeric:tabular-nums; text-align:right; } .muted{ color:var(--muted); } .pill{ display:inline-flex; align-items:center; gap:.45rem; padding:.32rem .65rem; border:1px solid var(--border); color:var(--text); border-radius:999px; font-size:.78rem; font-weight:600; background:linear-gradient(90deg,#ede9fe,#cffafe); margin:.20rem .35rem .55rem 0; } .footnote{ color:var(--muted); font-size:.85rem; margin-top:8px; } </style> """, unsafe_allow_html=True, )

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root{
  --ui-bg:#ffffff;
  --ui-bg-soft:#f8fafc;
  --ui-border:#d9e0ea;
  --ui-border-strong:#cbd5e1;
  --ui-text:#101828;
  --ui-muted:#667085;
  --ui-focus:#2563eb;
  --ui-shadow:0 8px 20px rgba(15,23,42,.06);
  --brand-font:Manrope, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --brand-mono:"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* Global typography rhythm */
.stApp,
.stApp *,
.stApp button,
.stApp input,
.stApp textarea,
.stApp select,
.stApp [data-baseweb],
.stApp [data-baseweb] *,
.stApp [data-testid],
.stApp [data-testid] *,
.stApp [role],
.stApp [role] *{
  font-family:var(--brand-font) !important;
  letter-spacing:0;
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{
  color:var(--ui-text);
  font-family:var(--brand-font) !important;
  font-weight:800;
  letter-spacing:-.01em;
}
.stApp h1{
  font-size:clamp(1.7rem, 2.4vw, 2.5rem);
}
.stApp h2{
  font-size:1.35rem;
}
.stApp h3{
  font-size:1.08rem;
}
.stMarkdown p, .stCaptionContainer, [data-testid="stCaptionContainer"]{
  color:var(--ui-muted);
  font-family:var(--brand-font) !important;
  line-height:1.55;
}
.stMarkdown strong,
.stMarkdown b{
  color:#172033;
  font-weight:800;
}
.stApp code,
.stApp pre,
.stApp kbd,
.stApp samp,
.stCode,
.stCode *,
.codebox,
.codebox *,
.file-cell{
  font-family:var(--brand-mono) !important;
}

/* DataFrames: professional shell around every Streamlit table */
[data-testid="stDataFrame"], .stDataFrame{
  border:1px solid var(--ui-border) !important;
  border-radius:12px !important;
  overflow:hidden !important;
  background:var(--ui-bg) !important;
  box-shadow:var(--ui-shadow) !important;
}
[data-testid="stDataFrame"] *{
  font-family:var(--brand-font) !important;
  font-size:13px !important;
}
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] thead th{
  background:var(--ui-bg-soft) !important;
  color:#475467 !important;
  font-weight:800 !important;
  text-transform:uppercase !important;
  letter-spacing:.045em !important;
  font-size:11px !important;
  border-bottom:1px solid var(--ui-border) !important;
}
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] tbody td{
  color:#1d2939 !important;
  font-weight:500 !important;
  border-color:#eef2f6 !important;
}

/* Forms and fields */
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stFileUploader"] label,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label,
[data-testid="stSlider"] label{
  color:#344054 !important;
  font-size:13px !important;
  font-weight:750 !important;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div{
  border:1px solid var(--ui-border) !important;
  border-radius:10px !important;
  background:var(--ui-bg) !important;
  color:var(--ui-text) !important;
  box-shadow:0 1px 2px rgba(16,24,40,.04) !important;
  min-height:40px !important;
}
[data-testid="stTextArea"] textarea{
  font-family:var(--brand-mono) !important;
  font-size:12.5px !important;
  line-height:1.55 !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-baseweb="select"] > div:focus-within{
  border-color:var(--ui-focus) !important;
  box-shadow:0 0 0 3px rgba(37,99,235,.14) !important;
}

/* Uploaders */
[data-testid="stFileUploader"] section{
  border:1px dashed var(--ui-border-strong) !important;
  border-radius:12px !important;
  background:linear-gradient(180deg,#fff,#f8fafc) !important;
  box-shadow:0 1px 2px rgba(16,24,40,.04) !important;
}
[data-testid="stFileUploader"] button{
  border-radius:9px !important;
  border:1px solid var(--ui-border-strong) !important;
  background:#fff !important;
  color:#344054 !important;
  font-weight:750 !important;
}

/* Tabs, expanders, radios */
[data-testid="stTabs"] [role="tablist"]{
  gap:4px;
  border-bottom:1px solid var(--ui-border);
}
[data-testid="stTabs"] [role="tab"]{
  color:#475467;
  font-weight:750;
  border-radius:9px 9px 0 0;
  padding:10px 14px;
}
[data-testid="stTabs"] [aria-selected="true"]{
  color:#1d4ed8 !important;
  background:#eff6ff;
}
[data-testid="stExpander"]{
  border:1px solid var(--ui-border) !important;
  border-radius:12px !important;
  overflow:hidden !important;
  background:#fff !important;
  box-shadow:0 1px 2px rgba(16,24,40,.04) !important;
}
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label p{
  color:#344054 !important;
  font-weight:650 !important;
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] div{
  font-family:var(--brand-font) !important;
}

/* Alerts */
[data-testid="stAlert"]{
  border-radius:12px !important;
  border:1px solid rgba(15,23,42,.08) !important;
  box-shadow:0 1px 2px rgba(16,24,40,.04) !important;
}

/* Buttons: keep primary action vivid, make utility buttons quieter */
.stDownloadButton button,
[data-testid="baseButton-secondary"]{
  border-radius:10px !important;
  border:1px solid var(--ui-border-strong) !important;
  background:#fff !important;
  color:#344054 !important;
  font-weight:800 !important;
  box-shadow:0 1px 2px rgba(16,24,40,.04) !important;
}
.stDownloadButton button:hover,
[data-testid="baseButton-secondary"]:hover{
  border-color:#94a3b8 !important;
  background:#f8fafc !important;
}
.stButton > button,
.stButton > button *,
[data-testid="baseButton-primary"],
[data-testid="baseButton-primary"] *{
  color:#ffffff !important;
  text-shadow:0 1px 1px rgba(15,23,42,.22) !important;
}
.stButton > button p,
[data-testid="baseButton-primary"] p{
  color:#ffffff !important;
}

/* Sidebar polish */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#ffffff,#f8fafc) !important;
  border-right:1px solid var(--ui-border) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{
  color:#475467;
}

/* Brand typography overrides for Streamlit/BaseWeb defaults */
[data-testid="stSidebar"] *,
[data-testid="stTabs"] *,
[data-testid="stExpander"] *,
[data-testid="stAlert"] *,
[data-testid="stMetric"] *,
[data-testid="stPopover"] *,
[data-testid="stTooltipHoverTarget"] *,
[data-baseweb="popover"] *,
[data-baseweb="menu"] *,
[data-baseweb="select"] *,
[data-baseweb="tag"] *,
[data-baseweb="slider"] *,
[data-baseweb="checkbox"] *,
[data-baseweb="radio"] *{
  font-family:var(--brand-font) !important;
}
[data-testid="stSidebar"]{
  font-size:13px;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
  letter-spacing:-.01em;
}
[data-testid="stTabs"] [role="tab"] p,
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label p,
[data-testid="stSelectbox"] label p,
[data-testid="stMultiSelect"] label p,
[data-testid="stTextInput"] label p,
[data-testid="stTextArea"] label p,
[data-testid="stFileUploader"] label p{
  font-family:var(--brand-font) !important;
}
.hero h1,
.hero p,
.badge,
.pill,
.kpi,
.kpi *,
.section,
.section *,
.pro-table,
.pro-table *{
  font-family:var(--brand-font) !important;
}
.pro-table .file-cell,
.codebox,
.codebox *,
textarea{
  font-family:var(--brand-mono) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Enterprise app shell */
.block-container{
  padding-top:1.1rem !important;
  max-width:100% !important;
}
.app-topbar{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:18px;
  margin:-18px -46px 18px;
  padding:13px 28px;
  background:#1f497d;
  border-bottom:1px solid rgba(255,255,255,.14);
  color:#fff;
  box-shadow:0 10px 24px rgba(15,23,42,.14);
}
.app-topbar-left,.app-topbar-right{
  display:flex;
  align-items:center;
  gap:10px;
  flex-wrap:wrap;
}
.app-brand{
  display:flex;
  align-items:center;
  gap:10px;
  font-weight:850;
  font-size:18px;
  letter-spacing:-.01em;
}
.app-brand img{
  width:28px;
  height:28px;
  border-radius:8px;
  box-shadow:0 4px 10px rgba(0,0,0,.18);
}
.nav-chip,.status-chip{
  display:inline-flex;
  align-items:center;
  gap:7px;
  min-height:30px;
  padding:6px 11px;
  border-radius:7px;
  background:rgba(255,255,255,.10);
  border:1px solid rgba(255,255,255,.16);
  color:#eaf2ff;
  font-size:13px;
  font-weight:750;
}
.nav-chip.active{
  background:#3f7fbd;
  color:#fff;
  box-shadow:inset 0 -2px 0 rgba(255,255,255,.18);
}
.status-chip{
  background:#2a5d91;
  color:#dbeafe;
}
.status-dot{
  width:8px;
  height:8px;
  border-radius:999px;
  background:#22c55e;
  box-shadow:0 0 0 3px rgba(34,197,94,.16);
}
[data-testid="stSidebar"]{
  background:#f3f6fa !important;
  border-right:1px solid #d9e2ec !important;
}
[data-testid="stSidebar"] > div:first-child{
  padding-top:1.15rem;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  gap:.75rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{
  color:#0f172a !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3{
  margin:0;
  padding:10px 12px;
  border-radius:8px 8px 0 0;
  background:#244f82;
  color:#fff !important;
  font-size:15px;
}
[data-testid="stTabs"]{
  margin-top:0;
}
[data-testid="stTabs"] [role="tablist"]{
  background:#244f82;
  padding:7px 8px;
  border-radius:0;
  border:none;
  gap:6px;
  margin:0 -46px 16px;
}
[data-testid="stTabs"] [role="tab"]{
  border-radius:7px;
  color:#dbeafe !important;
  padding:8px 13px;
  border:1px solid transparent;
}
[data-testid="stTabs"] [role="tab"] p{
  color:inherit !important;
  font-weight:800;
}
[data-testid="stTabs"] [aria-selected="true"]{
  background:#3f7fbd !important;
  color:#ffffff !important;
  border-color:rgba(255,255,255,.22);
}
.hero{
  border-radius:8px !important;
  box-shadow:0 8px 18px rgba(15,23,42,.08) !important;
  background:#ffffff !important;
}
.section{
  border-radius:8px !important;
  background:#ffffff !important;
  box-shadow:0 8px 18px rgba(15,23,42,.06) !important;
}
.stButton > button{
  background:#1f497d !important;
  border:1px solid #1f497d !important;
  color:#ffffff !important;
  border-radius:7px !important;
  box-shadow:0 8px 16px rgba(31,73,125,.18) !important;
}
.stButton > button:hover{
  background:#183b66 !important;
  border-color:#183b66 !important;
  color:#ffffff !important;
}
.stButton > button:focus{
  box-shadow:0 0 0 3px rgba(31,73,125,.22), 0 8px 16px rgba(31,73,125,.18) !important;
}
[data-testid="baseButton-primary"]{
  background:#1f497d !important;
  border-color:#1f497d !important;
}
[data-testid="baseButton-primary"]:hover{
  background:#183b66 !important;
  border-color:#183b66 !important;
}
[data-testid="stAlert"]{
  border-radius:8px !important;
}
@media (max-width: 900px){
  .app-topbar{
    margin-left:-22px;
    margin-right:-22px;
    align-items:flex-start;
    flex-direction:column;
  }
  [data-testid="stTabs"] [role="tablist"]{
    margin-left:-22px;
    margin-right:-22px;
    overflow:auto;
  }
}

/* Right-side settings panel */
section[data-testid="stSidebar"]{ display:none !important; }
.right-settings{
  border:1px solid #d9e2ec;
  border-radius:8px;
  background:#f8fafc;
  box-shadow:0 10px 24px rgba(15,23,42,.08);
  padding:0 14px 14px;
}
.right-settings-title{
  margin:0 -14px 12px;
  padding:11px 13px;
  border-radius:8px 8px 0 0;
  background:#1f497d;
  color:#fff;
  font-weight:850;
  font-size:15px;
}
@media (min-width: 901px){
  .block-container{
    padding-right:2.4rem !important;
    padding-left:2.4rem !important;
  }
  .app-topbar{
    margin-right:-46px;
  }
}
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def kpi(title: str, value: str, tone: str = "ok"):
    cls = {"ok":"kpi-ok", "warn":"kpi-warn", "bad":"kpi-bad"}.get(tone,"")
    st.markdown(f"""
    <div class="kpi {cls}">
      <div class="kpi-label">{title}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def pct(x: float) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "0.00%"

METRIC_INFO = {
    "f1": {
        "label": "Balanced detection score",
        "short": "Balance",
        "help": "Balances finding real analyzer issues and avoiding false alarms.",
        "format": "pct",
    },
    "precision": {
        "label": "Trustworthiness",
        "short": "Precision",
        "help": "Of the issues the LLM reports, how many match the analyzer ground truth.",
        "format": "pct",
    },
    "recall": {
        "label": "Coverage of real issues",
        "short": "Recall",
        "help": "Of the analyzer issues in the sampled files, how many the LLM finds.",
        "format": "pct",
    },
    "accuracy": {
        "label": "Overall agreement",
        "short": "Accuracy",
        "help": "Share of all human-oracle decisions that match the label.",
        "format": "pct",
    },
    "mrr": {
        "label": "First correct file rank",
        "short": "MRR",
        "help": "Rewards prompts that put a correct file near the very top of the ranked list.",
        "format": "score",
    },
    "map": {
        "label": "Overall ranking quality",
        "short": "MAP",
        "help": "Rewards ranking many correct files early, not just finding one.",
        "format": "score",
    },
    "hit@1": {
        "label": "Top answer is correct",
        "short": "Hit@1",
        "help": "Whether the first-ranked file is relevant.",
        "format": "pct",
    },
    "hit@5": {
        "label": "Correct file in top 5",
        "short": "Hit@5",
        "help": "Whether at least one relevant file appears in the first five predictions.",
        "format": "pct",
    },
    "recall@10": {
        "label": "Relevant files covered in top 10",
        "short": "Recall@10",
        "help": "Share of relevant files recovered within the first ten predictions.",
        "format": "pct",
    },
}

def _metric_label(metric: str, *, short: bool = False) -> str:
    info = METRIC_INFO.get(metric, {})
    return str(info.get("short" if short else "label") or metric)

def _metric_help(metric: str) -> str:
    return str((METRIC_INFO.get(metric) or {}).get("help") or "")

def _metric_choice_label(metric: str) -> str:
    return f"{_metric_label(metric)} ({_metric_label(metric, short=True)})"

def _format_metric(metric: str, value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    if (METRIC_INFO.get(metric) or {}).get("format") == "pct":
        return pct(number)
    return f"{number:.3f}"

def _sidebar_prompt_preview(title: str, prompts: dict[str, str], selected: str | list[str] | tuple[str, ...], target=None) -> None:
    selected_items = [selected] if isinstance(selected, str) else list(selected or [])
    if not selected_items:
        return
    target = target or st.sidebar
    with target.expander(title, expanded=False):
        for idx, name in enumerate(selected_items):
            if idx:
                st.divider()
            st.caption(str(name))
            st.code(str(prompts.get(name, ""))[:3500], language="text")

def _render_prompt_preview_panel(
    title: str,
    prompts: dict[str, str],
    selected: str | list[str] | tuple[str, ...],
    *,
    preview_builder=None,
) -> None:
    selected_items = [selected] if isinstance(selected, str) else list(selected or [])
    selected_items = [item for item in selected_items if item]
    if not selected_items:
        return
    st.markdown(f"### {title}")
    st.caption("Same preview pattern for every maintenance activity: inspect the prompt before launching the experiment.")
    tabs = st.tabs([str(item) for item in selected_items[:4]])
    for tab, name in zip(tabs, selected_items[:4]):
        with tab:
            prompt_text = preview_builder(name) if preview_builder else prompts.get(name, "")
            st.code(str(prompt_text or "")[:6000], language="text")
    if len(selected_items) > 4:
        st.caption(f"{len(selected_items) - 4} additional prompt(s) selected; only the first four are previewed here.")

PROMPT_REPOSITORY_FILE = Path(__file__).with_name("prompt_repository.json")
PROMPT_TASKS = {
    "code_smell_detection": {
        "label": "Code smell detection",
        "description": "Span-level detection of maintainability issues and code smells against analyzer ground truth.",
    },
    "feature_location": {
        "label": "Feature location",
        "description": "File-level ranking of source files that implement or refine a requested feature.",
    },
    "bug_location": {
        "label": "Bug location",
        "description": "File-level ranking of source files likely to be changed for a bug report.",
    },
}

def _prompt_entry(template: str, *, tags: list[str] | None = None, source: str = "user") -> dict[str, Any]:
    return {"template": str(template), "tags": tags or [], "source": source}

def _default_prompt_repository() -> dict[str, Any]:
    smell_prompts = {
        name: _prompt_entry(text, tags=["span-level", "json"], source="strategies.json")
        for name, text in load_strategies().items()
    }
    feature_prompts = {
        name: _prompt_entry(text, tags=["file-ranking", "json"], source="built-in")
        for name, text in LOCATION_PROMPT_TEMPLATES.items()
        if name in {"baseline", "feature_evidence", "terse_ranking"}
    }
    bug_prompts = {
        name: _prompt_entry(text, tags=["file-ranking", "json"], source="built-in")
        for name, text in LOCATION_PROMPT_TEMPLATES.items()
        if name in {"baseline", "bug_report", "terse_ranking"}
    }
    return {
        task_key: {
            "label": meta["label"],
            "description": meta["description"],
            "prompts": {
                "code_smell_detection": smell_prompts,
                "feature_location": feature_prompts,
                "bug_location": bug_prompts,
            }[task_key],
        }
        for task_key, meta in PROMPT_TASKS.items()
    }

def _load_prompt_repository() -> dict[str, Any]:
    repo = _default_prompt_repository()
    if PROMPT_REPOSITORY_FILE.exists():
        try:
            data = json.loads(PROMPT_REPOSITORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for task_key, task_data in data.items():
                    if task_key not in repo or not isinstance(task_data, dict):
                        continue
                    repo[task_key]["label"] = str(task_data.get("label") or repo[task_key]["label"])
                    repo[task_key]["description"] = str(task_data.get("description") or repo[task_key]["description"])
                    prompts = task_data.get("prompts") or {}
                    if isinstance(prompts, dict):
                        for name, entry in prompts.items():
                            if isinstance(entry, dict):
                                repo[task_key]["prompts"][str(name)] = {
                                    "template": str(entry.get("template", "")),
                                    "tags": list(entry.get("tags") or []),
                                    "source": str(entry.get("source") or "repository"),
                                }
                            else:
                                repo[task_key]["prompts"][str(name)] = _prompt_entry(str(entry), source="legacy")
        except Exception:
            pass
    return repo

def _save_prompt_repository(repo: dict[str, Any]) -> None:
    PROMPT_REPOSITORY_FILE.write_text(json.dumps(repo, indent=2, ensure_ascii=False), encoding="utf-8")
    smell_prompts = _prompt_templates_for_task("code_smell_detection", repo=repo)
    if smell_prompts:
        save_strategies(smell_prompts)

def _prompt_templates_for_task(task_key: str, *, repo: dict[str, Any] | None = None) -> dict[str, str]:
    source = repo or _load_prompt_repository()
    prompts = ((source.get(task_key) or {}).get("prompts") or {})
    return {str(name): str((entry or {}).get("template", "")) for name, entry in prompts.items() if isinstance(entry, dict)}


def _dacos_prompt_templates() -> dict[str, str]:
    return {
        "binary_oracle_baseline": DACOS_PROMPT_TEMPLATE,
        "conservative_evidence": """You are evaluating a human code-smell oracle.

Question: is the target smell present in this snippet?
Target smell: {smell}
Language: {language}

Use only visible evidence in the snippet. Do not report unrelated smells.
Return JSON only:
{{"present": true|false, "confidence": 0.0, "rationale": "one short evidence sentence"}}

Code:
{code}
""",
        "definition_driven": """You are a software-maintenance researcher.

Decide whether the code contains this exact smell: {smell}.

Interpretation guide:
- complex_method: a method is too complex because of branching, nested logic, mixed responsibilities, or difficult control flow.
- long_method: a method is too long or does too much to be easily understood.
- long_parameter_list: a method/constructor exposes too many parameters or parameter groups.
- multifaceted_abstraction: a class mixes multiple responsibilities or represents an overly broad abstraction.
- god_class: a class centralizes too many responsibilities and collaborators.
- data_class: a class mostly stores data with little meaningful behavior.
- feature_envy: a method relies more on data or behavior from another class than its own class.

Return strict JSON only, no markdown:
{{"present": true|false, "confidence": 0.0, "rationale": "specific evidence or why absent"}}

Language: {language}
Code:
{code}
""",
    }

def _render_bar_chart(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str | None = None,
    y_title: str | None = None,
    value_format: str = ".2f",
    height: int = 320,
) -> None:
    if df.empty:
        st.info("No chart data available.")
        return
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x}:N", sort=None, axis=alt.Axis(labelAngle=-25, title=None)),
            y=alt.Y(f"{y}:Q", title=y_title or y, axis=alt.Axis(grid=True)),
            tooltip=[
                alt.Tooltip(f"{x}:N", title=x),
                alt.Tooltip(f"{y}:Q", title=y_title or y, format=value_format),
            ],
        )
        .properties(height=height)
        .configure_axis(
            labelFont=CHART_FONT,
            titleFont=CHART_FONT,
            labelColor="#475569",
            titleColor="#334155",
            gridColor="#e2e8f0",
        )
        .configure_legend(
            labelFont=CHART_FONT,
            titleFont=CHART_FONT,
            labelColor="#334155",
        )
        .configure_view(stroke=None)
    )
    if color:
        chart = chart.encode(
            color=alt.Color(
                f"{color}:N",
                scale=alt.Scale(
                    range=["#2563eb", "#14b8a6", "#7c3aed", "#f59e0b", "#ef4444", "#64748b"]
                ),
                legend=alt.Legend(title=None, orient="top"),
            )
        )
    else:
        chart = chart.encode(color=alt.value("#2563eb"))
    st.altair_chart(chart, use_container_width=True)

def _render_wide_metric_chart(
    df: pd.DataFrame,
    columns: list[str],
    *,
    group_label: str,
    value_label: str,
    value_format: str = ".2f",
    height: int = 330,
) -> None:
    plot_df = df[columns].copy()
    plot_df.index = plot_df.index.astype(str)
    plot_df = (
        plot_df.reset_index(names=group_label)
        .melt(id_vars=group_label, var_name="Metric", value_name=value_label)
    )
    base = (
        alt.Chart(plot_df)
        .encode(
            x=alt.X(
                f"{group_label}:N",
                sort=None,
                axis=alt.Axis(labelAngle=-25, title=None),
            ),
            xOffset=alt.XOffset("Metric:N", sort=columns),
            y=alt.Y(
                f"{value_label}:Q",
                title=value_label,
                axis=alt.Axis(format=value_format if value_format.endswith("%") else None, grid=True),
                scale=alt.Scale(zero=True),
            ),
            color=alt.Color(
                "Metric:N",
                sort=columns,
                scale=alt.Scale(
                    domain=columns,
                    range=["#2563eb", "#14b8a6", "#7c3aed", "#f59e0b", "#ef4444", "#64748b"][:len(columns)],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip(f"{group_label}:N", title=group_label),
                alt.Tooltip("Metric:N", title="Metric"),
                alt.Tooltip(f"{value_label}:Q", title=value_label, format=value_format),
            ],
        )
        .properties(height=height)
    )
    bars = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=28)
    labels = base.mark_text(
        dy=-8,
        font=CHART_FONT,
        fontSize=12,
        fontWeight="bold",
        color="#334155",
    ).encode(
        text=alt.Text(f"{value_label}:Q", format=value_format)
    )
    chart = (
        (bars + labels)
        .configure_axis(
            labelFont=CHART_FONT,
            titleFont=CHART_FONT,
            labelColor="#475569",
            titleColor="#334155",
            gridColor="#e2e8f0",
        )
        .configure_legend(
            labelFont=CHART_FONT,
            titleFont=CHART_FONT,
            labelColor="#334155",
        )
        .configure_view(stroke=None)
    )
    st.altair_chart(chart, use_container_width=True)

def render_code_with_highlight(title: str, code: str, hl_ranges: list[tuple[int,int]]):
    """
    Render code with 1-based line numbers and highlight ranges inclusive.
    hl_ranges: list of (start,end) inclusive; use (n,n) for single line.
    """
    lines = code.splitlines()
    hl_set = set()
    for a,b in hl_ranges:
        if a > b: a,b = b,a
        for i in range(max(1,a), min(len(lines), b)+1):
            hl_set.add(i)
    rows_html = []
    for idx, txt in enumerate(lines, start=1):
        safe = txt.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        cls = "line hl" if idx in hl_set else "line"
        rows_html.append(
            f"<tr><td class='gutter ln'>{idx}</td><td class='code {cls}'><code>{safe}</code></td></tr>"
        )
    html = f"""
    <div class='section'><b>{title}</b>
    <div class='codebox'><table>{"".join(rows_html)}</table></div></div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_code_context(title: str, excerpt: dict[str, Any]) -> None:
    lines = excerpt.get("lines") or []
    if not lines:
        st.caption("No code context available for this file. Rerun the analysis if this result was produced before code context was added.")
        return
    start_line = int(excerpt.get("start_line") or 1)
    gt_lines = set(int(x) for x in excerpt.get("gt_lines", []) if x)
    llm_lines = set(int(x) for x in excerpt.get("llm_lines", []) if x)
    rows_html = []
    for offset, txt in enumerate(lines):
        line_no = start_line + offset
        safe = escape(str(txt))
        if line_no in gt_lines and line_no in llm_lines:
            cls = "line both"
        elif line_no in gt_lines:
            cls = "line gt"
        elif line_no in llm_lines:
            cls = "line llm"
        else:
            cls = "line"
        rows_html.append(
            f"<tr><td class='gutter ln'>{line_no}</td><td class='code {cls}'><code>{safe}</code></td></tr>"
        )
    st.markdown(f"""
    <div class='section'><b>{escape(title)}</b>
      <div class='code-legend'>
        <span class='legend-chip'><span class='legend-dot dot-gt'></span>Ground truth</span>
        <span class='legend-chip'><span class='legend-dot dot-llm'></span>LLM finding</span>
        <span class='legend-chip'><span class='legend-dot dot-both'></span>Overlap</span>
      </div>
      <div class='codebox'><table>{"".join(rows_html)}</table></div>
    </div>
    """, unsafe_allow_html=True)

def _demo_datasets() -> dict[str, tuple[Path, Path]]:
    base = Path(__file__).parent / "data" / "apps"
    candidates = {
        "ArgoUML (Java / SonarQube)": (
            base / "ArgoUML" / "ArgoUML.zip",
            base / "ArgoUML" / "ArgoUml-sonarqube-quality-analysis.csv",
        ),
        "eShopOnWeb Catalog (C# / SonarQube)": (
            base / "eShopOnWeb" / "Catalog.zip",
            base / "eShopOnWeb" / "eShopOnContainers-sonarqube-quality-analysis.csv",
        ),
        "Magento (PHP / SonarQube)": (
            base / "Magento" / "magento.zip",
            base / "Magento" / "Magento-sonarqube-quality-analysis.csv",
        ),
    }
    return {name: paths for name, paths in candidates.items() if paths[0].exists() and paths[1].exists()}

def _llm_preflight(llm_obj: ChatModel | None) -> tuple[bool, str]:
    if llm_obj is None:
        return False, "No LLM selected."
    try:
        cfg = llm_obj.cfg
        provider = (cfg.provider or "").lower()
        if provider == "azure-openai":
            if not cfg.model:
                return False, "Azure deployment name is missing."
            if not (cfg.api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")):
                return False, "Azure/OpenAI API key is missing."
            if not (cfg.api_base or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE")):
                return False, "Azure endpoint is missing."
        elif provider == "openai":
            if not (cfg.api_key or os.getenv("OPENAI_API_KEY")):
                return False, "OpenAI API key is missing."
            if not cfg.model:
                return False, "OpenAI model is missing."
        elif provider == "ollama":
            if not cfg.model:
                return False, "Ollama model is missing."
            if not (cfg.api_base or getattr(llm_obj, "ollama_host", None) or os.getenv("OLLAMA_HOST")):
                return False, "Ollama host is missing."
        else:
            return False, f"Unknown provider: {provider or 'empty'}."
        return True, llm_obj.model_label()
    except Exception as e:
        return False, str(e)

def _preflight_inputs(zip_name: str | None, zip_bytes: bytes | None,
                      csv_name: str | None, csv_bytes: bytes | None,
                      top_k: int, pos_ratio: float,
                      llm_obj: ChatModel | None) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {
        "zip": {"ok": False, "message": "No ZIP selected."},
        "csv": {"ok": False, "message": "No CSV selected."},
        "language": {"ok": False, "message": "Unknown"},
        "universe": {"ok": False, "message": "Waiting for ZIP + CSV."},
        "llm": {"ok": False, "message": "No LLM selected."},
    }

    llm_ok, llm_msg = _llm_preflight(llm_obj)
    report["llm"] = {"ok": llm_ok, "message": llm_msg}

    if not zip_bytes:
        return False, report
    if not csv_bytes:
        return False, report

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, zip_name or "project.zip")
        csv_path = os.path.join(tmpdir, csv_name or "ground_truth.csv")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        with open(csv_path, "wb") as f:
            f.write(csv_bytes)

        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                files = [n for n in z.namelist() if not n.endswith("/") and not n.startswith("__MACOSX/")]
            report["zip"] = {"ok": True, "message": f"{len(files):,} files found."}
        except Exception as e:
            report["zip"] = {"ok": False, "message": f"Invalid ZIP: {e}"}
            return False, report

        try:
            language = detect_language_from_zip(zip_path)
            exts = find_exts_for_language(language)
            report["language"] = {
                "ok": language != "Unknown",
                "message": f"{language} ({', '.join(exts) if exts else 'no known extensions'})",
            }
        except Exception as e:
            report["language"] = {"ok": False, "message": f"Could not detect language: {e}"}
            exts = []

        try:
            df = _read_csv_robust(csv_path)
            report["csv"] = {"ok": True, "message": f"{len(df):,} rows, {len(df.columns)} columns."}
            gt = load_ground_truth_spans(csv_path, allowed_exts=exts or None)
            if gt.empty:
                report["universe"] = {"ok": False, "message": "No GT rows match detected source extensions."}
            else:
                available_gt_files = int(gt["basename"].nunique())
                effective_top_k = min(int(top_k), max(available_gt_files, int(round(top_k * pos_ratio))))
                report["universe"] = {
                    "ok": True,
                    "message": f"{available_gt_files:,} GT files mapped before sampling; requested Top-K={top_k}, positive ratio={pos_ratio:.2f}.",
                    "effective_top_k": effective_top_k,
                }
        except Exception as e:
            report["csv"] = {"ok": False, "message": f"CSV/GT error: {e}"}

    ok = all(report[k]["ok"] for k in ["zip", "csv", "language", "universe", "llm"])
    return ok, report

def _render_preflight(report: dict[str, Any]) -> None:
    st.markdown("### 🧪 Preflight checks")
    rows = []
    for label, key in [
        ("Project ZIP", "zip"),
        ("Static analyzer CSV", "csv"),
        ("Detected language", "language"),
        ("Experiment universe", "universe"),
        ("LLM configuration", "llm"),
    ]:
        item = report.get(key, {})
        rows.append({
            "Check": label,
            "Status": "OK" if item.get("ok") else "Needs attention",
            "Details": item.get("message", ""),
        })
    _render_pro_dataframe(pd.DataFrame(rows), hide_index=True)

def _fmt_span_examples(items: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items or []:
        start = item.get("startLine")
        end = item.get("endLine")
        if start and end and start != end:
            loc = f"{start}-{end}"
        elif start:
            loc = str(start)
        else:
            loc = ""
        rows.append({
            "Lines": loc,
            "Type": item.get("type") or "",
            "Description": item.get("description") or "",
        })
    return pd.DataFrame(rows)

def _safe_ratio(num: Any, den: Any) -> float:
    try:
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    except Exception:
        return 0.0

def _file_main_issue(row: pd.Series) -> str:
    tp = int(row.get("tp", 0))
    fp = int(row.get("fp", 0))
    fn = int(row.get("fn", 0))
    gt = int(row.get("gt_spans", 0))
    pred = int(row.get("llm_spans", 0))
    bucket = str(row.get("bucket", ""))
    if bucket == "matched":
        return "Good overlap"
    if bucket == "missed":
        return "LLM missed all GT spans"
    if bucket == "extra":
        return "LLM-only findings"
    if fn > fp and fn >= max(3, tp):
        return "Many false negatives"
    if fp > fn and fp >= max(3, tp):
        return "Many false positives"
    if gt > pred and fn:
        return "Under-detected"
    if pred > gt and fp:
        return "Over-detected"
    return "Needs review"

def _review_summary(df: pd.DataFrame) -> str:
    counts = df["bucket"].value_counts().to_dict()
    tp = int(df["tp"].sum())
    fp = int(df["fp"].sum())
    fn = int(df["fn"].sum())
    total = int(len(df))
    matched = int(counts.get("matched", 0))
    partial = int(counts.get("partial", 0) + counts.get("mismatch", 0))
    missed = int(counts.get("missed", 0))
    extra = int(counts.get("extra", 0))
    if fn >= fp and fn > tp:
        driver = "Most remaining error comes from false negatives, so recall is the main weakness."
    elif fp > fn and fp > tp:
        driver = "Most remaining error comes from false positives, so precision is the main weakness."
    elif tp:
        driver = "The model has useful overlap, but the file-level details still need review."
    else:
        driver = "There is no confirmed overlap in this sampled universe."
    return (
        f"Across {total} sampled files, the LLM fully matched {matched}, partially covered {partial}, "
        f"missed {missed}, and produced LLM-only findings on {extra}. "
        f"Span totals: TP={tp}, FP={fp}, FN={fn}. {driver}"
    )

def _safe_key(value: Any) -> str:
    return "".join(ch if str(ch).isalnum() else "_" for ch in str(value))

def _metric_bar(value: Any) -> str:
    val = max(0.0, min(1.0, _safe_ratio(value, 1)))
    return (
        "<div class='metric-bar'>"
        "<div class='metric-bar-track'>"
        f"<div class='metric-bar-fill' style='width:{val * 100:.1f}%'></div>"
        "</div>"
        f"<div class='metric-bar-label'>{val:.2f}</div>"
        "</div>"
    )

def _render_pro_review_table(df: pd.DataFrame, cols: list[str]) -> None:
    verdict_class = {
        "Matched": "verdict-matched",
        "Partial": "verdict-partial",
        "Missed": "verdict-missed",
        "Extra": "verdict-extra",
        "Mismatch": "verdict-mismatch",
        "True negative": "verdict-true_negative",
    }
    headers = "".join(f"<th>{escape(col)}</th>" for col in cols)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in cols:
            value = row.get(col, "")
            if col == "Verdict":
                cls = verdict_class.get(str(value), "verdict-true_negative")
                cells.append(f"<td><span class='verdict-badge {cls}'>{escape(str(value))}</span></td>")
            elif col == "Main issue":
                cells.append(f"<td><span class='issue-pill'>{escape(str(value))}</span></td>")
            elif col == "File":
                cells.append(f"<td><div class='file-cell' title='{escape(str(value))}'>{escape(str(value))}</div></td>")
            elif col in {"File precision", "File recall", "File F1"}:
                cells.append(f"<td>{_metric_bar(value)}</td>")
            elif col in {"GT spans", "LLM spans", "TP", "FP", "FN"}:
                cells.append(f"<td class='num'>{escape(str(value))}</td>")
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f"<div class='pro-table-wrap'><table class='pro-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )

def _render_code_evidence_viewer(diagnostics: list[dict[str, Any]], key_prefix: str) -> None:
    if not diagnostics:
        return
    evidence_items = [
        item for item in diagnostics
        if (item.get("gt_code_excerpt") or item.get("llm_code_excerpt") or item.get("code_excerpt"))
    ]
    if not evidence_items:
        return

    st.markdown("### 🔎 Code evidence viewer")
    st.caption("Pick a file to inspect the ground-truth context and LLM context side by side.")
    labels = [
        f"{item.get('bucket', 'file')} · {item.get('file')} · TP {item.get('tp', 0)} / FP {item.get('fp', 0)} / FN {item.get('fn', 0)}"
        for item in evidence_items
    ]
    selected_label = st.selectbox("File to inspect", labels, key=f"{key_prefix}_evidence_file")
    selected = evidence_items[labels.index(selected_label)]

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Verdict", str(selected.get("bucket", "n/a")).replace("_", " ").title(), "warn")
    with c2: kpi("GT spans", str(selected.get("gt_spans", 0)), "ok")
    with c3: kpi("LLM spans", str(selected.get("llm_spans", 0)), "warn")
    with c4: kpi("TP / FP / FN", f"{selected.get('tp', 0)} / {selected.get('fp', 0)} / {selected.get('fn', 0)}", "bad")

    left, right = st.columns(2)
    with left:
        render_code_context("Ground Truth focus", selected.get("gt_code_excerpt") or {})
    with right:
        render_code_context("LLM focus", selected.get("llm_code_excerpt") or {})

def _format_table_value(value: Any) -> str:
    if not isinstance(value, (list, dict, tuple)):
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
    if isinstance(value, bool):
        cls = "verdict-matched" if value else "verdict-true_negative"
        label = "Yes" if value else "No"
        return f"<span class='verdict-badge {cls}'>{label}</span>"
    if isinstance(value, float):
        if 0 <= value <= 1:
            return f"{value:.4f}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    text = str(value)
    if len(text) > 140:
        short = text[:137] + "..."
        return f"<span title='{escape(text)}'>{escape(short)}</span>"
    return escape(text)

def _render_pro_dataframe(
    df: pd.DataFrame,
    *,
    hide_index: bool = False,
    max_rows: int = 200,
    monospace_cols: tuple[str, ...] = ("file", "filename", "path", "relpath", "basename", "File", "Filename", "Path"),
) -> None:
    if df is None:
        return
    data = df.copy()
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if data.empty:
        st.info("No rows to display.")
        return

    truncated = len(data) > max_rows
    if truncated:
        data = data.head(max_rows)

    display = data.reset_index() if not hide_index else data.reset_index(drop=True)
    cols = list(display.columns)
    headers = "".join(f"<th>{escape(str(col))}</th>" for col in cols)
    rows = []
    for _, row in display.iterrows():
        cells = []
        for col in cols:
            raw = row.get(col, "")
            col_name = str(col)
            is_num = isinstance(raw, (int, float)) and not isinstance(raw, bool)
            cls = "num" if is_num else ""
            value_html = _format_table_value(raw)
            if col_name in monospace_cols or any(token in col_name.lower() for token in ("path", "file", "base")):
                value_html = f"<div class='file-cell' title='{escape(str(raw))}'>{value_html}</div>"
            cells.append(f"<td class='{cls}'>{value_html}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    note = ""
    if truncated:
        note = f"<div class='footnote'>Showing first {max_rows:,} rows of {len(df):,}.</div>"
    st.markdown(
        f"<div class='pro-table-wrap'><table class='pro-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>{note}",
        unsafe_allow_html=True,
    )

def _report_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No rows.</p>"
    data = df.copy()
    truncated = len(data) > max_rows
    if truncated:
        data = data.head(max_rows)
    headers = "".join(f"<th>{escape(str(col))}</th>" for col in data.columns)
    rows = []
    for _, row in data.iterrows():
        cells = "".join(f"<td>{_format_table_value(row.get(col, ''))}</td>" for col in data.columns)
        rows.append(f"<tr>{cells}</tr>")
    note = f"<p class='muted'>Showing first {max_rows:,} rows of {len(df):,}.</p>" if truncated else ""
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>{note}"

def _review_df_from_metrics(metrics: dict[str, Any]) -> pd.DataFrame:
    diagnostics = metrics.get("file_diagnostics") or []
    if not diagnostics:
        return pd.DataFrame()
    df = pd.DataFrame(diagnostics)
    labels = {
        "matched": "Matched",
        "partial": "Partial",
        "missed": "Missed",
        "extra": "Extra",
        "mismatch": "Mismatch",
        "true_negative": "True negative",
    }
    df["Verdict"] = df["bucket"].map(labels).fillna(df["bucket"])
    df["Main issue"] = df.apply(_file_main_issue, axis=1)
    df["File"] = df["file"]
    df["GT spans"] = df["gt_spans"]
    df["LLM spans"] = df["llm_spans"]
    df["TP"] = df["tp"]
    df["FP"] = df["fp"]
    df["FN"] = df["fn"]
    df["File precision"] = df.apply(lambda r: _safe_ratio(r["tp"], r["tp"] + r["fp"]), axis=1)
    df["File recall"] = df.apply(lambda r: _safe_ratio(r["tp"], r["tp"] + r["fn"]), axis=1)
    df["File F1"] = df.apply(
        lambda r: _safe_ratio(2 * r["File precision"] * r["File recall"], r["File precision"] + r["File recall"]),
        axis=1,
    )
    return df[["Verdict", "Main issue", "File", "GT spans", "LLM spans", "TP", "FP", "FN", "File precision", "File recall", "File F1"]]

def _report_shell(title: str, body: str) -> str:
    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{escape(title)}</title>
<style>
body{{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;background:#f6f7fb;color:#101828;margin:0;padding:36px;}}
.page{{max-width:1180px;margin:0 auto;}}
h1{{font-size:34px;line-height:1.15;margin:0 0 8px;font-weight:850;}}
h2{{font-size:20px;margin:30px 0 10px;font-weight:800;}}
.muted{{color:#667085;font-size:13px;}}
.summary{{background:#fff;border:1px solid #d9e0ea;border-radius:14px;padding:16px 18px;box-shadow:0 8px 20px rgba(15,23,42,.06);margin:18px 0;}}
.kpis{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:10px;margin:16px 0;}}
.kpi{{background:#fff;border:1px solid #d9e0ea;border-radius:12px;padding:12px;box-shadow:0 4px 14px rgba(15,23,42,.05);}}
.kpi .label{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#667085;font-weight:800;}}
.kpi .value{{font-size:22px;font-weight:850;color:#101828;margin-top:4px;}}
table{{width:100%;border-collapse:separate;border-spacing:0;background:#fff;border:1px solid #d9e0ea;border-radius:12px;overflow:hidden;box-shadow:0 8px 20px rgba(15,23,42,.06);margin:10px 0 18px;}}
th{{background:#f8fafc;color:#475467;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.045em;font-weight:850;padding:10px 12px;border-bottom:1px solid #d9e0ea;}}
td{{padding:9px 12px;border-bottom:1px solid #eef2f6;font-size:13px;vertical-align:top;}}
tr:last-child td{{border-bottom:0;}}
code{{font-family:"SFMono-Regular",Consolas,monospace;font-size:12px;}}
pre{{white-space:pre-wrap;background:#0f172a;color:#e2e8f0;border-radius:12px;padding:14px;overflow:auto;}}
.code-line{{display:block;font-family:"SFMono-Regular",Consolas,monospace;font-size:12px;line-height:1.55;padding:0 8px;}}
.code-line.gt{{background:#fff7ed;border-left:3px solid #f97316;}}
.code-line.llm{{background:#eff6ff;border-left:3px solid #2563eb;}}
.code-line.both{{background:#ecfdf5;border-left:3px solid #16a34a;}}
.ln{{display:inline-block;width:48px;color:#64748b;user-select:none;}}
.code-pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px;}}
@media(max-width:900px){{.code-pair{{grid-template-columns:1fr;}}}}
</style>
</head>
<body><main class="page">
<h1>{escape(title)}</h1>
<p class="muted">Generated by StaLLM on {escape(generated)}</p>
{body}
</main></body></html>"""

def _kpi_html(items: dict[str, Any]) -> str:
    cards = []
    for label, value in items.items():
        cards.append(f"<div class='kpi'><div class='label'>{escape(str(label))}</div><div class='value'>{escape(str(value))}</div></div>")
    return f"<div class='kpis'>{''.join(cards)}</div>"

def _report_code_excerpt(excerpt: dict[str, Any]) -> str:
    lines = excerpt.get("lines") or []
    if not lines:
        return "<p class='muted'>No code context available.</p>"
    start_line = int(excerpt.get("start_line") or 1)
    gt_lines = set(int(x) for x in excerpt.get("gt_lines", []) if x)
    llm_lines = set(int(x) for x in excerpt.get("llm_lines", []) if x)
    rendered = []
    for offset, text in enumerate(lines):
        line_no = start_line + offset
        if line_no in gt_lines and line_no in llm_lines:
            cls = "both"
        elif line_no in gt_lines:
            cls = "gt"
        elif line_no in llm_lines:
            cls = "llm"
        else:
            cls = ""
        rendered.append(f"<span class='code-line {cls}'><span class='ln'>{line_no}</span>{escape(str(text))}</span>")
    return f"<pre>{''.join(rendered)}</pre>"

def _report_code_contexts(metrics: dict[str, Any], limit: int = 8) -> str:
    diagnostics = metrics.get("file_diagnostics") or []
    parts = []
    for item in diagnostics[:limit]:
        gt_excerpt = item.get("gt_code_excerpt") or item.get("code_excerpt", {})
        llm_excerpt = item.get("llm_code_excerpt") or item.get("code_excerpt", {})
        if not gt_excerpt.get("lines") and not llm_excerpt.get("lines"):
            continue
        parts.append(
            f"<h2>Code Context: {escape(str(item.get('file', 'file')))}</h2>"
            "<div class='code-pair'>"
            f"<div><h3>Ground Truth focus</h3>{_report_code_excerpt(gt_excerpt)}</div>"
            f"<div><h3>LLM focus</h3>{_report_code_excerpt(llm_excerpt)}</div>"
            "</div>"
        )
    return "".join(parts)

def _single_report_html(summary_U: pd.DataFrame, metrics: dict[str, Any], usage_totals: dict[str, Any],
                        *, strategy: str, language: str, model_label: str, preset: str) -> str:
    review_df = _review_df_from_metrics(metrics)
    summary_text = _review_summary(pd.DataFrame(metrics.get("file_diagnostics", []))) if metrics.get("file_diagnostics") else ""
    config_df = pd.DataFrame([{
        "Strategy": strategy,
        "Language": language,
        "Model": model_label,
        "Preset": preset,
        "Top-K": metrics.get("top_k_total_files", ""),
        "Require type": metrics.get("require_type", ""),
        "Use columns": metrics.get("use_cols_single", ""),
    }])
    metrics_df = pd.DataFrame([{
        "Precision": metrics.get("precision", 0),
        "Recall": metrics.get("recall", 0),
        "F1": metrics.get("f1", 0),
        "TP": metrics.get("tp", 0),
        "FP": metrics.get("fp", 0),
        "FN": metrics.get("fn", 0),
        "Prompt tokens": usage_totals.get("prompt_tokens", 0),
        "Completion tokens": usage_totals.get("completion_tokens", 0),
        "Total tokens": usage_totals.get("total_tokens", 0),
        "USD cost": usage_totals.get("usd_cost", 0),
        "Time (s)": metrics.get("time_s", ""),
    }])
    body = (
        f"<div class='summary'>{escape(summary_text)}</div>"
        + _kpi_html({
            "Precision": pct(metrics.get("precision", 0)),
            "Recall": pct(metrics.get("recall", 0)),
            "F1": pct(metrics.get("f1", 0)),
            "TP": metrics.get("tp", 0),
            "FP": metrics.get("fp", 0),
            "FN": metrics.get("fn", 0),
        })
        + "<h2>Run Configuration</h2>" + _report_table(config_df)
        + "<h2>Metrics</h2>" + _report_table(metrics_df)
        + "<h2>Universe U</h2>" + _report_table(summary_U)
        + "<h2>Review Board</h2>" + _report_table(review_df)
        + _report_code_contexts(metrics)
    )
    return _report_shell("StaLLM Run Report", body)

def _comparison_report_html(summary_U: pd.DataFrame, metrics_df: pd.DataFrame, samples: dict[str, Any], *, label: str) -> str:
    clean_metrics = _metrics_table_view(metrics_df)
    best_f1 = clean_metrics["f1"].idxmax() if "f1" in clean_metrics.columns and not clean_metrics.empty else "n/a"
    body = (
        f"<div class='summary'>Comparison across {len(clean_metrics)} {escape(label)} item(s). Best F1: <b>{escape(str(best_f1))}</b>.</div>"
        + "<h2>Comparison Metrics</h2>" + _report_table(clean_metrics.reset_index())
        + "<h2>Universe U</h2>" + _report_table(summary_U)
    )
    if "file_diagnostics" in metrics_df.columns:
        for name, row in metrics_df.iterrows():
            metrics = row.to_dict()
            review_df = _review_df_from_metrics(metrics)
            if not review_df.empty:
                body += f"<h2>Review Board: {escape(str(name))}</h2>" + _report_table(review_df)
                body += _report_code_contexts(metrics, limit=4)
    if samples:
        sample_rows = [{"Item": k, "Sample findings": str(v[:3] if isinstance(v, list) else v)} for k, v in samples.items()]
        body += "<h2>Sample Findings</h2>" + _report_table(pd.DataFrame(sample_rows))
    return _report_shell("StaLLM Comparison Report", body)

def _report_download_button(html: str, *, file_name: str, key: str) -> None:
    st.download_button(
        "Export HTML report",
        data=html.encode("utf-8"),
        file_name=file_name,
        mime="text/html",
        key=key,
    )

def _render_review_board(metrics: dict[str, Any], key_prefix: str = "review") -> None:
    diagnostics = metrics.get("file_diagnostics") or []
    if not diagnostics:
        return

    df = pd.DataFrame(diagnostics)
    labels = {
        "matched": "Matched",
        "partial": "Partial",
        "missed": "Missed",
        "extra": "Extra",
        "mismatch": "Mismatch",
        "true_negative": "True negative",
    }
    df["Verdict"] = df["bucket"].map(labels).fillna(df["bucket"])
    df["File"] = df["file"]
    df["GT spans"] = df["gt_spans"]
    df["LLM spans"] = df["llm_spans"]
    df["TP"] = df["tp"]
    df["FP"] = df["fp"]
    df["FN"] = df["fn"]
    df["File precision"] = df.apply(lambda r: _safe_ratio(r["tp"], r["tp"] + r["fp"]), axis=1)
    df["File recall"] = df.apply(lambda r: _safe_ratio(r["tp"], r["tp"] + r["fn"]), axis=1)
    df["File F1"] = df.apply(
        lambda r: _safe_ratio(2 * r["File precision"] * r["File recall"], r["File precision"] + r["File recall"]),
        axis=1,
    )
    df["Main issue"] = df.apply(_file_main_issue, axis=1)
    severity_order = {"mismatch": 0, "missed": 1, "partial": 2, "extra": 3, "matched": 4, "true_negative": 5}
    df["_severity"] = df["bucket"].map(severity_order).fillna(9).astype(int)
    df["_error_total"] = df["FP"] + df["FN"]
    df = df.sort_values(["_severity", "_error_total", "File"], ascending=[True, False, True])

    st.markdown("### 🧭 Review board")
    st.caption("File-by-file diagnostic over the sampled universe U. TP/FP/FN are span-level counts; verdicts group files for fast inspection.")
    st.info(_review_summary(df))

    counts = df["bucket"].value_counts().to_dict()
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Matched files", str(int(counts.get("matched", 0))), "ok")
    with c2: kpi("Partial / mismatch", str(int(counts.get("partial", 0) + counts.get("mismatch", 0))), "warn")
    with c3: kpi("Missed files", str(int(counts.get("missed", 0))), "bad")
    with c4: kpi("Extra files", str(int(counts.get("extra", 0))), "warn")

    filter_choice = st.selectbox(
        "Review filter",
        ["All files", "Only errors", "False negatives", "False positives", "Matched only"],
        index=0,
        key=f"{key_prefix}_filter",
    )
    visible_df = df.copy()
    if filter_choice == "Only errors":
        visible_df = visible_df[(visible_df["FP"] > 0) | (visible_df["FN"] > 0)]
    elif filter_choice == "False negatives":
        visible_df = visible_df[visible_df["FN"] > 0]
    elif filter_choice == "False positives":
        visible_df = visible_df[visible_df["FP"] > 0]
    elif filter_choice == "Matched only":
        visible_df = visible_df[visible_df["bucket"] == "matched"]

    board_cols = [
        "Verdict", "Main issue", "File", "GT spans", "LLM spans",
        "TP", "FP", "FN", "File precision", "File recall", "File F1",
    ]
    _render_pro_review_table(visible_df[board_cols], board_cols)

    _render_code_evidence_viewer(diagnostics, key_prefix)

    chart_df = pd.DataFrame({
        "Type": ["TP", "FP", "FN"],
        "Count": [int(df["TP"].sum()), int(df["FP"].sum()), int(df["FN"].sum())],
    })
    st.markdown("#### Error mix")
    _render_bar_chart(chart_df, x="Type", y="Count", color="Type", y_title="Count", value_format=",.0f", height=260)
    st.download_button(
        "Download review CSV",
        data=visible_df[board_cols].to_csv(index=False).encode("utf-8"),
        file_name="stallm_review_board.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )

    tab_matched, tab_partial, tab_missed, tab_extra = st.tabs(["Best matches", "Needs review", "Missed by LLM", "LLM-only findings"])
    groups = [
        (tab_matched, ["matched"]),
        (tab_partial, ["partial", "mismatch"]),
        (tab_missed, ["missed"]),
        (tab_extra, ["extra"]),
    ]
    for tab, buckets in groups:
        with tab:
            sub = [item for item in diagnostics if item.get("bucket") in buckets]
            if not sub:
                st.info("No files in this bucket.")
                continue
            for item in sub:
                title = (
                    f"{labels.get(item.get('bucket'), item.get('bucket'))} · "
                    f"{item.get('file')} · TP {item.get('tp', 0)} / FP {item.get('fp', 0)} / FN {item.get('fn', 0)}"
                )
                with st.expander(title):
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Ground truth spans**")
                        gt_df = _fmt_span_examples(item.get("gt_examples", []))
                        if gt_df.empty:
                            st.caption("No GT span in U for this file.")
                        else:
                            _render_pro_dataframe(gt_df, hide_index=True)
                    with right:
                        st.markdown("**LLM findings**")
                        llm_df = _fmt_span_examples(item.get("llm_examples", []))
                        if llm_df.empty:
                            st.caption("No LLM finding for this file.")
                        else:
                            _render_pro_dataframe(llm_df, hide_index=True)
                    gt_ctx, llm_ctx = st.columns(2)
                    with gt_ctx:
                        render_code_context("GT code context", item.get("gt_code_excerpt") or item.get("code_excerpt", {}))
                    with llm_ctx:
                        render_code_context("LLM code context", item.get("llm_code_excerpt") or item.get("code_excerpt", {}))

def _metrics_table_view(metrics_df: pd.DataFrame) -> pd.DataFrame:
    hidden_cols = {"diagnostics", "file_diagnostics"}
    return metrics_df.drop(columns=[c for c in hidden_cols if c in metrics_df.columns], errors="ignore")

def _render_comparison_review_boards(metrics_df: pd.DataFrame, label: str) -> None:
    if metrics_df is None or metrics_df.empty or "file_diagnostics" not in metrics_df.columns:
        return
    available = [
        idx for idx, row in metrics_df.iterrows()
        if isinstance(row.get("file_diagnostics"), list) and row.get("file_diagnostics")
    ]
    if not available:
        return
    has_context = any(
        bool((item.get("code_excerpt") or {}).get("lines"))
        for _, row in metrics_df.iterrows()
        for item in (row.get("file_diagnostics") or [])
        if isinstance(row.get("file_diagnostics"), list)
    )
    if not has_context:
        st.warning("These cached results do not contain code excerpts yet. Rerun the analysis to populate Code context blocks.")

    st.markdown("### 🧭 Review board by prompt/model")
    st.caption(f"Pick one {label} below. The board, filters, error mix, and file details update for that selected {label}.")

    summary_cols = [c for c in ["precision", "recall", "f1", "tp", "fp", "fn", "time_s", "total_tokens", "usd_cost"] if c in metrics_df.columns]
    if summary_cols:
        st.markdown(f"#### Available {label}s")
        _render_pro_dataframe(_metrics_table_view(metrics_df)[summary_cols])

    def _render_one_board(name: str) -> None:
        row = metrics_df.loc[name].to_dict()
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: kpi("Precision", pct(row.get("precision", 0.0)), "ok")
        with c2: kpi("Recall", pct(row.get("recall", 0.0)), "warn")
        with c3: kpi("F1", pct(row.get("f1", 0.0)), "ok")
        with c4: kpi("TP", str(int(row.get("tp", 0))), "ok")
        with c5: kpi("FP", str(int(row.get("fp", 0))), "warn")
        with c6: kpi("FN", str(int(row.get("fn", 0))), "bad")
        _render_review_board(row, key_prefix=f"{label}_{_safe_key(name)}")

    selected = st.radio(
        f"Choose {label} board",
        available,
        horizontal=True,
        key=f"review_board_choice_{label}",
    )
    st.success(f"Currently showing review board for: {selected}")
    _render_one_board(selected)

def _render_prompt_comparison_results(summary_U: pd.DataFrame, metrics_df: pd.DataFrame, samples: dict[str, Any]) -> None:
    st.markdown("### 📁 Universe U (files considered)")
    _render_pro_dataframe(summary_U)

    st.markdown("### 📐 Comparison Metrics (All Strategies)")
    _render_pro_dataframe(_metrics_table_view(metrics_df))

    st.markdown("### 📈 Precision / Recall / F1 by Strategy")
    _render_wide_metric_chart(
        metrics_df,
        ["precision", "recall", "f1"],
        group_label="Strategy",
        value_label="Score",
        value_format=".2%",
    )

    _render_comparison_review_boards(metrics_df, "strategy")

    _report_download_button(
        _comparison_report_html(summary_U, metrics_df, samples, label="strategy"),
        file_name="stallm_prompt_comparison_report.html",
        key="download_prompt_comparison_report",
    )

    st.markdown("## 🤖 LLM-based Findings per Strategy")
    for p, sample in (samples or {}).items():
        with st.expander(f"🔹 {p} (sample)"):
            st.json(sample)

def _render_model_comparison_results(summary_U: pd.DataFrame, metrics_df: pd.DataFrame, samples: dict[str, Any]) -> None:
    st.markdown("### 📁 Universe U (files considered)")
    _render_pro_dataframe(summary_U)
    st.markdown("### 📐 Metrics per Model")
    _render_pro_dataframe(_metrics_table_view(metrics_df))
    st.markdown("### 📈 Precision / Recall / F1 by Model")
    _render_wide_metric_chart(
        metrics_df,
        ["precision", "recall", "f1"],
        group_label="Model",
        value_label="Score",
        value_format=".2%",
    )
    st.markdown("### 💰 Tokens & Cost by Model")
    _render_wide_metric_chart(
        metrics_df,
        ["prompt_tokens", "completion_tokens", "total_tokens"],
        group_label="Model",
        value_label="Tokens",
        value_format=",.0f",
    )
    _render_wide_metric_chart(
        metrics_df,
        ["usd_cost"],
        group_label="Model",
        value_label="USD cost",
        value_format="$.6f",
        height=260,
    )
    _render_comparison_review_boards(metrics_df, "model")
    _report_download_button(
        _comparison_report_html(summary_U, metrics_df, samples, label="model"),
        file_name="stallm_model_comparison_report.html",
        key="download_model_comparison_report",
    )
    st.markdown("## 🤖 Samples per Model")
    for model_label, sample in (samples or {}).items():
        with st.expander(f"🔹 {model_label} (sample)"):
            st.json(sample)

def _render_single_run_results(summary_U: pd.DataFrame, metrics: dict[str, Any], usage_totals: dict[str, Any],
                               *, prompt_label: str, language: str, model_label: str, preset: str) -> None:
    st.markdown("### 📁 Universe U (files considered)")
    _render_pro_dataframe(summary_U)
    st.caption("`is_positive=True` → the file has ≥1 GT span. `gt_lines` = number of GT rows for that file.")

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Precision", pct(metrics.get("precision", 0.0)), "ok")
    with c2: kpi("Recall",    pct(metrics.get("recall", 0.0)), "warn")
    with c3: kpi("F1 Score",  pct(metrics.get("f1", 0.0)), "ok")
    with c4: kpi("Elapsed",   f"{metrics.get('time_s', 0.0):.2f}s", "ok")

    metrics_df_single = pd.DataFrame([{
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1": metrics.get("f1", 0.0)}], index=[prompt_label])
    _render_wide_metric_chart(
        metrics_df_single,
        ["precision", "recall", "f1"],
        group_label="Prompt",
        value_label="Score",
        value_format=".2%",
        height=260,
    )

    _render_review_board(metrics)

    t1,t2,t3,t4 = st.columns(4)
    with t1: kpi("Prompt tokens", f"{usage_totals.get('prompt_tokens', 0):,}")
    with t2: kpi("Completion tokens", f"{usage_totals.get('completion_tokens', 0):,}")
    with t3: kpi("Total tokens", f"{usage_totals.get('total_tokens', 0):,}")
    with t4: kpi("USD cost", f"${float(usage_totals.get('usd_cost', 0.0)):.6f}", "warn")

    with st.expander("🔬 Diagnostics (mapping & sampling)"):
        diag = metrics.get("diagnostics", {})
        st.json(diag)
        cov = float(diag.get("mapping_coverage", 0.0))
        if cov == 0.0:
            st.error("No GT rows could be mapped to files in the ZIP. Compare CSV basenames vs ZIP basenames.")
        elif cov < 0.2:
            st.warning(f"Low mapping coverage: {cov:.1%}. Metrics may be unstable.")

    _report_download_button(
        _single_report_html(
            summary_U,
            metrics,
            usage_totals,
            strategy=prompt_label,
            language=language,
            model_label=model_label,
            preset=preset,
        ),
        file_name="stallm_run_report.html",
        key="download_single_run_report",
    )

# =========================
# HERO
# =========================
st.markdown(f"""
<div class="app-topbar">
  <div class="app-topbar-left">
    <div class="app-brand">
      <img src="{BRAND_LOGO_DATA_URI}" alt="StarLLM logo" />
      <span>StarLLM</span>
    </div>
    <span class="nav-chip active">Maintenance Lab</span>
    <span class="nav-chip">Prompt Benchmarks</span>
    <span class="nav-chip">Model Comparison</span>
  </div>
  <div class="app-topbar-right">
    <span class="status-chip"><span class="status-dot"></span> Branch: feature/autoresearch-benchmark</span>
    <span class="status-chip">ArgoUML FL ready</span>
  </div>
</div>
<div class="hero">
  <div class="hero-brand">
    <img class="hero-logo" src="{BRAND_LOGO_DATA_URI}" alt="StarLLM logo" />
    <h1>StarLLM: Static Analysis meets LLMs</h1>
  </div>
  <p>LLM-guided code-smell detection with span-level evaluation, token/cost accounting, and crisp benchmarking workflows.</p>
  <div class="badges">
    <span class="badge">Span-level scoring</span>
    <span class="badge">Universe sampling (Top-K)</span>
    <span class="badge">IoU + line tolerance</span>
    <span class="badge">Prompt/Model comparison</span>
  </div>
</div>
""", unsafe_allow_html=True)

# =========================
# LLM builders (enhanced with Ollama host + connectivity test)
# =========================
def build_llm(sidebar_prefix: str = "", target=None) -> ChatModel:
    target = target or st.sidebar
    registry = load_llm_registry()
    if registry:
        keys = list(registry.keys())
        labels = [f"{k} · {registry[k]['label']} ({registry[k]['config'].provider})" for k in keys]
        choice = target.selectbox(f"{sidebar_prefix}LLM Slot (.env)", labels, index=0,
                                      help="Select a slot configured in your .env")
        slot = keys[labels.index(choice)]
        llm_obj = build_llm_from_slot(slot)
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> {llm_obj.model_label()}</span>", unsafe_allow_html=True)
        return llm_obj

    target.markdown("**Manual configuration (no .env slots detected)**")
    prov = target.selectbox(f"{sidebar_prefix}LLM Provider", ["azure-openai", "openai", "ollama"], index=0)
    mdl_list = available_models(prov)

    if prov == "azure-openai":
        base_default = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE", "")
        ver_default  = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
        dep_default  = os.getenv("OPENAI_DEPLOYMENT_NAME", "")
        with target.expander(f"{sidebar_prefix}Advanced Azure settings", expanded=False):
            api_base    = st.text_input("Azure Resource endpoint", value=base_default, help="e.g., https://<resource>.openai.azure.com/")
            api_version = st.text_input("API version", value=ver_default)
        deployment = target.text_input("Azure deployment name", value=dep_default, help="Exact name in Azure OpenAI Studio")
        api_key_env = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not api_key_env:
            api_key_env = target.text_input("Azure API Key (paste if .env missing)", value="", type="password")
            if api_key_env: os.environ["AZURE_OPENAI_API_KEY"] = api_key_env
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> azure:{deployment or 'deployment'}</span>", unsafe_allow_html=True)
        return ChatModel(LLMConfig(provider="azure-openai", model=deployment, api_base=api_base, api_version=api_version, api_key=api_key_env))

    elif prov == "openai":
        model = target.selectbox(f"{sidebar_prefix}Model", mdl_list, index=0)
        api_key_env = os.getenv("OPENAI_API_KEY") or ""
        if not api_key_env:
            api_key_env = target.text_input("OpenAI API Key (paste if .env missing)", value="", type="password")
            if api_key_env: os.environ["OPENAI_API_KEY"] = api_key_env
        base_url = os.getenv("OPENAI_BASE_URL") or None
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> openai:{model}</span>", unsafe_allow_html=True)
        return ChatModel(LLMConfig(provider="openai", model=model, api_base=base_url, api_key=api_key_env))

    else:
        # Ollama: host + connectivity + dynamic models from that host
        default_host = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        host = target.text_input(f"{sidebar_prefix}Ollama Host", value=default_host,
                                     help="e.g., http://localhost:11434 or http://192.168.1.100:11434")
        if target.button("🔍 Test Connection", key=f"test_conn_{sidebar_prefix}"):
            with target.spinner("Testing connection..."):
                ok, msg = test_ollama_connectivity(host)
                (target.success if ok else target.error)(msg)

        try:
            models_for_host = available_models("ollama", host)
            if not models_for_host:
                target.warning("⚠️ No models returned by host. Falling back to env list.")
                models_for_host = available_models("ollama", None)
        except Exception as e:
            target.warning(f"⚠️ Could not list models from host: {e}")
            models_for_host = available_models("ollama", None)

        model = target.selectbox(f"{sidebar_prefix}Model", models_for_host or mdl_list, index=0)
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> ollama:{model}@{host.replace('http://','').replace('https://','')}</span>", unsafe_allow_html=True)
        return ChatModel(LLMConfig(provider="ollama", model=model, api_base=host))

def build_llms_for_comparison(sidebar_prefix: str = "", target=None) -> list[ChatModel]:
    target = target or st.sidebar
    llms: list[ChatModel] = []
    registry = load_llm_registry()
    target.markdown(f"### {sidebar_prefix}LLM selection for comparison")

    if registry:
        keys = list(registry.keys())
        labels = [f"{k} · {registry[k]['label']} ({registry[k]['config'].provider})" for k in keys]
        picked = target.multiselect("LLM Slots (.env)", labels, default=labels[:2] if len(labels) >= 2 else labels)
        for lab in picked:
            slot = keys[labels.index(lab)]
            cm = build_llm_from_slot(slot)
            llms.append(cm)
        if llms:
            pills = " ".join(f"<span class='pill'>🧠 <b>Model</b> {cm.model_label()}</span>" for cm in llms)
            st.markdown(pills, unsafe_allow_html=True)
        return llms

    # Manual multi-model selection
    prov = target.selectbox(f"{sidebar_prefix}LLM Provider", ["azure-openai", "openai", "ollama"], index=0, key="cmp_prov")
    if prov == "ollama":
        default_host = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        host = target.text_input(f"{sidebar_prefix}Ollama Host", value=default_host,
                                     help="Ollama server URL for model comparison", key="cmp_ollama_host")
        if target.button("🔍 Test Connection", key="test_conn_cmp"):
            with target.spinner("Testing connection..."):
                ok, msg = test_ollama_connectivity(host)
                (target.success if ok else target.error)(msg)
        mdl_list = available_models("ollama", host) or available_models("ollama", None)
        picked_models = target.multiselect("Models to compare", mdl_list, default=mdl_list[:2] if len(mdl_list) >= 2 else mdl_list)
        for m in picked_models:
            llms.append(ChatModel(LLMConfig(provider="ollama", model=m, api_base=host)))
    else:
        mdl_list = available_models(prov)
        picked_models = target.multiselect("Models to compare", mdl_list, default=mdl_list[:2] if len(mdl_list) >= 2 else mdl_list)
        if prov == "azure-openai":
            base_default = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE", "")
            ver_default  = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
            with target.expander(f"{sidebar_prefix}Advanced Azure settings", expanded=False):
                api_base    = st.text_input("Azure Resource endpoint", value=base_default, key="cmp_api_base")
                api_version = st.text_input("API version", value=ver_default, key="cmp_api_ver")
            api_key_env = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            if not api_key_env:
                api_key_env = target.text_input("Azure API Key", value="", type="password", key="cmp_api_key")
                if api_key_env: os.environ["AZURE_OPENAI_API_KEY"] = api_key_env
            for dep in picked_models:
                llms.append(ChatModel(LLMConfig(provider="azure-openai", model=dep, api_base=api_base, api_version=api_version, api_key=api_key_env)))
        else:
            base_url = os.getenv("OPENAI_BASE_URL") or None
            api_key_env = os.getenv("OPENAI_API_KEY") or ""
            if not api_key_env:
                api_key_env = target.text_input("OpenAI API Key", value="", type="password", key="cmp_openai_key")
                if api_key_env: os.environ["OPENAI_API_KEY"] = api_key_env
            for m in picked_models:
                llms.append(ChatModel(LLMConfig(provider="openai", model=m, api_base=base_url, api_key=api_key_env)))

    if llms:
        pills = " ".join(f"<span class='pill'>🧠 <b>Model</b> {cm.model_label()}</span>" for cm in llms)
        st.markdown(pills, unsafe_allow_html=True)
    return llms

# =========================
# Navigation
# =========================
PAGES = [
    "🧭 Maintenance Tasks",
    "⚙️ Run Experiments",
    "🧪 AutoResearch",
    "📝 Manage Prompts",
    "🗃️ Stored Results (DB)",
    "🔄 Batch Experiments",
    "📘 Guide & Examples",
]
if "workspace_page" not in st.session_state:
    st.session_state.workspace_page = PAGES[0]

nav_cols = st.columns(len(PAGES))
for idx, page_name in enumerate(PAGES):
    with nav_cols[idx]:
        if st.button(page_name, key=f"nav_{idx}", use_container_width=True):
            st.session_state.workspace_page = page_name
page = st.session_state.workspace_page

# ==========================================================
# Tab 1 : Run Experiments (SPAN-LEVEL)
# ==========================================================
def render_tab_run_experiments(render_sidebar: bool = True, settings_target=None):
    sidebar_target = settings_target or (st.sidebar if render_sidebar else st.container())
    if render_sidebar and settings_target is None:
        st.sidebar.header("⚙️ Settings")
    if "run_top_k" not in st.session_state:
        st.session_state.run_top_k = 20
    if "run_pos_ratio" not in st.session_state:
        st.session_state.run_pos_ratio = 0.5
    if "run_preset_last" not in st.session_state:
        st.session_state.run_preset_last = "Balanced"
    if "use_bundled_demo" not in st.session_state:
        st.session_state.use_bundled_demo = False

    if sidebar_target.button("⚡ Load demo config"):
        st.session_state.use_bundled_demo = True
        st.session_state.run_top_k = 5
        st.session_state.run_pos_ratio = 0.6
        st.session_state.run_preset_last = "Balanced"

    mode_run = sidebar_target.radio(
        "Execution mode",
        ["Single prompt", "Compare selected prompts", "Compare LLM models"],
        index=0,
        help="Pick your execution scenario."
    )
    top_k = sidebar_target.slider("Total files in U (Top-K)", 5, 50, key="run_top_k", step=1, help="Universe size U (positives + negatives).")
    pos_ratio = sidebar_target.slider("Positive ratio in U", 0.0, 1.0, key="run_pos_ratio", step=0.05, help="Share of positive files in U.")
    preset_options = ["Lenient", "Balanced", "Strict"]
    preset_default = st.session_state.get("run_preset_last", "Balanced")
    preset = sidebar_target.selectbox(
        "Evaluation strictness",
        preset_options,
        index=preset_options.index(preset_default) if preset_default in preset_options else 1,
        help="Controls IoU threshold and line tolerance (δ).",
    )
    st.session_state.run_preset_last = preset

    strategies = _prompt_templates_for_task("code_smell_detection")
    selected_prompt_names: list[str] = []
    if mode_run == "Single prompt":
        if not strategies:
            st.error("⚠️ No strategies available. Please add one in ‘Manage Prompts’.")
            return
        prompt_mode = sidebar_target.selectbox("🧩 Select LLM Prompt Strategy", list(strategies.keys()))
        selected_prompt_names = [prompt_mode]
        if render_sidebar:
            _sidebar_prompt_preview("View selected prompt", strategies, prompt_mode, target=sidebar_target)
    elif mode_run == "Compare selected prompts":
        selected_modes = sidebar_target.multiselect(
            "🧩 Select Prompt Strategies to Compare",
            list(strategies.keys()),
            default=[k for k in ["baseline", "scanner", "hybrid"] if k in strategies],
        )
        selected_prompt_names = list(selected_modes)
        if render_sidebar:
            _sidebar_prompt_preview("View selected prompts", strategies, selected_modes, target=sidebar_target)
    else:
        if not strategies:
            st.error("⚠️ No strategies available. Please add one in ‘Manage Prompts’.")
            return
        prompt_mode = sidebar_target.selectbox("🧩 Fixed Prompt Strategy (for model comparison)", list(strategies.keys()))
        selected_prompt_names = [prompt_mode]
        if render_sidebar:
            _sidebar_prompt_preview("View fixed prompt", strategies, prompt_mode, target=sidebar_target)

    if mode_run in ("Single prompt", "Compare selected prompts"):
        llm = build_llm(target=sidebar_target) if render_sidebar else None
    else:
        llms_to_compare = build_llms_for_comparison(target=sidebar_target) if render_sidebar else []
        if not llms_to_compare:
            if render_sidebar:
                st.warning("Select at least one LLM to compare.")
            return

    if not render_sidebar:
        st.info("Open this tab directly to configure and run code-smell detection experiments.")
        return

    st.markdown("### Input Dataset")
    st.caption("Choose a bundled demo or upload your own project and analyzer ground truth.")
    demo_sets = _demo_datasets()
    use_demo = st.checkbox(
        "Use bundled demo dataset",
        key="use_bundled_demo",
        disabled=not bool(demo_sets),
        help="Loads one of the small included project/CSV pairs so the POC can run immediately.",
    )

    uploaded_zip = None
    uploaded_static = None
    zip_name = None
    csv_name = None
    zip_bytes = None
    csv_bytes = None

    if use_demo and demo_sets:
        demo_name = st.selectbox("Demo dataset", list(demo_sets.keys()))
        demo_zip, demo_csv = demo_sets[demo_name]
        zip_name = demo_zip.name
        csv_name = demo_csv.name
        zip_bytes = demo_zip.read_bytes()
        csv_bytes = demo_csv.read_bytes()
        st.caption(f"Using `{demo_zip.relative_to(Path(__file__).parent)}` + `{demo_csv.relative_to(Path(__file__).parent)}`")
    else:
        colu1, colu2 = st.columns([1,1])
        with colu1:
            uploaded_zip = st.file_uploader("📦 Project (ZIP)", type="zip", help="ZIP with source files.")
        with colu2:
            uploaded_static = st.file_uploader("📑 Static Analyzer CSV (Ground Truth)", type="csv",
                                               help="GT spans: file, startLine, endLine[, columns, type].")

        zip_bytes = uploaded_zip.getvalue() if uploaded_zip is not None else None
        csv_bytes = uploaded_static.getvalue() if uploaded_static is not None else None
        zip_name = uploaded_zip.name if uploaded_zip is not None else None
        csv_name = uploaded_static.name if uploaded_static is not None else None

    llm_for_preflight = llm if mode_run in ("Single prompt", "Compare selected prompts") else (llms_to_compare[0] if llms_to_compare else None)
    preflight_ok, preflight_report = _preflight_inputs(zip_name, zip_bytes, csv_name, csv_bytes, top_k, pos_ratio, llm_for_preflight)
    if mode_run == "Compare LLM models" and llms_to_compare:
        labels = [m.model_label() for m in llms_to_compare]
        preflight_report["llm"] = {"ok": True, "message": f"{len(labels)} model(s): {', '.join(labels[:3])}{'...' if len(labels) > 3 else ''}"}
        preflight_ok = all(preflight_report[k]["ok"] for k in ["zip", "csv", "language", "universe", "llm"])
    _render_preflight(preflight_report)

    _render_prompt_preview_panel("Prompt Preview", strategies, selected_prompt_names)

    # Capabilities (type/line/column)
    user_require_type = False
    user_use_line_span = True
    user_use_cols_single = False

    if csv_bytes:
        try:
            with tempfile.TemporaryDirectory() as _tmp:
                tmp_csv = os.path.join(_tmp, csv_name or "ground_truth.csv")
                with open(tmp_csv, "wb") as f: f.write(csv_bytes)
                df_cap = _read_csv_robust(tmp_csv)
            caps = detect_gt_capabilities(df_cap)
        except Exception as e:
            st.warning(f"Could not inspect GT capabilities: {e}")
            caps = None

        if caps:
            st.markdown("### ✅ Detected GT capabilities")
            sample = caps.get("sample", {})
            c1, c2, c3 = st.columns(3)

            with c1:
                if caps.get("has_type", False):
                    user_require_type = st.checkbox(
                        f"Match by rule/type (e.g., “{sample.get('rule/type', '…')}”)",
                        value=False
                    )
                else:
                    st.caption("No rule/type column detected.")
                    user_require_type = False

            with c2:
                if caps.get("has_line_span", False):
                    user_use_line_span = st.checkbox(
                        f"Use line spans (start–end) (e.g., {sample.get('startLine','?')}–{sample.get('endLine','?')})",
                        value=True
                    )
                else:
                    st.caption("No endLine column detected.")
                    user_use_line_span = False

            with c3:
                if caps.get("has_col_span", False):
                    user_use_cols_single = st.checkbox(
                        "Use column spans on single-line (if available)",
                        value=False
                    )
                else:
                    st.caption("No column spans detected.")
                    user_use_cols_single = False

    # Run
    run_clicked = st.button("🚀 Run Analysis")
    if run_clicked:
        if not (zip_bytes and csv_bytes):
            st.warning("📥 Please select demo data or upload both files (ZIP + CSV).")
            return
        if not preflight_ok:
            st.warning("Resolve the preflight checks before running the analysis.")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, zip_name or "project.zip"); open(zip_path, "wb").write(zip_bytes)
            csv_path = os.path.join(tmpdir, csv_name or "ground_truth.csv"); open(csv_path, "wb").write(csv_bytes)

            language = detect_language_from_zip(zip_path)
            exts = find_exts_for_language(language)
            st.markdown(f"### 🌐 Detected Language: **{language}**")
            st.caption("Span-level evaluation with a mixed universe U (positives + negatives).")

            # sanity read GT
            try:
                _ = load_ground_truth_spans(csv_path, allowed_exts=exts)
            except Exception as e:
                st.error(f"⚠️ Error reading Static Analyzer CSV: {e}")
                return

            if mode_run == "Single prompt":
                st.markdown(f"### 🤖 Running LLM analysis with `{prompt_mode}` strategy…")
                start = time.time()
                try:
                    with st.spinner(f"⏳ Running `{prompt_mode}`…"):
                        results_llm, summary_U, metrics, usage_totals = run_experiment(
                            zip_path, csv_path, prompt_mode,
                            top_k=top_k, pos_ratio=pos_ratio, preset=preset,
                            user_require_type=user_require_type,
                            user_use_line_span=user_use_line_span,
                            user_use_cols_single=user_use_cols_single,
                            llm=llm
                        )
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
                    return

                elapsed = time.time() - start
                metrics["time_s"] = round(elapsed, 2)

                # best-effort DB
                try:
                    save_run_result(
                        project=os.path.basename(zip_path), filename="ALL",
                        strategy=f"{prompt_mode}@span", language=language,
                        f1=float(metrics.get("f1", 0.0)),
                        precision=float(metrics.get("precision", 0.0)),
                        recall=float(metrics.get("recall", 0.0)),
                        top_k=int(metrics.get("top_k_total_files", 0)),
                        issues_detected=results_llm, time_elapsed=elapsed,
                        llm_used=llm.model_label(),
                        sonar_detected_smells=summary_U.reset_index().to_dict(orient="records"),
                        prompt_tokens=int(usage_totals.get("prompt_tokens", 0)),
                        completion_tokens=int(usage_totals.get("completion_tokens", 0)),
                        total_tokens=int(usage_totals.get("total_tokens", 0)),
                        usd_cost=float(usage_totals.get("usd_cost", 0.0)),
                    )
                    st.success("✅ Run saved to database.")
                except Exception:
                    pass

                prompt_label = f"{prompt_mode}@span"
                st.session_state.last_single_run = {
                    "summary_U": summary_U,
                    "metrics": metrics,
                    "usage_totals": usage_totals,
                    "prompt_label": prompt_label,
                    "language": language,
                    "model_label": llm.model_label(),
                    "preset": preset,
                    "schema_version": RESULTS_SCHEMA_VERSION,
                }
                _render_single_run_results(
                    summary_U,
                    metrics,
                    usage_totals,
                    prompt_label=prompt_label,
                    language=language,
                    model_label=llm.model_label(),
                    preset=preset,
                )

                st.caption(f"⏱️ Completed in {elapsed:.1f}s • Model: {llm.model_label()} • Preset: {preset}")

            elif mode_run == "Compare selected prompts":
                if not selected_modes:
                    st.warning("⚠️ Please select at least one prompt strategy.")
                    return
                st.markdown("### 🔄 Running Multiple Prompt Strategies (span-level)")
                progress = st.progress(0); status_text = st.empty(); timer_text = st.empty()
                try:
                    summary_U, metrics_df, samples = run_selected_experiments(
                        zip_path, csv_path, selected_modes,
                        progress=progress, status=status_text, timer=timer_text,
                        top_k=top_k, pos_ratio=pos_ratio, preset=preset,
                        user_require_type=user_require_type,
                        user_use_line_span=user_use_line_span,
                        user_use_cols_single=user_use_cols_single,
                        llm=llm
                    )
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
                    return

                st.session_state.last_prompt_comparison = {
                    "summary_U": summary_U,
                    "metrics_df": metrics_df,
                    "samples": samples,
                    "mode": mode_run,
                    "schema_version": RESULTS_SCHEMA_VERSION,
                }
                _render_prompt_comparison_results(summary_U, metrics_df, samples)

            else:
                if not llms_to_compare:
                    st.warning("⚠️ Please select at least one LLM to compare.")
                    return
                st.markdown(f"### 🔄 Comparing LLM models (strategy: `{prompt_mode}`, span-level)")
                progress = st.progress(0); status_text = st.empty(); timer_text = st.empty()
                try:
                    summary_U, metrics_df, samples = run_selected_models_experiments(
                        zip_path, csv_path, prompt_mode, llms_to_compare,
                        progress=progress, status=status_text, timer=timer_text,
                        top_k=top_k, pos_ratio=pos_ratio, preset=preset,
                        user_require_type=user_require_type,
                        user_use_line_span=user_use_line_span,
                        user_use_cols_single=user_use_cols_single
                    )
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
                    return

                st.session_state.last_model_comparison = {
                    "summary_U": summary_U,
                    "metrics_df": metrics_df,
                    "samples": samples,
                    "mode": mode_run,
                    "schema_version": RESULTS_SCHEMA_VERSION,
                }
                _render_model_comparison_results(summary_U, metrics_df, samples)

    if not run_clicked:
        if mode_run == "Single prompt" and "last_single_run" in st.session_state:
            cached = st.session_state.last_single_run
            if cached.get("schema_version") == RESULTS_SCHEMA_VERSION:
                st.caption("Showing cached single-run results. Changing the file below does not rerun the LLM.")
                _render_single_run_results(
                    cached["summary_U"],
                    cached["metrics"],
                    cached["usage_totals"],
                    prompt_label=cached["prompt_label"],
                    language=cached["language"],
                    model_label=cached["model_label"],
                    preset=cached["preset"],
                )
            else:
                st.info("Cached single-run results are from an older format. Rerun the analysis to enable Code context.")
        elif mode_run == "Compare selected prompts" and "last_prompt_comparison" in st.session_state:
            cached = st.session_state.last_prompt_comparison
            if cached.get("schema_version") == RESULTS_SCHEMA_VERSION:
                st.caption("Showing cached comparison results. Changing the selected board below does not rerun the LLM.")
                _render_prompt_comparison_results(cached["summary_U"], cached["metrics_df"], cached["samples"])
            else:
                st.info("Cached comparison results are from an older format. Rerun the analysis to enable Code context.")
        elif mode_run == "Compare LLM models" and "last_model_comparison" in st.session_state:
            cached = st.session_state.last_model_comparison
            if cached.get("schema_version") == RESULTS_SCHEMA_VERSION:
                st.caption("Showing cached model-comparison results. Changing the selected board below does not rerun the LLM.")
                _render_model_comparison_results(cached["summary_U"], cached["metrics_df"], cached["samples"])
            else:
                st.info("Cached model-comparison results are from an older format. Rerun the analysis to enable Code context.")

# ==========================================================
# Tab 1 : Maintenance Tasks (Feature/Bug Location)
# ==========================================================
def render_tab_maintenance_tasks(settings_target=None):
    st.markdown("## 🧭 Maintenance Tasks")
    st.caption("Experimental branch: feature location and bug localization use file-level ranking metrics.")

    settings = settings_target or st.sidebar
    activity = settings.selectbox(
        "Maintenance activity",
        ["Code smell detection", "Feature location", "Bug location"],
        index=0,
        help="Choose the software maintenance activity before selecting prompts or models.",
    )
    _render_maintenance_activity_intro(activity)
    if activity == "Code smell detection":
        oracle = settings.radio(
            "Reference oracle",
            ["Analyzer reference (SonarQube CSV)", "Human smell oracle (DACOS/DACOSX/MLCQ)"],
            index=0,
            help="Analyzer reference measures agreement with a static analyzer. Human-oracle mode uses DACOS/DACOSX or MLCQ labels.",
        )
        if oracle.startswith("Analyzer"):
            render_tab_run_experiments(render_sidebar=True, settings_target=settings)
        else:
            render_tab_dacos_smell_benchmark(settings_target=settings)
        return
    if activity == "Bug location":
        st.info("Bug location is the next activity to wire in the UI. The backend adapter is ready for Bench4BL-style records, but no bug-location dataset is configured in this workspace yet.")
        st.markdown("### Expected bug-location benchmark format")
        st.code(
            "project,bug_id,title,body,changed_files\n"
            "commons-lang,LANG-1,Crash on parse,Parser throws exception,src/main/java/Foo.java;src/main/java/Bar.java",
            language="csv",
        )
        return

    if activity == "Feature location":
        default_gt = "data/apps/Feature Location-ArgoUML"
        default_zip = "data/apps/ArgoUML/ArgoUML.zip"
        st.session_state.setdefault("fl_gt_path", default_gt)
        st.session_state.setdefault("fl_repo_zip", default_zip)
        st.session_state.setdefault("fl_top_k", 10)
        if settings.button("⚡ Load demo config", key="fl_load_demo_config"):
            st.session_state.fl_gt_path = default_gt
            st.session_state.fl_repo_zip = default_zip
            st.session_state.fl_top_k = 10

        technique = settings.selectbox(
            "Technique / benchmark",
            ["ArgoUML SPL benchmark"],
            index=0,
            help="Benchmark or technique used for this maintenance activity.",
        )
        settings.markdown("**Run configuration**")
        execution_mode = settings.radio(
            "Execution mode",
            ["Single prompt", "Compare selected prompts", "Compare LLM models"],
            index=0,
            key="maint_execution_mode",
            help="Same experimental scenarios as code-smell detection runs.",
        )
        with settings.expander("Dataset paths", expanded=False):
            gt_path = st.text_input("ArgoUML GT folder", key="fl_gt_path")
            repo_zip = st.text_input("ArgoUML source ZIP", key="fl_repo_zip")

        st.markdown("### Input Dataset")
        st.caption("Same workflow as code-smell detection: select the benchmark data, inspect the prompt, then launch the run.")
        dataset_cols = st.columns(3)
        with dataset_cols[0]:
            kpi("Benchmark", technique.replace(" benchmark", ""), "ok")
        with dataset_cols[1]:
            kpi("GT folder", Path(gt_path).name or "not set", "ok")
        with dataset_cols[2]:
            kpi("Source archive", Path(repo_zip).name or "not set", "warn")

        try:
            tasks = load_argouml_feature_tasks(gt_path, repo_zip=repo_zip)
        except Exception as e:
            st.error(f"Could not load ArgoUML feature-location benchmark: {e}")
            return
        if not tasks:
            st.warning("No feature-location tasks found.")
            return

        labels = [f"{t.metadata.get('feature', t.task_id)} · {len(t.gold_locations)} gold files" for t in tasks]
        task_by_label = dict(zip(labels, tasks))
        feature_scope = settings.radio(
            "Feature scope",
            ["Single feature", "Selected features", "All features"],
            index=0,
            help="Choose whether to evaluate one feature scenario or a batch of feature scenarios.",
        )
        if feature_scope == "Single feature":
            selected_label = settings.selectbox("Feature scenario", labels, index=0)
            selected_tasks = [task_by_label[selected_label]]
        elif feature_scope == "Selected features":
            selected_labels = settings.multiselect("Feature scenarios", labels, default=labels[:3])
            selected_tasks = [task_by_label[label] for label in selected_labels]
        else:
            selected_tasks = tasks
        if not selected_tasks:
            st.warning("Select at least one feature scenario.")
            return
        task = selected_tasks[0]

        top_k = settings.slider("Top-K predictions", 1, 30, key="fl_top_k", step=1)
        try:
            candidates = list_repo_candidates(repo_zip, allowed_exts=[".java"])
        except Exception as e:
            st.error(f"Could not list source candidates from ZIP: {e}")
            return
        if not candidates:
            st.warning("No Java source candidates found in the ZIP.")
            return
        max_candidates = settings.slider(
            "Candidate budget",
            50,
            min(2000, len(candidates)),
            min(600, len(candidates)),
            50,
            help="Number of candidate file paths included in the prompt.",
        )
        location_prompt_templates = _prompt_templates_for_task("feature_location") or LOCATION_PROMPT_TEMPLATES
        prompt_options = list(location_prompt_templates.keys())
        if execution_mode == "Compare selected prompts":
            prompt_styles = settings.multiselect("Prompt styles", prompt_options, default=prompt_options[:2])
            _sidebar_prompt_preview("View selected prompt templates", location_prompt_templates, prompt_styles, target=settings)
            llm = build_llm("Maintenance ", target=settings)
            llms = []
        elif execution_mode == "Compare LLM models":
            prompt_style = settings.selectbox("Fixed prompt style", prompt_options, index=0)
            prompt_styles = [prompt_style]
            _sidebar_prompt_preview("View fixed prompt template", location_prompt_templates, prompt_style, target=settings)
            llm = None
            llms = build_llms_for_comparison("Maintenance ", target=settings)
        else:
            prompt_style = settings.selectbox("Prompt style", prompt_options, index=0)
            prompt_styles = [prompt_style]
            _sidebar_prompt_preview("View selected prompt template", location_prompt_templates, prompt_style, target=settings)
            llm = build_llm("Maintenance ", target=settings)
            llms = []

        st.markdown("### Feature Location Scope")
        if len(selected_tasks) == 1:
            st.info(task.query)
        else:
            st.info(f"Batch evaluation over {len(selected_tasks)} feature scenarios. The prompt preview below shows the first selected feature.")
            scope_df = pd.DataFrame([
                {
                    "Feature": t.metadata.get("feature", t.task_id),
                    "Gold files": len(t.gold_locations),
                    "Query": t.query.splitlines()[0] if t.query else "",
                }
                for t in selected_tasks
            ])
            _render_pro_dataframe(scope_df, hide_index=True)

        gold_df = pd.DataFrame([{"Gold file": g.file, "Symbol": g.symbol or ""} for g in task.gold_locations])
        with st.expander(f"Gold locations for preview feature ({len(gold_df)})", expanded=False):
            _render_pro_dataframe(gold_df, hide_index=True)

        focused_candidates = _rank_feature_candidates(task.query, candidates)[:max_candidates]
        c_info1, c_info2, c_info3 = st.columns(3)
        with c_info1:
            kpi("Selected features", str(len(selected_tasks)), "ok")
        with c_info2:
            kpi("Preview gold files", str(len(task.gold_locations)), "ok")
        with c_info3:
            kpi("Prompt candidates", str(len(focused_candidates)), "warn")
        st.caption(f"Candidate pool: {len(candidates)} Java files. Prompt includes top {len(focused_candidates)} lexical candidates.")

        from StaLLM_tasks import build_location_prompt
        _render_prompt_preview_panel(
            "Prompt Preview",
            location_prompt_templates,
            prompt_styles,
            preview_builder=lambda style: build_location_prompt(
                task,
                focused_candidates,
                top_k=top_k,
                prompt_style=style,
                prompt_templates=location_prompt_templates,
            ),
        )

        if execution_mode == "Compare selected prompts" and not prompt_styles:
            st.warning("Select at least one prompt style.")
            return
        if execution_mode == "Compare LLM models" and not llms:
            st.warning("Select at least one model.")
            return

        run_label = {
            "Single prompt": "Run Feature Location",
            "Compare selected prompts": "Compare Prompt Styles",
            "Compare LLM models": "Compare LLM Models",
        }[execution_mode]
        if st.button(run_label, type="primary"):
            results = []
            if execution_mode == "Compare selected prompts":
                for style in prompt_styles:
                    with st.spinner(f"Running prompt style: {style} on {len(selected_tasks)} feature(s)"):
                        try:
                            result = _run_feature_location_batch(selected_tasks, candidates, llm, top_k, max_candidates, style, location_prompt_templates)
                        except Exception as e:
                            st.error(f"Feature-location run failed for {style}: {e}")
                            return
                    result["label"] = style
                    results.append(result)
            elif execution_mode == "Compare LLM models":
                for model_llm in llms:
                    label = model_llm.model_label()
                    with st.spinner(f"Running model: {label} on {len(selected_tasks)} feature(s)"):
                        try:
                            result = _run_feature_location_batch(selected_tasks, candidates, model_llm, top_k, max_candidates, prompt_styles[0], location_prompt_templates)
                        except Exception as e:
                            st.error(f"Feature-location run failed for {label}: {e}")
                            return
                    result["label"] = label
                    results.append(result)
            else:
                with st.spinner(f"Ranking files for {len(selected_tasks)} feature scenario(s)..."):
                    try:
                        result = _run_feature_location_batch(selected_tasks, candidates, llm, top_k, max_candidates, prompt_styles[0], location_prompt_templates)
                    except Exception as e:
                        st.error(f"Feature-location run failed: {e}")
                        return
                result["label"] = prompt_styles[0]
                results.append(result)

            _render_location_results(results, task)


def render_tab_dacos_smell_benchmark(settings_target=None):
    st.markdown("### Human Code-Smell Oracle")
    st.caption("Evaluate against human smell labels instead of treating a static analyzer as ground truth.")

    settings = settings_target or st.sidebar
    settings.markdown("**Human oracle dataset**")
    dataset_name = settings.selectbox(
        "Dataset",
        ["DACOS/DACOSX", "MLCQ"],
        index=0,
        key="human_oracle_dataset",
        help="DACOS/DACOSX provides focused binary smell labels. MLCQ provides professional Java smell labels with severity.",
    )
    execution_mode = settings.radio(
        "Execution mode",
        ["Single prompt", "Compare selected prompts", "Compare LLM models"],
        index=0,
        key="dacos_execution_mode",
        help="Use the human-oracle benchmark with the same StaLLM workflows: one run, prompt comparison, or model comparison.",
    )
    prompt_templates = _dacos_prompt_templates()
    prompt_names = list(prompt_templates.keys())
    if execution_mode == "Compare selected prompts":
        selected_prompts = settings.multiselect(
            "Human-oracle prompt variants",
            prompt_names,
            default=prompt_names[: min(3, len(prompt_names))],
            key="dacos_prompt_compare",
        )
        fixed_prompt = selected_prompts[0] if selected_prompts else prompt_names[0]
        llm = build_llm("Human oracle ", target=settings)
        llms_to_compare: list[Any] = []
    elif execution_mode == "Compare LLM models":
        fixed_prompt = settings.selectbox("Fixed human-oracle prompt", prompt_names, index=0, key="dacos_fixed_prompt")
        selected_prompts = [fixed_prompt]
        llms_to_compare = build_llms_for_comparison("Human oracle ", target=settings)
        llm = llms_to_compare[0] if llms_to_compare else None
    else:
        fixed_prompt = settings.selectbox("Human-oracle prompt", prompt_names, index=0, key="dacos_single_prompt")
        selected_prompts = [fixed_prompt]
        llm = build_llm("Human oracle ", target=settings)
        llms_to_compare = []

    if dataset_name == "MLCQ":
        dacos_path = settings.text_input("MLCQ samples CSV", "data/apps/MLCQ/MLCQCodeSmellSamples.csv", key="mlcq_path")
        files_root = None
        positive_threshold = settings.selectbox(
            "Positive severity threshold",
            ["minor", "major", "critical"],
            index=0,
            key="mlcq_positive_threshold",
            help="Labels at or above this severity are treated as smell-present.",
        )
    else:
        dacos_path = settings.text_input("DACOSMain/DACOSX file", "data/apps/DACOS/DACOSMain.sql", key="dacos_path")
        files_root = settings.text_input("Files folder or files.zip", "data/apps/DACOS/files", key="dacos_files_root")
        positive_threshold = "minor"
    sample_limit = settings.slider("Samples to evaluate", 1, 100, 10, 1, key="dacos_sample_limit")

    st.markdown("""
    <div class="section">
      <b>What this benchmark measures</b>
      <p class="muted">StaLLM asks the LLM whether a specific code smell is present in each snippet, then compares the answer to human annotations. This is a human-oracle benchmark, not analyzer replication.</p>
      <div class="badges">
        <span class="badge"><b>Oracle</b> Human labels</span>
        <span class="badge"><b>Dataset</b> DACOS/DACOSX or MLCQ</span>
        <span class="badge"><b>Output</b> present / absent</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    path_obj = Path(dacos_path)
    if not path_obj.exists():
        st.warning(f"{dataset_name} files are not present in this workspace yet.")
        if dataset_name == "MLCQ":
            st.markdown("Download `MLCQCodeSmellSamples.csv` from Zenodo, then place it under `data/apps/MLCQ/`.")
            st.link_button("Open MLCQ on Zenodo", "https://zenodo.org/records/3666840")
        else:
            st.markdown("Download DACOS from Zenodo, then place `DACOSMain.sql`, `DACOSExtended.sql`, and extracted `files.zip` under `data/apps/DACOS/`.")
            st.link_button("Open DACOS on Zenodo", "https://zenodo.org/records/7570428")
        return

    try:
        if dataset_name == "MLCQ":
            samples = load_mlcq_smell_samples(dacos_path, limit=sample_limit, positive_threshold=positive_threshold)
        else:
            samples = load_dacos_smell_samples(dacos_path, files_root=files_root, limit=sample_limit)
    except Exception as e:
        st.error(f"Could not load {dataset_name} dataset: {e}")
        return
    if not samples:
        if dataset_name == "MLCQ":
            st.warning("No MLCQ samples could be loaded. MLCQ stores GitHub links rather than inline code, so the first run needs internet access to fetch and cache snippets.")
        else:
            st.warning("No DACOS samples could be loaded. Check that `files.zip` is next to the SQL dump, or point the files field to an extracted DACOS files folder.")
        return

    preview_df = pd.DataFrame([
        {
            "Sample": sample.sample_id,
            "Smell": sample.smell,
            "Human label": "present" if sample.label else "absent",
            "Chars": len(sample.code),
        }
        for sample in samples[:20]
    ])
    st.markdown("### Dataset Preview")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Loaded samples", str(len(samples)), "ok")
    with c2:
        kpi("Smell types", str(len({s.smell for s in samples})), "ok")
    with c3:
        positives = sum(1 for s in samples if s.label)
        kpi("Positive labels", pct(positives / len(samples) if samples else 0.0), "warn")
    _render_pro_dataframe(preview_df, hide_index=True)
    _render_prompt_preview_panel(f"{dataset_name} Prompt Preview", prompt_templates, selected_prompts)

    button_label = {
        "Single prompt": f"Run {dataset_name} Human-Oracle Benchmark",
        "Compare selected prompts": f"Compare {dataset_name} Prompt Variants",
        "Compare LLM models": f"Compare {dataset_name} Models",
    }[execution_mode]
    if st.button(button_label, type="primary"):
        if execution_mode == "Compare selected prompts" and not selected_prompts:
            st.warning("Select at least one human-oracle prompt variant.")
            return
        if execution_mode == "Compare LLM models" and not llms_to_compare:
            st.warning("Select at least one LLM model.")
            return
        with st.spinner(f"Evaluating {len(samples)} {dataset_name} sample(s)..."):
            try:
                if execution_mode == "Compare selected prompts":
                    results = []
                    progress = st.progress(0)
                    for idx, prompt_name in enumerate(selected_prompts, start=1):
                        result = run_dacos_smell_benchmark(
                            samples,
                            llm,
                            prompt_template=prompt_templates[prompt_name],
                            max_samples=sample_limit,
                        )
                        result["label"] = prompt_name
                        result["prompt"] = prompt_name
                        result["model"] = llm.model_label()
                        results.append(result)
                        progress.progress(idx / max(1, len(selected_prompts)))
                    st.session_state.last_dacos_result = {"mode": execution_mode, "results": results}
                    _render_dacos_comparison_results(results, compare_by="Prompt")
                    return
                if execution_mode == "Compare LLM models":
                    results = []
                    progress = st.progress(0)
                    for idx, model_llm in enumerate(llms_to_compare, start=1):
                        result = run_dacos_smell_benchmark(
                            samples,
                            model_llm,
                            prompt_template=prompt_templates[fixed_prompt],
                            max_samples=sample_limit,
                        )
                        result["label"] = model_llm.model_label()
                        result["prompt"] = fixed_prompt
                        result["model"] = model_llm.model_label()
                        results.append(result)
                        progress.progress(idx / max(1, len(llms_to_compare)))
                    st.session_state.last_dacos_result = {"mode": execution_mode, "results": results}
                    _render_dacos_comparison_results(results, compare_by="Model")
                    return
                result = run_dacos_smell_benchmark(
                    samples,
                    llm,
                    prompt_template=prompt_templates[fixed_prompt],
                    max_samples=sample_limit,
                )
                result["label"] = fixed_prompt
                result["prompt"] = fixed_prompt
                result["model"] = llm.model_label()
            except Exception as e:
                st.error(f"Human-oracle benchmark failed: {e}")
                return
        st.session_state.last_dacos_result = result
        _render_dacos_results(result)

    cached = st.session_state.get("last_dacos_result")
    if cached:
        st.markdown("### Last Human-Oracle Run")
        if isinstance(cached, dict) and cached.get("results"):
            _render_dacos_comparison_results(
                cached["results"],
                compare_by="Model" if cached.get("mode") == "Compare LLM models" else "Prompt",
            )
        else:
            _render_dacos_results(cached)


def _render_dacos_results(result: dict[str, Any]) -> None:
    metrics = result.get("metrics") or {}
    usage = result.get("usage") or {}
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Human-label F1", pct(metrics.get("f1", 0.0)), "ok")
    with c2:
        kpi("Human-label precision", pct(metrics.get("precision", 0.0)), "ok")
    with c3:
        kpi("Human-label recall", pct(metrics.get("recall", 0.0)), "warn")
    with c4:
        kpi("Accuracy", pct(metrics.get("accuracy", 0.0)), "warn")

    st.markdown("### Confusion Counts")
    count_df = pd.DataFrame([{
        "Agreement present": metrics.get("tp", 0),
        "LLM-only present": metrics.get("fp", 0),
        "Human-only present": metrics.get("fn", 0),
        "Agreement absent": metrics.get("tn", 0),
        "Samples": metrics.get("samples", 0),
        "Tokens": usage.get("total_tokens", 0),
    }])
    _render_pro_dataframe(count_df, hide_index=True)

    rows = result.get("rows") or []
    if rows:
        detail_df = pd.DataFrame(rows)
        detail_df["Human label"] = detail_df["gold"].map(lambda x: "present" if x else "absent")
        detail_df["LLM label"] = detail_df["predicted"].map(lambda x: "present" if x else "absent")
        st.markdown("### Sample Decisions")
        _render_pro_dataframe(detail_df[["sample_id", "smell", "Human label", "LLM label", "confidence", "rationale"]], hide_index=True)


def _render_dacos_comparison_results(results: list[dict[str, Any]], *, compare_by: str) -> None:
    if not results:
        st.info("No human-oracle comparison result to display.")
        return
    rows = []
    for result in results:
        metrics = result.get("metrics") or {}
        usage = result.get("usage") or {}
        rows.append({
            compare_by: result.get("label", ""),
            "Prompt": result.get("prompt", ""),
            "Model": result.get("model", ""),
            "Human-label F1": metrics.get("f1", 0.0),
            "Precision": metrics.get("precision", 0.0),
            "Recall": metrics.get("recall", 0.0),
            "Accuracy": metrics.get("accuracy", 0.0),
            "TP": metrics.get("tp", 0),
            "FP": metrics.get("fp", 0),
            "FN": metrics.get("fn", 0),
            "TN": metrics.get("tn", 0),
            "Tokens": usage.get("total_tokens", 0),
        })
    summary_df = pd.DataFrame(rows)
    best = summary_df.sort_values(["Human-label F1", "Recall", "Precision"], ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(f"Best {compare_by.lower()}", str(best.get(compare_by, "n/a")), "ok")
    with c2:
        kpi("Best human-label F1", pct(best.get("Human-label F1", 0.0)), "ok")
    with c3:
        kpi("Best recall", pct(best.get("Recall", 0.0)), "warn")
    with c4:
        kpi("Runs", str(len(results)), "ok")

    st.markdown("### Human-Oracle Comparison")
    _render_pro_dataframe(summary_df, hide_index=True)
    metric_df = summary_df[[compare_by, "Human-label F1", "Precision", "Recall", "Accuracy"]].melt(
        compare_by,
        var_name="Metric",
        value_name="Score",
    )
    metric_df["zero"] = 0.0
    order = list(summary_df.sort_values("Human-label F1", ascending=True)[compare_by])
    ranking_base = alt.Chart(metric_df).encode(
        y=alt.Y(f"{compare_by}:N", title=None, sort=order, axis=alt.Axis(labelLimit=240)),
        x=alt.X("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color(
            "Metric:N",
            scale=alt.Scale(
                domain=["Human-label F1", "Precision", "Recall", "Accuracy"],
                range=["#2563eb", "#ef4444", "#f59e0b", "#10b981"],
            ),
        ),
        tooltip=[f"{compare_by}:N", "Metric:N", alt.Tooltip("Score:Q", format=".2%")],
    )
    ranking_chart = (
        ranking_base.mark_rule(opacity=0.28, strokeWidth=2).encode(x=alt.X("zero:Q"), x2="Score:Q")
        + ranking_base.mark_circle(size=115, opacity=0.95)
    ).properties(height=max(180, 58 * len(summary_df)))

    confusion_df = summary_df[[compare_by, "TP", "FP", "FN", "TN"]].melt(compare_by, var_name="Outcome", value_name="Count")
    confusion_chart = (
        alt.Chart(confusion_df)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("Outcome:N", title=None, sort=["TP", "FP", "FN", "TN"]),
            y=alt.Y(f"{compare_by}:N", title=None, sort=order, axis=alt.Axis(labelLimit=240)),
            color=alt.Color(
                "Count:Q",
                title="Count",
                scale=alt.Scale(scheme="blues"),
            ),
            tooltip=[f"{compare_by}:N", "Outcome:N", "Count:Q"],
        )
        .properties(height=max(180, 58 * len(summary_df)))
    )
    confusion_text = (
        alt.Chart(confusion_df)
        .mark_text(fontSize=13, fontWeight="bold")
        .encode(
            x=alt.X("Outcome:N", sort=["TP", "FP", "FN", "TN"]),
            y=alt.Y(f"{compare_by}:N", sort=order),
            text="Count:Q",
            color=alt.condition(alt.datum.Count > max(1, float(confusion_df["Count"].max()) * 0.55), alt.value("white"), alt.value("#0f172a")),
        )
    )
    c_left, c_right = st.columns([1.35, 1.0], gap="large")
    with c_left:
        st.markdown("### Score Profile")
        st.altair_chart(ranking_chart, use_container_width=True)
    with c_right:
        st.markdown("### Error Shape")
        st.altair_chart(confusion_chart + confusion_text, use_container_width=True)

    pareto = (
        alt.Chart(summary_df)
        .mark_circle(size=220, opacity=0.88, stroke="#0f172a", strokeWidth=0.6)
        .encode(
            x=alt.X("Precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            size=alt.Size("Human-label F1:Q", title="F1", scale=alt.Scale(range=[120, 520])),
            color=alt.Color(f"{compare_by}:N", title=compare_by),
            tooltip=[
                f"{compare_by}:N",
                alt.Tooltip("Human-label F1:Q", format=".2%"),
                alt.Tooltip("Precision:Q", format=".2%"),
                alt.Tooltip("Recall:Q", format=".2%"),
                alt.Tooltip("Accuracy:Q", format=".2%"),
                "Tokens:Q",
            ],
        )
        .properties(height=330)
    )
    label_layer = (
        alt.Chart(summary_df)
        .mark_text(align="left", dx=9, dy=-5, fontSize=12)
        .encode(
            x="Precision:Q",
            y="Recall:Q",
            text=f"{compare_by}:N",
            color=alt.value("#334155"),
        )
    )
    st.markdown("### Precision/Recall Pareto")
    st.altair_chart(pareto + label_layer, use_container_width=True)

    smell_rows = []
    for result in results:
        label = str(result.get("label", ""))
        for item in result.get("rows") or []:
            gold = bool(item.get("gold"))
            predicted = bool(item.get("predicted"))
            if gold and predicted:
                outcome = "TP"
            elif not gold and predicted:
                outcome = "FP"
            elif gold and not predicted:
                outcome = "FN"
            else:
                outcome = "TN"
            smell_rows.append({
                compare_by: label,
                "Smell": str(item.get("smell", "unknown")),
                "Outcome": outcome,
                "Count": 1,
            })
    if smell_rows:
        smell_df = pd.DataFrame(smell_rows).groupby([compare_by, "Smell", "Outcome"], as_index=False)["Count"].sum()
        smell_df["Prompt / smell"] = smell_df[compare_by].astype(str) + "  |  " + smell_df["Smell"].astype(str)
        smell_order = (
            smell_df[[compare_by, "Smell", "Prompt / smell"]]
            .drop_duplicates()
            .sort_values([compare_by, "Smell"], ascending=[True, True])["Prompt / smell"]
            .tolist()
        )
        smell_chart = (
            alt.Chart(smell_df)
            .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
            .encode(
                x=alt.X("Count:Q", title="Decisions"),
                y=alt.Y("Prompt / smell:N", title=None, sort=smell_order, axis=alt.Axis(labelLimit=360)),
                color=alt.Color(
                    "Outcome:N",
                    title="Outcome",
                    scale=alt.Scale(domain=["TP", "FP", "FN", "TN"], range=["#16a34a", "#ef4444", "#f59e0b", "#64748b"]),
                ),
                order=alt.Order("Outcome:N", sort="ascending"),
                tooltip=[f"{compare_by}:N", "Smell:N", "Outcome:N", "Count:Q"],
            )
            .properties(height=max(260, 34 * len(smell_order)))
        )
        smell_text = (
            alt.Chart(smell_df)
            .mark_text(dx=-8, color="white", fontSize=12, fontWeight="bold")
            .encode(
                x=alt.X("sum(Count):Q", stack="zero"),
                y=alt.Y("Prompt / smell:N", sort=smell_order),
                detail="Outcome:N",
                text=alt.Text("Count:Q"),
                color=alt.condition(alt.datum.Count > 0, alt.value("white"), alt.value("transparent")),
            )
        )
        st.markdown("### Error Distribution By Smell")
        st.altair_chart(smell_chart + smell_text, use_container_width=True)

    selected = st.selectbox(
        "Review sample decisions",
        [str(result.get("label", idx)) for idx, result in enumerate(results)],
        key=f"dacos_review_{compare_by}",
    )
    result = next((item for item in results if str(item.get("label", "")) == selected), results[0])
    detail_df = pd.DataFrame(result.get("rows") or [])
    if not detail_df.empty:
        detail_df["Human label"] = detail_df["gold"].map(lambda x: "present" if x else "absent")
        detail_df["LLM label"] = detail_df["predicted"].map(lambda x: "present" if x else "absent")
        _render_pro_dataframe(detail_df[["sample_id", "smell", "Human label", "LLM label", "confidence", "rationale"]], hide_index=True)


def _render_maintenance_activity_intro(activity: str) -> None:
    descriptions = {
        "Code smell detection": {
            "goal": "Detect code-quality issues and compare LLM findings against static-analyzer ground truth.",
            "input": "Project ZIP + analyzer CSV with file/span locations.",
            "output": "Detected smells or maintainability findings with file/span evidence.",
            "metrics": "Precision, recall, F1, TP/FP/FN, token usage, cost.",
        },
        "Feature location": {
            "goal": "Locate the source files that implement or refine a requested feature.",
            "input": "Feature description + repository candidates + feature-location ground truth.",
            "output": "Ranked files likely to implement the feature.",
            "metrics": "Hit@K, Recall@K, MRR, MAP.",
        },
        "Bug location": {
            "goal": "Locate files likely to require a fix from a bug report or issue description.",
            "input": "Bug report title/body + repository candidates + files changed by the fix.",
            "output": "Ranked suspicious files for the bug.",
            "metrics": "Hit@K, Recall@K, MRR, MAP.",
        },
    }
    info = descriptions.get(activity)
    if not info:
        return
    st.markdown(f"""
    <div class="section">
      <h3 style="margin-top:0">{escape(activity)}</h3>
      <p class="muted">{escape(info["goal"])}</p>
      <div class="badges">
        <span class="badge"><b>Input</b> {escape(info["input"])}</span>
        <span class="badge"><b>Output</b> {escape(info["output"])}</span>
        <span class="badge"><b>Metrics</b> {escape(info["metrics"])}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _rank_feature_candidates(query: str, candidates: list[str]) -> list[str]:
    terms = {
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z0-9]+", query)
        if len(t) >= 4 and t.lower() not in {"feature", "description", "diagram", "diagrams"}
    }
    def score(path: str) -> tuple[int, int, str]:
        low = path.lower()
        hits = sum(1 for term in terms if term in low)
        package_bonus = 2 if "org/argouml" in low else 0
        return (hits + package_bonus, -len(path), path)
    return sorted(candidates, key=score, reverse=True)


def _run_feature_location_batch(
    tasks: list[Any],
    candidates: list[str],
    llm: ChatModel,
    top_k: int,
    max_candidates: int,
    prompt_style: str,
    prompt_templates: dict[str, str] | None = None,
) -> dict[str, Any]:
    task_results = []
    for task in tasks:
        focused_candidates = _rank_feature_candidates(task.query, candidates)[:max_candidates]
        task_results.append(run_location_task(
            task,
            focused_candidates,
            llm,
            top_k=top_k,
            prompt_style=prompt_style,
            prompt_templates=prompt_templates,
        ))

    metric_keys = sorted({k for row in task_results for k in (row.get("metrics") or {})})
    metrics = {}
    for key in metric_keys:
        vals = [float((row.get("metrics") or {}).get(key, 0.0)) for row in task_results]
        metrics[key] = sum(vals) / len(vals) if vals else 0.0

    usage = {
        "prompt_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens", 0)) for row in task_results),
        "completion_tokens": sum(int((row.get("usage") or {}).get("completion_tokens", 0)) for row in task_results),
        "total_tokens": sum(int((row.get("usage") or {}).get("total_tokens", 0)) for row in task_results),
    }
    return {
        "task_id": "feature-location-batch" if len(tasks) > 1 else task_results[0].get("task_id"),
        "task_type": "feature_location",
        "project": "ArgoUML",
        "prompt_style": prompt_style,
        "metrics": metrics,
        "usage": usage,
        "predictions": task_results[0].get("predictions", []) if task_results else [],
        "raw_response": task_results[0].get("raw_response", "") if task_results else "",
        "task_results": task_results,
        "task_count": len(tasks),
    }


def _render_location_results(results: list[dict[str, Any]], task: Any) -> None:
    if not results:
        st.info("No location results to display.")
        return
    summary_rows = []
    for result in results:
        metrics = result.get("metrics", {})
        usage = result.get("usage", {})
        summary_rows.append({
            "Item": result.get("label", result.get("prompt_style", "run")),
            "Prompt": result.get("prompt_style", ""),
            "Hit@1": metrics.get("hit@1", 0.0),
            "Hit@5": metrics.get("hit@5", 0.0),
            "Hit@10": metrics.get("hit@10", 0.0),
            "Recall@10": metrics.get("recall@10", 0.0),
            "MRR": metrics.get("mrr", 0.0),
            "MAP": metrics.get("map", 0.0),
            "Total tokens": usage.get("total_tokens", 0),
        })
    summary_df = pd.DataFrame(summary_rows)

    best = summary_df.sort_values(["MRR", "MAP", "Hit@1"], ascending=False).head(1)
    metrics = results[0].get("metrics", {}) if results else {}
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Best item", str(best["Item"].iloc[0]) if len(best) else "n/a", "ok")
    with c2: kpi("Best Hit@1", pct(float(best["Hit@1"].iloc[0])) if len(best) else "0.00%", "ok")
    with c3: kpi("Best MRR", f"{float(best['MRR'].iloc[0]):.3f}" if len(best) else "0.000", "ok")
    with c4: kpi("Gold files", str(len(task.gold_locations)), "warn")

    st.markdown("### Location metrics")
    _render_pro_dataframe(summary_df, hide_index=True)
    if len(summary_df) > 1:
        chart_df = summary_df.set_index("Item")
        _render_wide_metric_chart(
            chart_df,
            ["Hit@1", "Hit@5", "Hit@10", "MRR", "MAP"],
            group_label="Item",
            value_label="Score",
            value_format=".2%",
            height=320,
        )

    result_labels = [str(r.get("label", idx)) for idx, r in enumerate(results)]
    selected = st.selectbox("Review ranked predictions", result_labels, index=0)
    result = results[result_labels.index(selected)]
    pred_rows = []
    for p in result.get("predictions", []):
        file_name = str(p.get("file", ""))
        pred_rows.append({
            "Rank": p.get("rank"),
            "File": file_name,
            "Hit": any(paths_match(file_name, g.file) for g in task.gold_locations),
            "Confidence": p.get("confidence"),
            "Rationale": p.get("rationale", ""),
        })
    st.markdown("### Ranked predictions")
    if result.get("task_count", 1) > 1:
        detail_rows = []
        for row in result.get("task_results", []):
            metrics = row.get("metrics", {})
            detail_rows.append({
                "Feature": row.get("task_id", ""),
                "Hit@1": metrics.get("hit@1", 0.0),
                "Hit@5": metrics.get("hit@5", 0.0),
                "MRR": metrics.get("mrr", 0.0),
                "MAP": metrics.get("map", 0.0),
            })
        with st.expander("Per-feature results", expanded=False):
            _render_pro_dataframe(pd.DataFrame(detail_rows), hide_index=True)
        st.caption("Ranked predictions below show the first selected feature as a preview.")
    _render_pro_dataframe(pd.DataFrame(pred_rows), hide_index=True)
    with st.expander("Raw LLM response", expanded=False):
        st.code(result.get("raw_response", ""), language="json")

if page in ("🧭 Maintenance Tasks", "⚙️ Run Experiments"):
    main_col, settings_col = st.columns([3.2, 1.05], gap="large")
    with settings_col:
        st.markdown('<div class="right-settings-title">Settings</div>', unsafe_allow_html=True)
        settings_box = st.container()
    with main_col:
        if page == "🧭 Maintenance Tasks":
            render_tab_maintenance_tasks(settings_target=settings_box)
        else:
            render_tab_run_experiments(render_sidebar=True, settings_target=settings_box)

# ==========================================================
# Tab 3 : Results DB
# ==========================================================
def render_tab_results_db():
    st.markdown("## 🗃️ Stored Experiment Results (Database)")
    session = Session()
    try:
        results = session.query(SmellDetectionResult).all()
    except Exception as e:
        st.error(f"DB error: {e}")
        session.close()
        return

    if not results:
        st.info("ℹ️ No results stored in the database.")
        session.close()
        return

    df = pd.DataFrame([{
        "Project": r.project, "Language": r.language, "Filename": r.filename,
        "Strategy": r.strategy, "Precision": r.precision, "Recall": r.recall,
        "F1 Score": r.f1, "Top-K (files in U)": r.top_k, "Time (s)": r.time_elapsed,
        "LLM Used": r.llm_used, "Prompt tokens": r.prompt_tokens,
        "Completion tokens": r.completion_tokens, "Total tokens": r.total_tokens,
        "USD cost": r.usd_cost, "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    } for r in results])
    _render_pro_dataframe(df, hide_index=True)

    try:
        pivot_df = df.pivot_table(index="Project", columns=["LLM Used","Strategy"], values="F1 Score", aggfunc="mean").sort_index(axis=1)
        if isinstance(pivot_df.columns, pd.MultiIndex):
            pivot_df.columns = [f"{(llm or 'Unknown')} · {(strategy or 'n/a')}" for (llm,strategy) in pivot_df.columns.to_list()]
        else:
            pivot_df.columns = pivot_df.columns.astype(str)
        pivot_df = pivot_df.fillna(0)
        st.markdown("### 📈 Global Comparison (F1 across Projects • by LLM & Strategy)")
        global_chart_df = (
            pivot_df.reset_index()
            .melt(id_vars="Project", var_name="LLM / Strategy", value_name="F1 Score")
        )
        _render_bar_chart(
            global_chart_df,
            x="Project",
            y="F1 Score",
            color="LLM / Strategy",
            y_title="F1 Score",
            value_format=".2%",
            height=360,
        )
    except Exception:
        st.info("Pivot not available (insufficient data).")
    session.close()

# ==========================================================
# Tab 3 : Manage Prompts
# ==========================================================
def render_tab_manage_prompts():
    st.markdown("## 📝 Prompt Repository")
    st.caption("Manage prompts as a real repository: one category per maintenance task, with tags, descriptions, and reusable templates.")

    repo = _load_prompt_repository()
    task_options = list(PROMPT_TASKS.keys())
    task_labels = [repo[key]["label"] for key in task_options]
    left, right = st.columns([0.92, 1.55], gap="large")

    with left:
        selected_label = st.selectbox("Task category", task_labels, index=0)
        task_key = task_options[task_labels.index(selected_label)]
        task_repo = repo[task_key]
        prompts = task_repo.get("prompts", {})
        kpi("Prompt templates", str(len(prompts)), "ok")
        st.caption(task_repo.get("description", ""))

        rows = [
            {
                "Name": name,
                "Tags": ", ".join(entry.get("tags", [])),
                "Source": entry.get("source", "repository"),
                "Chars": len(entry.get("template", "")),
            }
            for name, entry in prompts.items()
        ]
        if rows:
            _render_pro_dataframe(pd.DataFrame(rows), hide_index=True)
        else:
            st.info("No prompt in this category yet.")

        st.markdown("### Add Template")
        new_name = st.text_input("Template id", key=f"new_prompt_name_{task_key}")
        new_tags = st.text_input("Tags", value="json", help="Comma-separated tags.", key=f"new_prompt_tags_{task_key}")
        new_prompt = st.text_area("Template", height=180, key=f"new_prompt_text_{task_key}")
        if st.button("➕ Add to repository", key=f"add_prompt_{task_key}", use_container_width=True):
            clean_name = re.sub(r"[^A-Za-z0-9_-]+", "_", new_name.strip()).strip("_")
            if not clean_name:
                st.error("Provide a template id.")
            elif clean_name in prompts:
                st.error("This template id already exists in the selected category.")
            elif not new_prompt.strip():
                st.error("Provide a prompt template.")
            else:
                prompts[clean_name] = _prompt_entry(
                    new_prompt,
                    tags=[tag.strip() for tag in new_tags.split(",") if tag.strip()],
                    source="repository",
                )
                _save_prompt_repository(repo)
                st.success(f"Prompt `{clean_name}` added to {task_repo['label']}.")
                st.rerun()

    with right:
        prompts = repo[task_key].get("prompts", {})
        if not prompts:
            return
        selected_prompt = st.selectbox("Template", list(prompts.keys()), key=f"edit_prompt_select_{task_key}")
        selected_entry = prompts[selected_prompt]
        c1, c2 = st.columns([1, 1])
        with c1:
            edited_name = st.text_input("Template id", value=selected_prompt, key=f"edit_prompt_name_{task_key}")
        with c2:
            edited_tags = st.text_input(
                "Tags",
                value=", ".join(selected_entry.get("tags", [])),
                key=f"edit_prompt_tags_{task_key}",
            )
        edited_prompt = st.text_area(
            "Prompt template",
            value=selected_entry.get("template", ""),
            height=360,
            key=f"edit_prompt_text_{task_key}",
        )
        st.caption("Available placeholders depend on the task. Code smell uses `{language}` and `{code}`. Localization uses `{task_name}`, `{project}`, `{language}`, `{candidate_level}`, `{query}`, `{candidate_text}`, `{top_k}`.")

        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            if st.button("💾 Save template", key=f"save_prompt_{task_key}", type="primary", use_container_width=True):
                clean_name = re.sub(r"[^A-Za-z0-9_-]+", "_", edited_name.strip()).strip("_")
                if not clean_name:
                    st.error("Template id cannot be empty.")
                    return
                if clean_name != selected_prompt and clean_name in prompts:
                    st.error("Another template already uses this id.")
                    return
                entry = _prompt_entry(
                    edited_prompt,
                    tags=[tag.strip() for tag in edited_tags.split(",") if tag.strip()],
                    source="repository",
                )
                if clean_name != selected_prompt:
                    prompts.pop(selected_prompt, None)
                prompts[clean_name] = entry
                _save_prompt_repository(repo)
                st.success(f"Prompt `{clean_name}` saved.")
                st.rerun()
        with b2:
            if st.button("📄 Duplicate", key=f"duplicate_prompt_{task_key}", use_container_width=True):
                base = f"{selected_prompt}_copy"
                candidate = base
                i = 2
                while candidate in prompts:
                    candidate = f"{base}_{i}"
                    i += 1
                prompts[candidate] = dict(selected_entry)
                prompts[candidate]["source"] = "repository"
                _save_prompt_repository(repo)
                st.success(f"Prompt duplicated as `{candidate}`.")
                st.rerun()
        with b3:
            if st.button("🗑️ Delete", key=f"delete_prompt_{task_key}", use_container_width=True):
                if len(prompts) <= 1:
                    st.error("Keep at least one prompt in the category.")
                else:
                    prompts.pop(selected_prompt, None)
                    _save_prompt_repository(repo)
                    st.success(f"Prompt `{selected_prompt}` deleted.")
                    st.rerun()

        _render_prompt_preview_panel("Repository Preview", {selected_prompt: edited_prompt}, selected_prompt)

def render_tab_autoresearch():
    st.markdown("## 🧪 AutoResearch")
    st.caption("Compare prompt candidates on a fixed StarLLM benchmark, then keep the prompt that performs best under the chosen research objective.")

    settings, main = st.columns([1.05, 2.4], gap="large")
    with settings:
        st.markdown("### Settings")
        task_label = st.selectbox(
            "Benchmark task",
            ["Feature location", "Code smell detection"],
            index=0,
            help="Choose the StarLLM benchmark surface to optimize prompts against.",
        )
        task_key = "feature_location" if task_label == "Feature location" else "code_smell_detection"
        code_smell_oracle = "Analyzer reference (SonarQube CSV)"
        human_dataset_name = "MLCQ"
        if task_key == "code_smell_detection":
            code_smell_oracle = st.selectbox(
                "Code-smell oracle",
                ["Analyzer reference (SonarQube CSV)", "Human oracle (DACOS/DACOSX/MLCQ)"],
                index=1,
                key="ar_code_smell_oracle",
                help="Use analyzer replication for SonarQube-style prompts, or human labels for DACOS/MLCQ binary smell prompts.",
            )
            if code_smell_oracle.startswith("Human"):
                task_key = "human_smell_oracle"
                task_label = "Human smell oracle"
                human_dataset_name = st.selectbox(
                    "Human oracle dataset",
                    ["MLCQ", "DACOS/DACOSX"],
                    index=0,
                    key="ar_human_oracle_dataset",
                )
        llm = build_llm("AutoResearch ", target=st.container())
        search_mode = st.radio(
            "Search mode",
            ["Iterative improvement loop", "Compare selected prompts"],
            index=0,
            help="The loop mutates the best prompt so far and accepts only improvements. Compare mode just evaluates selected prompts once.",
        )

        st.markdown("**Prompt candidates**")
        prompt_templates = _dacos_prompt_templates() if task_key == "human_smell_oracle" else _prompt_templates_for_task(task_key)
        if not prompt_templates:
            fallback_key = "feature_location" if task_key == "feature_location" else "code_smell_detection"
            prompt_templates = _prompt_templates_for_task(fallback_key)
        metric_options = ["mrr", "map", "hit@1", "hit@5", "recall@10"] if task_key == "feature_location" else ["f1", "precision", "recall", "accuracy"]
        metric_label_to_key = {_metric_choice_label(metric): metric for metric in metric_options}
        selected_metric_label = st.selectbox(
            "Research objective",
            list(metric_label_to_key.keys()),
            index=0,
            help="This decides which prompt candidate wins the benchmark.",
        )
        primary_metric = metric_label_to_key[selected_metric_label]
        st.info(_metric_help(primary_metric))
        generated_key = f"ar_generated_prompts_{task_key}"
        generated_prompts = st.session_state.setdefault(generated_key, {})
        if generated_prompts:
            prompt_templates = {**prompt_templates, **generated_prompts}

        prompt_names = list(prompt_templates.keys())
        if search_mode == "Iterative improvement loop":
            seed_prompt = st.selectbox("Starting prompt", prompt_names, index=0, key=f"ar_loop_seed_{task_key}") if prompt_names else ""
            loop_iterations = st.slider("Improvement attempts", 1, 10, 3, 1, key=f"ar_loop_iterations_{task_key}")
            candidates_per_attempt = st.slider(
                "Mutations per attempt",
                1,
                4,
                2,
                1,
                key=f"ar_loop_candidates_{task_key}",
                help="Generate several mutations at each step, benchmark them, and let the best compete with the current best prompt.",
            )
            focus_options = (
                ["Auto", "Improve coverage", "Reduce false positives", "Improve JSON/location discipline", "Use analyzer taxonomy"]
                if task_key == "code_smell_detection"
                else ["Auto", "Improve MLCQ severity alignment", "Reduce false positives", "Improve recall on positive smells", "Use smell definitions"]
                if task_key == "human_smell_oracle"
                else ["Auto", "Improve top-ranked file", "Improve coverage of relevant files", "Diversify ranked files", "Use query terminology"]
            )
            mutation_focus = st.selectbox(
                "Mutation focus",
                focus_options,
                index=0,
                key=f"ar_loop_focus_{task_key}",
                help="Guides the LLM when it proposes the next prompt mutation.",
            )
            min_gain = st.slider(
                "Minimum gain to accept",
                0.0,
                0.10,
                0.0,
                0.005,
                key=f"ar_loop_min_gain_{task_key}",
                help="A mutation must improve the selected objective by at least this absolute amount.",
            )
            selected_prompts = [seed_prompt] if seed_prompt else []
        else:
            with st.expander("Generate prompt variants", expanded=False):
                st.caption("Generate temporary prompt candidates with the selected LLM. They are benchmarked like repository prompts, but are not saved unless you promote them later.")
                seed_options = list(prompt_templates.keys())
                seed_prompt = st.selectbox("Seed prompt", seed_options, index=0, key=f"ar_seed_{task_key}") if seed_options else ""
                variant_count = st.slider("Variants", 1, 5, 3, 1, key=f"ar_variant_count_{task_key}")
                if st.button("Generate variants", key=f"ar_generate_{task_key}", use_container_width=True):
                    if not seed_prompt:
                        st.error("Select a seed prompt first.")
                    else:
                        try:
                            variants = _generate_autoresearch_prompt_variants(
                                task_key,
                                seed_prompt,
                                prompt_templates[seed_prompt],
                                primary_metric,
                                variant_count,
                                llm,
                            )
                        except Exception as e:
                            st.error(f"Could not generate prompt variants: {e}")
                        else:
                            st.session_state[generated_key] = {**generated_prompts, **variants}
                            st.success(f"Generated {len(variants)} temporary prompt variant(s).")
                            st.rerun()
                if generated_prompts and st.button("Clear generated variants", key=f"ar_clear_generated_{task_key}", use_container_width=True):
                    st.session_state[generated_key] = {}
                    st.rerun()

            default_prompts = [name for name in prompt_names if name.startswith("auto_")][:3] or prompt_names[: min(3, len(prompt_names))]
            selected_prompts = st.multiselect("Prompt templates", prompt_names, default=default_prompts)
            loop_iterations = 0
            candidates_per_attempt = 1
            mutation_focus = "Auto"
            min_gain = 0.0

        st.markdown("**Benchmark scope**")
        if task_key == "feature_location":
            gt_path = st.text_input("GT folder", "data/apps/Feature Location-ArgoUML")
            repo_zip = st.text_input("Source ZIP", "data/apps/ArgoUML/ArgoUML.zip")
            max_tasks = st.slider("Feature scenarios", 1, 20, 3, 1)
            candidate_budget = st.slider("Candidate budget", 50, 1000, 300, 50)
            top_k = st.slider("Top-K predictions", 1, 30, 10, 1, key="ar_fl_top_k")
            args = SimpleNamespace(
                task="feature_location",
                gt_path=gt_path,
                repo_zip=repo_zip,
                max_tasks=max_tasks,
                candidate_budget=candidate_budget,
                top_k=top_k,
                primary_metric=primary_metric,
                output_dir="output/autoresearch",
            )
        elif task_key == "human_smell_oracle":
            sample_limit = st.slider("Human-oracle samples", 3, 100, 10, 1, key="ar_human_sample_limit")
            if human_dataset_name == "MLCQ":
                human_path = st.text_input("MLCQ samples CSV", "data/apps/MLCQ/MLCQCodeSmellSamples.csv", key="ar_mlcq_path")
                positive_threshold = st.selectbox(
                    "Positive severity threshold",
                    ["minor", "major", "critical"],
                    index=0,
                    key="ar_mlcq_threshold",
                )
                samples = load_mlcq_smell_samples(human_path, limit=sample_limit, positive_threshold=positive_threshold)
                source_path = human_path
            else:
                human_path = st.text_input("DACOSMain/DACOSX file", "data/apps/DACOS/DACOSMain.sql", key="ar_dacos_path")
                files_root = st.text_input("Files folder or files.zip", "data/apps/DACOS/files", key="ar_dacos_files_root")
                samples = load_dacos_smell_samples(human_path, files_root=files_root, limit=sample_limit)
                source_path = human_path
                positive_threshold = "minor"
            if not samples:
                st.warning(f"No {human_dataset_name} samples could be loaded for AutoResearch.")
            args = SimpleNamespace(
                task="human_smell_oracle",
                dataset_name=human_dataset_name,
                source_path=source_path,
                samples=samples,
                sample_limit=sample_limit,
                positive_threshold=positive_threshold,
                primary_metric=primary_metric,
                output_dir="output/autoresearch",
            )
        else:
            demo_sets = _demo_datasets()
            dataset_mode = st.radio("Dataset source", ["Bundled demo", "Custom paths"], index=0, key="ar_smell_dataset_mode")
            if dataset_mode == "Bundled demo" and demo_sets:
                dataset_name = st.selectbox("Demo dataset", list(demo_sets.keys()), key="ar_smell_demo_dataset")
                repo_zip_path, static_csv_path = demo_sets[dataset_name]
                repo_zip = str(repo_zip_path)
                static_csv = str(static_csv_path)
                st.caption(f"ZIP: {repo_zip_path.name}")
                st.caption(f"GT: {static_csv_path.name}")
            else:
                if dataset_mode == "Bundled demo":
                    st.warning("No bundled demo datasets were found. Use custom paths.")
                repo_zip = st.text_input("Source ZIP", "data/apps/ArgoUML/ArgoUML.zip")
                static_csv = st.text_input("Analyzer CSV", "data/apps/ArgoUML/ArgoUml-sonarqube-quality-analysis.csv")
            top_k = st.slider(
                "Sampled files",
                1,
                20,
                5,
                1,
                key="ar_smell_top_k",
                help="How many source files StarLLM samples for this quick benchmark run.",
            )
            pos_ratio = st.slider(
                "Known-issue share",
                0.0,
                1.0,
                0.6,
                0.05,
                key="ar_smell_pos_ratio",
                help="Target share of sampled files that contain analyzer findings.",
            )
            preset = st.selectbox(
                "Match strictness",
                ["Lenient", "Balanced", "Strict"],
                index=1,
                key="ar_smell_preset",
                help="Controls how close LLM-reported lines must be to analyzer lines.",
            )
            require_type = st.checkbox("Require issue category match", value=False)
            args = SimpleNamespace(
                task="code_smell",
                repo_zip=repo_zip,
                static_csv=static_csv,
                top_k=top_k,
                pos_ratio=pos_ratio,
                preset=preset,
                require_type=require_type,
                primary_metric=primary_metric,
                output_dir="output/autoresearch",
            )

    with main:
        st.markdown("### Candidate Review")
        if not prompt_names:
            st.warning("No prompts are available for this benchmark task.")
            return
        if not selected_prompts:
            st.warning("Select a starting prompt." if search_mode == "Iterative improvement loop" else "Select at least one prompt candidate.")
            return
        if task_key == "human_smell_oracle" and not getattr(args, "samples", []):
            st.warning("Load at least one human-oracle sample before running AutoResearch.")
            return

        preview_title = "Starting Prompt" if search_mode == "Iterative improvement loop" else "Prompt Preview"
        _render_prompt_preview_panel(preview_title, prompt_templates, selected_prompts)
        prompt_candidates = [
            PromptCandidate(name=name, template=prompt_templates[name], source="prompt_repository")
            for name in selected_prompts
            if name in prompt_templates
        ]

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi("Task", task_label, "ok")
        with c2:
            kpi("Mode", "Loop" if search_mode == "Iterative improvement loop" else "Compare", "ok")
        with c3:
            kpi("Objective", _metric_label(primary_metric, short=True), "warn")

        st.markdown("### Benchmark Plan")
        if task_key == "feature_location":
            st.markdown(
                f"""
                <div class="section">
                  <b>What this run measures</b>
                  <p class="muted">Each prompt receives the same feature-location scenarios and candidate file list. StarLLM asks the LLM to rank source files, then scores whether relevant files appear early in the ranking.</p>
                  <div class="badges">
                    <span class="badge"><b>Dataset</b> {escape(Path(args.gt_path).name)}</span>
                    <span class="badge"><b>Scenarios</b> {int(args.max_tasks)}</span>
                    <span class="badge"><b>Candidates/prompt</b> {int(args.candidate_budget)}</span>
                    <span class="badge"><b>Winner</b> {escape(_metric_label(primary_metric))}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif task_key == "human_smell_oracle":
            st.markdown(
                f"""
                <div class="section">
                  <b>What this run measures</b>
                  <p class="muted">Each prompt receives the same human-labeled smell snippets. StarLLM asks the LLM for a present/absent decision, then scores agreement with the human oracle.</p>
                  <div class="badges">
                    <span class="badge"><b>Dataset</b> {escape(str(args.dataset_name))}</span>
                    <span class="badge"><b>Samples</b> {int(len(args.samples))}</span>
                    <span class="badge"><b>Oracle</b> Human labels</span>
                    <span class="badge"><b>Winner</b> {escape(_metric_label(primary_metric))}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="section">
                  <b>What this run measures</b>
                  <p class="muted">Each prompt analyzes the same sampled files and is compared against the static analyzer CSV. StarLLM scores how well the LLM findings match the ground-truth issue locations.</p>
                  <div class="badges">
                    <span class="badge"><b>Project</b> {escape(Path(args.repo_zip).stem)}</span>
                    <span class="badge"><b>Sampled files</b> {int(args.top_k)}</span>
                    <span class="badge"><b>Positive ratio</b> {float(args.pos_ratio):.2f}</span>
                    <span class="badge"><b>Winner</b> {escape(_metric_label(primary_metric))}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Run")
        if search_mode == "Iterative improvement loop":
            st.caption("This benchmarks the starting prompt, then repeatedly asks the LLM to mutate the best prompt so far. Several mutations can compete at each attempt; only an improving winner is accepted.")
            button_label = "Run Improvement Loop"
        else:
            st.caption("This evaluates selected prompt candidates once. It does not overwrite the prompt repository or modify Python code.")
            button_label = "Run Prompt Comparison"
        run_clicked = st.button(button_label, type="primary")
        if run_clicked:
            progress = st.progress(0)
            status = st.empty()
            try:
                if search_mode == "Iterative improvement loop":
                    rows, best_candidate = _run_autoresearch_loop(
                        task_key,
                        prompt_candidates[0],
                        primary_metric,
                        int(loop_iterations),
                        int(candidates_per_attempt),
                        str(mutation_focus),
                        float(min_gain),
                        args,
                        llm,
                        progress,
                        status,
                    )
                    prompt_candidates_for_output = [PromptCandidate(str(row.get("prompt")), str(row.get("template", "")), str(row.get("source", "loop"))) for row in rows]
                else:
                    rows = []
                    for idx, candidate in enumerate(prompt_candidates, start=1):
                        status.markdown(f"Running `{candidate.name}` ({idx}/{len(prompt_candidates)})...")
                        result_rows = _evaluate_autoresearch_candidate(task_key, candidate, args, llm)
                        rows.extend(result_rows)
                        progress.progress(idx / max(1, len(prompt_candidates)))
                    best_row = max(rows, key=lambda row: float(row.get("score", 0.0))) if rows else {}
                    best_candidate = next((c for c in prompt_candidates if c.name == best_row.get("prompt")), None)
                    prompt_candidates_for_output = prompt_candidates
            except Exception as e:
                st.error(f"AutoResearch benchmark failed: {e}")
                return

            if not rows:
                st.warning("No results produced.")
                return

            out_dir = write_autoresearch_outputs(rows, prompt_candidates_for_output, args)
            result_df = pd.DataFrame(rows).sort_values("score", ascending=False)
            st.session_state.last_autoresearch_results = {
                "rows": rows,
                "out_dir": str(out_dir),
                "metric": primary_metric,
                "mode": search_mode,
                "best_prompt": best_candidate.name if best_candidate else "",
                "best_template": best_candidate.template if best_candidate else "",
                "task_key": task_key,
            }
            st.success(f"AutoResearch run saved to {out_dir}")
            if search_mode == "Iterative improvement loop":
                _render_autoresearch_loop_results(result_df, primary_metric)
            else:
                _render_autoresearch_results(result_df, primary_metric)
            _render_autoresearch_promote_button(st.session_state.last_autoresearch_results)

        cached = st.session_state.get("last_autoresearch_results")
        if cached and not run_clicked:
            st.markdown("### Last AutoResearch Run")
            st.caption(f"Artifacts: {cached.get('out_dir')}")
            if cached.get("mode") == "Iterative improvement loop":
                _render_autoresearch_loop_results(pd.DataFrame(cached.get("rows", [])), cached.get("metric", primary_metric))
            else:
                _render_autoresearch_results(pd.DataFrame(cached.get("rows", [])), cached.get("metric", primary_metric))
            _render_autoresearch_promote_button(cached)

def _evaluate_autoresearch_candidate(task_key: str, candidate: PromptCandidate, args: Any, llm: ChatModel) -> list[dict[str, Any]]:
    if task_key == "feature_location":
        rows = evaluate_feature_location(args, [candidate], llm)
    elif task_key == "human_smell_oracle":
        started = time.time()
        result = run_dacos_smell_benchmark(
            args.samples,
            llm,
            prompt_template=candidate.template,
            max_samples=getattr(args, "sample_limit", None),
        )
        metrics = result.get("metrics") or {}
        usage = result.get("usage") or {}
        rows = [{
            "rank_order": 1,
            "task": "human_smell_oracle",
            "dataset": getattr(args, "dataset_name", "human"),
            "prompt": candidate.name,
            "source": candidate.source,
            "primary_metric": args.primary_metric,
            "score": metrics.get(args.primary_metric, 0.0),
            "task_count": metrics.get("samples", len(getattr(args, "samples", []))),
            "elapsed_s": round(time.time() - started, 2),
            **metrics,
            **usage,
        }]
    else:
        rows = evaluate_code_smell(args, [candidate], llm)
    for row in rows:
        row["template"] = candidate.template
        row["source"] = candidate.source
    return rows

def _run_autoresearch_loop(
    task_key: str,
    seed_candidate: PromptCandidate,
    primary_metric: str,
    iterations: int,
    candidates_per_attempt: int,
    mutation_focus: str,
    min_gain: float,
    args: Any,
    llm: ChatModel,
    progress: Any,
    status: Any,
) -> tuple[list[dict[str, Any]], PromptCandidate]:
    rows: list[dict[str, Any]] = []
    status.markdown(f"Benchmarking starting prompt `{seed_candidate.name}`...")
    seed_rows = _evaluate_autoresearch_candidate(task_key, seed_candidate, args, llm)
    seed_row = seed_rows[0]
    seed_row.update({
        "iteration": 0,
        "decision": "seed",
        "parent": "",
        "gain_vs_best": 0.0,
    })
    rows.append(seed_row)

    best_candidate = seed_candidate
    best_score = float(seed_row.get("score", 0.0))
    progress.progress(1 / max(1, iterations + 1))

    for iteration in range(1, iterations + 1):
        status.markdown(f"Iteration {iteration}: mutating `{best_candidate.name}`...")
        variants = _generate_autoresearch_prompt_variants(
            task_key,
            best_candidate.name,
            best_candidate.template,
            primary_metric,
            candidates_per_attempt,
            llm,
            history=rows,
            mutation_focus=mutation_focus,
        )
        attempt_rows: list[tuple[dict[str, Any], PromptCandidate]] = []
        for candidate_idx, (variant_name, variant_template) in enumerate(variants.items(), start=1):
            candidate = PromptCandidate(
                name=f"{variant_name}_i{iteration}_{candidate_idx}",
                template=variant_template,
                source=f"mutation_of:{best_candidate.name}",
            )
            status.markdown(f"Iteration {iteration}: benchmarking `{candidate.name}`...")
            candidate_rows = _evaluate_autoresearch_candidate(task_key, candidate, args, llm)
            row = candidate_rows[0]
            score = float(row.get("score", 0.0))
            gain = score - best_score
            row.update({
                "iteration": iteration,
                "decision": "pending",
                "parent": best_candidate.name,
                "gain_vs_best": gain,
                "attempt_candidate": candidate_idx,
            })
            attempt_rows.append((row, candidate))

        winner_row, winner_candidate = max(attempt_rows, key=lambda pair: float(pair[0].get("score", 0.0)))
        winner_score = float(winner_row.get("score", 0.0))
        accepted = (winner_score - best_score) >= min_gain and winner_score > best_score
        for row, candidate in attempt_rows:
            if candidate.name == winner_candidate.name:
                row["decision"] = "accepted" if accepted else "best rejected"
            else:
                row["decision"] = "rejected"
            rows.append(row)
        if accepted:
            best_candidate = winner_candidate
            best_score = winner_score
        progress.progress((iteration + 1) / max(1, iterations + 1))

    status.markdown(f"Done. Best prompt: `{best_candidate.name}`.")
    return rows, best_candidate

def _render_autoresearch_loop_results(result_df: pd.DataFrame, primary_metric: str) -> None:
    if result_df.empty:
        return
    history_df = result_df.sort_values("iteration") if "iteration" in result_df.columns else result_df
    accepted_df = history_df[history_df["decision"].isin(["seed", "accepted"])] if "decision" in history_df.columns else history_df
    best = accepted_df.sort_values("score", ascending=False).iloc[0] if not accepted_df.empty else history_df.sort_values("score", ascending=False).iloc[0]
    seed = history_df.iloc[0]
    improvement = float(best.get("score", 0.0)) - float(seed.get("score", 0.0))

    st.markdown(
        "<div class='section'><b>Loop behavior</b><p class='muted'>StarLLM benchmarks the seed prompt, generates competing mutations from the best prompt so far, and accepts only the winning mutation when it improves the selected objective.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Best prompt", str(best.get("prompt", "n/a")), "ok")
    with c2:
        kpi(_metric_label(primary_metric), _format_metric(primary_metric, best.get("score", 0.0)), "ok")
    with c3:
        kpi("Gain vs seed", _format_metric(primary_metric, improvement), "warn" if improvement > 0 else "bad")
    with c4:
        accepted_count = int((history_df.get("decision", pd.Series(dtype=str)) == "accepted").sum()) if "decision" in history_df.columns else 0
        kpi("Accepted", str(accepted_count), "ok")

    display_rows = []
    for _, row in history_df.iterrows():
        attempt_candidate = row.get("attempt_candidate", "")
        try:
            candidate_display = "" if pd.isna(attempt_candidate) else int(attempt_candidate)
        except Exception:
            candidate_display = ""
        display_rows.append({
            "Iteration": int(row.get("iteration", 0)),
            "Candidate": candidate_display,
            "Prompt": row.get("prompt", ""),
            "Decision": str(row.get("decision", "")),
            "Score": _format_metric(primary_metric, row.get("score", 0.0)),
            "Gain vs previous best": _format_metric(primary_metric, row.get("gain_vs_best", 0.0)),
            "Parent": row.get("parent", ""),
            "Tokens": f"{int(row.get('total_tokens', 0)):,}",
            "Time": f"{float(row.get('elapsed_s', 0.0)):.1f}s",
        })
    st.markdown("### Attempt History")
    _render_pro_dataframe(pd.DataFrame(display_rows), hide_index=True)

    if "iteration" in history_df.columns:
        chart_df = history_df[["iteration", "score", "decision", "prompt"]].copy()
        chart_df["Score"] = chart_df["score"].astype(float)
        chart = alt.Chart(chart_df).mark_line(point=True).encode(
            x=alt.X("iteration:O", title="Iteration"),
            y=alt.Y("Score:Q", title=_metric_label(primary_metric), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("decision:N", title="Decision"),
            tooltip=["iteration:O", "prompt:N", "decision:N", alt.Tooltip("Score:Q", format=".3f")],
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)

    metric_cols = [col for col in ["f1", "precision", "recall", "accuracy", "mrr", "map", "hit@1", "hit@5"] if col in history_df.columns]
    if metric_cols:
        slope_rows = []
        for metric in metric_cols:
            slope_rows.append({"Stage": "Seed", "Metric": _metric_label(metric, short=True), "Score": float(seed.get(metric, 0.0))})
            slope_rows.append({"Stage": "Best", "Metric": _metric_label(metric, short=True), "Score": float(best.get(metric, 0.0))})
        slope_df = pd.DataFrame(slope_rows)
        slope = (
            alt.Chart(slope_df)
            .mark_line(point=alt.OverlayMarkDef(size=95, filled=True), strokeWidth=3)
            .encode(
                x=alt.X("Stage:N", title=None, sort=["Seed", "Best"]),
                y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                color=alt.Color("Metric:N", title="Metric"),
                detail="Metric:N",
                tooltip=["Metric:N", "Stage:N", alt.Tooltip("Score:Q", format=".2%")],
            )
            .properties(height=260)
        )
        st.markdown("### Seed vs Best Shift")
        st.altair_chart(slope, use_container_width=True)

def _render_autoresearch_promote_button(cached: dict[str, Any]) -> None:
    best_template = str(cached.get("best_template") or "")
    best_name = str(cached.get("best_prompt") or "autoresearch_best")
    task_key = str(cached.get("task_key") or "")
    if not best_template or task_key not in PROMPT_TASKS:
        return
    with st.expander("Promote best prompt", expanded=False):
        st.caption("Save the best prompt from this AutoResearch run into the StarLLM prompt repository.")
        clean_default = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{best_name}_promoted").strip("_").lower()
        target_name = st.text_input("Repository prompt id", value=clean_default, key=f"promote_{task_key}_{best_name}")
        if st.button("Save best prompt to repository", key=f"promote_btn_{task_key}_{best_name}", type="primary"):
            clean_name = re.sub(r"[^A-Za-z0-9_-]+", "_", target_name.strip()).strip("_")
            if not clean_name:
                st.error("Provide a prompt id.")
                return
            repo = _load_prompt_repository()
            prompts = (repo.get(task_key) or {}).get("prompts") or {}
            if clean_name in prompts:
                st.error("A prompt with this id already exists.")
                return
            prompts[clean_name] = _prompt_entry(best_template, tags=["autoresearch", "promoted"], source="autoresearch")
            repo[task_key]["prompts"] = prompts
            _save_prompt_repository(repo)
            st.success(f"Saved `{clean_name}` to the prompt repository.")

def _render_autoresearch_results(result_df: pd.DataFrame, primary_metric: str) -> None:
    if result_df.empty:
        return
    result_df = result_df.sort_values("score", ascending=False)
    best = result_df.iloc[0]
    task_name = str(best.get("task", ""))
    if task_name == "code_smell":
        summary = (
            "For code-smell detection, a strong prompt should find analyzer-confirmed issues "
            "without flooding the review board with false positives."
        )
    else:
        summary = (
            "For localization, a strong prompt should rank at least one relevant source file "
            "near the top and keep other relevant files early in the list."
        )
    st.markdown(f"<div class='section'><b>How to read this result</b><p class='muted'>{escape(summary)}</p></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Best prompt", str(best.get("prompt", "n/a")), "ok")
    with c2:
        kpi(_metric_label(primary_metric), _format_metric(primary_metric, best.get("score", 0.0)), "ok")
    with c3:
        kpi("Total tokens", f"{int(best.get('total_tokens', 0)):,}", "warn")
    with c4:
        kpi("Elapsed", f"{float(best.get('elapsed_s', 0.0)):.1f}s", "warn")

    display_df = _autoresearch_display_df(result_df, primary_metric)
    st.markdown("### Ranked Candidates")
    _render_pro_dataframe(display_df, hide_index=True)

    chart_cols = [
        col for col in ["precision", "recall", "f1", "accuracy", "mrr", "map", "hit@1", "hit@5", "recall@10"]
        if col in result_df.columns
    ]
    if len(result_df) > 1 and chart_cols:
        chart_df = result_df[["prompt", *chart_cols]].melt("prompt", var_name="Metric", value_name="Score")
        chart_df["Metric"] = chart_df["Metric"].map(lambda m: _metric_label(str(m), short=True))
        chart_df["zero"] = 0.0
        order = list(result_df.sort_values("score", ascending=True)["prompt"])
        base = alt.Chart(chart_df).encode(
            y=alt.Y("prompt:N", title=None, sort=order, axis=alt.Axis(labelLimit=260)),
            x=alt.X("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["prompt:N", "Metric:N", alt.Tooltip("Score:Q", format=".3f")],
        )
        chart = (
            base.mark_rule(opacity=0.25, strokeWidth=2).encode(x=alt.X("zero:Q"), x2="Score:Q")
            + base.mark_circle(size=105, opacity=0.95)
        ).properties(height=max(220, 54 * len(result_df)))
        st.altair_chart(chart, use_container_width=True)

def _autoresearch_display_df(result_df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    rows = []
    for _, row in result_df.iterrows():
        item = {
            "Prompt candidate": row.get("prompt", ""),
            "Winning objective": _format_metric(primary_metric, row.get("score", 0.0)),
        }
        if "precision" in result_df.columns:
            item["Trustworthiness"] = _format_metric("precision", row.get("precision", 0.0))
        if "recall" in result_df.columns:
            item["Coverage"] = _format_metric("recall", row.get("recall", 0.0))
        if "f1" in result_df.columns:
            item["Balanced score"] = _format_metric("f1", row.get("f1", 0.0))
        if "hit@1" in result_df.columns:
            item["Top answer correct"] = _format_metric("hit@1", row.get("hit@1", 0.0))
        if "hit@5" in result_df.columns:
            item["Correct file in top 5"] = _format_metric("hit@5", row.get("hit@5", 0.0))
        if "mrr" in result_df.columns:
            item["First correct rank score"] = _format_metric("mrr", row.get("mrr", 0.0))
        if "map" in result_df.columns:
            item["Overall ranking quality"] = _format_metric("map", row.get("map", 0.0))
        item["Tokens"] = f"{int(row.get('total_tokens', 0)):,}"
        item["Time"] = f"{float(row.get('elapsed_s', 0.0)):.1f}s"
        item["Scope"] = int(row.get("task_count", 0))
        rows.append(item)
    return pd.DataFrame(rows)

def _generate_autoresearch_prompt_variants(
    task_key: str,
    seed_name: str,
    seed_template: str,
    primary_metric: str,
    count: int,
    llm: ChatModel,
    *,
    history: list[dict[str, Any]] | None = None,
    mutation_focus: str = "Auto",
) -> dict[str, str]:
    if task_key == "human_smell_oracle":
        placeholders = "{smell}, {language}, {code}"
        task_label = "binary code-smell classification against human oracle labels"
    else:
        placeholders = "{language}, {code}" if task_key == "code_smell_detection" else "{task_name}, {project}, {language}, {candidate_level}, {query}, {candidate_text}, {top_k}"
        task_label = "code-smell detection against static analyzer ground truth" if task_key == "code_smell_detection" else "file-level feature/bug localization ranking"
    objective = f"{_metric_label(primary_metric)}: {_metric_help(primary_metric)}"
    history_rows = history or []
    history_summary = "\n".join(
        f"- iter {row.get('iteration', '?')} {row.get('prompt', '')}: score={float(row.get('score', 0.0)):.4f}, "
        f"precision={float(row.get('precision', 0.0)):.4f}, recall={float(row.get('recall', 0.0)):.4f}, "
        f"decision={row.get('decision', '')}"
        for row in history_rows[-8:]
    ) or "- no previous attempts"
    if task_key == "human_smell_oracle":
        domain_guidance = """
Human-oracle smell benchmark guidance:
- The ground truth is a human label such as MLCQ or DACOS, not a static analyzer issue list.
- The prompt must answer only whether the target smell is present in the snippet.
- Preserve strict JSON output with keys: present, confidence, rationale.
- Use MLCQ-style smell definitions: god_class/blob, long_method, data_class, feature_envy, complex_method, long_parameter_list, multifaceted_abstraction.
- For positive recall, emphasize structural evidence such as size, centralization, many fields/methods, branching, parameter count, external-class reliance, and data-holder behavior.
- For precision, require visible evidence and avoid declaring unrelated smells.
"""
    elif task_key == "code_smell_detection":
        domain_guidance = """
Code-smell benchmark guidance:
- The ground truth comes from static analyzer CSVs such as SonarQube.
- Improve line/span discipline: prefer exact startLine/endLine near the concrete issue.
- Avoid broad architectural commentary unless it points to a concrete code location.
- Name issue types with common analyzer-like categories: Long Method, Long Parameter List, Magic Number, Duplicated Code, Deeply Nested Ifs, God Class, Tight Coupling, Unused Variable, Resource Leak, Null Dereference Risk.
- If optimizing coverage, ask for more concrete findings but still require evidence.
- If reducing false positives, ask for fewer high-confidence findings with specific code evidence.
"""
    else:
        domain_guidance = """
Localization benchmark guidance:
- Rank concrete files, not packages or vague components.
- Use query terms, class/file names, UI concepts, model/controller names, and likely implementation responsibilities.
- Put the most likely correct file first, then diversify related files.
"""
    prompt = f"""
Generate {int(count)} improved prompt templates for StarLLM AutoResearch.

Task: {task_label}
Optimization objective: {objective}
Mutation focus: {mutation_focus}
Seed prompt name: {seed_name}
Required placeholders: {placeholders}

Recent benchmark history:
{history_summary}

{domain_guidance}

Rules:
- Return JSON only, no markdown.
- JSON shape: [{{"name":"short_id","template":"prompt template text"}}]
- Keep every generated template compatible with the required placeholders.
- Do not include ground-truth answers, dataset-specific file names, or benchmark leakage.
- Prefer prompts that improve measurable benchmark behavior, not verbose explanations.
- Make each variant meaningfully different from the seed and from the recent rejected attempts.
- Keep templates concise enough for repeated benchmarking.

Seed template:
{seed_template}
"""
    content, _ = llm.chat(
        messages=[
            {"role": "system", "content": "You design benchmarkable prompt variants. Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2500,
        return_meta=True,
    )
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            raise ValueError("The LLM did not return a JSON array of prompt variants.")
        data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("The LLM response must be a JSON array.")

    out: dict[str, str] = {}
    if task_key == "human_smell_oracle":
        required = ["smell", "language", "code"]
    else:
        required = ["language", "code"] if task_key == "code_smell_detection" else ["query", "candidate_text", "top_k"]
    for idx, item in enumerate(data[:count], start=1):
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("name") or f"variant_{idx}")
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name.strip()).strip("_").lower() or f"variant_{idx}"
        template = str(item.get("template") or "").strip()
        if not template:
            continue
        missing = [placeholder for placeholder in required if "{" + placeholder + "}" not in template]
        if missing:
            continue
        candidate_name = f"auto_{name}"
        suffix = 2
        while candidate_name in out:
            candidate_name = f"auto_{name}_{suffix}"
            suffix += 1
        out[candidate_name] = template
    if not out:
        raise ValueError("No usable prompt variants were generated. Try another seed prompt or objective.")
    return out

if page == "📝 Manage Prompts":
    render_tab_manage_prompts()

if page == "🧪 AutoResearch":
    render_tab_autoresearch()

# ==========================================================
# Tab 4 : Results DB
# ==========================================================
if page == "🗃️ Stored Results (DB)":
    render_tab_results_db()

# ==========================================================
# Tab 5 : Batch Experiments (span-level)
# ==========================================================
def render_tab_batch():
    st.markdown("## 🔄 Batch Experiments (span-level)")
    data_dir = st.text_input("📂 Data folder", "data/apps/")
    output_dir = st.text_input("📂 Output folder", "output/apps/")
    strategies = _prompt_templates_for_task("code_smell_detection")
    selected_strategies = st.multiselect("🧩 Select strategies", list(strategies.keys()), default=list(strategies.keys()))
    top_k_values = st.multiselect("Top-K values (files in U)", [10,20,30,50], default=[20,30])
    pos_ratio = st.slider("Positive ratio in U", 0.0, 1.0, 0.5, 0.05, key="batch_pos_ratio")
    preset_b = st.selectbox("Evaluation strictness", ["Lenient","Balanced","Strict"], index=1, key="batch_preset")

    st.markdown("### 🤖 LLM for Batch")
    registry = load_llm_registry()
    if registry:
        keys = list(registry.keys())
        labels = [f"{k} · {registry[k]['label']} ({registry[k]['config'].provider})" for k in keys]
        choice_b = st.selectbox("LLM Slot for batch (.env)", labels, index=0, key="prov_b")
        llm_batch = build_llm_from_slot(keys[labels.index(choice_b)])
    else:
        st.info("No .env slots detected. Falling back to manual selection.")
        prov_b = st.selectbox("LLM Provider (batch)", ["azure-openai","openai","ollama"], index=0, key="prov_b2")
        if prov_b == "ollama":
            default_host_b = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
            host_b = st.text_input("Ollama Host (batch)", value=default_host_b, key="batch_ollama_host",
                                   help="Ollama server URL for batch processing")
            if st.button("🔍 Test Connection", key="test_conn_batch"):
                with st.spinner("Testing connection..."):
                    ok, msg = test_ollama_connectivity(host_b)
                    (st.success if ok else st.error)(msg)
            mdl_b_list = available_models("ollama", host_b) or available_models("ollama", None)
            mdl_b = st.selectbox("Model (batch)", mdl_b_list, index=0, key="mdl_b_ol")
            llm_batch = ChatModel(LLMConfig(provider="ollama", model=mdl_b, api_base=host_b))
        elif prov_b == "azure-openai":
            mdl_b_list = available_models("azure-openai")
            base_default = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE", "")
            ver_default  = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
            with st.expander("Advanced Azure settings (batch)", expanded=False):
                api_base_b = st.text_input("Azure Resource endpoint (batch)", value=base_default, key="api_base_b")
                api_version_b = st.text_input("API version (batch)", value=ver_default, key="api_version_b")
            deployment_b = st.selectbox("Azure deployment name (batch)", mdl_b_list or [""], index=0, key="deployment_b")
            api_key_env_b = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            if not api_key_env_b:
                api_key_env_b = st.text_input("Azure API Key (batch)", value="", type="password", key="api_key_b")
                if api_key_env_b: os.environ["AZURE_OPENAI_API_KEY"] = api_key_env_b
            llm_batch = ChatModel(LLMConfig(provider="azure-openai", model=deployment_b, api_base=api_base_b, api_version=api_version_b, api_key=api_key_env_b))
        else:
            mdl_b_list = available_models("openai")
            mdl_b = st.selectbox("Model (batch)", mdl_b_list, index=0, key="mdl_b")
            api_key_env_b = os.getenv("OPENAI_API_KEY") or ""
            if not api_key_env_b:
                api_key_env_b = st.text_input("OpenAI API Key (batch)", value="", type="password", key="openai_key_b")
                if api_key_env_b: os.environ["OPENAI_API_KEY"] = api_key_env_b
            base_url_b = os.getenv("OPENAI_BASE_URL") or None
            llm_batch = ChatModel(LLMConfig(provider="openai", model=mdl_b, api_base=base_url_b, api_key=api_key_env_b))

    if st.button("🚀 Run Batch Analysis"):
        if not selected_strategies:
            st.warning("⚠️ Please select at least one strategy.")
            return

        projects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for project in projects:
            st.write(f"▶️ Running project: {project}")
            project_dir = Path(data_dir, project)
            zip_files = [str(project_dir / f) for f in os.listdir(project_dir) if f.endswith(".zip")]
            csv_files = [str(project_dir / f) for f in os.listdir(project_dir) if f.endswith(".csv")]
            if not zip_files or not csv_files:
                st.warning(f"❌ Missing files for {project} (need at least one .zip and one .csv)")
                continue
            csv_path = csv_files[0]
            proj_out = Path(output_dir, project); proj_out.mkdir(parents=True, exist_ok=True)
            all_results = []
            for zip_path in zip_files:
                app_name = Path(zip_path).stem
                st.write(f"🔍 Analyzing ZIP: {app_name}")
                for k in top_k_values:
                    summary_U, metrics_df, samples = run_selected_experiments(
                        zip_path, csv_path, selected_strategies,
                        top_k=int(k), pos_ratio=pos_ratio, preset=preset_b, llm=llm_batch
                    )
                    (proj_out / f"{app_name}_results_top{k}.csv").write_text(metrics_df.to_csv())
                    md = metrics_df.copy(); md["Project"]=project; md["ZIP"]=app_name; md["Top-K"]=int(k)
                    all_results.append(md); st.success(f"✅ {project}/{app_name} done (Top-K={k})")
            if all_results:
                import pandas as _pd
                global_df = _pd.concat(all_results)
                (proj_out / "global_results.csv").write_text(global_df.to_csv())
                st.info(f"📁 Global results saved for {project}: {proj_out/'global_results.csv'}")

if page == "🔄 Batch Experiments":
    render_tab_batch()

# ==========================================================
# Tab 6 : Guide & Examples (English, with visuals)
# ==========================================================
def render_tab_guide():
    st.markdown("## 📘 How to Read the Metrics (Span-Level, with Negatives)")
    st.markdown("""
**What we evaluate**  
LLM **code-smell spans** vs. a **ground truth (GT)** CSV containing `(file, startLine, endLine, startColumn, endColumn, rule/type)`.

**Universe of files U**  
Mixed **U** (Top-K files): **positives** (≥1 GT) + **negatives** (0 GT). You control **Top-K** and **Positive ratio**.

**Matching (tolerant)**  
Greedy 1-to-1 **on the same file**. Accepted if:  
• (optional) **rule/type matches** • for lines: `IoU ≥ threshold` **or** distance ≤ `δ` • (optional) for single-line spans: also check **columns**.

**Metrics**  
TP, FP, FN → Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R).
""")

    # --- Strictness block (table + example with line numbers)
    st.markdown("### 🎛️ Evaluation strictness")
    strict_df = pd.DataFrame(
        {"IoU min": [0.10, 0.25, 0.50],
         "Line tolerance δ": [3, 2, 1]},
        index=["Lenient","Balanced","Strict"]
    )
    st.table(strict_df)

    st.caption("A (Prediction P, GT G) pair is accepted if (IoU≥threshold) OR (distance≤δ). "
               "If both are single-line and columns exist (and enabled), apply the same rule on columns.")

    # Mini example
    st.markdown("#### 🧪 Mini example (A.java)")
    code_eval = """ 18:   // …
 19:   // …
 20:   int sum = 0;                 // GT start
 21:   for (int i=0; i<n; i++) {
 22:       sum += arr[i];
 23:       // ...
 24:       // ...
 25:       // ...
 26:       // ...
 27:       // ...
 28:       // ...
 29:       // ...
 30:       // ...
 31:       // ...
 32:       // ...
 33:       // ...
 34:       // ...
 35:       // ...
 36:       // ...
 37:       // ...
 38:       // ...
 39:       // ...
 40:   }                            // GT end
 41:   // …
 42:   if (sum > 1000) {
 43:       sum -= 1000;
 44:   }"""
    render_code_with_highlight("A.java — GT: 20–40", code_eval, [(20,40)])

    st.markdown("""
**GT**: 20–40  
**P1**: 18–22 → IoU ≈ 0.13, distance=0 ⇒ **match** (all presets, due to distance=0)  
**P2**: 21–39 → IoU ≈ 0.90 ⇒ **match** (all presets)  
**P3**: 41–41 → IoU=0, distance=1 ⇒ **match** if δ≥1 (Lenient/Balanced/Strict as configured).  
→ If you want to **reject** off-by-one cases like P3, use **δ=0**.
""")

    # --- Java code examples with highlight (A/B/C)
    st.markdown("### 🧩 Illustrative Example (code & CSV)")
    code_a = """public class A {
    // Long Method starting around line 10 to ~70
    public int compute(int[] arr) {                // ← GT: Long Method (10–70)
        int sum = 0;
        for (int i = 0; i < arr.length; i++) {
            sum += arr[i];
            // ... many lines ...
            if (sum > 1000) {                      // ← GT: Magic Number (42–42)
                sum -= 1000;                       // should use a named constant
            }
        }
        return sum;
    }
}"""
    code_b = """public class B {
    public boolean check(int x, int y, int z) {
        if (x > 0) {               // if-1
            if (y > 0) {           // if-2
                if (z > 0) {       // ← GT: Deeply Nested Ifs (15–20 approx)
                    return true;
                }
            }
        }
        return false;
    }
}"""
    code_c = """public class C {
    // No ground-truth smells in this example file
    public String echo(String s) { return s; }
}"""
    st.markdown("#### 🔎 Code excerpts (line numbers & highlighted spans)")
    ca, cb, cc = st.columns(3)
    with ca:
        render_code_with_highlight("A.java (Long Method 10–70; Magic Number 42)", code_a, [(10,70),(42,42)])
    with cb:
        render_code_with_highlight("B.java (Deeply Nested Ifs 15–20)", code_b, [(15,20)])
    with cc:
        render_code_with_highlight("C.java (no GT)", code_c, [])

    # --- GT CSV sample
    st.markdown("#### 📑 Ground-Truth CSV (sample)")
    gt_df = pd.DataFrame([
        {"file":"a.java", "startLine":10, "endLine":70, "type":"Long Method", "description":"Method too long"},
        {"file":"a.java", "startLine":42, "endLine":42, "type":"Magic Number", "description":"Use named constant"},
        {"file":"b.java", "startLine":15, "endLine":20, "type":"Deeply Nested Ifs", "description":"Depth >= 3"},
    ])
    _render_pro_dataframe(gt_df, hide_index=True)
    st.caption("GT total = 3 spans (A:2, B:1, C:0).")

    # --- Preds (to show type ON/OFF)
    st.markdown("#### 🤖 Predicted findings (illustration)")
    preds_noisy = pd.DataFrame([
        {"file":"a.java","startLine":12,"endLine":68,"type":"Long Method","desc":"big method"},        # match
        {"file":"b.java","startLine":16,"endLine":20,"type":"Deeply Nested Ifs","desc":"nested ifs"},  # match
        {"file":"a.java","startLine":42,"endLine":42,"type":"Maintainability","desc":"generic vs Magic Number"},  # type-mismatch
        {"file":"c.java","startLine":5,"endLine":5,"type":"Magic Number","desc":"false positive"},     # FP on negative file
        {"file":"a.java","startLine":80,"endLine":85,"type":"Long Method","desc":"far from GT"},       # far span
    ])
    _render_pro_dataframe(preds_noisy, hide_index=True)
    st.caption("We keep a few imperfect predictions to illustrate the effect of the “Match by rule/type” option.")

    type_on = st.toggle("Apply type constraint (same as ticking “Match by rule/type”)", value=False)
    if not type_on:
        TP, FP, FN = 3, 2, 0
    else:
        TP, FP, FN = 2, 3, 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall    = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("TP", str(TP), "ok")
    with c2: kpi("FP", str(FP), "warn")
    with c3: kpi("FN", str(FN), "warn")
    with c4: st.markdown("&nbsp;"); st.code("Precision = TP/(TP+FP)\nRecall    = TP/(TP+FN)\nF1        = 2PR/(P+R)")

    c5,c6,c7 = st.columns(3)
    with c5: kpi("Precision", pct(precision), "ok")
    with c6: kpi("Recall",    pct(recall), "ok")
    with c7: kpi("F1 Score",  pct(f1), "ok")

    # --- Top-K & Positive ratio illustration widget
    st.markdown("---")
    st.markdown("### 🖼️ Universe U configuration (Top-K & Positive ratio) — illustration")
    st.caption("Toy widget that emulates U sampling to make the Run-tab sliders intuitive.")
    colA, colB = st.columns(2)
    with colA:
        demo_topk = st.slider("Top-K (illustration only)", 3, 30, 10, 1, key="demo_topk")
    with colB:
        demo_pos_ratio = st.slider("Positive ratio in U (illustration only)", 0.0, 1.0, 0.5, 0.05, key="demo_pos_ratio")

    P_available, N_available = 2, 1  # toy repo: A,B positive; C negative
    want_pos = min(P_available, round(demo_topk * demo_pos_ratio))
    want_neg = max(0, demo_topk - want_pos)
    want_neg = min(N_available, want_neg)
    total_u  = want_pos + want_neg

    st.markdown(f"- Positives in repository: **{P_available}** (A.java, B.java)")
    st.markdown(f"- Negatives in repository: **{N_available}** (C.java)")
    st.markdown(f"- Requested U: **Top-K = {demo_topk}**, **Positive ratio = {demo_pos_ratio:.2f}**")
    st.markdown(f"➡️ Sampled **U**: **positives = {want_pos}**, **negatives = {want_neg}**, **total = {total_u}**")

    comp_df = pd.DataFrame({
        "Type": ["positives (>=1 GT)", "negatives (0 GT)"],
        "Count": [want_pos, want_neg],
    })
    _render_bar_chart(comp_df, x="Type", y="Count", color="Type", y_title="Files", value_format=",.0f", height=260)

    st.markdown("<div class='footnote'>In real runs, positives are selected by highest GT count; negatives are random. This widget only illustrates the knobs.</div>", unsafe_allow_html=True)

if page == "📘 Guide & Examples":
    render_tab_guide()
