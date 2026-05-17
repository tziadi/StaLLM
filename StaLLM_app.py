# StaLLM_UI.py
# Pro UI: Inter/JetBrains Mono fonts, premium hero, glass cards, KPI chips,
# code viewer with line numbers + highlights, strictness guide, Top-K/Positive illustration.
import time
import os
import tempfile
import zipfile
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

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
st.set_page_config(page_title="⚡ StaLLM: Static Analysis meets LLMs", layout="wide")
_load_dotenv_robust(Path(__file__).with_name(".env"), override=False)
init_db()

RESULTS_SCHEMA_VERSION = 2

# =========================
# Styles (UI) — English + Pro look
# =========================
st.markdown( """ <style> /* Fonts */ @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap'); :root{ --text:#0e1220; --muted:#667085; --border:rgba(14,18,32,.12); --card-bg:rgba(255,255,255,.55); --glass:linear-gradient(180deg, rgba(255,255,255,.80), rgba(255,255,255,.60)); --accent:#7c3aed; --accent2:#06b6d4; --accent3:#22c55e; --warn:#f59e0b; --danger:#ef4444; --hl: rgba(252, 211, 77, .35); --ring: 0 8px 22px rgba(35, 38, 47, .10), 0 2px 6px rgba(35, 38, 47, .06); } html, body, .stApp { background:#f6f7fb !important; color:var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; } .stApp { padding-top: 10px; } @media (prefers-color-scheme: dark){ :root{ --text:#e6e9ef; --muted:#a3afc6; --border:rgba(255,255,255,.09); --card-bg:rgba(20,24,35,.55); --glass:linear-gradient(180deg, rgba(20,24,35,.75), rgba(20,24,35,.55)); --hl: rgba(234, 179, 8, .20); } html, body, .stApp{ background: radial-gradient(1200px 600px at 10% -10%, rgba(124,58,237,.16), transparent), radial-gradient(1000px 500px at 110% -20%, rgba(6,182,212,.12), transparent), #0b1220 !important; } } /* HERO */ .hero{ position:relative; border:1px solid var(--border); border-radius:22px; padding:28px 24px; overflow:hidden; margin-bottom:14px; background: radial-gradient(900px 220px at -10% -20%, rgba(124,58,237,.18), transparent 55%), radial-gradient(700px 220px at 110% -10%, rgba(6,182,212,.15), transparent 55%), var(--glass); box-shadow: var(--ring); } .hero h1{ margin:0 0 8px 0; font-weight:800; letter-spacing:.2px; } .hero p{ margin:0; color:var(--muted); font-size:1.0rem; } .badges{ margin-top:8px; } .badge{ display:inline-flex; align-items:center; gap:.45rem; padding:.38rem .70rem; border:1px solid var(--border); border-radius:999px; font-size:.78rem; font-weight:600; background:linear-gradient(90deg, rgba(124,58,237,.10), rgba(6,182,212,.10)); margin-right:.45rem; } /* KPI CARDS */ .kpi{ background:var(--glass); border:1px solid var(--border); border-radius:16px; padding:14px 16px; box-shadow: var(--ring); } .kpi-label{ color:var(--muted); font-size:.85rem; margin-bottom:.25rem;} .kpi-value{ font-size:1.6rem; font-weight:800; letter-spacing:.2px;} .kpi-ok{ box-shadow:0 0 0 1px rgba(34,197,94,.28) inset; } .kpi-warn{ box-shadow:0 0 0 1px rgba(245,158,11,.28) inset; } .kpi-bad{ box-shadow:0 0 0 1px rgba(239,68,68,.28) inset; } /* CODE BOX with line numbers + highlight */ .codebox{ font-family:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12.5px; border:1px solid var(--border); border-radius:14px; overflow:hidden; background: var(--glass); box-shadow: var(--ring); } .codebox table{ border-collapse: collapse; width:100%; } .codebox td{ vertical-align: top; padding: 0; } .codebox .gutter{ width: 3.2em; background: rgba(2,6,23,.05); color: var(--muted); text-align: right; user-select:none; } .codebox .code{ white-space: pre; } .codebox .line{ padding: 2px 10px; border-bottom: 1px solid rgba(2,6,23,.06); } .codebox .ln{ padding: 2px 8px; border-bottom: 1px solid rgba(2,6,23,.08); } .codebox .hl{ background: var(--hl); } .codebox .gt{ background:#fff7ed; box-shadow:inset 3px 0 #f97316; } .codebox .llm{ background:#eff6ff; box-shadow:inset 3px 0 #2563eb; } .codebox .both{ background:#ecfdf5; box-shadow:inset 3px 0 #16a34a; } .code-legend{ display:flex; gap:8px; margin:.4rem 0 .6rem; flex-wrap:wrap; } .legend-chip{ display:inline-flex; align-items:center; border-radius:999px; padding:4px 8px; font-size:12px; font-weight:750; border:1px solid #d9e0ea; background:#fff; color:#344054; } .legend-dot{ width:8px; height:8px; border-radius:999px; margin-right:6px; } .dot-gt{ background:#f97316; } .dot-llm{ background:#2563eb; } .dot-both{ background:#16a34a; } /* SECTION CARD */ .section{ background:var(--glass); border:1px solid var(--border); border-radius:18px; padding:18px 18px; margin:10px 0 16px; box-shadow: var(--ring); } /* BUTTONS */ .stButton > button { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; color: white !important; border-radius: 12px; font-size: 16px; font-weight: 800; padding: 0.70em 1.25em; border: none; letter-spacing:.2px; box-shadow: 0 10px 18px rgba(124,58,237,.20), 0 6px 14px rgba(6,182,212,.12); } .stButton > button:focus { outline: none; box-shadow: 0 0 0 2px rgba(124,58,237,.45); } /* TABLE polish */ .stDataFrame thead tr th { background:#eef2ff !important; color:#0f172a !important; font-weight:700; text-align:center; } .stDataFrame tbody tr:nth-child(even) { background-color: rgba(2, 6, 23, .03); } .pro-table-wrap{ border:1px solid rgba(15,23,42,.10); border-radius:12px; overflow:auto; background:#fff; box-shadow:0 10px 24px rgba(15,23,42,.06); margin:.6rem 0 1rem; } .pro-table{ width:100%; border-collapse:separate; border-spacing:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size:13px; line-height:1.35; color:#111827; } .pro-table th{ position:sticky; top:0; z-index:1; background:#f8fafc; color:#475569; text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.05em; font-weight:800; padding:11px 12px; border-bottom:1px solid #e5e7eb; white-space:nowrap; } .pro-table td{ padding:10px 12px; border-bottom:1px solid #eef2f7; vertical-align:middle; } .pro-table tr:hover td{ background:#f8fbff; } .pro-table td.num{ text-align:right; font-variant-numeric:tabular-nums; color:#0f172a; font-weight:650; } .pro-table .file-cell{ font-family:"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size:12px; color:#1f2937; max-width:620px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .verdict-badge{ display:inline-flex; align-items:center; border-radius:999px; padding:4px 9px; font-size:12px; font-weight:800; border:1px solid transparent; white-space:nowrap; } .verdict-matched{ color:#166534; background:#dcfce7; border-color:#bbf7d0; } .verdict-partial,.verdict-mismatch{ color:#92400e; background:#fef3c7; border-color:#fde68a; } .verdict-missed{ color:#991b1b; background:#fee2e2; border-color:#fecaca; } .verdict-extra{ color:#6d28d9; background:#ede9fe; border-color:#ddd6fe; } .verdict-true_negative{ color:#334155; background:#e2e8f0; border-color:#cbd5e1; } .issue-pill{ display:inline-flex; border-radius:7px; padding:4px 7px; background:#f1f5f9; color:#334155; font-weight:650; white-space:nowrap; } .metric-bar{ min-width:92px; } .metric-bar-track{ height:7px; border-radius:999px; background:#e5e7eb; overflow:hidden; } .metric-bar-fill{ height:100%; border-radius:999px; background:linear-gradient(90deg,#22c55e,#06b6d4); } .metric-bar-label{ margin-top:3px; color:#475569; font-size:11px; font-variant-numeric:tabular-nums; text-align:right; } .muted{ color:var(--muted); } .pill{ display:inline-flex; align-items:center; gap:.45rem; padding:.32rem .65rem; border:1px solid var(--border); color:var(--text); border-radius:999px; font-size:.78rem; font-weight:600; background:linear-gradient(90deg,#ede9fe,#cffafe); margin:.20rem .35rem .55rem 0; } .footnote{ color:var(--muted); font-size:.85rem; margin-top:8px; } </style> """, unsafe_allow_html=True, )

st.markdown("""
<style>
:root{
  --ui-bg:#ffffff;
  --ui-bg-soft:#f8fafc;
  --ui-border:#d9e0ea;
  --ui-border-strong:#cbd5e1;
  --ui-text:#101828;
  --ui-muted:#667085;
  --ui-focus:#2563eb;
  --ui-shadow:0 8px 20px rgba(15,23,42,.06);
}

/* Global typography rhythm */
.stApp, .stApp *{
  font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  letter-spacing:0;
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{
  color:var(--ui-text);
  font-weight:800;
}
.stMarkdown p, .stCaptionContainer, [data-testid="stCaptionContainer"]{
  color:var(--ui-muted);
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
  font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
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
  font-family:"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
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
  font-family:Inter, ui-sans-serif, system-ui !important;
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

/* Sidebar polish */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#ffffff,#f8fafc) !important;
  border-right:1px solid var(--ui-border) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{
  color:#475467;
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
        "Count": {
            "TP": int(df["TP"].sum()),
            "FP": int(df["FP"].sum()),
            "FN": int(df["FN"].sum()),
        }
    })
    st.markdown("#### Error mix")
    st.bar_chart(chart_df)
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
    st.bar_chart(metrics_df[["precision", "recall", "f1"]])

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
    st.bar_chart(metrics_df[["precision", "recall", "f1"]])
    st.markdown("### 💰 Tokens & Cost by Model")
    st.bar_chart(metrics_df[["prompt_tokens","completion_tokens","total_tokens"]])
    st.bar_chart(metrics_df[["usd_cost"]])
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
    st.bar_chart(metrics_df_single[["precision","recall","f1"]])

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
st.markdown("""
<div class="hero">
  <h1>⚡ StaLLM: Static Analysis meets LLMs</h1>
  <p>LLM-guided static analysis with span-level evaluation, token/cost accounting, and crisp benchmarking workflows.</p>
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
def build_llm(sidebar_prefix: str = "") -> ChatModel:
    registry = load_llm_registry()
    if registry:
        keys = list(registry.keys())
        labels = [f"{k} · {registry[k]['label']} ({registry[k]['config'].provider})" for k in keys]
        choice = st.sidebar.selectbox(f"{sidebar_prefix}LLM Slot (.env)", labels, index=0,
                                      help="Select a slot configured in your .env")
        slot = keys[labels.index(choice)]
        llm_obj = build_llm_from_slot(slot)
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> {llm_obj.model_label()}</span>", unsafe_allow_html=True)
        return llm_obj

    st.sidebar.markdown("**Manual configuration (no .env slots detected)**")
    prov = st.sidebar.selectbox(f"{sidebar_prefix}LLM Provider", ["azure-openai", "openai", "ollama"], index=0)
    mdl_list = available_models(prov)

    if prov == "azure-openai":
        base_default = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE", "")
        ver_default  = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
        dep_default  = os.getenv("OPENAI_DEPLOYMENT_NAME", "")
        with st.sidebar.expander(f"{sidebar_prefix}Advanced Azure settings", expanded=False):
            api_base    = st.text_input("Azure Resource endpoint", value=base_default, help="e.g., https://<resource>.openai.azure.com/")
            api_version = st.text_input("API version", value=ver_default)
        deployment = st.sidebar.text_input("Azure deployment name", value=dep_default, help="Exact name in Azure OpenAI Studio")
        api_key_env = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not api_key_env:
            api_key_env = st.sidebar.text_input("Azure API Key (paste if .env missing)", value="", type="password")
            if api_key_env: os.environ["AZURE_OPENAI_API_KEY"] = api_key_env
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> azure:{deployment or 'deployment'}</span>", unsafe_allow_html=True)
        return ChatModel(LLMConfig(provider="azure-openai", model=deployment, api_base=api_base, api_version=api_version, api_key=api_key_env))

    elif prov == "openai":
        model = st.sidebar.selectbox(f"{sidebar_prefix}Model", mdl_list, index=0)
        api_key_env = os.getenv("OPENAI_API_KEY") or ""
        if not api_key_env:
            api_key_env = st.sidebar.text_input("OpenAI API Key (paste if .env missing)", value="", type="password")
            if api_key_env: os.environ["OPENAI_API_KEY"] = api_key_env
        base_url = os.getenv("OPENAI_BASE_URL") or None
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> openai:{model}</span>", unsafe_allow_html=True)
        return ChatModel(LLMConfig(provider="openai", model=model, api_base=base_url, api_key=api_key_env))

    else:
        # Ollama: host + connectivity + dynamic models from that host
        default_host = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        host = st.sidebar.text_input(f"{sidebar_prefix}Ollama Host", value=default_host,
                                     help="e.g., http://localhost:11434 or http://192.168.1.100:11434")
        if st.sidebar.button("🔍 Test Connection", key=f"test_conn_{sidebar_prefix}"):
            with st.sidebar.spinner("Testing connection..."):
                ok, msg = test_ollama_connectivity(host)
                (st.sidebar.success if ok else st.sidebar.error)(msg)

        try:
            models_for_host = available_models("ollama", host)
            if not models_for_host:
                st.sidebar.warning("⚠️ No models returned by host. Falling back to env list.")
                models_for_host = available_models("ollama", None)
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not list models from host: {e}")
            models_for_host = available_models("ollama", None)

        model = st.sidebar.selectbox(f"{sidebar_prefix}Model", models_for_host or mdl_list, index=0)
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> ollama:{model}@{host.replace('http://','').replace('https://','')}</span>", unsafe_allow_html=True)
        return ChatModel(LLMConfig(provider="ollama", model=model, api_base=host))

def build_llms_for_comparison(sidebar_prefix: str = "") -> list[ChatModel]:
    llms: list[ChatModel] = []
    registry = load_llm_registry()
    st.sidebar.markdown(f"### {sidebar_prefix}LLM selection for comparison")

    if registry:
        keys = list(registry.keys())
        labels = [f"{k} · {registry[k]['label']} ({registry[k]['config'].provider})" for k in keys]
        picked = st.sidebar.multiselect("LLM Slots (.env)", labels, default=labels[:2] if len(labels) >= 2 else labels)
        for lab in picked:
            slot = keys[labels.index(lab)]
            cm = build_llm_from_slot(slot)
            llms.append(cm)
        if llms:
            pills = " ".join(f"<span class='pill'>🧠 <b>Model</b> {cm.model_label()}</span>" for cm in llms)
            st.markdown(pills, unsafe_allow_html=True)
        return llms

    # Manual multi-model selection
    prov = st.sidebar.selectbox(f"{sidebar_prefix}LLM Provider", ["azure-openai", "openai", "ollama"], index=0, key="cmp_prov")
    if prov == "ollama":
        default_host = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        host = st.sidebar.text_input(f"{sidebar_prefix}Ollama Host", value=default_host,
                                     help="Ollama server URL for model comparison", key="cmp_ollama_host")
        if st.sidebar.button("🔍 Test Connection", key="test_conn_cmp"):
            with st.sidebar.spinner("Testing connection..."):
                ok, msg = test_ollama_connectivity(host)
                (st.sidebar.success if ok else st.sidebar.error)(msg)
        mdl_list = available_models("ollama", host) or available_models("ollama", None)
        picked_models = st.sidebar.multiselect("Models to compare", mdl_list, default=mdl_list[:2] if len(mdl_list) >= 2 else mdl_list)
        for m in picked_models:
            llms.append(ChatModel(LLMConfig(provider="ollama", model=m, api_base=host)))
    else:
        mdl_list = available_models(prov)
        picked_models = st.sidebar.multiselect("Models to compare", mdl_list, default=mdl_list[:2] if len(mdl_list) >= 2 else mdl_list)
        if prov == "azure-openai":
            base_default = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE", "")
            ver_default  = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
            with st.sidebar.expander(f"{sidebar_prefix}Advanced Azure settings", expanded=False):
                api_base    = st.text_input("Azure Resource endpoint", value=base_default, key="cmp_api_base")
                api_version = st.text_input("API version", value=ver_default, key="cmp_api_ver")
            api_key_env = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            if not api_key_env:
                api_key_env = st.sidebar.text_input("Azure API Key", value="", type="password", key="cmp_api_key")
                if api_key_env: os.environ["AZURE_OPENAI_API_KEY"] = api_key_env
            for dep in picked_models:
                llms.append(ChatModel(LLMConfig(provider="azure-openai", model=dep, api_base=api_base, api_version=api_version, api_key=api_key_env)))
        else:
            base_url = os.getenv("OPENAI_BASE_URL") or None
            api_key_env = os.getenv("OPENAI_API_KEY") or ""
            if not api_key_env:
                api_key_env = st.sidebar.text_input("OpenAI API Key", value="", type="password", key="cmp_openai_key")
                if api_key_env: os.environ["OPENAI_API_KEY"] = api_key_env
            for m in picked_models:
                llms.append(ChatModel(LLMConfig(provider="openai", model=m, api_base=base_url, api_key=api_key_env)))

    if llms:
        pills = " ".join(f"<span class='pill'>🧠 <b>Model</b> {cm.model_label()}</span>" for cm in llms)
        st.markdown(pills, unsafe_allow_html=True)
    return llms

# =========================
# Tabs
# =========================
tabs = st.tabs([
    "⚙️ Run Experiments",
    "🗃️ Stored Results (DB)",
    "📝 Manage Prompts",
    "🔄 Batch Experiments",
    "📘 Guide & Examples"
])

# ==========================================================
# Tab 1 : Run Experiments (SPAN-LEVEL)
# ==========================================================
def render_tab_run_experiments():
    st.sidebar.header("⚙️ Settings")
    if "run_top_k" not in st.session_state:
        st.session_state.run_top_k = 20
    if "run_pos_ratio" not in st.session_state:
        st.session_state.run_pos_ratio = 0.5
    if "run_preset" not in st.session_state:
        st.session_state.run_preset = "Balanced"
    if "use_bundled_demo" not in st.session_state:
        st.session_state.use_bundled_demo = False

    if st.sidebar.button("⚡ Load demo config"):
        st.session_state.use_bundled_demo = True
        st.session_state.run_top_k = 5
        st.session_state.run_pos_ratio = 0.6
        st.session_state.run_preset = "Balanced"

    mode_run = st.sidebar.radio(
        "Execution mode",
        ["Single prompt", "Compare selected prompts", "Compare LLM models"],
        index=0,
        help="Pick your execution scenario."
    )
    top_k = st.sidebar.slider("Total files in U (Top-K)", 5, 50, key="run_top_k", step=1, help="Universe size U (positives + negatives).")
    pos_ratio = st.sidebar.slider("Positive ratio in U", 0.0, 1.0, key="run_pos_ratio", step=0.05, help="Share of positive files in U.")
    preset_options = ["Lenient", "Balanced", "Strict"]
    preset = st.sidebar.selectbox("Evaluation strictness", preset_options,
                                  index=preset_options.index(st.session_state.run_preset),
                                  key="run_preset",
                                  help="Controls IoU threshold and line tolerance (δ).")

    strategies = load_strategies()
    if mode_run == "Single prompt":
        if not strategies:
            st.error("⚠️ No strategies available. Please add one in ‘Manage Prompts’.")
            return
        prompt_mode = st.sidebar.selectbox("🧩 Select LLM Prompt Strategy", list(strategies.keys()))
    elif mode_run == "Compare selected prompts":
        selected_modes = st.sidebar.multiselect(
            "🧩 Select Prompt Strategies to Compare",
            list(strategies.keys()),
            default=[k for k in ["baseline", "scanner", "hybrid"] if k in strategies],
        )
    else:
        if not strategies:
            st.error("⚠️ No strategies available. Please add one in ‘Manage Prompts’.")
            return
        prompt_mode = st.sidebar.selectbox("🧩 Fixed Prompt Strategy (for model comparison)", list(strategies.keys()))

    if mode_run in ("Single prompt", "Compare selected prompts"):
        llm = build_llm()
    else:
        llms_to_compare = build_llms_for_comparison()
        if not llms_to_compare:
            st.warning("Select at least one LLM to compare.")
            return

    st.markdown("### 📂 Input data")
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

with tabs[0]:
    render_tab_run_experiments()

# ==========================================================
# Tab 2 : Results DB
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
        st.bar_chart(pivot_df)
    except Exception:
        st.info("Pivot not available (insufficient data).")
    session.close()

with tabs[1]:
    render_tab_results_db()

# ==========================================================
# Tab 3 : Manage Prompts
# ==========================================================
def render_tab_manage_prompts():
    st.markdown("## 📝 Manage Prompt Strategies")
    strategies = load_strategies()
    if strategies:
        st.markdown("### ✏️ Edit Existing Strategy")
        selected = st.selectbox("Select strategy", list(strategies.keys()))
        new_text = st.text_area("Edit prompt", value=strategies[selected], height=220)
        if st.button("💾 Save changes"):
            strategies[selected] = new_text; save_strategies(strategies)
            st.success(f"Strategy '{selected}' updated!"); st.rerun()
    st.markdown("---")
    st.markdown("### ➕ Add New Strategy")
    new_name = st.text_input("Strategy name")
    new_prompt = st.text_area("Prompt template", height=200)
    if st.button("➕ Create strategy"):
        if new_name in strategies:
            st.error("Strategy already exists!")
        elif new_name.strip() == "":
            st.error("Please provide a name.")
        else:
            strategies[new_name] = new_prompt; save_strategies(strategies)
            st.success(f"Strategy '{new_name}' added!"); st.rerun()

with tabs[2]:
    render_tab_manage_prompts()

# ==========================================================
# Tab 4 : Batch Experiments (span-level)
# ==========================================================
def render_tab_batch():
    st.markdown("## 🔄 Batch Experiments (span-level)")
    data_dir = st.text_input("📂 Data folder", "data/apps/")
    output_dir = st.text_input("📂 Output folder", "output/apps/")
    strategies = load_strategies()
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

with tabs[3]:
    render_tab_batch()

# ==========================================================
# Tab 5 : Guide & Examples (English, with visuals)
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

    comp_df = pd.DataFrame({"count":[want_pos, want_neg]}, index=["positives (≥1 GT)","negatives (0 GT)"])
    st.bar_chart(comp_df)

    st.markdown("<div class='footnote'>In real runs, positives are selected by highest GT count; negatives are random. This widget only illustrates the knobs.</div>", unsafe_allow_html=True)

with tabs[4]:
    render_tab_guide()
