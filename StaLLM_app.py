# StaLLM_UI.py
# Pro UI: Inter/JetBrains Mono fonts, premium hero, glass cards, KPI chips,
# code viewer with line numbers + highlights, strictness guide, Top-K/Positive illustration.
import time
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt  # ⬅️ pour des graphs avec labels lisibles

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
_load_dotenv_robust(Path(__file__).with_name(".env"), override=False)  # ⬅️ micro-fix __file__
init_db()

# =========================
# Styles (UI) — English + Pro look
# =========================
st.markdown(
    """ <style>
/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
:root{
 --text:#0e1220; --muted:#667085; --border:rgba(14,18,32,.12);
 --card-bg:rgba(255,255,255,.55); --glass:linear-gradient(180deg, rgba(255,255,255,.80), rgba(255,255,255,.60));
 --accent:#7c3aed; --accent2:#06b6d4; --accent3:#22c55e; --warn:#f59e0b; --danger:#ef4444;
 --hl: rgba(252, 211, 77, .35); --ring: 0 8px 22px rgba(35, 38, 47, .10), 0 2px 6px rgba(35, 38, 47, .06);
}
html, body, .stApp { background:#f6f7fb !important; color:var(--text);
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto,
  'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji'; }
.stApp { padding-top: 10px; }
@media (prefers-color-scheme: dark){
 :root{ --text:#e6e9ef; --muted:#a3afc6; --border:rgba(255,255,255,.09);
   --card-bg:rgba(20,24,35,.55); --glass:linear-gradient(180deg, rgba(20,24,35,.75), rgba(20,24,35,.55));
   --hl: rgba(234, 179, 8, .20);
 }
 html, body, .stApp{
   background: radial-gradient(1200px 600px at 10% -10%, rgba(124,58,237,.16), transparent),
               radial-gradient(1000px 500px at 110% -20%, rgba(6,182,212,.12), transparent),
               #0b1220 !important;
 }
}
/* HERO */
.hero{
  position:relative; border:1px solid var(--border); border-radius:22px; padding:28px 24px;
  overflow:hidden; margin-bottom:14px;
  background: radial-gradient(900px 220px at -10% -20%, rgba(124,58,237,.18), transparent 55%),
              radial-gradient(700px 220px at 110% -10%, rgba(6,182,212,.15), transparent 55%), var(--glass);
  box-shadow: var(--ring);
}
.hero h1{ margin:0 0 8px 0; font-weight:800; letter-spacing:.2px; }
.hero p{ margin:0; color:var(--muted); font-size:1.0rem; }
.badges{ margin-top:8px; }
.badge{
 display:inline-flex; align-items:center; gap:.45rem; padding:.38rem .70rem; border:1px solid var(--border);
 border-radius:999px; font-size:.78rem; font-weight:600;
 background:linear-gradient(90deg, rgba(124,58,237,.10), rgba(6,182,212,.10));
 margin-right:.45rem;
}
/* KPI CARDS */
.kpi{ background:var(--glass); border:1px solid var(--border); border-radius:16px; padding:14px 16px; box-shadow: var(--ring); }
.kpi-label{ color:var(--muted); font-size:.85rem; margin-bottom:.25rem;}
.kpi-value{ font-size:1.6rem; font-weight:800; letter-spacing:.2px;}
.kpi-ok{ box-shadow:0 0 0 1px rgba(34,197,94,.28) inset; }
.kpi-warn{ box-shadow:0 0 0 1px rgba(245,158,11,.28) inset; }
.kpi-bad{ box-shadow:0 0 0 1px rgba(239,68,68,.28) inset; }
/* CODE BOX with line numbers + highlight */
.codebox{
  font-family:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 12.5px; border:1px solid var(--border); border-radius:14px; overflow:hidden;
  background: var(--glass); box-shadow: var(--ring);
}
.codebox table{ border-collapse: collapse; width:100%; }
.codebox td{ vertical-align: top; padding: 0; }
.codebox .gutter{ width: 3.2em; background: rgba(2,6,23,.05); color: var(--muted); text-align: right; user-select:none; }
.codebox .code{ white-space: pre; }
.codebox .line{ padding: 2px 10px; border-bottom: 1px solid rgba(2,6,23,.06); }
.codebox .ln{ padding: 2px 8px; border-bottom: 1px solid rgba(2,6,23,.08); }
.codebox .hl{ background: var(--hl); }
/* SECTION CARD */
.section{ background:var(--glass); border:1px solid var(--border); border-radius:18px; padding:18px 18px; margin:10px 0 16px; box-shadow: var(--ring); }
/* BUTTONS */
.stButton > button {
 background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; color: white !important; border-radius: 12px;
 font-size: 16px; font-weight: 800; padding: 0.70em 1.25em; border: none; letter-spacing:.2px;
 box-shadow: 0 10px 18px rgba(124,58,237,.20), 0 6px 14px rgba(6,182,212,.12);
}
.stButton > button:focus { outline: none; box-shadow: 0 0 0 2px rgba(124,58,237,.45); }
/* TABLE polish */
.stDataFrame thead tr th { background:#eef2ff !important; color:#0f172a !important; font-weight:700; text-align:center; }
.stDataFrame tbody tr:nth-child(even) { background-color: rgba(2, 6, 23, .03); }
.muted{ color:var(--muted); }
.pill{
 display:inline-flex; align-items:center; gap:.45rem; padding:.32rem .65rem; border:1px solid var(--border); color:var(--text);
 border-radius:999px; font-size:.78rem; font-weight:600; background:linear-gradient(90deg,#ede9fe,#cffafe);
 margin:.20rem .35rem .55rem 0;
}
.footnote{ color:var(--muted); font-size:.85rem; margin-top:8px; }
</style> """,
    unsafe_allow_html=True,
)

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
        safe = (txt
                .replace("&","&amp;")
                .replace("<","&lt;")
                .replace(">","&gt;"))
        cls = "line hl" if idx in hl_set else "line"
        rows_html.append(
            f"<tr><td class='gutter ln'>{idx}</td><td class='code {cls}'><code>{safe}</code></td></tr>"
        )
    html = f"""
    <div class='section'><b>{title}</b>
    <div class='codebox'><table>{"".join(rows_html)}</table></div></div>
    """
    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# NEW: helper to draw bars with bigger, readable labels (minimal change)
# -------------------------
def show_bar_with_bigger_labels(df: pd.DataFrame, value_cols, x_label_col=None, title=""):
    """
    Remplace st.bar_chart pour avoir des labels X lisibles :
    - labelAngle=0, labelFontSize=16, labelLimit/padding généreux.
    - df : DataFrame (index = noms → on reset en 'name' si x_label_col absent)
    """
    data = df.copy()
    if x_label_col is None:
        data = data.reset_index().rename(columns={"index": "name"})
        x = "name"
    else:
        x = x_label_col

    # Ne garder que les colonnes utiles (évite Altair warnings)
    cols = [x] + list(value_cols)
    data = data[cols]

    melted = data.melt(id_vars=[x], value_vars=value_cols, var_name="metric", value_name="value")

    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(
                title=None, labelAngle=0, labelFontSize=16, labelLimit=260, labelPadding=10
            )),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("metric:N", legend=alt.Legend(title=None, labelFontSize=13)),
        )
        .properties(height=320, title=title or None)
        .configure_title(fontSize=16, fontWeight="bold")
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)

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
with tabs[0]:
    st.sidebar.header("⚙️ Settings")
    mode_run = st.sidebar.radio(
        "Execution mode",
        ["Single prompt", "Compare selected prompts", "Compare LLM models"],
        index=0,
        help="Pick your execution scenario."
    )
    top_k = st.sidebar.slider("Total files in U (Top-K)", 5, 50, 20, 1, help="Universe size U (positives + negatives).")
    pos_ratio = st.sidebar.slider("Positive ratio in U", 0.0, 1.0, 0.5, 0.05, help="Share of positive files in U.")
    preset = st.sidebar.selectbox("Evaluation strictness", ["Lenient", "Balanced", "Strict"], index=1,
                                  help="Controls IoU threshold and line tolerance (δ).")

    strategies = load_strategies()
    if mode_run == "Single prompt":
        if not strategies:
            st.error("⚠️ No strategies available. Please add one in ‘Manage Prompts’."); st.stop()
        prompt_mode = st.sidebar.selectbox("🧩 Select LLM Prompt Strategy", list(strategies.keys()))
    elif mode_run == "Compare selected prompts":
        selected_modes = st.sidebar.multiselect(
            "🧩 Select Prompt Strategies to Compare",
            list(strategies.keys()),
            default=[k for k in ["baseline", "scanner", "hybrid"] if k in strategies],
        )
    else:
        if not strategies:
            st.error("⚠️ No strategies available. Please add one in ‘Manage Prompts’."); st.stop()
        prompt_mode = st.sidebar.selectbox("🧩 Fixed Prompt Strategy (for model comparison)", list(strategies.keys()))

    if mode_run in ("Single prompt", "Compare selected prompts"):
        llm = build_llm()
    else:
        llms_to_compare = build_llms_for_comparison()
        if not llms_to_compare:
            st.warning("Select at least one LLM to compare."); st.stop()

    st.markdown("### 📂 Upload Required Files")
    colu1, colu2 = st.columns([1,1])
    with colu1:
        uploaded_zip = st.file_uploader("📦 Project (ZIP)", type="zip", help="ZIP with source files.")
    with colu2:
        uploaded_static = st.file_uploader("📑 Static Analyzer CSV (Ground Truth)", type="csv",
                                           help="GT spans: file, startLine, endLine[, columns, type].")

    # read bytes once
    zip_bytes = uploaded_zip.getvalue() if uploaded_zip is not None else None
    csv_bytes = uploaded_static.getvalue() if uploaded_static is not None else None

    # Capabilities (type/line/column)
    user_require_type = False
    user_use_line_span = True
    user_use_cols_single = False

    if uploaded_static is not None and csv_bytes:
        try:
            with tempfile.TemporaryDirectory() as _tmp:
                tmp_csv = os.path.join(_tmp, uploaded_static.name)
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
                    # default OFF
                    user_require_type = st.checkbox(
                        f"Match by rule/type (e.g., “{sample.get('rule/type', '…')}”)",
                        value=False
                    )
                else:
                    st.caption("No rule/type column detected.")
                    user_require_type = False

            with c2:
                if caps.get("has_line_span", False):
                    # default ON
                    user_use_line_span = st.checkbox(
                        f"Use line spans (start–end) (e.g., {sample.get('startLine','?')}–{sample.get('endLine','?')})",
                        value=True
                    )
                else:
                    st.caption("No endLine column detected.")
                    user_use_line_span = False

            with c3:
                if caps.get("has_col_span", False):
                    # default OFF
                    user_use_cols_single = st.checkbox(
                        "Use column spans on single-line (if available)",
                        value=False
                    )
                else:
                    st.caption("No column spans detected.")
                    user_use_cols_single = False

    # Run
    if st.button("🚀 Run Analysis"):
        if not (uploaded_zip and uploaded_static and zip_bytes and csv_bytes):
            st.warning("📥 Please upload both files (ZIP + CSV)."); st.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, uploaded_zip.name); open(zip_path, "wb").write(zip_bytes)
            csv_path = os.path.join(tmpdir, uploaded_static.name); open(csv_path, "wb").write(csv_bytes)

            language = detect_language_from_zip(zip_path)
            exts = find_exts_for_language(language)
            st.markdown(f"### 🌐 Detected Language: **{language}**")
            st.caption("Span-level evaluation with a mixed universe U (positives + negatives).")

            # sanity read GT
            try:
                _ = load_ground_truth_spans(csv_path, allowed_exts=exts)
            except Exception as e:
                st.error(f"⚠️ Error reading Static Analyzer CSV: {e}"); st.stop()

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
                    st.error(f"LLM call failed: {e}"); st.stop()

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

                st.markdown("### 📁 Universe U (files considered)")
                st.dataframe(summary_U, use_container_width=True)
                st.caption("`is_positive=True` → the file has ≥1 GT span. `gt_lines` = number of GT rows for that file.")

                c1,c2,c3,c4 = st.columns(4)
                with c1: kpi("Precision", pct(metrics.get("precision", 0.0)), "ok")
                with c2: kpi("Recall",    pct(metrics.get("recall", 0.0)), "warn")
                with c3: kpi("F1 Score",  pct(metrics.get("f1", 0.0)), "ok")
                with c4: kpi("Elapsed",   f"{metrics.get('time_s', 0.0):.2f}s", "ok")

                # ⬇️ Labels lisibles
                metrics_df_single = pd.DataFrame([{
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1": metrics.get("f1", 0.0)}], index=[f"{prompt_mode}@span"])
                show_bar_with_bigger_labels(metrics_df_single, ["precision","recall","f1"], title="Precision / Recall / F1")

                t1,t2,t3,t4 = st.columns(4)
                with t1: kpi("Prompt tokens", f"{usage_totals['prompt_tokens']:,}")
                with t2: kpi("Completion tokens", f"{usage_totals['completion_tokens']:,}")
                with t3: kpi("Total tokens", f"{usage_totals['total_tokens']:,}")
                with t4: kpi("USD cost", f"${usage_totals['usd_cost']:.6f}", "warn")

                with st.expander("🔬 Diagnostics (mapping & sampling)"):
                    diag = metrics.get("diagnostics", {})
                    st.json(diag)
                    cov = float(diag.get("mapping_coverage", 0.0))
                    if cov == 0.0:
                        st.error("No GT rows could be mapped to files in the ZIP. Compare CSV basenames vs ZIP basenames.")
                    elif cov < 0.2:
                        st.warning(f"Low mapping coverage: {cov:.1%}. Metrics may be unstable.")

                st.caption(f"⏱️ Completed in {elapsed:.1f}s • Model: {llm.model_label()} • Preset: {preset}")

            elif mode_run == "Compare selected prompts":
                if not selected_modes:
                    st.warning("⚠️ Please select at least one prompt strategy."); st.stop()
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
                    st.error(f"LLM call failed: {e}"); st.stop()

                st.markdown("### 📁 Universe U (files considered)")
                st.dataframe(summary_U, use_container_width=True)

                st.markdown("### 📐 Comparison Metrics (All Strategies)")
                st.dataframe(metrics_df, use_container_width=True)

                # ⬇️ Labels lisibles pour stratégies
                show_bar_with_bigger_labels(
                    metrics_df.reset_index().rename(columns={"index": "strategy"}),
                    ["precision","recall","f1"],
                    x_label_col="strategy",
                    title="Precision / Recall / F1 by Strategy"
                )

                st.markdown("## 🤖 LLM-based Findings per Strategy")
                for p, sample in (samples or {}).items():
                    with st.expander(f"🔹 {p} (sample)"): st.json(sample)

            else:
                if not llms_to_compare:
                    st.warning("⚠️ Please select at least one LLM to compare."); st.stop()
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
                    st.error(f"LLM call failed: {e}"); st.stop()

                st.markdown("### 📁 Universe U (files considered)")
                st.dataframe(summary_U, use_container_width=True)
                st.markdown("### 📐 Metrics per Model")
                st.dataframe(metrics_df, use_container_width=True)

                # ⬇️ Labels lisibles pour modèles
                show_bar_with_bigger_labels(
                    metrics_df.reset_index().rename(columns={"index": "model"}),
                    ["precision", "recall", "f1"],
                    x_label_col="model",
                    title="Precision / Recall / F1 by Model"
                )

                st.markdown("### 💰 Tokens & Cost by Model")
                show_bar_with_bigger_labels(
                    metrics_df.reset_index().rename(columns={"index": "model"}),
                    ["prompt_tokens","completion_tokens","total_tokens"],
                    x_label_col="model",
                    title="Tokens by Model"
                )
                show_bar_with_bigger_labels(
                    metrics_df.reset_index().rename(columns={"index": "model"}),
                    ["usd_cost"],
                    x_label_col="model",
                    title="USD Cost by Model"
                )

                st.markdown("## 🤖 Samples per Model")
                for model_label, sample in (samples or {}).items():
                    with st.expander(f"🔹 {model_label} (sample)"): st.json(sample)

# ==========================================================
# Tab 2 : Results DB
# ==========================================================
with tabs[1]:
    st.markdown("## 🗃️ Stored Experiment Results (Database)")
    session = Session()
    try:
        results = session.query(SmellDetectionResult).all()
    except Exception as e:
        st.error(f"DB error: {e}"); session.close(); st.stop()

    if not results:
        st.info("ℹ️ No results stored in the database."); session.close(); st.stop()

    df = pd.DataFrame([{
        "Project": r.project, "Language": r.language, "Filename": r.filename,
        "Strategy": r.strategy, "Precision": r.precision, "Recall": r.recall,
        "F1 Score": r.f1, "Top-K (files in U)": r.top_k, "Time (s)": r.time_elapsed,
        "LLM Used": r.llm_used, "Prompt tokens": r.prompt_tokens,
        "Completion tokens": r.completion_tokens, "Total tokens": r.total_tokens,
        "USD cost": r.usd_cost, "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    } for r in results])
    st.dataframe(df, use_container_width=True)

    try:
        pivot_df = df.pivot_table(index="Project", columns=["LLM Used","Strategy"], values="F1 Score", aggfunc="mean").sort_index(axis=1)
        if isinstance(pivot_df.columns, pd.MultiIndex):
            pivot_df.columns = [f"{(llm or 'Unknown')} · {(strategy or 'n/a')}" for (llm,strategy) in pivot_df.columns.to_list()]
        else:
            pivot_df.columns = pivot_df.columns.astype(str)
        pivot_df = pivot_df.fillna(0)
        st.markdown("### 📈 Global Comparison (F1 across Projects • by LLM & Strategy)")
        # labels lisibles
        show_bar_with_bigger_labels(
            pivot_df,
            list(pivot_df.columns),  # chaque colonne = une série différente, on les dessinera autrement
            # Ici, pour rester simple et ne pas "exploser" les barres, on laisse le tableau. Tu peux aussi
            # reformater pour un chart multi-séries si besoin.
            title="(Pivot) F1 by Project / (LLM · Strategy)"
        )
    except Exception:
        st.info("Pivot not available (insufficient data).")
    session.close()

# ==========================================================
# Tab 3 : Manage Prompts
# ==========================================================
with tabs[2]:
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
        if new_name in strategies: st.error("Strategy already exists!")
        elif new_name.strip() == "": st.error("Please provide a name.")
        else:
            strategies[new_name] = new_prompt; save_strategies(strategies)
            st.success(f"Strategy '{new_name}' added!"); st.rerun()

# ==========================================================
# Tab 4 : Batch Experiments (span-level)
# ==========================================================
with tabs[3]:
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
            st.warning("⚠️ Please select at least one strategy."); st.stop()

        projects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        for project in projects:
            st.write(f"▶️ Running project: {project}")
            project_dir = Path(data_dir, project)
            zip_files = [str(project_dir / f) for f in os.listdir(project_dir) if f.endswith(".zip")]
            csv_files = [str(project_dir / f) for f in os.listdir(project_dir) if f.endswith(".csv")]
            if not zip_files or not csv_files:
                st.warning(f"❌ Missing files for {project} (need at least one .zip and one .csv)"); continue
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

# ==========================================================
# Tab 5 : Guide & Examples (English, with visuals)
# ==========================================================
with tabs[4]:
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
    st.dataframe(gt_df, use_container_width=True)
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
    st.dataframe(preds_noisy, use_container_width=True)
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
    show_bar_with_bigger_labels(
        comp_df.reset_index().rename(columns={"index": "class"}),
        ["count"],
        x_label_col="class",
        title="U composition (illustration)"
    )
