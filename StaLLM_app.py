# StaLLM_UI.py
import time
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from StaLLM_core import (
    run_experiment,
    run_selected_experiments,
    run_selected_models_experiments,  # <-- NEW
    load_ground_truth,
    detect_language_from_zip,
    load_strategies,
    save_strategies,
    find_exts_for_language,   # filtre CSV selon la langue
)
from StaLLM_models import Session, SmellDetectionResult, init_db, save_run_result
from StaLLM_llm import (
    ChatModel, LLMConfig, available_models,
    load_llm_registry, build_llm_from_slot
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
st.set_page_config(page_title="StaLLM - Static Analysis meets LLMs", layout="wide")
_load_dotenv_robust(Path(__file__).with_name(".env"), override=False)
init_db()

# =========================
# Styles (UI améliorée)
# =========================
st.markdown(
    """
    <style>
    :root{
      --text:#111827; --muted:#475569; --border:rgba(2,6,23,.12);
      --card-bg:rgba(2,6,23,.02); --accent:#7c3aed; --accent2:#06b6d4;
    }
    html, body, .stApp { background:#ffffff !important; color:var(--text); }
    @media (prefers-color-scheme: dark){
      :root{ --text:#e5e7eb; --muted:#94a3b8; --border:rgba(148,163,184,.25);
             --card-bg:linear-gradient(180deg, rgba(255,255,255,.035), rgba(255,255,255,.015)); }
      html, body, .stApp{
        background:
          radial-gradient(1200px 600px at 10% -10%, rgba(124,58,237,.12), transparent),
          radial-gradient(1000px 500px at 110% -20%, rgba(6,182,212,.10), transparent),
          #0f172a !important;
        color:var(--text);
      }
    }
    h1, h2, h3 { color: var(--text) !important; font-weight: 800; }
    .pill{ display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .75rem;
      border:1px solid #c7d2fe; color:#111827; border-radius:999px;
      background:linear-gradient(90deg,#ede9fe,#cffafe); margin:.25rem .25rem .75rem 0; font-size:0.85rem; }
    .pill b{ color:#0f172a; }
    .metric-card{ background:var(--card-bg); border:1px solid var(--border);
      border-radius:12px; padding:14px 16px; }
    .metric-title{ color:var(--muted); font-size:.85rem; margin-bottom:.35rem;}
    .metric-value{ font-size:1.55rem; font-weight:800; letter-spacing:.2px;}
    .muted{ color:var(--muted); }
    .section-card{ background:var(--card-bg); border:1px solid var(--border);
      border-radius:14px; padding:14px 16px; margin:6px 0 14px; }
    .stButton > button {
      background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
      color: white !important; border-radius: 10px; font-size: 16px; font-weight: 700;
      padding: 0.65em 1.2em; border: none;
    }
    .stDataFrame thead tr th { background:#eef2ff !important; color:#0f172a !important;
      font-weight:700; text-align:center; }
    .stDataFrame tbody tr:nth-child(even) { background-color: rgba(2, 6, 23, .03); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚡ StaLLM: Static Analysis meets LLMs")
st.markdown("<div class='muted'>LLM-guided static analysis • precision/recall tracking • token & cost accounting</div>", unsafe_allow_html=True)

# Small helpers
def metric_card(title: str, value: str):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-title">{title}</div>
      <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def pct(x: float) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "0.00%"

# ==========================================================
# Helper: Build LLM (single)
# ==========================================================
def build_llm(sidebar_prefix: str = "") -> ChatModel:
    registry = load_llm_registry()
    if registry:
        keys = list(registry.keys())
        labels = [f"{k} · {registry[k]['label']} ({registry[k]['config'].provider})" for k in keys]
        choice = st.sidebar.selectbox(f"{sidebar_prefix}LLM Slot (.env)", labels, index=0)
        slot = keys[labels.index(choice)]
        llm_obj = build_llm_from_slot(slot)
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> {llm_obj.model_label()}</span>", unsafe_allow_html=True)
        return llm_obj

    # Fallback manuel
    st.sidebar.markdown("**Advanced manual configuration (no .env slots detected)**")
    prov = st.sidebar.selectbox(f"{sidebar_prefix}LLM Provider", ["azure-openai", "openai", "ollama"], index=0)
    mdl_list = available_models(prov)

    if prov == "azure-openai":
        base_default = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE", "")
        ver_default  = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
        dep_default  = os.getenv("OPENAI_DEPLOYMENT_NAME", "")

        with st.sidebar.expander(f"{sidebar_prefix}Advanced Azure settings", expanded=False):
            api_base    = st.text_input("Azure Resource endpoint", value=base_default, help="ex: https://<resource>.openai.azure.com/")
            api_version = st.text_input("API version", value=ver_default)
        deployment = st.sidebar.text_input("Azure deployment name", value=dep_default, help="Nom exact dans Azure OpenAI Studio.")

        api_key_env = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not api_key_env:
            api_key_env = st.sidebar.text_input("Azure API Key (.env absent ? colle-la ici)", value="", type="password")
            if api_key_env:
                os.environ["AZURE_OPENAI_API_KEY"] = api_key_env

        st.markdown(f"<span class='pill'>🧠 <b>Model</b> azure:{deployment or 'deployment'}</span>", unsafe_allow_html=True)

        return ChatModel(LLMConfig(
            provider="azure-openai",
            model=deployment,
            api_base=api_base,
            api_version=api_version,
            api_key=api_key_env,
        ))

    elif prov == "openai":
        model = st.sidebar.selectbox(f"{sidebar_prefix}Model", mdl_list, index=0)
        api_key_env = os.getenv("OPENAI_API_KEY") or ""
        if not api_key_env:
            api_key_env = st.sidebar.text_input("OpenAI API Key (.env absent ? colle-la ici)", value="", type="password")
            if api_key_env:
                os.environ["OPENAI_API_KEY"] = api_key_env
        base_url    = os.getenv("OPENAI_BASE_URL") or None
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> openai:{model}</span>", unsafe_allow_html=True)
        return ChatModel(LLMConfig(provider="openai", model=model, api_base=base_url, api_key=api_key_env))

    else:
        model = st.sidebar.selectbox(f"{sidebar_prefix}Model", mdl_list, index=0)
        host  = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        st.markdown(f"<span class='pill'>🧠 <b>Model</b> ollama:{model}</span>", unsafe_allow_html=True)
        return ChatModel(LLMConfig(provider="ollama", model=model, api_base=host))

# ==========================================================
# Helper: Build MULTIPLE LLMs for comparison
# ==========================================================
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

    # Fallback manuel (sans slots) : même provider, multi-modèles
    prov = st.sidebar.selectbox(f"{sidebar_prefix}LLM Provider", ["azure-openai", "openai", "ollama"], index=0, key="cmp_prov")
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
            if api_key_env:
                os.environ["AZURE_OPENAI_API_KEY"] = api_key_env
        for dep in picked_models:
            llms.append(ChatModel(LLMConfig(provider="azure-openai", model=dep, api_base=api_base, api_version=api_version, api_key=api_key_env)))
    elif prov == "openai":
        base_url = os.getenv("OPENAI_BASE_URL") or None
        api_key_env = os.getenv("OPENAI_API_KEY") or ""
        if not api_key_env:
            api_key_env = st.sidebar.text_input("OpenAI API Key", value="", type="password", key="cmp_openai_key")
            if api_key_env:
                os.environ["OPENAI_API_KEY"] = api_key_env
        for m in picked_models:
            llms.append(ChatModel(LLMConfig(provider="openai", model=m, api_base=base_url, api_key=api_key_env)))
    else:
        host = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        for m in picked_models:
            llms.append(ChatModel(LLMConfig(provider="ollama", model=m, api_base=host)))

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
    "🔄 Batch Experiments"
])

# ==========================================================
# Tab 1 : Run Experiments
# ==========================================================
with tabs[0]:
    st.sidebar.header("⚙️ Settings")
    mode_run = st.sidebar.radio(
        "Execution mode",
        ["Single prompt", "Compare selected prompts", "Compare LLM models"],  # <-- NEW
        index=0
    )
    top_k = st.sidebar.slider("Top-K files to analyze", min_value=3, max_value=20, value=5)

    strategies = load_strategies()

    # --- Prompt selection depends on mode ---
    if mode_run == "Single prompt":
        if not strategies:
            st.error("⚠️ No strategies available. Please add one in ‘Manage Prompts’.")
            st.stop()
        prompt_mode = st.sidebar.selectbox("🧩 Select LLM Prompt Strategy", list(strategies.keys()))
    elif mode_run == "Compare selected prompts":
        selected_modes = st.sidebar.multiselect(
            "🧩 Select Prompt Strategies to Compare",
            list(strategies.keys()),
            default=[k for k in ["baseline", "scanner", "hybrid"] if k in strategies],
        )
    else:  # Compare LLM models
        if not strategies:
            st.error("⚠️ No strategies available. Please add one in ‘Manage Prompts’.")
            st.stop()
        prompt_mode = st.sidebar.selectbox("🧩 Fixed Prompt Strategy (for model comparison)", list(strategies.keys()))

    # Build LLM (single) for Single/Compare Prompts; for Model Comparison we build many
    if mode_run in ("Single prompt", "Compare selected prompts"):
        llm = build_llm()
    else:
        llms_to_compare = build_llms_for_comparison()
        if not llms_to_compare:
            st.warning("Select at least one LLM to compare.")
            st.stop()

    st.markdown("### 📂 Upload Required Files")
    with st.container():
        colu1, colu2 = st.columns([1,1])
        with colu1:
            uploaded_zip = st.file_uploader("📦 Upload Project (ZIP)", type="zip")
        with colu2:
            uploaded_static = st.file_uploader("📑 Upload Static Analyzer Results (CSV)", type="csv")

    if st.button("🚀 Run Analysis"):
        if not (uploaded_zip and uploaded_static):
            st.warning("📥 Please upload both files (ZIP + Static Analyzer CSV).")
            st.stop()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {}
            for label, file in {"zip": uploaded_zip, "static": uploaded_static}.items():
                path = os.path.join(tmpdir, file.name)
                with open(path, "wb") as f:
                    f.write(file.read())
                paths[label] = path

            language = detect_language_from_zip(paths["zip"])
            exts = find_exts_for_language(language)
            st.markdown(f"### 🌐 Detected Language: **{language}**")
            st.caption("The CSV is filtered to match the detected language extensions.")

            try:
                static_results = load_ground_truth(paths["static"], allowed_exts=exts)
            except Exception as e:
                st.error(f"⚠️ Error reading Static Analyzer CSV: {e}")
                st.stop()

            st.markdown("## 📊 Static Analysis Results")
            st.info("Results from an external static analyzer (e.g., SonarQube, SpotBugs, ESLint...).")
            st.dataframe(static_results.head(top_k), use_container_width=True)

            # ----- Single prompt -----
            if mode_run == "Single prompt":
                st.markdown(f"### 🤖 Running LLM analysis with `{prompt_mode}` strategy…")
                start = time.time()
                try:
                    with st.spinner(f"⏳ Running `{prompt_mode}`…"):
                        results_llm, issues_top, metrics, usage_totals = run_experiment(
                            paths["zip"], paths["static"], prompt_mode, top_k=top_k, llm=llm
                        )
                except Exception as e:
                    st.error(f"LLM call failed: {e}\nIf you use Azure, double-check the **deployment name** and **resource endpoint**.")
                    st.stop()

                elapsed = time.time() - start
                metrics = dict(metrics or {})
                metrics["time_s"] = round(elapsed, 2)
                metrics["language"] = language

                # --- Persist in DB (Single prompt) ---
                try:
                    save_run_result(
                        project=os.path.basename(paths["zip"]),
                        filename="ALL",
                        strategy=prompt_mode,
                        language=language,
                        f1=float(metrics.get("f1", 0.0)),
                        precision=float(metrics.get("precision", 0.0)),
                        recall=float(metrics.get("recall", 0.0)),
                        top_k=top_k,
                        issues_detected=results_llm,
                        time_elapsed=elapsed,
                        llm_used=llm.model_label(),
                        sonar_detected_smells=static_results.head(top_k).to_dict(orient="records"),
                        prompt_tokens=int(usage_totals.get("prompt_tokens", 0)),
                        completion_tokens=int(usage_totals.get("completion_tokens", 0)),
                        total_tokens=int(usage_totals.get("total_tokens", 0)),
                        usd_cost=float(usage_totals.get("usd_cost", 0.0)),
                    )
                    st.success("✅ Run saved to database.")
                except Exception as e:
                    st.warning(f"⚠️ Could not save to DB: {e}")

                st.markdown("### 📁 Files Considered")
                st.dataframe(static_results.head(top_k), use_container_width=True)

                st.markdown("### 📐 Metrics (cards)")
                c1, c2, c3, c4 = st.columns(4)
                with c1: metric_card("Precision", pct(metrics.get("precision", 0.0)))
                with c2: metric_card("Recall",    pct(metrics.get("recall", 0.0)))
                with c3: metric_card("F1 Score",  pct(metrics.get("f1", 0.0)))
                with c4: metric_card("Elapsed",   f"{metrics.get('time_s', 0.0):.2f}s")

                st.markdown("### 📈 Precision / Recall / F1")
                metrics_df_single = pd.DataFrame(
                    [{"precision": metrics.get("precision", 0.0),
                      "recall": metrics.get("recall", 0.0),
                      "f1": metrics.get("f1", 0.0)}],
                    index=[prompt_mode]
                )
                st.bar_chart(metrics_df_single[["precision", "recall", "f1"]])

                st.markdown("### 💰 Tokens & Cost")
                t1, t2, t3, t4 = st.columns(4)
                with t1: metric_card("Prompt tokens", f"{usage_totals['prompt_tokens']:,}")
                with t2: metric_card("Completion tokens", f"{usage_totals['completion_tokens']:,}")
                with t3: metric_card("Total tokens", f"{usage_totals['total_tokens']:,}")
                with t4: metric_card("USD cost", f"${usage_totals['usd_cost']:.6f}")

                with st.expander("🔎 Détails des métriques (tableau)"):
                    st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}), use_container_width=True)
                with st.expander("🔎 Tokens & Coût (tableau)"):
                    st.dataframe(pd.DataFrame([usage_totals]).T.rename(columns={0: "value"}), use_container_width=True)

                st.caption(f"⏱️ Completed in {elapsed:.1f}s • Model: {llm.model_label()}")

            # ----- Multi prompt -----
            elif mode_run == "Compare selected prompts":
                if not selected_modes:
                    st.warning("⚠️ Please select at least one prompt strategy.")
                    st.stop()

                st.markdown("### 🔄 Running Multiple Prompt Strategies")
                progress = st.progress(0); status_text = st.empty(); timer_text = st.empty()

                start = time.time()
                try:
                    issues_top, metrics_df, samples = run_selected_experiments(
                        paths["zip"], paths["static"], selected_modes,
                        progress=progress, status=status_text, timer=timer_text, top_k=top_k, llm=llm
                    )
                except Exception as e:
                    st.error(f"LLM call failed: {e}\nIf you use Azure, double-check the **deployment name** and **resource endpoint**.")
                    st.stop()

                st.markdown("### 📁 Files Considered (Top-K)")
                st.dataframe(issues_top, use_container_width=True)

                st.markdown("### 📐 Comparison Metrics (All Strategies)")
                st.dataframe(metrics_df, use_container_width=True)

                st.markdown("### 📈 Precision / Recall / F1 by Strategy")
                st.bar_chart(metrics_df[["precision", "recall", "f1"]])

                st.markdown("## 🤖 LLM-based Findings per Strategy")
                for p, sample in (samples or {}).items():
                    with st.expander(f"🔹 {p} (sample)"):
                        st.json(sample)

            # ----- NEW: Compare LLM models (fixed prompt) -----
            else:
                if not llms_to_compare:
                    st.warning("⚠️ Please select at least one LLM to compare.")
                    st.stop()

                st.markdown(f"### 🔄 Comparing LLM models (prompt strategy: `{prompt_mode}`)")
                progress = st.progress(0); status_text = st.empty(); timer_text = st.empty()

                try:
                    issues_top, metrics_df, samples = run_selected_models_experiments(
                        paths["zip"], paths["static"], prompt_mode, llms_to_compare,
                        progress=progress, status=status_text, timer=timer_text, top_k=top_k
                    )
                except Exception as e:
                    st.error(f"LLM call failed: {e}")
                    st.stop()

                st.markdown("### 📁 Files Considered (Top-K)")
                st.dataframe(issues_top, use_container_width=True)

                st.markdown("### 📐 Metrics per Model")
                st.dataframe(metrics_df, use_container_width=True)

                st.markdown("### 📈 Precision / Recall / F1 by Model")
                st.bar_chart(metrics_df[["precision", "recall", "f1"]])

                st.markdown("### 💰 Tokens & Cost by Model")
                st.bar_chart(metrics_df[["prompt_tokens", "completion_tokens", "total_tokens"]])
                st.bar_chart(metrics_df[["usd_cost"]])

                st.markdown("## 🤖 Samples per Model")
                for model_label, sample in (samples or {}).items():
                    with st.expander(f"🔹 {model_label} (sample)"):
                        st.json(sample)

# ==========================================================
# Tab 2 : Results DB
# ==========================================================
with tabs[1]:
    st.markdown("## 🗃️ Stored Experiment Results (Database)")
    session = Session()
    try:
        results = session.query(SmellDetectionResult).all()
    except Exception as e:
        st.error(f"DB error: {e}")
        session.close(); st.stop()

    if not results:
        st.info("ℹ️ No results stored in the database.")
        session.close(); st.stop()

    df = pd.DataFrame(
        [
            {
                "Project": r.project,
                "Language": r.language,
                "Filename": r.filename,
                "Strategy": r.strategy,
                "Precision": r.precision,
                "Recall": r.recall,
                "F1 Score": r.f1,
                "Top-K": r.top_k,
                "Time (s)": r.time_elapsed,
                "LLM Used": r.llm_used,
                "Prompt tokens": r.prompt_tokens,
                "Completion tokens": r.completion_tokens,
                "Total tokens": r.total_tokens,
                "USD cost": r.usd_cost,
                "Timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True)

    # ---------- FIX: éviter MultiIndex pour st.bar_chart ----------
    pivot_df = df.pivot_table(
        index="Project",
        columns=["LLM Used", "Strategy"],
        values="F1 Score",
        aggfunc="mean"
    ).sort_index(axis=1)

    if isinstance(pivot_df.columns, pd.MultiIndex):
        pivot_df.columns = [
            f"{(llm or 'Unknown')} · {(strategy or 'n/a')}"
            for (llm, strategy) in pivot_df.columns.to_list()
        ]
    else:
        pivot_df.columns = pivot_df.columns.astype(str)

    pivot_df = pivot_df.fillna(0)

    st.markdown("### 📈 Global Comparison (F1 Score across Projects • by LLM & Strategy)")
    st.bar_chart(pivot_df)
    # --------------------------------------------------------------

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
        new_text = st.text_area("Edit prompt", value=strategies[selected], height=200)
        if st.button("💾 Save changes"):
            strategies[selected] = new_text
            save_strategies(strategies)
            st.success(f"Strategy '{selected}' updated!")
            st.experimental_rerun()

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
            strategies[new_name] = new_prompt
            save_strategies(strategies)
            st.success(f"Strategy '{new_name}' added!")
            st.experimental_rerun()

# ==========================================================
# Tab 4 : Batch Experiments (inchangé)
# ==========================================================
with tabs[3]:
    st.markdown("## 🔄 Batch Experiments")

    data_dir = st.text_input("📂 Data folder", "data/apps/")
    output_dir = st.text_input("📂 Output folder", "output/apps/")
    strategies = load_strategies()
    selected_strategies = st.multiselect("🧩 Select strategies", list(strategies.keys()), default=list(strategies.keys()))
    top_k_values = st.multiselect("Top-K values", [5, 10, 20], default=[5, 10, 20])

    # Batch LLM picker
    st.markdown("### 🤖 LLM for Batch")
    registry = load_llm_registry()
    if registry:
        keys = list(registry.keys())
        labels = [f"{k} · {registry[k]['label']} ({registry[k]['config'].provider})" for k in keys]
        choice_b = st.selectbox("LLM Slot for batch (.env)", labels, index=0, key="prov_b")
        llm_batch = build_llm_from_slot(keys[labels.index(choice_b)])
    else:
        st.info("No .env slots detected. Falling back to manual selection.")
        prov_b = st.selectbox("LLM Provider (batch)", ["azure-openai", "openai", "ollama"], index=0, key="prov_b2")
        mdl_b_list = available_models(prov_b)
        if prov_b == "azure-openai":
            base_default = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE", "")
            ver_default  = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
            dep_default  = os.getenv("OPENAI_DEPLOYMENT_NAME", "")
            with st.expander("Advanced Azure settings (batch)", expanded=False):
                api_base_b = st.text_input("Azure Resource endpoint (batch)", value=base_default, key="api_base_b")
                api_version_b = st.text_input("API version (batch)", value=ver_default, key="api_version_b")
            deployment_b = st.text_input("Azure deployment name (batch)", value=dep_default, key="deployment_b")
            api_key_env_b = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            if not api_key_env_b:
                api_key_env_b = st.text_input("Azure API Key (batch)", value="", type="password", key="api_key_b")
                if api_key_env_b:
                    os.environ["AZURE_OPENAI_API_KEY"] = api_key_env_b
            llm_batch = ChatModel(LLMConfig(
                provider="azure-openai", model=deployment_b, api_base=api_base_b, api_version=api_version_b, api_key=api_key_env_b
            ))
        elif prov_b == "openai":
            mdl_b = st.selectbox("Model (batch)", mdl_b_list, index=0, key="mdl_b")
            api_key_env_b = os.getenv("OPENAI_API_KEY") or ""
            if not api_key_env_b:
                api_key_env_b = st.text_input("OpenAI API Key (batch)", value="", type="password", key="openai_key_b")
                if api_key_env_b:
                    os.environ["OPENAI_API_KEY"] = api_key_env_b
            base_url_b    = os.getenv("OPENAI_BASE_URL") or None
            llm_batch = ChatModel(LLMConfig(provider="openai", model=mdl_b, api_base=base_url_b, api_key=api_key_env_b))
        else:
            mdl_b = st.selectbox("Model (batch)", mdl_b_list, index=0, key="mdl_b_ol")
            host_b  = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
            llm_batch = ChatModel(LLMConfig(provider="ollama", model=mdl_b, api_base=host_b))

    if st.button("🚀 Run Batch Analysis"):
        if not selected_strategies:
            st.warning("⚠️ Please select at least one strategy.")
            st.stop()

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
                    issues_top, metrics_df, samples = run_selected_experiments(
                        zip_path, csv_path, selected_strategies, top_k=k, llm=llm_batch
                    )
                    (proj_out / f"{app_name}_results_top{k}.csv").write_text(metrics_df.to_csv())

                    metrics_df = metrics_df.copy()
                    metrics_df["Project"] = project
                    metrics_df["ZIP"] = app_name
                    metrics_df["Top-K"] = k
                    all_results.append(metrics_df)
                    st.success(f"✅ {project}/{app_name} done (Top-K={k})")

            if all_results:
                import pandas as _pd
                global_df = _pd.concat(all_results)
                (proj_out / "global_results.csv").write_text(global_df.to_csv())
                st.info(f"📁 Global results saved for {project}: {proj_out/'global_results.csv'}")
