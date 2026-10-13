"""Benchmark pages: overview, results per research question, built-in controls.

Everything here is read-only: the pages render the artefacts written by
``run_paper_experiments.py`` and ``analyze_paper_stats.py`` under
``output/paper_results``. Nothing is recomputed, so what the workbench shows is
exactly what the paper reports.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from .reproducibility import CITATION_BIBTEX, CITATION_TEXT, environment_info, git_info
from .theme import BASELINE, SERIES, card, chip, note, page_header, stat_row

RESULTS_DIR = Path(__file__).resolve().parent.parent / "output" / "paper_results"
STATS_DIR = RESULTS_DIR / "stats"
ALPHA = 0.05

# Paper names for the model slots (deployment names in .env are personal).
MODEL_NAMES = {
    "AZ1": "GPT-4o",
    "AZ2": "GPT-4.1",
    "AZ3": "o3",
    "AZ4": "Qwen2.5-Coder 14B",
    "AZ5": "Llama 3.1 8B",
    "AZ6": "DeepSeek-Coder 6.7B",
}
MODEL_ORDER = list(MODEL_NAMES.values())
PROMPT_NAMES = {
    "baseline": "Baseline",
    "hybrid": "Hybrid",
    "scanner": "Scanner",
    "binary_oracle_baseline": "Binary oracle",
    "conservative_evidence": "Conservative evidence",
    "definition_driven": "Definition-driven",
    "feature_evidence": "Feature evidence",
    "threshold_guided": "Threshold-guided (generated)",
    "prompts-averaged": "Prompts averaged",
}
ORDERING_NAMES = {
    "alphabetical": "Alphabetical",
    "keyword_sorted": "Keyword-sorted",
    "bm25_sorted": "BM25-sorted",
    "bm25_masked": "BM25, identifiers masked",
}
BASELINE_NAMES = {
    "keyword_sorted_only": "Keyword ranking (no LLM)",
    "bm25_only": "BM25 ranking (no LLM)",
    "bm25_masked_only": "BM25 masked (no LLM)",
    "always_present": "Always 'smelly'",
    "metric_heuristic": "Metric heuristic",
}
DATASET_NAMES = {"DACOS_heldout": "DACOS (held-out split)"}


# ---------------------------------------------------------------- loading
@st.cache_data(show_spinner=False, ttl=60)
def _read(rel: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / rel
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _stats(name: str) -> pd.DataFrame | None:
    return _read(f"stats/{name}")


def _model(slot: Any) -> str:
    return MODEL_NAMES.get(str(slot), str(slot))


def _prompt(name: Any) -> str:
    return PROMPT_NAMES.get(str(name), str(name).replace("_", " ").capitalize())


def _ci(value: Any) -> tuple[float, float]:
    try:
        lo, hi = ast.literal_eval(str(value))
        return float(lo), float(hi)
    except Exception:
        return float("nan"), float("nan")


def _p(p: Any) -> str:
    try:
        p = float(p)
    except Exception:
        return "–"
    if pd.isna(p):
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _missing(what: str) -> None:
    note(f"No results found for <b>{what}</b>. Run <code>python run_paper_experiments.py</code> "
         f"then <code>python analyze_paper_stats.py</code> to populate <code>output/paper_results</code>.")


def _latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * 1 + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("–" if pd.isna(v) else f"{v:.3f}")
            else:
                cells.append(esc(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _table(df: pd.DataFrame, *, key: str, caption: str, label: str, fmt: dict[str, str] | None = None) -> None:
    config = {c: st.column_config.NumberColumn(c, format=f) for c, f in (fmt or {}).items()}
    st.dataframe(df, hide_index=True, use_container_width=True, column_config=config)
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("CSV", df.to_csv(index=False), file_name=f"{label.replace(':', '_')}.csv",
                       mime="text/csv", key=f"{key}_csv", use_container_width=True)
    with c2.popover("LaTeX", use_container_width=True):
        st.code(_latex(df, caption, label), language="latex")


def _sig(p: Any) -> str:
    try:
        return "yes" if float(p) < ALPHA else "no"
    except Exception:
        return "–"


# ---------------------------------------------------------------- overview
def _result_counts() -> dict[str, int]:
    counts = {}
    for label, folders in {
        "RQ1–RQ2": ["rq2", "rq2_numbered"],
        "RQ3": ["rq3"],
        "RQ4": ["rq4", "rq4_sorted", "rq4_bm25", "rq4_masked", "eshop"],
        "RQ5": ["rq5"],
    }.items():
        counts[label] = sum(len(list((RESULTS_DIR / f).rglob("*.json"))) for f in folders if (RESULTS_DIR / f).exists())
    return counts


def render_overview() -> None:
    page_header(
        "StaLLM",
        "A benchmark for evaluating large language models on software maintenance tasks, "
        "with built-in controls that tell genuine model signal apart from artefacts of the "
        "evaluation protocol: chance placement, oracle choice, retrieval shortcuts and "
        "optimistic prompt selection.",
        eyebrow="Research workbench",
    )

    counts = _result_counts()
    rq3 = _stats("rq3_by_slot.csv")
    models = sorted({_model(s) for df in (_stats("rq12_rerun.csv"), rq3, _stats("rq4_conditions.csv"))
                     if df is not None for s in df["slot"].unique()}, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    stat_row([
        ("Maintenance tasks", "4", "span recovery, smells, feature & bug location"),
        ("Models evaluated", str(len(models)), ", ".join(models[:4]) + ("…" if len(models) > 4 else "")),
        ("Built-in controls", "4", "chance, oracle, baselines, held-out"),
        ("Stored result files", str(sum(counts.values())), " · ".join(f"{k} {v}" for k, v in counts.items())),
    ])

    st.markdown("## What the benchmark measures")
    matrix = pd.DataFrame([
        ["RQ1–RQ2", "Static-analysis finding recovery", "Span", "SonarQube report (eShopOnContainers Catalog)", "P / R / F1, IoU + line tolerance", "Random placement"],
        ["RQ3", "Code-smell classification", "Method / class", "Human labels (DACOS, MLCQ)", "F1, Cohen's κ", "Oracle substitution, non-LLM baselines"],
        ["RQ4", "Feature location", "File ranking", "Feature traces (ArgoUML, eShopOnContainers)", "Hit@k, MRR, P@10", "Retrieval baselines, identifier masking"],
        ["RQ5", "Automated prompt selection", "Prompt", "Human labels, held-out split", "F1, McNemar", "Held-out evaluation"],
    ], columns=["RQ", "Task", "Granularity", "Oracle", "Metrics", "Control"])
    head = "".join(f"<th>{c}</th>" for c in matrix.columns)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in matrix.itertuples(index=False))
    st.markdown(f'<div class="pro-table-wrap"><table class="pro-table wrap"><thead><tr>{head}</tr></thead>'
                f'<tbody>{body}</tbody></table></div>', unsafe_allow_html=True)

    st.markdown("## Why scores need controls")
    cols = st.columns(4)
    blocks = [
        ("Random placement", "C1", "Would spans placed at random in the same files score as well? Separates localisation from file-level luck."),
        ("Oracle substitution", "C2", "Do the conclusions survive when the human oracle is swapped for a metric heuristic?"),
        ("Non-LLM baselines", "C3", "Does the LLM beat keyword / BM25 ranking and trivial classifiers, also with identifiers masked?"),
        ("Held-out selection", "C4", "Is the best prompt still best on data it was not selected on?"),
    ]
    for col, (title, k, body) in zip(cols, blocks):
        col.markdown(card(title, body, kicker=f"Control {k}"), unsafe_allow_html=True)
    st.markdown("")
    note("Open <b>Built-in Controls</b> in the sidebar for the verdict of each control on the stored results, "
         "and <b>Benchmark Results</b> for the full tables with confidence intervals and corrected p-values.")

    st.markdown("## Workbench")
    cols = st.columns(3)
    for col, (title, body) in zip(cols, [
        ("Run tasks", "Configure a model slot, a prompt and an oracle, then run any maintenance task interactively. "
                      "Every run can be exported with a provenance manifest."),
        ("Optimise prompts", "Generate and score prompt variants on a training split, then confirm the winner on a held-out split."),
        ("Inspect and export", "Browse stored runs, compare prompts and models, and export tables as CSV or LaTeX."),
    ]):
        col.markdown(card(title, body), unsafe_allow_html=True)


# ---------------------------------------------------------------- RQ1/RQ2
def _rq12_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        cfg = f"{_model(r['slot'])} · {_prompt(r['prompt'])}"
        lo, hi = _ci(r.get("f1_random_ci"))
        flo, fhi = _ci(r.get("f1_random_files_ci"))
        common = {"Protocol": "Numbered lines" if r["protocol"] == "numbered" else "Original", "Configuration": cfg}
        rows.append({**common, "Series": "LLM", "F1": r["f1_mean"], "lo": r["f1_mean"] - r.get("f1_sd", 0), "hi": r["f1_mean"] + r.get("f1_sd", 0)})
        rows.append({**common, "Series": "Random span in same file", "F1": r["f1_random_mean"], "lo": lo, "hi": hi})
        rows.append({**common, "Series": "Random file", "F1": r["f1_random_files_mean"], "lo": flo, "hi": fhi})
    return pd.DataFrame(rows)


def _rq12_chart(df: pd.DataFrame) -> alt.Chart:
    long = _rq12_frame(df)
    domain = ["LLM", "Random span in same file", "Random file"]
    color = alt.Color("Series:N", scale=alt.Scale(domain=domain, range=[SERIES[0], BASELINE, SERIES[1]]), title=None)
    base = alt.Chart(long).encode(
        y=alt.Y("Configuration:N", title=None, sort=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("Series:N", sort=domain),
    )
    rules = base.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="F1"), x2="hi:Q", color=color)
    pts = base.mark_point(filled=True, size=80, opacity=1).encode(
        x="F1:Q", color=color, shape=alt.Shape("Series:N", sort=domain, legend=None),
        tooltip=["Configuration", "Series", alt.Tooltip("F1:Q", format=".3f"),
                 alt.Tooltip("lo:Q", format=".3f", title="interval low"), alt.Tooltip("hi:Q", format=".3f", title="interval high")],
    )
    return (rules + pts).properties(height=alt.Step(11), width=400).facet(spacing=36, 
        column=alt.Column("Protocol:N", title=None, sort=["Original", "Numbered lines"])
    ).resolve_scale(x="shared")


def _render_rq12() -> None:
    df = _stats("rq12_rerun.csv")
    if df is None:
        _missing("RQ1–RQ2")
        return
    st.markdown("Each model–prompt configuration is run three times on the eShopOnContainers Catalog service. "
                "The LLM's spans are compared with spans placed **at random in the same files** and with spans "
                "in **randomly chosen files** (1,000 draws each). A score that does not exceed the first "
                "baseline reflects file selection, not localisation.")
    st.altair_chart(_rq12_chart(df), use_container_width=False)
    st.caption("Dots: mean F1. Bars: ± 1 SD across runs for the LLM, 95% interval of the random baselines.")
    out = pd.DataFrame({
        "Protocol": df["protocol"].str.capitalize(),
        "Model": df["slot"].map(_model),
        "Prompt": df["prompt"].map(_prompt),
        "Runs": df["runs"],
        "Precision": df["precision"], "Recall": df["recall"],
        "F1 (mean)": df["f1_mean"], "F1 (SD)": df["f1_sd"],
        "Random-in-file F1": df["f1_random_mean"], "p vs in-file": df["p_random"].map(_p),
        "Random-file F1": df["f1_random_files_mean"], "p vs file": df["p_random_files"].map(_p),
    })
    _table(out, key="rq12", caption="Span-level recovery of static-analysis findings against random placement",
           label="tab:rq12", fmt={c: "%.3f" for c in ["Precision", "Recall", "F1 (mean)", "F1 (SD)", "Random-in-file F1", "Random-file F1"]})


# ---------------------------------------------------------------- RQ3
def _render_rq3() -> None:
    df = _stats("rq3_by_slot.csv")
    swap = _stats("rq3_oracle_swap.csv")
    base = _read("summary_rq3_baselines.csv")
    if df is None:
        _missing("RQ3")
        return
    st.markdown("Binary smell classification against **human labels** (DACOS, MLCQ; 200 samples each). "
                "Confidence intervals are bootstrap 95% intervals over samples.")
    plot = df.copy()
    plot["Model"] = plot["slot"].map(_model)
    plot["Prompt"] = plot["prompt"].map(_prompt)
    plot[["lo", "hi"]] = plot["f1_ci"].apply(lambda v: pd.Series(_ci(v)))
    models = [m for m in MODEL_ORDER if m in set(plot["Model"])]
    color = alt.Color("Model:N", scale=alt.Scale(domain=models, range=SERIES[:len(models)]))
    enc = alt.Chart().encode(y=alt.Y("Prompt:N", title=None, axis=alt.Axis(labelLimit=220)), yOffset=alt.YOffset("Model:N", sort=models))
    layers = [enc.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="F1 vs human labels", scale=alt.Scale(domain=[0, 1])), x2="hi:Q", color=color),
              enc.mark_point(filled=True, size=80, opacity=1).encode(
                  x="f1:Q", color=color,
                  tooltip=["Model", "Prompt", "dataset", alt.Tooltip("f1:Q", format=".3f"), alt.Tooltip("kappa:Q", format=".3f", title="κ")])]
    if base is not None:
        # One rule per facet: encode it on the faceted data itself, not on a separate frame.
        rules = base[base["method"] == "metric_heuristic"][["dataset", "f1"]].rename(columns={"f1": "heur"})
        plot = plot.merge(rules, on="dataset", how="left")
        layers.append(alt.Chart().mark_rule(strokeDash=[4, 3], color=BASELINE, strokeWidth=1.5).encode(
            x="mean(heur):Q", tooltip=[alt.Tooltip("mean(heur):Q", format=".3f", title="Metric heuristic F1")]))
    chart = alt.layer(*layers, data=plot).properties(height=alt.Step(14), width=380).facet(
        column=alt.Column("dataset:N", title=None))
    st.altair_chart(chart, use_container_width=False)
    st.caption("Dashed line: non-LLM metric heuristic (Designite-style thresholds) on the same samples.")

    out = pd.DataFrame({
        "Dataset": df["dataset"], "Model": df["slot"].map(_model), "Prompt": df["prompt"].map(_prompt),
        "Precision": df["precision"], "Recall": df["recall"], "F1": df["f1"], "F1 95% CI": df["f1_ci"],
        "κ": df["kappa"], "Accuracy": df["accuracy"],
    })
    _table(out, key="rq3", caption="Code-smell classification against human labels", label="tab:rq3",
           fmt={c: "%.3f" for c in ["Precision", "Recall", "F1", "κ", "Accuracy"]})

    if base is not None:
        st.markdown("#### Non-LLM baselines")
        b = pd.DataFrame({"Dataset": base["dataset"].replace(DATASET_NAMES), "Baseline": base["method"].map(BASELINE_NAMES),
                          "Precision": base["precision"], "Recall": base["recall"], "F1": base["f1"],
                          "F1 95% CI": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(base["f1_ci_lo"], base["f1_ci_hi"])],
                          "κ": base["kappa"]})
        _table(b, key="rq3b", caption="Non-LLM baselines for smell classification", label="tab:rq3-baselines",
               fmt={c: "%.3f" for c in ["Precision", "Recall", "F1", "κ"]})

    mc = _stats("rq3_mcnemar.csv")
    if mc is not None:
        st.markdown("#### Pairwise comparisons (McNemar, Holm-corrected)")
        m = pd.DataFrame({
            "Factor": mc["factor"], "Held fixed": mc["fixed"].map(lambda v: _model(v) if str(v).startswith("AZ") else _prompt(v)),
            "Dataset": mc["dataset"],
            "A": mc["a"].map(lambda v: _model(v) if str(v).startswith("AZ") else _prompt(v)),
            "B": mc["b"].map(lambda v: _model(v) if str(v).startswith("AZ") else _prompt(v)),
            "Only A correct": mc["a_only_correct"], "Only B correct": mc["b_only_correct"],
            "p": mc["p"].map(_p), "p (Holm)": mc["p_holm"].map(_p), "Significant": mc["p_holm"].map(_sig),
        })
        _table(m, key="rq3mc", caption="McNemar tests between prompts and between models", label="tab:rq3-mcnemar")


# ---------------------------------------------------------------- RQ4
def _render_rq4() -> None:
    system = st.radio("System", ["ArgoUML (8 features)", "eShopOnContainers (5 features)"], horizontal=True, key="rq4_sys")
    suffix = "" if system.startswith("Argo") else "_eshop"
    df = _stats(f"rq4_conditions{suffix}.csv")
    wil = _stats(f"rq4_wilcoxon{suffix}.csv")
    if df is None:
        _missing("RQ4")
        return
    st.markdown("The LLM re-ranks a candidate list whose initial **ordering** varies. Comparing the LLM with "
                "the same ordering *without* the LLM isolates what the model adds; masking identifiers removes "
                "lexical shortcuts.")
    d = df[df["method"] == "LLM"].copy()
    d["Model"] = d["slot"].map(_model)
    d["Ordering"] = d["ordering"].map(ORDERING_NAMES).fillna(d["ordering"])
    agg = d.groupby(["Model", "Ordering"], as_index=False)["p@10"].mean()
    models = [m for m in MODEL_ORDER if m in set(agg["Model"])]
    orders = [o for o in ORDERING_NAMES.values() if o in set(agg["Ordering"])]
    color = alt.Color("Model:N", scale=alt.Scale(domain=models, range=SERIES[:len(models)]))
    pts = alt.Chart(agg).mark_point(filled=True, size=90, opacity=1).encode(
        y=alt.Y("Ordering:N", sort=orders, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("Model:N", sort=models),
        x=alt.X("p@10:Q", title="Precision@10 (mean over features and prompts)", scale=alt.Scale(domain=[0, 1])),
        color=color, tooltip=["Model", "Ordering", alt.Tooltip("p@10:Q", format=".3f")])
    layers = [pts]
    if wil is not None:
        b = wil[(wil["metric"] == "p@10") & wil["a"].str.contains("prompts-averaged")]
        b = b.groupby("b", as_index=False)["mean_b"].first()
        if not b.empty:
            b["Baseline"] = b["b"].map(BASELINE_NAMES).fillna(b["b"])
            layers.append(alt.Chart(b).mark_rule(strokeDash=[4, 3], color=BASELINE).encode(
                x="mean_b:Q", tooltip=["Baseline", alt.Tooltip("mean_b:Q", format=".3f", title="P@10")]))
            layers.append(alt.Chart(b).mark_text(angle=270, dx=-6, align="right", baseline="bottom", fontSize=10, color="#5b6270").encode(
                x="mean_b:Q", y=alt.value(0), text="Baseline"))
    st.altair_chart(alt.layer(*layers).properties(height=alt.Step(13), width=640), use_container_width=False)
    st.caption("Dashed lines: the same candidate orderings without the LLM.")

    out = pd.DataFrame({"Model": d["Model"], "Prompt": d["prompt"].map(_prompt), "Ordering": d["Ordering"],
                        "Features": d["features"], "Runs": d["runs"], "Hit@1": d["hit@1"], "Hit@5": d["hit@5"],
                        "MRR": d["mrr"], "P@10": d["p@10"]})
    _table(out, key=f"rq4{suffix}", caption=f"Feature location on {system}", label=f"tab:rq4{suffix}",
           fmt={c: "%.3f" for c in ["Hit@1", "Hit@5", "MRR", "P@10"]})
    if wil is not None:
        st.markdown("#### LLM vs the same ordering without the LLM (Wilcoxon signed-rank)")
        metric = st.segmented_control("Metric", ["p@10", "mrr", "map"], default="p@10", key=f"rq4m{suffix}") \
            if hasattr(st, "segmented_control") else st.radio("Metric", ["p@10", "mrr", "map"], horizontal=True, key=f"rq4m{suffix}")
        w = wil[wil["metric"] == (metric or "p@10")].copy()
        parts = w["a"].str.split("/", expand=True)
        w2 = pd.DataFrame({"Model": parts[0].map(_model), "Prompt": parts[1].map(_prompt),
                           "Ordering": parts[2].map(ORDERING_NAMES).fillna(parts[2]),
                           "Compared with": w["b"].map(BASELINE_NAMES).fillna(w["b"]), "n": w["n"],
                           "LLM": w["mean_a"], "No LLM": w["mean_b"], "LLM wins": w["wins_a"], "Baseline wins": w["wins_b"],
                           "p": w["p"].map(_p), "p (Holm)": w["p_holm"].map(_p)})
        _table(w2, key=f"rq4w{suffix}", caption="Wilcoxon tests, LLM re-ranking vs non-LLM ordering",
               label=f"tab:rq4-wilcoxon{suffix}", fmt={"LLM": "%.3f", "No LLM": "%.3f"})


# ---------------------------------------------------------------- RQ5
def _render_rq5() -> None:
    df = _stats("rq5_heldout.csv")
    split = _stats("sensitivity_rq5_splits.csv")
    if df is None:
        _missing("RQ5")
        return
    st.markdown("The AutoResearch loop generates prompt variants and selects one on a training split. "
                "The selected prompt is then scored on a **held-out** DACOS split alongside the hand-written prompts.")
    p = df.copy()
    p["Prompt"] = p["prompt"].map(_prompt)
    p["Source"] = p["source"].map({"RQ3_manual": "Hand-written", "RQ5_generated": "Generated (selected on train)"}).fillna(p["source"])
    p[["lo", "hi"]] = p["f1_ci"].apply(lambda v: pd.Series(_ci(v)))
    dom = ["Hand-written", "Generated (selected on train)"]
    color = alt.Color("Source:N", scale=alt.Scale(domain=dom, range=[BASELINE, SERIES[0]]), title=None)
    enc = alt.Chart(p).encode(y=alt.Y("Prompt:N", title=None, sort="-x"))
    chart = (enc.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="F1 on held-out split", scale=alt.Scale(domain=[0, 1])), x2="hi:Q", color=color)
             + enc.mark_point(filled=True, size=90, opacity=1).encode(x="f1:Q", color=color,
                                                                     tooltip=["Prompt", "Source", alt.Tooltip("f1:Q", format=".3f"), "f1_ci"]))
    st.altair_chart(chart.properties(height=alt.Step(26), width=560), use_container_width=False)
    out = pd.DataFrame({"Source": p["Source"], "Prompt": p["Prompt"], "Precision": p["precision"], "Recall": p["recall"],
                        "F1": p["f1"], "F1 95% CI": p["f1_ci"], "κ": p["kappa"],
                        "Only generated correct": p["generated_only_correct"], "Only manual correct": p["manual_only_correct"],
                        "McNemar p": p["mcnemar_p"].map(_p)})
    _table(out, key="rq5", caption="Held-out evaluation of the generated prompt", label="tab:rq5",
           fmt={c: "%.3f" for c in ["Precision", "Recall", "F1", "κ"]})
    if split is not None and not split.empty:
        r = split.iloc[0]
        lo, hi = _ci(r["optimism_ci"])
        st.markdown("#### Selection optimism over random splits")
        stat_row([
            ("Random splits", f"{int(r['repeats']):,}", f"{int(r['prompts'])} candidate prompts"),
            ("Mean optimism (train − test F1)", f"{r['optimism_mean']:+.3f}", f"95% interval [{lo:+.3f}, {hi:+.3f}]"),
            ("Train winner also best on test", f"{r['winner_also_best_on_test']:.0%}", "of splits"),
            ("Generated prompt wins selection", f"{r['generated_wins_selection']:.0%}", "of splits"),
        ])


def render_benchmark_results() -> None:
    page_header("Benchmark Results",
                "Stored results for each research question, with confidence intervals, non-LLM and chance "
                "baselines, and multiplicity-corrected tests. Tables export to CSV and LaTeX.",
                eyebrow="Evidence · RQ1–RQ5")
    tabs = st.tabs(["RQ1–RQ2 · Finding recovery", "RQ3 · Smell classification", "RQ4 · Feature location",
                    "RQ5 · Prompt selection", "Sensitivity analyses"])
    with tabs[0]:
        _render_rq12()
    with tabs[1]:
        _render_rq3()
    with tabs[2]:
        _render_rq4()
    with tabs[3]:
        _render_rq5()
    with tabs[4]:
        _render_sensitivity()


def _render_sensitivity() -> None:
    st.markdown("How much each conclusion depends on analysis choices: random seed and matching preset "
                "(RQ1–RQ2), heuristic thresholds (RQ3), retrieval depth (RQ4) and split (RQ5).")
    items = [
        ("Chance baselines under other seeds and presets", "sensitivity_chance.csv"),
        ("Metric-heuristic thresholds (oracle substitution)", "sensitivity_rq3_thresholds.csv"),
        ("Candidate retrieval ceiling", "sensitivity_rq4_retrieval.csv"),
        ("Prompt-selection splits", "sensitivity_rq5_splits.csv"),
    ]
    for title, name in items:
        df = _stats(name)
        if df is None:
            continue
        with st.expander(title, expanded=False):
            _table(df, key=f"sens_{name}", caption=title, label=f"tab:{name.removesuffix('.csv')}")
    md = STATS_DIR / "sensitivity.md"
    if md.exists():
        with st.expander("Full sensitivity report (generated)", expanded=False):
            st.markdown(md.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- controls
def _control(title: str, threat: str, finding: str, verdict: tuple[str, str], source: str) -> None:
    st.markdown(
        f'<div class="control"><div class="head"><div class="title">{title}</div>{chip(*verdict)}</div>'
        f'<div class="threat"><b>Threat addressed.</b> {threat}</div>'
        f'<div class="finding">{finding}</div><div class="src">source: {source}</div></div>',
        unsafe_allow_html=True,
    )


def _verdict(passed: int, total: int) -> tuple[str, str]:
    if total == 0:
        return ("no data", "neutral")
    if passed == total:
        return (f"signal confirmed · {passed}/{total}", "good")
    if passed == 0:
        return (f"not distinguishable · 0/{total}", "bad")
    return (f"mixed · {passed}/{total}", "warn")


def render_controls() -> None:
    page_header("Built-in Controls",
                "Each control re-reads the stored results against an alternative explanation of the score. "
                "A configuration counts as passing when it beats the control at α = 0.05.",
                eyebrow="Validity")

    # C1 - random placement
    df = _stats("rq12_rerun.csv")
    if df is not None:
        rows = []
        for proto, g in df.groupby("protocol"):
            infile = int((g["p_random"] < ALPHA).sum())
            files = int((g["p_random_files"] < ALPHA).sum())
            rows.append((proto, infile, files, len(g)))
        best = df.loc[df["f1_mean"].idxmax()]
        txt = " ".join(
            f"<b>{p.capitalize()} protocol:</b> {i}/{n} configurations beat random spans in the same files, "
            f"{f}/{n} beat random files." for p, i, f, n in rows)
        txt += (f" Best configuration: {_model(best['slot'])} · {_prompt(best['prompt'])} "
                f"(F1 {best['f1_mean']:.3f} vs {best['f1_random_mean']:.3f} at random, p {_p(best['p_random'])}).")
        _control("C1 · Random placement (RQ1–RQ2)",
                 "Spans scored against a lenient matcher can look correct because they land in the right file.",
                 txt, _verdict(sum(r[1] for r in rows), sum(r[3] for r in rows)), "stats/rq12_rerun.csv")

    # C2 - oracle substitution
    sens = _stats("sensitivity_rq3_thresholds.csv")
    if sens is not None:
        d = sens[sens["thresholds"] == "default"]
        parts, stable = [], 0
        for _, r in d.iterrows():
            same = r["best_vs_human"] == r["best_vs_metric"]
            stable += int(same)
            bh, bm = (str(r["best_vs_human"]).split("/"), str(r["best_vs_metric"]).split("/"))
            parts.append(f"<b>{r['dataset']}:</b> Spearman ρ = {r['spearman_rho']:.2f} (p {_p(r['spearman_p'])}) between "
                         f"the two rankings of {int(r['configs'])} configurations; best under human labels "
                         f"{_model(bh[0])} · {_prompt(bh[-1])}, under the heuristic {_model(bm[0])} · {_prompt(bm[-1])}.")
        _control("C2 · Oracle substitution (RQ3)",
                 "A ranking of models and prompts may reflect the oracle that scored them rather than the task.",
                 " ".join(parts) + (" The best configuration is the same under both oracles." if stable == len(d)
                                    else " The best configuration changes with the oracle."),
                 ("ranking stable", "good") if stable == len(d) else (f"oracle-dependent · {len(d) - stable}/{len(d)} datasets", "warn"),
                 "stats/sensitivity_rq3_thresholds.csv")

    # C3 - non-LLM baselines and masking
    wil = _stats("rq4_wilcoxon.csv")
    rq3 = _stats("rq3_by_slot.csv")
    base = _read("summary_rq3_baselines.csv")
    parts, passed, total = [], 0, 0
    if rq3 is not None and base is not None:
        for ds, g in rq3.groupby("dataset"):
            heur = base[(base["method"] == "metric_heuristic") & (base["dataset"] == ds)]
            if heur.empty:
                continue
            h = float(heur["f1"].iloc[0])
            hlo = float(heur["f1_ci_hi"].iloc[0])
            above = int((g["f1"] > h).sum())
            best = float(g["f1"].max())
            where = "inside" if best <= hlo else "above"
            parts.append(f"<b>RQ3 {ds}:</b> {above}/{len(g)} configurations exceed the metric heuristic's F1 ({h:.3f}); "
                         f"best {best:.3f}, {where} the heuristic's 95% interval (upper bound {hlo:.3f}).")
    if wil is not None:
        avg = wil[(wil["metric"] == "p@10") & wil["a"].str.contains("prompts-averaged")]
        for b, g in avg.groupby("b"):
            sig = int((g["p"] < ALPHA).sum())
            passed += sig
            total += len(g)
            parts.append(f"<b>RQ4 vs {BASELINE_NAMES.get(b, b)}:</b> LLM re-ranking significant for {sig}/{len(g)} models "
                         f"(P@10 {g['mean_a'].min():.2f}–{g['mean_a'].max():.2f} vs {g['mean_b'].iloc[0]:.2f}).")
    if parts:
        _control("C3 · Non-LLM baselines and identifier masking (RQ3–RQ4)",
                 "Lexical retrieval or trivial classifiers may already explain the score; identifiers may leak the answer.",
                 " ".join(parts), _verdict(passed, total), "stats/rq4_wilcoxon.csv · summary_rq3_baselines.csv")

    # C4 - held-out selection
    split = _stats("sensitivity_rq5_splits.csv")
    held = _stats("rq5_heldout.csv")
    if split is not None and not split.empty:
        r = split.iloc[0]
        lo, hi = _ci(r["optimism_ci"])
        txt = (f"Over {int(r['repeats']):,} random splits, the prompt selected on train is also best on test in "
               f"{r['winner_also_best_on_test']:.0%} of splits; mean optimism {r['optimism_mean']:+.3f} F1 "
               f"(95% interval [{lo:+.3f}, {hi:+.3f}]).")
        if held is not None:
            gen = held[held["source"] == "RQ5_generated"]
            man = held[held["source"] == "RQ3_manual"]
            if not gen.empty and not man.empty:
                txt += (f" On the fixed held-out split the generated prompt scores F1 {gen['f1'].iloc[0]:.3f} vs "
                        f"{man['f1'].max():.3f} for the best hand-written prompt.")
        ok = r["winner_also_best_on_test"] >= 0.5 and lo > 0
        _control("C4 · Held-out selection (RQ5)",
                 "A prompt chosen because it scored best on some data is optimistically biased on that same data.",
                 txt, ("selection holds", "good") if ok else ("selection not confirmed", "warn"),
                 "stats/sensitivity_rq5_splits.csv · stats/rq5_heldout.csv")


# ---------------------------------------------------------------- about
def render_about() -> None:
    page_header("About & Citation", "Provenance of this workbench and how to cite the benchmark.", eyebrow="Reproducibility")
    g = git_info()
    env = environment_info()
    stat_row([
        ("Code version", g["short"] + (" (modified)" if g["dirty"] else ""), f"{g['branch']} · {g['date']}"),
        ("Python", env["python"], env["platform"]),
        ("Streamlit", env["streamlit"], f"pandas {env['pandas']}"),
    ])
    st.markdown("## Cite")
    st.markdown(CITATION_TEXT)
    st.code(CITATION_BIBTEX, language="bibtex")
    st.markdown("## Reproducing the paper")
    st.code(
        "python run_paper_experiments.py --rq 2 --slots AZ1,AZ2,AZ3 --runs 3   # RQ1–RQ2\n"
        "python run_paper_experiments.py --rq 3 --slots AZ1,AZ2,AZ3            # RQ3\n"
        "python run_paper_experiments.py --rq 4 --slots AZ1,AZ2,AZ3,AZ4        # RQ4\n"
        "python run_paper_experiments.py --rq 5h                               # RQ5 held-out\n"
        "python analyze_paper_stats.py && python sensitivity_analyses.py\n"
        "python generate_paper_figures.py",
        language="bash",
    )
    note("Model slots are configured in <code>.env</code> (see <code>env.example</code>). Hosted models run at "
         "temperature 0; local Ollama runs must be serialised. Every interactive run can be exported with a "
         "<b>run manifest</b> recording code version, environment, prompt hashes and evaluation settings.")
    st.markdown("## Authors")
    st.markdown("Tewfik Ziadi, Seifeddine Bouallegue, Reda Bendraou — University of Doha for Science and Technology.")
