"""Run History page: stored runs, aggregated per configuration."""
from __future__ import annotations

import math

import altair as alt
import pandas as pd
import streamlit as st

from StaLLM_models import Session, SmellDetectionResult

from .paper_results import _table
from .theme import SERIES, page_header, stat_row

# Two-sided 95% Student t quantiles for small n (df = n - 1).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def _t95(df: int) -> float:
    keys = [k for k in sorted(_T95) if k <= df]
    return _T95[keys[-1]] if keys else float("nan")


def _load_runs() -> pd.DataFrame:
    session = Session()
    try:
        rows = session.query(SmellDetectionResult).all()
        return pd.DataFrame([{
            "id": r.id, "Timestamp": r.timestamp, "Project": r.project or "", "Language": r.language or "",
            "Strategy": (r.strategy or "").removesuffix("@span"), "Model": r.llm_used or "unknown",
            "Top-K": r.top_k, "Precision": r.precision, "Recall": r.recall, "F1": r.f1,
            "Time (s)": r.time_elapsed, "Total tokens": r.total_tokens, "USD cost": r.usd_cost,
        } for r in rows])
    finally:
        session.close()


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["Project", "Model", "Strategy", "Top-K"]
    g = df.groupby(keys, dropna=False)
    out = g.agg(Runs=("F1", "size"), Precision=("Precision", "mean"), Recall=("Recall", "mean"),
                F1=("F1", "mean"), F1_SD=("F1", "std"), Tokens=("Total tokens", "mean"),
                Cost=("USD cost", "sum")).reset_index()
    half = [(_t95(int(n) - 1) * sd / math.sqrt(n)) if n > 1 and pd.notna(sd) else float("nan")
            for n, sd in zip(out["Runs"], out["F1_SD"])]
    out["F1 CI low"] = out["F1"] - pd.Series(half)
    out["F1 CI high"] = out["F1"] + pd.Series(half)
    return out.sort_values(["Project", "F1"], ascending=[True, False])


def render_run_history() -> None:
    page_header("Run History",
                "Runs stored in the local SQLite database, aggregated per configuration "
                "(project × model × prompt × universe size) with mean, standard deviation and a 95% "
                "t-interval over repeated runs.", eyebrow="Resources")
    try:
        df = _load_runs()
    except Exception as exc:
        st.error(f"Could not read the run database: {exc}")
        return
    if df.empty:
        st.info("No runs stored yet. Runs launched from Run Tasks or Batch Runs are saved here automatically.")
        return

    c1, c2, c3 = st.columns(3)
    projects = c1.multiselect("Project", sorted(df["Project"].unique()), placeholder="All projects")
    models = c2.multiselect("Model", sorted(df["Model"].unique()), placeholder="All models")
    strategies = c3.multiselect("Prompt strategy", sorted(df["Strategy"].unique()), placeholder="All strategies")
    f = df
    if projects:
        f = f[f["Project"].isin(projects)]
    if models:
        f = f[f["Model"].isin(models)]
    if strategies:
        f = f[f["Strategy"].isin(strategies)]
    if f.empty:
        st.info("No runs match the current filters.")
        return

    agg = _aggregate(f)
    stat_row([
        ("Runs", f"{len(f):,}", f"{f['Timestamp'].min():%Y-%m-%d} → {f['Timestamp'].max():%Y-%m-%d}"),
        ("Configurations", f"{len(agg):,}", f"{int((agg['Runs'] > 1).sum())} with repeated runs"),
        ("Best mean F1", f"{agg['F1'].max():.3f}", agg.loc[agg['F1'].idxmax(), 'Strategy'] + " · " + agg.loc[agg['F1'].idxmax(), 'Project']),
        ("Total cost", f"${f['USD cost'].fillna(0).sum():.2f}", f"{int(f['Total tokens'].fillna(0).sum()):,} tokens"),
    ])

    st.markdown("## Mean F1 per configuration")
    plot = agg.copy()
    plot["Configuration"] = plot["Strategy"] + " · K=" + plot["Top-K"].astype(str)
    plot["lo"] = plot["F1 CI low"].fillna(plot["F1"]).clip(lower=0)
    plot["hi"] = plot["F1 CI high"].fillna(plot["F1"])
    model_dom = sorted(plot["Model"].unique())
    if len(model_dom) <= len(SERIES):
        color = alt.Color("Model:N", scale=alt.Scale(domain=model_dom, range=SERIES[:len(model_dom)]))
        enc = alt.Chart(plot).encode(y=alt.Y("Configuration:N", title=None), yOffset=alt.YOffset("Model:N"))
        chart = (enc.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="F1 (mean, 95% t-interval)"), x2="hi:Q", color=color)
                 + enc.mark_point(filled=True, size=70, opacity=1).encode(
                     x="F1:Q", color=color,
                     tooltip=["Project", "Model", "Strategy", "Top-K", "Runs",
                              alt.Tooltip("F1:Q", format=".3f"), alt.Tooltip("F1_SD:Q", format=".3f", title="SD")]))
        rows = alt.Row("Project:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left", labelFontWeight=600))
        st.altair_chart(chart.properties(width=620, height=alt.Step(12)).facet(row=rows), use_container_width=False)
        st.caption("Configurations with a single run show a dot without an interval.")

    table = agg.rename(columns={"F1_SD": "F1 SD", "Tokens": "Mean tokens", "Cost": "Total cost ($)"})
    _table(table, key="hist_agg", caption="Stored runs aggregated per configuration", label="tab:run-history",
           fmt={c: "%.3f" for c in ["Precision", "Recall", "F1", "F1 SD", "F1 CI low", "F1 CI high"]}
           | {"Mean tokens": "%d", "Total cost ($)": "%.4f"})

    with st.expander(f"Individual runs ({len(f)})", expanded=False):
        raw = f.sort_values("Timestamp", ascending=False)
        st.dataframe(raw, hide_index=True, use_container_width=True,
                     column_config={"Timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                                    **{c: st.column_config.NumberColumn(format="%.3f") for c in ["Precision", "Recall", "F1"]}})
        st.download_button("Download runs (CSV)", raw.to_csv(index=False), file_name="stallm_runs.csv",
                           mime="text/csv", key="hist_raw_csv")
