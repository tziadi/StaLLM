"""
Generate publication-quality figures for the StaLLM journal paper.

The script reads experiment summaries from output/paper_results by default and
writes figures to Publications/StaLLM___Journal/images. The publication folder
remains ignored by git; this script only makes the local Overleaf assets easier
to regenerate from experiment outputs.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/stallm_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/stallm_cache")

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS = Path("output/paper_results")
DEFAULT_OUT = Path("Publications/StaLLM___Journal/images")

PROMPT_ORDER = ["binary_oracle_baseline", "conservative_evidence", "definition_driven"]
PROMPT_LABELS = {
    "binary_oracle_baseline": "Baseline",
    "conservative_evidence": "Conservative",
    "definition_driven": "Def.-driven",
}

RQ4_FEATURE_ORDER = [
    "ACTIVITYDIAGRAM",
    "COGNITIVE",
    "COLLABORATIONDIAGRAM",
    "DEPLOYMENTDIAGRAM",
    "LOGGING",
    "SEQUENCEDIAGRAM",
    "STATEDIAGRAM",
    "USECASEDIAGRAM",
]
RQ4_FEATURE_LABELS = ["Activity", "Cognitive", "Collab.", "Deploy.", "Logging", "Sequence", "State", "UseCase"]
RQ4_CONDITIONS = [
    ("baseline", "alphabetical", "Alph.\n+Baseline"),
    ("feature_evidence", "alphabetical", "Alph.\n+Feat."),
    ("baseline", "keyword_sorted", "KW\n+Baseline"),
    ("feature_evidence", "keyword_sorted", "KW\n+Feat."),
]

# Colorblind-friendly Okabe-Ito palette.
C1 = "#0072B2"
C2 = "#D55E00"
C3 = "#009E73"
C4 = "#CC79A7"
GRAY = "#5F6368"
LIGHT_GRID = "#E6E8EB"
PAPER_CMAP = LinearSegmentedColormap.from_list("stallm_blue_orange", ["#F7F7F7", "#9ECAE1", C1])


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 8.8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.6,
    })


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"  [skip] missing {path}")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _single_feature_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r.get("feature") in RQ4_FEATURE_ORDER]


def _compound_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if "_and_" in r.get("feature", "")]


def _rq4_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        (r.get("feature", ""), r.get("prompt", ""), r.get("candidate_order", "")): r
        for r in rows
    }


def _avg(rows: list[dict[str, str]], metric: str) -> float:
    if not rows:
        return 0.0
    return sum(_float(r, metric) for r in rows) / len(rows)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


def fig_rq3(results_dir: Path, out_dir: Path) -> None:
    rows = _read_csv(results_dir / "summary_rq3.csv")
    if not rows:
        return

    by_key = {(r.get("prompt"), r.get("dataset")): r for r in rows}
    prompts = [PROMPT_LABELS[p] for p in PROMPT_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharey=True)
    x = np.arange(len(prompts))
    width = 0.32
    for ax, dataset, panel in zip(axes, ["DACOS", "MLCQ"], ["A", "B"]):
        f1 = [_float(by_key.get((p, dataset), {}), "f1") for p in PROMPT_ORDER]
        kappa = [_float(by_key.get((p, dataset), {}), "kappa") for p in PROMPT_ORDER]
        f1_err = np.array([
            [
                max(0.0, _float(by_key.get((p, dataset), {}), "f1") - _float(by_key.get((p, dataset), {}), "f1_ci_lo")),
                max(0.0, _float(by_key.get((p, dataset), {}), "f1_ci_hi") - _float(by_key.get((p, dataset), {}), "f1")),
            ]
            for p in PROMPT_ORDER
        ]).T
        ax.bar(x - width / 2, f1, width, color=C1, edgecolor="white", linewidth=0.5,
               label="F1", yerr=f1_err, capsize=2, error_kw={"elinewidth": 0.6, "capthick": 0.6})
        ax.bar(x + width / 2, kappa, width, color=C2, edgecolor="white", linewidth=0.5,
               label="Cohen's $\\kappa$")
        for xi, value in zip(x - width / 2, f1):
            ax.text(xi, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=6.3)
        for xi, value in zip(x + width / 2, kappa):
            ax.text(xi, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=6.3)
        ax.set_title(f"{panel}. {dataset}", loc="left", fontweight="bold", pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(prompts, rotation=18, ha="right")
        ax.set_ylim(0, 0.70)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, loc="upper right", handlelength=1.0)
    _save(fig, out_dir / "fig_rq3_smells.pdf")


def fig_rq4_hit1(rows: list[dict[str, str]], out_dir: Path) -> None:
    lookup = _rq4_lookup(_single_feature_rows(rows))
    matrix = np.array([
        [_float(lookup.get((feature, prompt, order), {}), "hit@1") for feature in RQ4_FEATURE_ORDER]
        for prompt, order, _ in RQ4_CONDITIONS
    ])
    fig, ax = plt.subplots(figsize=(6.8, 2.65))
    im = ax.imshow(matrix, cmap=PAPER_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(RQ4_FEATURE_LABELS)))
    ax.set_xticklabels(RQ4_FEATURE_LABELS, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(RQ4_CONDITIONS)))
    ax.set_yticklabels([label.replace("\n", " ") for _, _, label in RQ4_CONDITIONS])
    ax.set_xticks(np.arange(-.5, len(RQ4_FEATURE_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(RQ4_CONDITIONS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                    color="white" if value > 0.55 else "#222222", fontsize=8, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_label("Hit@1", rotation=270, labelpad=10)
    ax.set_title("RQ4 - Hit@1 across feature-location conditions", loc="left", fontweight="bold", pad=4)
    _save(fig, out_dir / "fig_rq4_hit1.pdf")


def fig_rq4_map(rows: list[dict[str, str]], out_dir: Path) -> None:
    lookup = _rq4_lookup(_single_feature_rows(rows))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
    prompt_panels = [("baseline", "A. Baseline"), ("feature_evidence", "B. Feature-evidence")]
    for ax, (prompt, title) in zip(axes, prompt_panels):
        alpha_vals = [_float(lookup.get((feature, prompt, "alphabetical"), {}), "map") for feature in RQ4_FEATURE_ORDER]
        kw_vals = [_float(lookup.get((feature, prompt, "keyword_sorted"), {}), "map") for feature in RQ4_FEATURE_ORDER]
        for idx, (feature_label, a_val, k_val) in enumerate(zip(RQ4_FEATURE_LABELS, alpha_vals, kw_vals)):
            color = C3 if k_val >= a_val else C4
            ax.plot([0, 1], [a_val, k_val], color=color, linewidth=1.4, alpha=0.85)
            ax.scatter([0, 1], [a_val, k_val], color=[C1, C2], s=20, zorder=3,
                       edgecolor="white", linewidth=0.4)
            if abs(k_val - a_val) >= 0.06 or feature_label in {"Cognitive", "Deploy."}:
                ax.text(1.04, k_val, feature_label, va="center", fontsize=6.8, color=GRAY)
        ax.set_xlim(-0.12, 1.35)
        ax.set_ylim(0, 0.23)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Alphabetical", "Keyword"])
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
        ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, loc="left", fontweight="bold", pad=3)
    axes[0].set_ylabel("MAP")
    _save(fig, out_dir / "fig_rq4_map.pdf")


def fig_rq4_aggregate(rows: list[dict[str, str]], out_dir: Path) -> None:
    singles = _single_feature_rows(rows)
    labels = ["Hit@1", "Hit@5", "MRR", "MAP"]
    metrics = ["hit@1", "hit@5", "mrr", "map"]
    data = []
    for prompt, order, _ in RQ4_CONDITIONS:
        condition_rows = [r for r in singles if r.get("prompt") == prompt and r.get("candidate_order") == order]
        data.append([_avg(condition_rows, metric) for metric in metrics])
    data_arr = np.array(data)

    colors = [C1, C1, C2, C2]
    alphas = [1.0, 0.55, 1.0, 0.55]
    hatches = ["", "//", "", "//"]
    x = np.arange(len(labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for i, (vals, color, alpha, hatch, (_, _, label)) in enumerate(zip(data_arr, colors, alphas, hatches, RQ4_CONDITIONS)):
        ax.bar(x + (i - 1.5) * w, vals, w, color=color, alpha=alpha, hatch=hatch,
               edgecolor="white", linewidth=0.4, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="upper left", handlelength=1.0, fontsize=7.0,
              columnspacing=0.8)
    ax.set_title(f"RQ4 - Aggregate ranking metrics over {len(RQ4_FEATURE_ORDER)} features", loc="left", fontweight="bold", pad=4)
    for i, vals in enumerate(data_arr):
        for j, value in enumerate(vals):
            ax.text(x[j] + (i - 1.5) * w, value + 0.012, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=6, rotation=90)
    _save(fig, out_dir / "fig_rq4_aggregate.pdf")


def fig_rq4_single_vs_compound(rows: list[dict[str, str]], out_dir: Path) -> None:
    singles = _single_feature_rows(rows)
    compounds = _compound_rows(rows)
    if not compounds:
        print("  [skip] no compound RQ4 rows available for fig_rq4_single_vs_compound.pdf")
        return

    metrics = ["hit@1", "mrr", "map"]
    metric_labels = ["Hit@1", "MRR", "MAP"]
    cond_labels = [label for _, _, label in RQ4_CONDITIONS]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=False)
    x = np.arange(len(cond_labels))
    w = 0.32
    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        single_vals = []
        compound_vals = []
        for prompt, order, _ in RQ4_CONDITIONS:
            single_vals.append(_avg([r for r in singles if r.get("prompt") == prompt and r.get("candidate_order") == order], metric))
            compound_vals.append(_avg([r for r in compounds if r.get("prompt") == prompt and r.get("candidate_order") == order], metric))
        b1 = ax.bar(x - w / 2, single_vals, w, color=C1, label="Single", edgecolor="white", linewidth=0.4)
        b2 = ax.bar(x + w / 2, compound_vals, w, color=C2, label="Compound", edgecolor="white", linewidth=0.4, alpha=0.80, hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels, fontsize=7)
        ax.set_ylabel(metric_label)
        ax.set_ylim(0, max(max(single_vals), max(compound_vals), 0.15) * 1.30)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(metric_label, fontsize=8, pad=3)
        for rect in list(b1) + list(b2):
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=5.5, rotation=90)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("RQ4 - Single vs. Compound Feature Location", fontsize=8.5, y=1.02)
    fig.tight_layout(pad=0.4)
    path = out_dir / "fig_rq4_single_vs_compound.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def fig_rq5(results_dir: Path, out_dir: Path) -> None:
    rows = _read_csv(results_dir / "summary_rq5.csv")
    if not rows:
        return

    rq3_dacos = [r for r in rows if r.get("source") == "RQ3_baseline" and r.get("dataset") == "DACOS"]
    rq5_dacos = [r for r in rows if r.get("source") == "RQ5_generated" and r.get("dataset") == "DACOS"]
    rq3_best_mlcq = max(
        (_float(r, "f1") for r in rows if r.get("source") == "RQ3_baseline" and r.get("dataset") == "MLCQ"),
        default=0.0,
    )
    rq5_best_mlcq = next(
        (_float(r, "f1") for r in rows if r.get("source") == "RQ5_generated" and r.get("dataset") == "MLCQ"),
        None,
    )

    PROMPT_SHORT = {
        "binary_oracle_baseline": "Baseline",
        "conservative_evidence": "Conservative",
        "definition_driven": "Def.-driven",
        "chain_of_thought_reasoning": "Chain-of-thought",
        "calibrated_certainty": "Calibrated",
        "contrastive_analysis": "Contrastive",
        "threshold_guided": "Threshold-guided [best]",
        "socratic_diagnostic": "Socratic",
    }

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    y_rq3 = np.arange(len(rq3_dacos))
    y_rq5 = np.arange(len(rq3_dacos), len(rq3_dacos) + len(rq5_dacos))
    gap = 0.25
    h = 0.55

    bars3 = ax.barh(
        y_rq3 + gap, [_float(r, "f1") for r in rq3_dacos], h,
        color=C1, edgecolor="white", linewidth=0.5, label="Manual (RQ3)",
    )
    bars5 = ax.barh(
        y_rq5, [_float(r, "f1") for r in rq5_dacos], h,
        color=C2, edgecolor="white", linewidth=0.5, label="AutoResearch (RQ5)",
    )

    labels = [PROMPT_SHORT.get(r["prompt"], r["prompt"]) for r in rq3_dacos] + \
             [PROMPT_SHORT.get(r["prompt"], r["prompt"]) for r in rq5_dacos]
    ticks = list(y_rq3 + gap) + list(y_rq5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.8)

    for bar in list(bars3) + list(bars5):
        w = bar.get_width()
        ax.text(w + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=7.0)

    ax.axvline(x=max(_float(r, "f1") for r in rq3_dacos), color=C1,
               linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("F1 on DACOS (optimisation set)")
    ax.set_xlim(0, 0.70)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color=LIGHT_GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("RQ5 - AutoResearch prompt optimisation (DACOS F1)", loc="left",
                 fontweight="bold", pad=4)

    if rq5_best_mlcq is not None:
        ax.annotate(
            f"Best RQ5 on MLCQ: {rq5_best_mlcq:.3f}  (best RQ3: {rq3_best_mlcq:.3f})",
            xy=(0.98, 0.03), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.5),
        )
    _save(fig, out_dir / "fig_rq5_autoresearch.pdf")


# ── Journal v2 figures (multi-model, baselines, held-out) ────────────────────

SLOT_STYLE = {  # validated categorical slots 1-3 (all-pairs); shapes as secondary encoding
    "AZ1": ("gpt-4o", "#2a78d6", "o"),
    "AZ2": ("gpt-4.1", "#eb6834", "s"),
    "AZ3": ("o3", "#1baf7a", "^"),
    "AZ4": ("qwen2.5-coder 14B", "#4a3aa7", "D"),
}
BASELINE_INK = "#52514e"
ORDER_LABELS = {
    "alphabetical": "Alpha-\nbetical",
    "keyword_sorted": "Keyword-\nsorted",
    "bm25_sorted": "BM25-\nsorted",
    "bm25_masked": "Name-\nmasked",
}
ORDER_BASELINE = {"keyword_sorted": "keyword_sorted_only", "bm25_sorted": "bm25_only",
                  "bm25_masked": "bm25_masked_only"}


def _style_axis(ax) -> None:
    ax.grid(axis="y", color=LIGHT_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_rq4_conditions(stats_dir: Path, out_dir: Path, metric: str = "mrr") -> None:
    panels = [("ArgoUML (8 features)", stats_dir / "rq4_conditions.csv"),
              ("eShopOnContainers (5 features)", stats_dir / "rq4_conditions_eshop.csv")]
    panels = [(t, _read_csv(p)) for t, p in panels]
    panels = [(t, rows) for t, rows in panels if rows]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    orders = list(ORDER_LABELS)
    slots = [s for s in SLOT_STYLE if any(r["slot"] == s for _, rows in panels for r in rows)]
    for ax, (title, rows) in zip(axes, panels):
        for j, slot in enumerate(slots):
            name, color, marker = SLOT_STYLE[slot]
            xs, ys = [], []
            for i, order in enumerate(orders):
                vals = [_float(r, metric) for r in rows
                        if r["method"] == "LLM" and r["slot"] == slot and r["ordering"] == order]
                if vals:  # mean over the two prompts
                    xs.append(i + (j - (len(slots) - 1) / 2) * 0.14)
                    ys.append(sum(vals) / len(vals))
            ax.plot(xs, ys, linestyle="none", marker=marker, markersize=6, color=color,
                    markeredgecolor="#fcfcfb", markeredgewidth=0.8, label=f"LLM: {name}")
        for i, order in enumerate(orders):
            base = next((r for r in rows if r["method"] == ORDER_BASELINE.get(order)), None)
            if base:
                ax.hlines(_float(base, metric), i - 0.32, i + 0.32, colors=BASELINE_INK,
                          linestyles=(0, (2, 1.5)), linewidth=1.2,
                          label="Same ranking without LLM" if i == 1 else None)
        ax.set_xticks(range(len(orders)))
        ax.set_xticklabels([ORDER_LABELS[o] for o in orders], fontsize=7.5)
        ax.set_xlim(-0.5, len(orders) - 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        _style_axis(ax)
    axes[0].set_ylabel("MRR" if metric == "mrr" else metric.upper())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    path = out_dir / f"fig_rq4_conditions_{metric}.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


def fig_rq3_models(stats_dir: Path, results_dir: Path, out_dir: Path, metric: str = "kappa") -> None:
    rows = [r for r in _read_csv(stats_dir / "rq3_by_slot.csv") if r["prompt"] == "binary_oracle_baseline"]
    base = _read_csv(results_dir / "summary_rq3_baselines.csv")
    if not rows:
        return
    datasets = ["DACOS", "MLCQ"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    slots = [s for s in SLOT_STYLE if any(r["slot"] == s for r in rows)]
    for j, slot in enumerate(slots):
        name, color, marker = SLOT_STYLE[slot]
        xs, ys = [], []
        for i, ds in enumerate(datasets):
            r = next((r for r in rows if r["slot"] == slot and r["dataset"] == ds), None)
            if r:
                xs.append(i + (j - (len(slots) - 1) / 2) * 0.12)
                ys.append(_float(r, metric))
        ax.plot(xs, ys, linestyle="none", marker=marker, markersize=6, color=color,
                markeredgecolor="#fcfcfb", markeredgewidth=0.8, label=name)
    for i, ds in enumerate(datasets):
        h = next((r for r in base if r["dataset"] == ds and r["method"] == "metric_heuristic"), None)
        if h:
            ax.hlines(_float(h, metric), i - 0.3, i + 0.3, colors=BASELINE_INK,
                      linestyles=(0, (2, 1.5)), linewidth=1.2,
                      label="Metric heuristic (no LLM)" if i == 0 else None)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Cohen's $\\kappa$" if metric == "kappa" else metric)
    _style_axis(ax)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_ylim(0, 0.65)
    _save(fig, out_dir / f"fig_rq3_models_{metric}.pdf")


def fig_rq5_heldout(results_dir: Path, out_dir: Path) -> None:
    sel = {r["prompt"]: _float(r, "f1") for r in _read_csv(results_dir / "summary_rq5.csv")
           if r["dataset"] == "DACOS"}
    test = {r["prompt"]: _float(r, "f1") for r in _read_csv(results_dir / "summary_rq5_heldout.csv")}
    mlcq = {r["prompt"]: _float(r, "f1") for r in _read_csv(results_dir / "summary_rq5.csv")
            if r["dataset"] == "MLCQ" and r["source"] in ("RQ3_baseline", "RQ5_generated")}
    prompts = [p for p in ("binary_oracle_baseline", "conservative_evidence", "definition_driven",
                           "threshold_guided") if p in sel and p in test]
    if not prompts:
        return
    labels = {"binary_oracle_baseline": "Baseline", "conservative_evidence": "Conservative",
              "definition_driven": "Def.-driven", "threshold_guided": "Threshold-guided\n(generated)"}
    stages = ["DACOS\nselection", "DACOS\nheld-out test", "MLCQ"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    # spread end labels so close MLCQ values stay legible
    ends = sorted(prompts, key=lambda p: mlcq.get(p, 0))
    label_y, last = {}, -1.0
    for p in ends:
        y = max(mlcq.get(p, 0), last + 0.014)
        label_y[p], last = y, y
    for p in prompts:
        ys = [sel[p], test[p], mlcq.get(p, np.nan)]
        gen = p == "threshold_guided"
        color = "#eb6834" if gen else "#a3a29b"
        ax.plot(range(3), ys, color=color, linewidth=2 if gen else 1.2,
                marker="o", markersize=5, markeredgecolor="#fcfcfb", zorder=3 if gen else 2)
        ax.annotate(labels[p], (2, ys[2]), xytext=(2.08, label_y[p]), textcoords="data",
                    va="center", fontsize=6.5, color="#0b0b0b" if gen else BASELINE_INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(stages, fontsize=7.5)
    ax.set_xlim(-0.2, 2.95)
    ax.set_ylabel("F$_1$")
    _style_axis(ax)
    _save(fig, out_dir / "fig_rq5_heldout.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate StaLLM paper figures from experiment summaries.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--rq4-summary",
        type=Path,
        default=None,
        help="Optional RQ4 summary CSV. Defaults to summary_rq4_all.csv when present, else summary_rq4.csv.",
    )
    args = parser.parse_args()

    _configure_style()
    print("Generating figures from summaries...")
    stats_dir = args.results_dir / "stats"
    fig_rq3_models(stats_dir, args.results_dir, args.out_dir)
    for metric in ("mrr", "map"):
        fig_rq4_conditions(stats_dir, args.out_dir, metric)
    fig_rq5_heldout(args.results_dir, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
