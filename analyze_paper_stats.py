"""
StaLLM — Statistical analysis of the paper experiments
=======================================================
Reads the per-sample / per-feature JSON results written by
run_paper_experiments.py and produces the significance tests and stability
measures reported in the journal paper.

Usage:
    python analyze_paper_stats.py

Outputs (in output/paper_results/stats/):
    rq3_by_slot.csv          F1 / kappa per model slot, prompt and dataset (run 1)
    rq3_mcnemar.csv          paired exact McNemar tests (prompts within a slot,
                             slots within a prompt) on per-sample correctness
    rq3_run_stability.csv    run-to-run prediction agreement and F1 spread
    rq4_conditions.csv       mean ranking metrics per slot x prompt x ordering,
                             plus the non-LLM keyword / BM25 baselines
    rq4_wilcoxon.csv         paired Wilcoxon tests across features
    rq5_heldout.csv          held-out DACOS results + McNemar vs. manual prompts
    report.md                human-readable digest of all of the above
"""

from __future__ import annotations

import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import binomtest, wilcoxon

OUT = Path("output/paper_results")
STATS = OUT / "stats"
RQ4_DIRS = {"alphabetical": OUT / "rq4", "keyword_sorted": OUT / "rq4_sorted",
            "bm25_sorted": OUT / "rq4_bm25", "bm25_masked": OUT / "rq4_masked"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _correct(rows: list[dict[str, Any]]) -> dict[str, bool]:
    return {r["sample_id"]: bool(r["predicted"]) == bool(r["gold"]) for r in rows}


def mcnemar_exact(a: dict[str, bool], b: dict[str, bool]) -> tuple[int, int, float]:
    """Exact McNemar test on paired per-sample correctness. Returns (b_only, a_only, p)."""
    ids = a.keys() & b.keys()
    a_only = sum(1 for i in ids if a[i] and not b[i])
    b_only = sum(1 for i in ids if b[i] and not a[i])
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return a_only, b_only, p


def holm(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as input)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def _r(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


# ── RQ3 ──────────────────────────────────────────────────────────────────────

def load_rq3() -> list[dict[str, Any]]:
    res = []
    for f in sorted((OUT / "rq3").glob("rq3_*.json")):
        d = _load(f)
        d.setdefault("run", 1)
        res.append(d)
    return res


def analyze_rq3(results: list[dict[str, Any]], md: list[str]) -> None:
    run1 = [d for d in results if d["run"] == 1]
    by_slot = [{
        "slot": d["slot"], "prompt": d["prompt"], "dataset": d["dataset"],
        "precision": _r(d["metrics"]["precision"]), "recall": _r(d["metrics"]["recall"]),
        "f1": _r(d["metrics"]["f1"]), "f1_ci": f"[{_r(d['metrics']['f1_ci_lo'])}, {_r(d['metrics']['f1_ci_hi'])}]",
        "kappa": _r(d["metrics"]["kappa"]), "accuracy": _r(d["metrics"]["accuracy"]),
        "tokens": d.get("usage", {}).get("total_tokens"),
    } for d in sorted(run1, key=lambda d: (d["dataset"], d["slot"], d["prompt"]))]
    _write_csv(STATS / "rq3_by_slot.csv", by_slot)

    md.append("## RQ3 — human-oracle smell classification (run 1)\n")
    md.append("| Dataset | Slot | Prompt | P | R | F1 [95% CI] | κ |\n|---|---|---|---|---|---|---|")
    for r in by_slot:
        md.append(f"| {r['dataset']} | {r['slot']} | {r['prompt']} | {r['precision']} | {r['recall']} "
                  f"| {r['f1']} {r['f1_ci']} | {r['kappa']} |")
    md.append("")

    # McNemar: prompts within slot/dataset, slots within prompt/dataset
    idx = {(d["slot"], d["prompt"], d["dataset"]): _correct(d["rows"]) for d in run1}
    tests = []
    groups: dict[tuple, list] = defaultdict(list)
    for (slot, prompt, ds) in idx:
        groups[("prompt", slot, ds)].append(prompt)
        groups[("slot", prompt, ds)].append(slot)
    for (kind, fixed, ds), members in sorted(groups.items()):
        for a, b in itertools.combinations(sorted(members), 2):
            ka = (fixed, a, ds) if kind == "prompt" else (a, fixed, ds)
            kb = (fixed, b, ds) if kind == "prompt" else (b, fixed, ds)
            a_only, b_only, p = mcnemar_exact(idx[ka], idx[kb])
            tests.append({"factor": kind, "fixed": fixed, "dataset": ds, "a": a, "b": b,
                          "a_only_correct": a_only, "b_only_correct": b_only, "p": p})
    for fam in ("prompt", "slot"):
        fam_tests = [t for t in tests if t["factor"] == fam]
        for t, adj in zip(fam_tests, holm([t["p"] for t in fam_tests])):
            t["p_holm"] = adj
    for t in tests:
        t["p"], t["p_holm"] = _r(t["p"], 4), _r(t["p_holm"], 4)
    _write_csv(STATS / "rq3_mcnemar.csv", tests)

    sig = [t for t in tests if t["p_holm"] < 0.05]
    md.append(f"**McNemar (Holm-adjusted within factor):** {len(sig)}/{len(tests)} pairwise "
              "comparisons significant at α=0.05.\n")
    for t in sig:
        md.append(f"- {t['factor']} | {t['fixed']} | {t['dataset']}: {t['a']} vs {t['b']} "
                  f"({t['a_only_correct']} vs {t['b_only_correct']} discordant, p_holm={t['p_holm']})")
    md.append("")

    # run-to-run stability
    by_cfg: dict[tuple, list] = defaultdict(list)
    for d in results:
        by_cfg[(d["slot"], d["prompt"], d["dataset"])].append(d)
    stab = []
    for cfg, runs in sorted(by_cfg.items()):
        if len(runs) < 2:
            continue
        preds = [{r["sample_id"]: r["predicted"] for r in d["rows"]} for d in runs]
        ids = set.intersection(*(set(p) for p in preds))
        agree = sum(1 for i in ids if len({p[i] for p in preds}) == 1) / len(ids)
        f1s = [d["metrics"]["f1"] for d in runs]
        stab.append({"slot": cfg[0], "prompt": cfg[1], "dataset": cfg[2], "runs": len(runs),
                     "all_runs_agree": _r(agree), "f1_mean": _r(statistics.mean(f1s)),
                     "f1_sd": _r(statistics.stdev(f1s)), "f1_min": _r(min(f1s)), "f1_max": _r(max(f1s))})
    _write_csv(STATS / "rq3_run_stability.csv", stab)
    if stab:
        md.append("**Run-to-run stability (temperature 0):**\n")
        md.append("| Slot | Prompt | Dataset | runs | identical predictions | F1 mean ± sd |\n|---|---|---|---|---|---|")
        for s in stab:
            md.append(f"| {s['slot']} | {s['prompt']} | {s['dataset']} | {s['runs']} | "
                      f"{s['all_runs_agree']:.1%} | {s['f1_mean']} ± {s['f1_sd']} |")
        md.append("")


# ── RQ4 ──────────────────────────────────────────────────────────────────────

METRICS4 = ("hit@1", "hit@5", "recall@10", "mrr", "map")


def analyze_rq4(md: list[str]) -> None:
    per_feature: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for label, folder in RQ4_DIRS.items():
        for f in sorted(folder.glob("rq4_*.json")):
            d = _load(f)
            if "_and_" in d.get("feature", ""):
                continue
            key = (d["slot"], d["prompt_style"], label)
            for m in METRICS4:
                per_feature[key][(d["feature"], m)].append(d["metrics"][m])

    rows, feat_means = [], {}
    for key, vals in sorted(per_feature.items()):
        features = sorted({f for f, _ in vals})
        fm = {f: {m: statistics.mean(vals[(f, m)]) for m in METRICS4} for f in features}
        feat_means[key] = fm
        runs = max(len(v) for v in vals.values())
        rows.append({"method": "LLM", "slot": key[0], "prompt": key[1], "ordering": key[2],
                     "features": len(features), "runs": runs,
                     **{m: _r(statistics.mean(fm[f][m] for f in features)) for m in METRICS4}})

    base_path = OUT / "summary_rq4_baselines.csv"
    baselines: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    if base_path.exists():
        with open(base_path, newline="") as f:
            for r in csv.DictReader(f):
                baselines[r["method"]][r["feature"]] = {m: float(r[m]) for m in METRICS4}
        for method, fm in baselines.items():
            rows.append({"method": method, "slot": "-", "prompt": "-", "ordering": "-",
                         "features": len(fm), "runs": 1,
                         **{m: _r(statistics.mean(v[m] for v in fm.values())) for m in METRICS4}})
    _write_csv(STATS / "rq4_conditions.csv", rows)

    md.append("## RQ4 — feature location (8 single features; mean over features, runs averaged)\n")
    md.append("| Method | Slot | Prompt | Ordering | runs | Hit@1 | Hit@5 | R@10 | MRR | MAP |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['method']} | {r['slot']} | {r['prompt']} | {r['ordering']} | {r['runs']} | "
                  + " | ".join(str(r[m]) for m in METRICS4) + " |")
    md.append("")

    # Wilcoxon: each LLM condition vs. keyword-only baseline, and orderings against each other
    tests = []
    kw = baselines.get("keyword_sorted_only", {})
    for key, fm in feat_means.items():
        for m in ("mrr", "map", "recall@10"):
            if kw:
                feats = sorted(fm.keys() & kw.keys())
                a = [fm[f][m] for f in feats]
                b = [kw[f][m] for f in feats]
                tests.append(_wilcoxon_row(f"{key[0]}/{key[1]}/{key[2]}", "keyword_sorted_only", m, a, b))
    for slot, prompt in {(k[0], k[1]) for k in feat_means}:
        for o1, o2 in itertools.combinations(RQ4_DIRS, 2):
            k1, k2 = (slot, prompt, o1), (slot, prompt, o2)
            if k1 in feat_means and k2 in feat_means:
                feats = sorted(feat_means[k1].keys() & feat_means[k2].keys())
                for m in ("mrr", "map"):
                    tests.append(_wilcoxon_row(f"{slot}/{prompt}/{o1}", f"{slot}/{prompt}/{o2}", m,
                                               [feat_means[k1][f][m] for f in feats],
                                               [feat_means[k2][f][m] for f in feats]))
    _write_csv(STATS / "rq4_wilcoxon.csv", tests)
    md.append("*Paired Wilcoxon tests across features are in `rq4_wilcoxon.csv`; with n=8 "
              "features the minimum attainable two-sided p is 0.0078.*\n")


def _wilcoxon_row(a_name: str, b_name: str, metric: str, a: list[float], b: list[float]) -> dict:
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs or all(d == 0 for d in diffs):
        p = 1.0
    else:
        p = wilcoxon(a, b, zero_method="zsplit").pvalue
    return {"a": a_name, "b": b_name, "metric": metric, "n": len(diffs),
            "mean_a": _r(statistics.mean(a)) if a else None,
            "mean_b": _r(statistics.mean(b)) if b else None,
            "wins_a": sum(d > 0 for d in diffs), "wins_b": sum(d < 0 for d in diffs), "p": _r(p, 4)}


# ── RQ5 held-out ─────────────────────────────────────────────────────────────

def analyze_rq5(md: list[str]) -> None:
    files = sorted((OUT / "rq5" / "heldout").glob("rq5ho_*.json"))
    if not files:
        return
    res = [_load(f) for f in files]
    rows = []
    gen = [d for d in res if d.get("source") == "RQ5_generated"]
    for d in res:
        m = d["metrics"]
        row = {"slot": d["slot"], "source": d["source"], "prompt": d["prompt"],
               "precision": _r(m["precision"]), "recall": _r(m["recall"]), "f1": _r(m["f1"]),
               "f1_ci": f"[{_r(m['f1_ci_lo'])}, {_r(m['f1_ci_hi'])}]", "kappa": _r(m["kappa"])}
        if gen and d is not gen[0]:
            a_only, b_only, p = mcnemar_exact(_correct(gen[0]["rows"]), _correct(d["rows"]))
            row.update({"generated_only_correct": a_only, "manual_only_correct": b_only, "mcnemar_p": _r(p, 4)})
        rows.append(row)
    _write_csv(STATS / "rq5_heldout.csv", rows)
    md.append("## RQ5 — held-out DACOS test split (disjoint from the selection split)\n")
    md.append("| Source | Prompt | P | R | F1 [95% CI] | κ | McNemar p vs generated |\n|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['source']} | {r['prompt']} | {r['precision']} | {r['recall']} | {r['f1']} {r['f1_ci']} "
                  f"| {r['kappa']} | {r.get('mcnemar_p', '-')} |")
    md.append("")


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    md = ["# StaLLM paper — statistical analysis\n"]
    analyze_rq3(load_rq3(), md)
    analyze_rq4(md)
    analyze_rq5(md)
    (STATS / "report.md").write_text("\n".join(md))
    print(f"Wrote {STATS / 'report.md'}")


if __name__ == "__main__":
    main()
