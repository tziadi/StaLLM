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
    rq3_oracle_swap.csv      each LLM configuration scored against the human
                             labels and against the metric heuristic (same samples)
    rq3_heuristic_by_smell.csv  metric-heuristic confusion counts per smell
    rq12_rerun.csv           RQ1/RQ2 span-level runs (mean over runs) with a
                             random-placement baseline: the same number and
                             lengths of predicted spans placed uniformly at
                             random in the same files, scored with the same matcher
    rq5_heldout.csv          held-out DACOS results + McNemar vs. manual prompts
    report.md                human-readable digest of all of the above
"""

from __future__ import annotations

import csv
import itertools
import random
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import binomtest, spearmanr, wilcoxon

OUT = Path("output/paper_results")
STATS = OUT / "stats"
RQ4_DIRS = {"alphabetical": "rq4", "keyword_sorted": "rq4_sorted",
            "bm25_sorted": "rq4_bm25", "bm25_masked": "rq4_masked"}
RQ4_SYSTEMS = {"ArgoUML (8 single features)": OUT, "eShopOnContainers (5 features)": OUT / "eshop"}


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


def analyze_rq4(md: list[str], root: Path = OUT, title: str = "ArgoUML", tag: str = "") -> None:
    per_feature: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for label, folder in RQ4_DIRS.items():
        for f in sorted((root / folder).glob("rq4_*.json")):
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

    base_path = root / "summary_rq4_baselines.csv"
    baselines: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    if base_path.exists():
        with open(base_path, newline="") as f:
            for r in csv.DictReader(f):
                baselines[r["method"]][r["feature"]] = {m: float(r[m]) for m in METRICS4}
        for method, fm in baselines.items():
            rows.append({"method": method, "slot": "-", "prompt": "-", "ordering": "-",
                         "features": len(fm), "runs": 1,
                         **{m: _r(statistics.mean(v[m] for v in fm.values())) for m in METRICS4}})
    _write_csv(STATS / f"rq4_conditions{tag}.csv", rows)

    md.append(f"## RQ4 — feature location, {title} (mean over features, runs averaged)\n")
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
    _write_csv(STATS / f"rq4_wilcoxon{tag}.csv", tests)
    md.append(f"*Paired Wilcoxon tests across features are in `rq4_wilcoxon{tag}.csv` (with n=8 "
              "features the minimum two-sided p is 0.0078; with n=5 it is 0.0625, so eShop "
              "comparisons are descriptive only).*\n")


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


def _binary_scores(gold: list[bool], pred: list[bool]) -> tuple[float, float]:
    """F1 and Cohen's kappa of `pred` against `gold`."""
    n = len(gold)
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum(p and not g for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    po = sum(g == p for g, p in zip(gold, pred)) / n
    pg, pp = sum(gold) / n, sum(pred) / n
    pe = pg * pp + (1 - pg) * (1 - pp)
    return f1, (po - pe) / (1 - pe) if pe < 1 else 0.0


def analyze_rq3_oracle_swap(results: list[dict[str, Any]], md: list[str]) -> None:
    """Score every RQ3 configuration against two oracles on the same samples:
    the human labels and the Designite-inspired metric heuristic."""
    base = OUT / "rq3" / "baselines"
    swap, by_smell = [], []
    md.append("## RQ3 — oracle substitution (human labels vs. metric heuristic, same samples)\n")
    for split in ("DACOS", "DACOS_heldout", "MLCQ"):
        hfile = base / f"rq3b_metric_heuristic_{split}.json"
        if not hfile.exists():
            continue
        hrows = _load(hfile)["rows"]
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for r in hrows:
            g, p = r["gold"], r["predicted"]
            counts[r["smell"]][0 if g and p else 1 if p else 2 if g else 3] += 1
        for smell, (tp, fp, fn, tn) in sorted(counts.items()):
            by_smell.append({"dataset": split, "smell": smell, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
        heur = {r["sample_id"]: bool(r["predicted"]) for r in hrows}
        configs = [d for d in results if d["run"] == 1 and d["dataset"] == split
                   and {r["sample_id"] for r in d["rows"]} == set(heur)]
        if len(configs) < 3:
            continue
        rows = []
        for d in sorted(configs, key=lambda d: (d["slot"], d["prompt"])):
            gold = [bool(r["gold"]) for r in d["rows"]]
            pred = [bool(r["predicted"]) for r in d["rows"]]
            hpred = [heur[r["sample_id"]] for r in d["rows"]]
            f1_h, k_h = _binary_scores(gold, pred)
            f1_m, k_m = _binary_scores(hpred, pred)
            rows.append({"dataset": split, "slot": d["slot"], "prompt": d["prompt"],
                         "f1_human": _r(f1_h), "kappa_human": _r(k_h),
                         "f1_heuristic": _r(f1_m), "kappa_heuristic": _r(k_m)})
        rho, p = spearmanr([r["kappa_human"] for r in rows], [r["kappa_heuristic"] for r in rows])
        best_h = max(rows, key=lambda r: r["kappa_human"])
        best_m = max(rows, key=lambda r: r["kappa_heuristic"])
        swap.extend(rows)
        md.append(f"**{split}** ({len(rows)} configurations): Spearman ρ(κ human, κ heuristic) = "
                  f"{_r(rho)} (p={_r(p, 3)}); best vs human: {best_h['slot']}/{best_h['prompt']}, "
                  f"best vs heuristic: {best_m['slot']}/{best_m['prompt']}\n")
        md.append("| Slot | Prompt | F1 human | κ human | F1 heuristic | κ heuristic |\n|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['slot']} | {r['prompt']} | {r['f1_human']} | {r['kappa_human']} "
                      f"| {r['f1_heuristic']} | {r['kappa_heuristic']} |")
        md.append("")
    _write_csv(STATS / "rq3_oracle_swap.csv", swap)
    _write_csv(STATS / "rq3_heuristic_by_smell.csv", by_smell)


# ── RQ3 baselines and RQ1/RQ2 reruns ─────────────────────────────────────────

def analyze_rq3_baselines(md: list[str]) -> None:
    path = OUT / "summary_rq3_baselines.csv"
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    md.append("## RQ3 — non-LLM baselines\n")
    md.append("| Dataset | Method | P | R | F1 | κ |\n|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['dataset']} | {r['method']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['kappa']} |")
    md.append("")


RANDOM_DRAWS = 1000


def _span_f1(spans_by_file: dict[str, Any], preds_by_file: dict[str, list], match) -> float:
    tp = fp = fn = 0
    for b, f in spans_by_file.items():
        a, p_, n = match(preds_by_file.get(b, []), [{"startLine": s, "endLine": e} for s, e in f["gt"]])
        tp, fp, fn = tp + a, fp + p_, fn + n
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def random_placement_f1(spans_by_file: dict[str, Any], rng: random.Random, match,
                        shuffle_files: bool = False) -> float:
    """F1 when every predicted span keeps its length but is moved to a uniformly
    random position, in the same file or, with shuffle_files, in a file drawn
    uniformly from the evaluation universe."""
    names = sorted(spans_by_file)
    preds: dict[str, list] = defaultdict(list)
    for b, f in spans_by_file.items():
        for s, e in f["pred"]:
            target = rng.choice(names) if shuffle_files else b
            n_lines = spans_by_file[target]["lines"]
            length = min(max(0, e - s), n_lines - 1)
            start = rng.randint(1, max(1, n_lines - length))
            preds[target].append({"startLine": start, "endLine": start + length})
    return _span_f1(spans_by_file, preds, match)


def analyze_rq12(md: list[str]) -> None:
    from StaLLM_core import _match_file

    def match(p, g):
        return _match_file([dict(x, startColumn=None, endColumn=None, type=None) for x in p],
                           [dict(x, startColumn=None, endColumn=None, type=None) for x in g],
                           require_type=False, use_cols_single=False, preset="Lenient")

    files = sorted((OUT / "rq2").glob("rq2_*.json"))
    if not files:
        return
    groups: dict[tuple, list] = defaultdict(list)
    for f in files:
        d = _load(f)
        groups[(d["slot"], d["prompt"])].append(d)
    rows = []
    for (slot, prompt), runs in sorted(groups.items()):
        f1 = [d["metrics"]["f1"] for d in runs]
        row = {"slot": slot, "prompt": prompt, "runs": len(runs),
               "model": runs[0].get("model"),
               "precision": _r(statistics.mean(d["metrics"]["precision"] for d in runs)),
               "recall": _r(statistics.mean(d["metrics"]["recall"] for d in runs)),
               "f1_mean": _r(statistics.mean(f1)),
               "f1_sd": _r(statistics.stdev(f1)) if len(f1) > 1 else 0.0,
               "llm_errors": sum(d.get("llm_errors", 0) for d in runs),
               "time_s": _r(statistics.median(d["elapsed_s"] for d in runs), 1)}
        if all("spans_by_file" in d["metrics"] for d in runs):
            for tag, shuffle in (("random", False), ("random_files", True)):
                rng = random.Random(42)
                draws = sorted(
                    statistics.mean(random_placement_f1(d["metrics"]["spans_by_file"], rng, match, shuffle)
                                    for d in runs)
                    for _ in range(RANDOM_DRAWS))
                row.update({f"f1_{tag}_mean": _r(statistics.mean(draws)),
                            f"f1_{tag}_ci": f"[{_r(draws[int(0.025 * RANDOM_DRAWS)])}, "
                                            f"{_r(draws[int(0.975 * RANDOM_DRAWS) - 1])}]",
                            f"p_{tag}": _r((1 + sum(x >= statistics.mean(f1) for x in draws)) / (1 + RANDOM_DRAWS), 4)})
        rows.append(row)
    _write_csv(STATS / "rq12_rerun.csv", rows)
    md.append("## RQ1/RQ2 — span-level reruns (eShop Catalog, path-matched oracle, Top-k=20, Lenient)\n")
    md.append("*Random placement: same number and lengths of spans, uniformly random lines in the same file; "
              "random files: also a uniformly random file of the universe. p = share of draws reaching the LLM F1.*\n")
    md.append("| Slot | Model | Prompt | runs | P | R | F1 mean ± sd | random placement [95%] | p "
              "| random files [95%] | p | LLM errors | median time (s) |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['slot']} | {r['model']} | {r['prompt']} | {r['runs']} | {r['precision']} | {r['recall']} "
                  f"| {r['f1_mean']} ± {r['f1_sd']} | {r.get('f1_random_mean', '-')} {r.get('f1_random_ci', '')} "
                  f"| {r.get('p_random', '-')} | {r.get('f1_random_files_mean', '-')} {r.get('f1_random_files_ci', '')} "
                  f"| {r.get('p_random_files', '-')} | {r['llm_errors']} | {r['time_s']} |")
    md.append("")


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

    mlcq_gen = OUT / "rq5" / "rq5_MLCQ_threshold_guided.json"
    if mlcq_gen.exists():
        g = _load(mlcq_gen)
        md.append(f"**MLCQ validation** — generated threshold_guided F1={_r(g['metrics']['f1'])}:\n")
        for f in sorted((OUT / "rq3").glob("rq3_AZ1_*_MLCQ.json")):
            d = _load(f)
            a_only, b_only, p = mcnemar_exact(_correct(g["rows"]), _correct(d["rows"]))
            md.append(f"- vs {d['prompt']} (F1={_r(d['metrics']['f1'])}): {a_only} vs {b_only} "
                      f"discordant, McNemar p={_r(p, 4)}")
        md.append("")


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    md = ["# StaLLM paper — statistical analysis\n"]
    rq3 = load_rq3()
    analyze_rq3(rq3, md)
    analyze_rq3_oracle_swap(rq3, md)
    for i, (title, root) in enumerate(RQ4_SYSTEMS.items()):
        analyze_rq4(md, root=root, title=title, tag="" if i == 0 else "_eshop")
    analyze_rq3_baselines(md)
    analyze_rq12(md)
    analyze_rq5(md)
    (STATS / "report.md").write_text("\n".join(md))
    print(f"Wrote {STATS / 'report.md'}")


if __name__ == "__main__":
    main()
