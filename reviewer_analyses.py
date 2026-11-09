"""Additional analyses requested in review, computed from the stored outputs only
(no model call).

    python reviewer_analyses.py                                     # legacy DACOS labels, all analyses
    python reviewer_analyses.py --root output/paper_results_annot   # RQ3 on annotated DACOS (paper)

Writes CSV files and a markdown summary (reviewer_analyses.md) to <root>/stats/.
With the default root, the DACOS part of analysis 1 and the RQ5 check use the
legacy DACOS labels of the preliminary version (sample table); the paper
reports RQ3/RQ5 from output/paper_results_annot/. Analyses:

1. RQ3 ranking stability with the oracle held fixed (reruns and paired bootstrap),
   compared with the oracle-substitution correlation.
2. RQ1/RQ2 without the empty-answer retry, for every model, and LLM errors.
3. RQ4 expected Precision@10 and MRR of a random ordering of each candidate list.
4. MLCQ: fragments, reviewers and inter-reviewer agreement of the 200 rows used.
5. Checks: time statistic of Table 2, run counts of the sensitivity analysis,
   direction of the McNemar tests reported in the paper.
"""
from __future__ import annotations

import csv
import itertools
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

import analyze_paper_stats as aps

OUT, STATS = aps.OUT, aps.STATS
BOOT = 1000
DATASETS = ("DACOS", "MLCQ")


def _q(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(q * len(xs))))]


def _rho(a: list[float], b: list[float]) -> float:
    r = spearmanr(a, b)[0]
    return 0.0 if math.isnan(r) else float(r)


# ── 1. RQ3 ranking stability ─────────────────────────────────────────────────

def rq3_stability(md: list[str]) -> None:
    results = aps.load_rq3()
    base = OUT / "rq3" / "baselines"
    rerun_rows, boot_rows = [], []
    md.append("## 1. RQ3 — ranking stability with the oracle held fixed\n")
    for ds in DATASETS:
        runs = [d for d in results if d["dataset"] == ds]
        run1 = sorted([d for d in runs if d["run"] == 1], key=lambda d: (d["slot"], d["prompt"]))
        names = [f"{d['slot']}/{d['prompt']}" for d in run1]

        # (a) Reruns: rank the configurations with run r of every model that has
        # three runs (gpt-4o, DeepSeek-V3.2, Mistral-Large-3), single-run models fixed.
        def scores(r: int, metric: str) -> list[float]:
            out = []
            for d in run1:
                d = next((x for x in runs if x["slot"] == d["slot"] and x["prompt"] == d["prompt"]
                          and x["run"] == r), d)
                out.append(d["metrics"][metric])
            return out
        for metric in ("f1", "kappa"):
            for r1, r2 in ((1, 2), (1, 3), (2, 3)):
                a, b = scores(r1, metric), scores(r2, metric)
                rerun_rows.append({"dataset": ds, "metric": metric, "runs": f"{r1} vs {r2}",
                                   "rho_all_configs": aps._r(_rho(a, b)),
                                   "best_same": names[a.index(max(a))] == names[b.index(max(b))],
                                   "rho_3_gpt4o_prompts": aps._r(_rho(a[:3], b[:3]))})

        # (b) Paired bootstrap over samples.
        heur = {r["sample_id"]: bool(r["predicted"])
                for r in aps._load(base / f"rq3b_metric_heuristic_{ds}.json")["rows"]}
        # Majority vote over the runs of each configuration, as in Table 3.
        voted = sorted(aps.rq3_voted(runs), key=lambda d: (d["slot"], d["prompt"]))
        assert [f"{d['slot']}/{d['prompt']}" for d in voted] == names
        ids = [r["sample_id"] for r in voted[0]["rows"]]
        gold = {r["sample_id"]: bool(r["gold"]) for r in voted[0]["rows"]}
        preds = [{r["sample_id"]: bool(r["predicted"]) for r in d["rows"]} for d in voted]

        def score_all(sample: list[str], oracle: dict[str, bool]) -> tuple[list[float], list[float]]:
            g = [oracle[i] for i in sample]
            f1s, ks = [], []
            for p in preds:
                f1, k = aps._binary_scores(g, [p[i] for i in sample])
                f1s.append(f1)
                ks.append(k)
            return f1s, ks

        f1_h0, k_h0 = score_all(ids, gold)
        f1_m0, k_m0 = score_all(ids, heur)
        rng = random.Random(42)
        stats_: dict[str, list] = defaultdict(list)
        for _ in range(BOOT):
            s = [rng.choice(ids) for _ in ids]
            s2 = [rng.choice(ids) for _ in ids]
            f1_h, k_h = score_all(s, gold)
            f1_h2, k_h2 = score_all(s2, gold)
            f1_m, k_m = score_all(s, heur)
            stats_["rho_k_human_vs_orig"].append(_rho(k_h, k_h0))
            stats_["rho_f1_human_vs_orig"].append(_rho(f1_h, f1_h0))
            stats_["rho_k_human_two_resamples"].append(_rho(k_h, k_h2))
            stats_["rho_k_metric_vs_orig"].append(_rho(k_m, k_m0))
            stats_["rho_k_human_vs_metric"].append(_rho(k_h, k_m))
            stats_["best_k_human_same"].append(k_h.index(max(k_h)) == k_h0.index(max(k_h0)))
            stats_["best_f1_human_same"].append(f1_h.index(max(f1_h)) == f1_h0.index(max(f1_h0)))
            stats_["best_k_metric_same"].append(k_m.index(max(k_m)) == k_m0.index(max(k_m0)))
            stats_["best_k_human_ne_metric"].append(k_h.index(max(k_h)) != k_m.index(max(k_m)))
        row = {"dataset": ds, "configs": len(run1),
               "rho_k_human_vs_metric_observed": aps._r(_rho(k_h0, k_m0)),
               "best_k_human": names[k_h0.index(max(k_h0))], "best_k_metric": names[k_m0.index(max(k_m0))]}
        for key, xs in stats_.items():
            if isinstance(xs[0], bool):
                row[f"share_{key}"] = aps._r(sum(xs) / len(xs))
            else:
                row[f"{key}_median"] = aps._r(statistics.median(xs))
                row[f"{key}_ci"] = f"[{aps._r(_q(xs, 0.025))}, {aps._r(_q(xs, 0.975))}]"
        boot_rows.append(row)
    aps._write_csv(STATS / "reviewer_rq3_rerun_rankings.csv", rerun_rows)
    aps._write_csv(STATS / "reviewer_rq3_bootstrap.csv", boot_rows)

    md.append("**(a) Reruns.** The configurations ranked with run r of the models that have three runs "
              "(gpt-4o, DeepSeek-V3.2, Mistral-Large-3; gpt-4.1 and o3 have one run). "
              "ρ over all configurations, and over the 3 gpt-4o prompts alone.\n")
    md.append("| Dataset | Metric | Runs | ρ (all configs) | same best | ρ (3 gpt-4o prompts) |\n|---|---|---|---|---|---|")
    for r in rerun_rows:
        md.append(f"| {r['dataset']} | {r['metric']} | {r['runs']} | {r['rho_all_configs']} | {r['best_same']} "
                  f"| {r['rho_3_gpt4o_prompts']} |")
    md.append("")
    md.append(f"**(b) Paired bootstrap** ({BOOT} resamples of the samples, run 1 of each configuration, seed 42). "
              "Median ρ and 95% interval.\n")
    md.append("| Dataset | ρ(κ human, κ metric) observed | ρ κ human: resample vs original | ρ κ human: two resamples "
              "| ρ F1 human: resample vs original | ρ κ metric: resample vs original | ρ(κ human, κ metric) in resamples "
              "| same human-best (κ) | same human-best (F1) | same metric-best (κ) | human-best ≠ metric-best |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    for r in boot_rows:
        md.append(f"| {r['dataset']} | {r['rho_k_human_vs_metric_observed']} "
                  f"| {r['rho_k_human_vs_orig_median']} {r['rho_k_human_vs_orig_ci']} "
                  f"| {r['rho_k_human_two_resamples_median']} {r['rho_k_human_two_resamples_ci']} "
                  f"| {r['rho_f1_human_vs_orig_median']} {r['rho_f1_human_vs_orig_ci']} "
                  f"| {r['rho_k_metric_vs_orig_median']} {r['rho_k_metric_vs_orig_ci']} "
                  f"| {r['rho_k_human_vs_metric_median']} {r['rho_k_human_vs_metric_ci']} "
                  f"| {r['share_best_k_human_same']:.0%} | {r['share_best_f1_human_same']:.0%} "
                  f"| {r['share_best_k_metric_same']:.0%} | {r['share_best_k_human_ne_metric']:.0%} |")
    md.append("")
    for r in boot_rows:
        md.append(f"- {r['dataset']}: best by human κ = {r['best_k_human']}, best by metric κ = {r['best_k_metric']}.")
    md.append("")


# ── 2. RQ1/RQ2 without the retry ─────────────────────────────────────────────

def _matcher():
    from StaLLM_core import _match_file

    def match(p, g):
        return _match_file([dict(x, startColumn=None, endColumn=None, type=None) for x in p],
                           [dict(x, startColumn=None, endColumn=None, type=None) for x in g],
                           require_type=False, use_cols_single=False, preset="Lenient")
    return match


def _file_counts(f: dict[str, Any], match) -> tuple[int, int, int]:
    return match([{"startLine": s, "endLine": e} for s, e in f["pred"]],
                 [{"startLine": s, "endLine": e} for s, e in f["gt"]])


def _without(spans: dict[str, Any], drop: set[str]) -> dict[str, Any]:
    return {b: (dict(f, pred=[]) if b in drop else f) for b, f in spans.items()}


def _dump_index(kind: str) -> set[frozenset]:
    """Prediction sets of the answers saved in llm_debug/ (first answers or retries)."""
    import json
    from StaLLM_core import _coerce_items_schema
    out = set()
    for f in Path("llm_debug").glob(f"*_{kind}_sanitized.txt"):
        try:
            items = _coerce_items_schema(json.loads(f.read_text()))
        except Exception:
            continue
        out.add(frozenset((int(i.get("startLine") or i.get("line") or 0),
                           int(i.get("endLine") or i.get("startLine") or i.get("line") or 0)) for i in items))
    return out


def _retry_choices(spans: dict[str, Any], k: int, retries: set, firsts: set) -> list[set[str]]:
    """Possible sets of files answered by the retry. The runs record only their number k,
    but every retry answer is saved in llm_debug/: a file whose predictions equal a saved
    retry answer and no saved first answer is a retry file (definite); one that matches
    both is ambiguous. Checked against the runs that record per-file provenance."""
    key = lambda f: frozenset(map(tuple, f["pred"]))
    matched = [b for b, f in spans.items() if f["pred"] and key(f) in retries]
    definite = {b for b in matched if key(spans[b]) not in firsts}
    ambiguous = [b for b in matched if b not in definite]
    need = k - len(definite)
    if need <= 0:
        return [definite]
    if need > len(ambiguous):
        raise ValueError("retry files cannot be reconstructed")
    return [definite | set(c) for c in itertools.combinations(ambiguous, need)]


def rq12_no_retry(md: list[str]) -> None:
    match = _matcher()
    oracle = aps._rq12_oracle()
    groups: dict[tuple, list] = defaultdict(list)
    for protocol, folder in aps.RQ12_DIRS.items():
        for f in sorted(folder.glob("rq2_*.json")):
            d = aps._load(f)
            groups[(protocol, d["slot"], d["prompt"])].append(d)
    rows = []
    retries, firsts = _dump_index("retry"), _dump_index("first")

    def f1_of(s):
        return aps._span_f1(s, {b: [{"startLine": x, "endLine": y} for x, y in v["pred"]] for b, v in s.items()}, match)

    def p_random(sp: list[dict], f1: float) -> tuple[float, float]:
        rng = random.Random(42)
        draws = [statistics.mean(aps.random_placement_f1(s, rng, match, False) for s in sp)
                 for _ in range(aps.RANDOM_DRAWS)]
        return statistics.mean(draws), (1 + sum(x >= f1 for x in draws)) / (1 + aps.RANDOM_DRAWS)

    for (protocol, slot, prompt), runs in sorted(groups.items()):
        spans = [aps._rescored_spans(d["metrics"]["spans_by_file"], oracle) for d in runs]
        retry = [d["metrics"].get("file_sources", {}).get("llm_retry", 0) for d in runs]
        exact = all("source" in f for s in spans for f in s.values())
        row = {"protocol": protocol, "slot": slot, "prompt": prompt, "runs": len(runs),
               "per_file_provenance": exact, "retry_files_mean": aps._r(statistics.mean(retry), 1),
               "llm_errors": sum(d.get("llm_errors", 0) for d in runs),
               "error_files": sum(d["metrics"].get("file_sources", {}).get("llm_error", 0) for d in runs),
               "f1_with_retry": aps._r(statistics.mean(f1_of(s) for s in spans))}
        if exact:
            nr = [aps._drop_retry(s) for s in spans]
            f1 = statistics.mean(f1_of(s) for s in nr)
            rnd, p = p_random(nr, f1)
            row.update({"f1_no_retry_low": aps._r(f1), "f1_no_retry_high": aps._r(f1),
                        "random_low": aps._r(rnd), "p_low": aps._r(p, 4),
                        "random_high": aps._r(rnd), "p_high": aps._r(p, 4)})
        else:
            per_run = []
            for s_, k in zip(spans, retry):
                options = [_without(s_, c) for c in _retry_choices(s_, k, retries, firsts)]
                per_run.append(sorted(options, key=f1_of))
            row["choices"] = "x".join(str(len(o)) for o in per_run)
            for tag, pick in (("low", 0), ("high", -1)):
                sp = [o[pick] for o in per_run]
                f1 = statistics.mean(f1_of(x) for x in sp)
                rnd, p = p_random(sp, f1)
                row.update({f"f1_no_retry_{tag}": aps._r(f1), f"random_{tag}": aps._r(rnd), f"p_{tag}": aps._r(p, 4)})
        rows.append(row)
    aps._write_csv(STATS / "reviewer_rq12_no_retry.csv", rows)
    md.append("## 2. RQ1/RQ2 — without the empty-answer retry\n")
    md.append("*Files answered only through the retry are counted as empty. Runs with per-file provenance "
              "(`source` in spans_by_file: the numbered runs of AZ4 and AZ6) give an exact value. The other runs record "
              "only the number k of such files, so the files are identified from the retry answers saved in llm_debug/ "
              "(a file whose predictions equal a saved retry answer and no saved first answer). This reconstruction "
              "recovers the recorded k in every run and agrees with the per-file provenance where it exists. When a "
              "file matches both a first and a retry answer, low/high give the extreme choices. Random placement and p are "
              f"recomputed on the same spans ({aps.RANDOM_DRAWS} draws, seed 42).*\n")
    md.append("| Protocol | Slot | Prompt | runs | exact | retry files | LLM errors | F1 with retry "
              "| F1 without retry (low–high) | random (low/high) | p (low/high) |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['protocol']} | {r['slot']} | {r['prompt']} | {r['runs']} | {r['per_file_provenance']} "
                  f"| {r['retry_files_mean']} | {r['llm_errors']} | {r['f1_with_retry']} "
                  f"| {r['f1_no_retry_low']}–{r['f1_no_retry_high']} | {r['random_low']}/{r['random_high']} "
                  f"| {r['p_low']}/{r['p_high']} |")
    md.append("")


# ── 3. RQ4 random ordering ───────────────────────────────────────────────────

def _expected_rr(n: int, g: int, depth: int = 10) -> float:
    """Expected reciprocal rank of the first gold file in the top `depth` of a
    uniformly random ordering of n candidates, g of which are gold."""
    if g == 0:
        return 0.0
    total, p_none = 0.0, 1.0
    for r in range(1, min(depth, n) + 1):
        p_hit = g / (n - r + 1)
        total += p_none * p_hit / r
        p_none *= 1 - p_hit
        if p_none <= 0:
            break
    return total


def rq4_random(md: list[str]) -> None:
    import run_paper_experiments as rpe
    from StaLLM_benchmarks import load_argouml_feature_tasks, load_feature_locations_csv_tasks
    from StaLLM_tasks import paths_match

    rows = []
    for system in ("argouml", "eshop"):
        rpe.SYSTEM = system
        cfg = rpe.SYSTEMS[system]
        if system == "eshop":
            tasks = load_feature_locations_csv_tasks(
                str(rpe.ESHOP_FL_CSV), repo_zip=str(rpe.ESHOP_FL_ZIP), project="eShopOnContainers",
                language="C#", descriptions=rpe.ESHOP_FEATURE_DESCRIPTIONS, extensions=[".cs"])
        else:
            tasks = load_argouml_feature_tasks(str(rpe.ARGOUML_GT), repo_zip=str(rpe.ARGOUML_ZIP), single_only=True)
        all_c = rpe.list_repo_candidates(str(cfg["zip"]), allowed_exts=[cfg["ext"]])
        masked = {t.task_id: rpe.mask_feature_task(t) for t in tasks}
        for label, (policy, _) in rpe.RQ4_CONDITIONS.items():
            for t in tasks:
                task = masked[t.task_id] if label == "bm25_masked" else t
                cands = rpe.rq4_candidates(policy, task, all_c)[:rpe.CANDIDATE_BUDGET]
                gold = [g.normalized_file for g in t.gold_locations]
                g = sum(1 for c in cands if any(paths_match(c, x) for x in gold))
                n = len(cands)
                rows.append({"system": system, "condition": label, "feature": t.metadata.get("feature", t.task_id),
                             "candidates": n, "gold": len(gold), "gold_in_list": g,
                             "expected_p@10": aps._r(g / n if n else 0.0),
                             "expected_mrr@10": aps._r(_expected_rr(n, g))})
    aps._write_csv(STATS / "reviewer_rq4_random_ordering.csv", rows)
    md.append("## 3. RQ4 — random ordering of the candidate list\n")
    md.append("*The model receives only the file paths of the candidate list (StaLLM_tasks.py, "
              "`candidate_text = \"- path\"` lines), never file contents. Expected Precision@10 = gold files in the "
              "list / list length (hypergeometric mean of the top ten). Expected MRR = exact expectation of the "
              "reciprocal rank of the first gold file within the top ten (0 if none), uniform random permutation.*\n")
    md.append("| System | Condition | candidates | gold in list (mean) | E[P@10] | E[MRR] |\n|---|---|---|---|---|---|")
    agg: dict[tuple, list] = defaultdict(list)
    for r in rows:
        agg[(r["system"], r["condition"])].append(r)
    for (system, cond), rs in agg.items():
        md.append(f"| {system} | {cond} | {statistics.mean(r['candidates'] for r in rs):.0f} "
                  f"| {statistics.mean(r['gold_in_list'] for r in rs):.1f} "
                  f"| {statistics.mean(r['expected_p@10'] for r in rs):.3f} "
                  f"| {statistics.mean(r['expected_mrr@10'] for r in rs):.3f} |")
    md.append("")
    md.append("Per feature: reviewer_rq4_random_ordering.csv.\n")


# ── 4. MLCQ reviewers ────────────────────────────────────────────────────────

def _alpha_nominal(units: list[list[int]]) -> float:
    """Krippendorff's alpha (nominal) for units with two or more ratings."""
    units = [u for u in units if len(u) >= 2]
    cats = sorted({v for u in units for v in u})
    o = defaultdict(float)
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[(a, b)] += 1 / (m - 1)
    n_c = {c: sum(o[(c, k)] for k in cats) for c in cats}
    n = sum(n_c.values())
    d_o = sum(o[(a, b)] for a in cats for b in cats if a != b)
    d_e = sum(n_c[a] * n_c[b] for a in cats for b in cats if a != b) / (n - 1)
    return 1 - d_o / d_e if d_e else 1.0


def mlcq_reviewers(md: list[str]) -> None:
    with open(OUT.parent.parent / "data" / "apps" / "MLCQ" / "MLCQCodeSmellSamples.csv", newline="", encoding="utf-8") as f:
        allrows = list(csv.DictReader(f, delimiter=";"))
    by_id = {r["id"]: r for r in allrows}
    used = aps._load(OUT / "rq3" / "rq3_AZ1_binary_oracle_baseline_MLCQ.json")["rows"]
    used_rows = [by_id[str(r["sample_id"])] for r in used if str(r["sample_id"]) in by_id]
    units: dict[tuple, list[int]] = defaultdict(list)
    for r in allrows:
        units[(r["sample_id"], r["smell"])].append(0 if r["severity"].strip().lower() == "none" else 1)
    used_units = {(r["sample_id"], r["smell"]) for r in used_rows}
    multi = [units[u] for u in used_units if len(units[u]) >= 2]
    disagree = sum(1 for u in multi if len(set(u)) > 1)
    label_vs_majority = 0
    for r in used_rows:
        votes = units[(r["sample_id"], r["smell"])]
        own = 0 if r["severity"].strip().lower() == "none" else 1
        if len(votes) >= 2 and sum(votes) * 2 != len(votes) and own != int(sum(votes) * 2 > len(votes)):
            label_vs_majority += 1
    all_multi = [u for u in units.values() if len(u) >= 2]
    row = {"rows_used": len(used), "rows_matched": len(used_rows),
           "distinct_fragments": len({r["sample_id"] for r in used_rows}),
           "distinct_fragment_smell_pairs": len(used_units),
           "reviewers": dict(sorted(defaultdict(int, {k: sum(1 for r in used_rows if r["reviewer_id"] == k)
                                                      for k in {r["reviewer_id"] for r in used_rows}}).items())),
           "pairs_with_2plus_reviewers": len(multi), "pairs_with_disagreement": disagree,
           "pct_agreement_used": aps._r(1 - disagree / len(multi)) if multi else None,
           "alpha_used": aps._r(_alpha_nominal(multi)) if multi else None,
           "rows_label_differs_from_majority": label_vs_majority,
           "alpha_full_dataset": aps._r(_alpha_nominal(all_multi)),
           "pct_agreement_full_dataset": aps._r(sum(1 for u in all_multi if len(set(u)) == 1) / len(all_multi))}
    aps._write_csv(STATS / "reviewer_mlcq_agreement.csv", [row])
    md.append("## 4. MLCQ — fragments and reviewers of the 200 rows used\n")
    md.append("*Presence = severity other than `none`. Agreement over (fragment, smell) pairs reviewed by two or more "
              "reviewers in the full MLCQ export; Krippendorff's α (nominal).*\n")
    for k, v in row.items():
        md.append(f"- {k}: {v}")
    md.append("")


# ── 5. Checks ────────────────────────────────────────────────────────────────

def checks(md: list[str]) -> None:
    md.append("## 5. Checks\n")
    md.append("- Table 2 time: analyze_paper_stats.analyze_rq12 writes `time_s = statistics.median(elapsed_s of the runs)`, "
              "i.e. the median over runs of the wall-clock time of a run.")
    with open(STATS / "sensitivity_chance.csv", newline="") as f:
        sens = {(r["protocol"], r["slot"], r["prompt"]): r["runs"] for r in csv.DictReader(f)}
    with open(STATS / "rq12_rerun.csv", newline="") as f:
        rer = {(r["protocol"], r["slot"], r["prompt"]): r["runs"] for r in csv.DictReader(f)}
    stale = [f"{k}: sensitivity {v} vs rerun {rer.get(k)}" for k, v in sens.items() if rer.get(k) != v]
    md.append(f"- Run counts, sensitivity_chance.csv vs rq12_rerun.csv: "
              f"{'; '.join(stale) if stale else 'identical'}. A difference means sensitivity_analyses.py was last "
              "run before all runs were stored and must be rerun.")

    # McNemar direction: RQ5 held-out and MLCQ (generated vs manual).
    gen_mlcq = aps._load(OUT / "rq5" / "rq5_MLCQ_threshold_guided.json")
    md.append("- RQ5 held-out (stats/rq5_heldout.csv): `generated_only_correct` counts samples where only the "
              "generated prompt is correct (mcnemar_exact(generated, manual)).")
    for p in ("binary_oracle_baseline", "conservative_evidence", "definition_driven"):
        man = aps._load(OUT / "rq3" / f"rq3_AZ1_{p}_MLCQ.json")
        a, b, pv = aps.mcnemar_exact(aps._correct(gen_mlcq["rows"]), aps._correct(man["rows"]))
        md.append(f"- RQ5 MLCQ, generated vs {p} (run 1): {a} vs {b} discordant, p={aps._r(pv, 4)}")
    md.append("")


def main() -> None:
    global OUT, STATS
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(aps.OUT))
    root = Path(parser.parse_args().root)
    annotated = root != aps.OUT
    OUT, STATS = root, root / "stats"
    aps.OUT, aps.STATS = OUT, STATS
    STATS.mkdir(parents=True, exist_ok=True)
    md = ["# Additional analyses (review)\n", "Computed from stored outputs, no model call.\n"]
    if annotated:
        md.append("RQ3 on DACOS labeled by the annotators' majority (complex method, long parameter list), "
                  "with the MLCQ runs of output/paper_results/. These are the numbers of the paper.\n")
        steps = (rq3_stability,)
    else:
        md.append("**Legacy DACOS labels.** In this root, DACOS is labeled by the `smells` column of the DACOS "
                  "sample table (preliminary version). The paper reports RQ3 and RQ5 on the annotated DACOS "
                  "sample: see output/paper_results_annot/stats/reviewer_analyses.md. Analyses 2–4 do not "
                  "depend on DACOS and are the numbers of the paper.\n")
        steps = (rq3_stability, rq12_no_retry, rq4_random, mlcq_reviewers, checks)
    for step in steps:
        step(md)
        print(f"done: {step.__name__}")
    (STATS / "reviewer_analyses.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"→ {STATS / 'reviewer_analyses.md'}")


if __name__ == "__main__":
    main()
