"""
StaLLM — offline sensitivity analyses for the journal paper
============================================================
Re-uses recorded predictions only; no model is called.

  1. RQ1/RQ2  robustness of the random-placement baselines (seed, universe
              restricted to files with oracle spans, matching preset)
  2. RQ3      sensitivity of the oracle substitution to the heuristic thresholds
  3. RQ4      candidate recall before the model, and P@10 relative to the gold
              files that were visible in the candidate list
  4. RQ5      selection bias under repeated stratified splits of the selection set

Usage:
    python sensitivity_analyses.py      # writes output/paper_results/stats/sensitivity*.csv and sensitivity.md
"""

from __future__ import annotations

import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

import analyze_paper_stats as aps

STATS = aps.STATS
OUT = aps.OUT


# ── 1. RQ1/RQ2 chance baselines ──────────────────────────────────────────────

def _matcher(preset: str):
    from StaLLM_core import _match_file

    def match(p, g):
        return _match_file([dict(x, startColumn=None, endColumn=None, type=None) for x in p],
                           [dict(x, startColumn=None, endColumn=None, type=None) for x in g],
                           require_type=False, use_cols_single=False, preset=preset)
    return match


def chance_sensitivity(md: list[str], draws: int = 500) -> None:
    oracle = aps._rq12_oracle()
    groups: dict[tuple, list] = defaultdict(list)
    for protocol, folder in aps.RQ12_DIRS.items():
        for f in sorted(folder.glob("rq2_*.json")):
            d = aps._load(f)
            groups[(protocol, d["slot"], d["prompt"])].append(
                aps._rescored_spans(d["metrics"]["spans_by_file"], oracle))
    variants = [("Lenient, all 20 files, seed 42", "Lenient", False, 42)]
    variants += [(f"Lenient, all 20 files, seed {s}", "Lenient", False, s) for s in (1, 2, 3, 4)]
    variants += [("Lenient, 13 positive files", "Lenient", True, 42),
                 ("Balanced, all 20 files", "Balanced", False, 42),
                 ("Strict, all 20 files", "Strict", False, 42)]
    rows = []
    for (protocol, slot, prompt), runs in sorted(groups.items()):
        for label, preset, positives_only, seed in variants:
            match = _matcher(preset)
            spans = [{b: f for b, f in s.items() if f["gt"] or not positives_only} for s in runs]
            preds = [{b: [{"startLine": x, "endLine": y} for x, y in v["pred"]] for b, v in s.items()} for s in spans]
            llm = statistics.mean(aps._span_f1(s, p, match) for s, p in zip(spans, preds))
            rng = random.Random(seed)
            dist = [statistics.mean(aps.random_placement_f1(s, rng, match) for s in spans) for _ in range(draws)]
            rows.append({"protocol": protocol, "slot": slot, "prompt": prompt, "runs": len(runs),
                         "variant": label, "f1_llm": aps._r(llm), "f1_random_mean": aps._r(statistics.mean(dist)),
                         "p_random": aps._r((1 + sum(x >= llm for x in dist)) / (1 + draws), 4)})
    aps._write_csv(STATS / "sensitivity_chance.csv", rows)
    md.append("## 1. RQ1/RQ2 — robustness of the random-placement baseline\n")
    md.append(f"{draws} draws per variant; p = share of draws reaching the model's mean F1.\n")
    md.append("| Protocol | Slot | Prompt | runs | Variant | F1 model | F1 random | p |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['protocol']} | {r['slot']} | {r['prompt']} | {r['runs']} | {r['variant']} | "
                  f"{r['f1_llm']} | {r['f1_random_mean']} | {r['p_random']} |")
    md.append("")


# ── 2. RQ3 threshold sensitivity ─────────────────────────────────────────────

THRESHOLDS = {
    "lenient": dict(lpl=4, cm=6, lm=80, ma_f=5, ma_m=5, gc_m=15, gc_loc=400, dc_f=2, dc_r=0.6, fe_ext=4, fe_ratio=1.5),
    "default": dict(lpl=5, cm=8, lm=100, ma_f=7, ma_m=7, gc_m=20, gc_loc=500, dc_f=3, dc_r=0.7, fe_ext=5, fe_ratio=2.0),
    "strict": dict(lpl=6, cm=10, lm=120, ma_f=9, ma_m=9, gc_m=25, gc_loc=600, dc_f=4, dc_r=0.8, fe_ext=6, fe_ratio=2.5),
}


def heuristic(smell: str, code: str, t: dict) -> bool:
    """Same rules as run_paper_experiments.smell_heuristic, with parameterised thresholds."""
    from run_paper_experiments import _DECISION_RE, _FIELD_RE, _SIG_RE
    sigs = _SIG_RE.findall(code)
    loc = sum(1 for line in code.splitlines() if line.strip())
    params = max((len([p for p in a.split(",") if p.strip()]) for _, a in sigs), default=0)
    fields, methods = len(_FIELD_RE.findall(code)), len(sigs)
    if smell == "long_parameter_list":
        return params >= t["lpl"]
    if smell == "complex_method":
        return len(_DECISION_RE.findall(code)) >= t["cm"]
    if smell == "long_method":
        return loc > t["lm"]
    if smell == "multifaceted_abstraction":
        return fields >= t["ma_f"] and methods >= t["ma_m"]
    if smell == "god_class":
        return methods >= t["gc_m"] or loc > t["gc_loc"]
    if smell == "data_class":
        acc = sum(1 for name, _ in sigs if re.match(r"(?:get|set|is)[A-Z]", name))
        return fields >= t["dc_f"] and methods > 0 and acc / methods >= t["dc_r"]
    if smell == "feature_envy":
        ext = len(re.findall(r"\b(?!this\b|super\b)[a-z]\w*\.\w+\s*\(", code))
        own = len(re.findall(r"\bthis\.", code))
        return ext >= t["fe_ext"] and ext > t["fe_ratio"] * own
    return False


def rq3_threshold_sensitivity(md: list[str]) -> None:
    import run_paper_experiments as rpe
    samples = {
        "DACOS": rpe.load_dacos_smell_samples(str(rpe.DACOS_MAIN), files_root=str(rpe.DACOS_FILES),
                                              limit=rpe.MAX_SAMPLES),
        "MLCQ": rpe.load_mlcq_smell_samples(str(rpe.MLCQ_CSV), limit=rpe.MAX_SAMPLES),
    }
    results = [d for d in aps.load_rq3() if d["run"] == 1]
    rows = []
    for ds, sset in samples.items():
        recorded = {r["sample_id"]: r["predicted"]
                    for r in aps._load(OUT / "rq3" / "baselines" / f"rq3b_metric_heuristic_{ds}.json")["rows"]}
        default = {s.sample_id: heuristic(s.smell, s.code, THRESHOLDS["default"]) for s in sset}
        if default != recorded:
            raise AssertionError(f"parameterised heuristic does not reproduce the recorded one on {ds}")
        gold = {s.sample_id: bool(s.label) for s in sset}
        configs = [d for d in results if d["dataset"] == ds and {r["sample_id"] for r in d["rows"]} == set(gold)]
        k_human = []
        for d in configs:
            ids = [r["sample_id"] for r in d["rows"]]
            pred = [bool(r["predicted"]) for r in d["rows"]]
            k_human.append(aps._binary_scores([gold[i] for i in ids], pred)[1])
        for level, t in THRESHOLDS.items():
            h = {s.sample_id: heuristic(s.smell, s.code, t) for s in sset}
            ids = sorted(gold)
            k_h_vs_gold = aps._binary_scores([gold[i] for i in ids], [h[i] for i in ids])[1]
            k_metric = []
            for d in configs:
                rid = [r["sample_id"] for r in d["rows"]]
                k_metric.append(aps._binary_scores([h[i] for i in rid], [bool(r["predicted"]) for r in d["rows"]])[1])
            rho, p = spearmanr(k_human, k_metric)
            best_h = max(range(len(configs)), key=lambda i: k_human[i])
            best_m = max(range(len(configs)), key=lambda i: k_metric[i])
            rows.append({"dataset": ds, "thresholds": level, "positives_flagged": sum(h.values()),
                         "kappa_heuristic_vs_human": aps._r(k_h_vs_gold), "configs": len(configs),
                         "spearman_rho": aps._r(rho), "spearman_p": aps._r(p, 3),
                         "best_vs_human": f"{configs[best_h]['slot']}/{configs[best_h]['prompt']}",
                         "best_vs_metric": f"{configs[best_m]['slot']}/{configs[best_m]['prompt']}"})
    aps._write_csv(STATS / "sensitivity_rq3_thresholds.csv", rows)
    md.append("## 2. RQ3 — oracle substitution under three threshold settings\n")
    md.append("The default setting reproduces the recorded heuristic exactly. Lenient and strict move "
              "every threshold one step (see THRESHOLDS in this script).\n")
    md.append("| Dataset | Thresholds | flagged | κ heuristic vs human | ρ (n configs) | p | best vs human | best vs metric |\n"
              "|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['dataset']} | {r['thresholds']} | {r['positives_flagged']} | {r['kappa_heuristic_vs_human']} | "
                  f"{r['spearman_rho']} ({r['configs']}) | {r['spearman_p']} | {r['best_vs_human']} | {r['best_vs_metric']} |")
    md.append("")


# ── 3. RQ4 retrieval vs ranking ──────────────────────────────────────────────

def rq4_retrieval(md: list[str]) -> None:
    import run_paper_experiments as rpe
    from StaLLM_benchmarks import load_argouml_feature_tasks, load_feature_locations_csv_tasks
    from StaLLM_tasks import list_repo_candidates, paths_match

    rows = []
    for system, root in (("argouml", OUT), ("eshop", OUT / "eshop")):
        rpe.SYSTEM = system
        cfg = rpe.SYSTEMS[system]
        if system == "eshop":
            tasks = load_feature_locations_csv_tasks(
                str(rpe.ESHOP_FL_CSV), repo_zip=str(rpe.ESHOP_FL_ZIP), project="eShopOnContainers",
                language="C#", descriptions=rpe.ESHOP_FEATURE_DESCRIPTIONS, extensions=[".cs"])
        else:
            tasks = load_argouml_feature_tasks(str(rpe.ARGOUML_GT), repo_zip=str(rpe.ARGOUML_ZIP), single_only=True)
        all_c = list_repo_candidates(str(cfg["zip"]), allowed_exts=[cfg["ext"]])
        for label, (policy, folder) in rpe.RQ4_CONDITIONS.items():
            task_list = [rpe.mask_feature_task(t) for t in tasks] if label == "bm25_masked" else tasks
            for task in task_list:
                feature = task.metadata.get("feature", task.task_id)
                cands = rpe.rq4_candidates(policy, task, all_c)[:rpe.CANDIDATE_BUDGET]
                gold = [g.normalized_file for g in task.gold_locations]
                visible = sum(1 for g in gold if any(paths_match(c, g) for c in cands))
                top10 = cands[:10]
                base_hits = sum(1 for g in gold if any(paths_match(c, g) for c in top10))
                folder_path = root / Path(folder).name
                per_slot = defaultdict(list)
                for f in folder_path.glob(f"rq4_*_{feature}*.json"):
                    d = aps._load(f)
                    if d.get("feature") != feature:
                        continue
                    hits = min(10, round(d["metrics"]["recall@10"] * d["gold_count"]))
                    per_slot[d["slot"]].append(hits)
                row = {"system": system, "ordering": label, "feature": feature, "gold": len(gold),
                       "candidates": len(cands), "visible_gold": visible,
                       "candidate_recall": aps._r(visible / len(gold)),
                       "p10_ceiling": aps._r(min(10, visible) / 10),
                       "p10_no_llm": aps._r(base_hits / 10) if label != "alphabetical" else None}
                for slot, hits in sorted(per_slot.items()):
                    row[f"p10_{slot}"] = aps._r(statistics.mean(hits) / 10)
                    row[f"p10_rel_{slot}"] = aps._r(statistics.mean(hits) / min(10, visible)) if visible else None
                rows.append(row)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (not k.startswith(("system", "ordering", "feature")), k))
    for r in rows:
        for k in keys:
            r.setdefault(k, None)
    aps._write_csv(STATS / "sensitivity_rq4_retrieval.csv", rows)

    md.append("## 3. RQ4 — candidate recall before the model and P@10 relative to visible gold\n")
    md.append("Candidate recall = share of gold files in the candidate list given to the model "
              "(budget 600). P@10 ceiling = min(10, visible gold)/10. Relative P@10 = hits in the top ten / "
              "min(10, visible gold), mean over features, prompts and runs.\n")
    md.append("| System | Ordering | candidate recall | P@10 ceiling | gpt-4o rel. | gpt-4.1 rel. | o3 rel. |\n|---|---|---|---|---|---|---|")
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("candidate_recall", "p10_ceiling", "p10_rel_AZ1", "p10_rel_AZ2", "p10_rel_AZ3"):
            if r.get(k) is not None:
                agg[(r["system"], r["ordering"])][k].append(r[k])
    for (system, ordering), v in agg.items():
        cells = [aps._r(statistics.mean(v[k])) if v[k] else "-" for k in
                 ("candidate_recall", "p10_ceiling", "p10_rel_AZ1", "p10_rel_AZ2", "p10_rel_AZ3")]
        md.append(f"| {system} | {ordering} | " + " | ".join(str(c) for c in cells) + " |")
    md.append("")


# ── 4. RQ5 selection bias under repeated splits ──────────────────────────────

def rq5_split_simulation(md: list[str], repeats: int = 1000) -> None:
    manual = {p: aps._load(OUT / "rq3" / f"rq3_AZ1_{p}_DACOS.json")
              for p in ("binary_oracle_baseline", "conservative_evidence", "definition_driven")}
    generated = {f.stem.replace("rq5_DACOS_", ""): aps._load(f) for f in sorted((OUT / "rq5").glob("rq5_DACOS_*.json"))}
    prompts = {**{f"manual:{k}": v for k, v in manual.items()}, **{f"generated:{k}": v for k, v in generated.items()}}
    gold = {r["sample_id"]: bool(r["gold"]) for r in manual["binary_oracle_baseline"]["rows"]}
    pred = {name: {r["sample_id"]: bool(r["predicted"]) for r in d["rows"]} for name, d in prompts.items()}
    if any(set(p) != set(gold) for p in pred.values()):
        raise AssertionError("prompts were not evaluated on the same selection samples")

    def f1(name, ids):
        return aps._binary_scores([gold[i] for i in ids], [pred[name][i] for i in ids])[0]

    pos = sorted(i for i, g in gold.items() if g)
    neg = sorted(i for i, g in gold.items() if not g)
    rng = random.Random(42)
    optimism, winner_best_on_test, gen_wins, gen_beats_manual = [], 0, 0, 0
    for _ in range(repeats):
        rng.shuffle(pos)
        rng.shuffle(neg)
        sel = pos[: len(pos) // 2] + neg[: len(neg) // 2]
        test = pos[len(pos) // 2:] + neg[len(neg) // 2:]
        scores_sel = {n: f1(n, sel) for n in pred}
        scores_test = {n: f1(n, test) for n in pred}
        winner = max(scores_sel, key=scores_sel.get)
        optimism.append(scores_sel[winner] - scores_test[winner])
        winner_best_on_test += scores_test[winner] >= max(scores_test.values())
        if winner.startswith("generated:"):
            gen_wins += 1
            best_manual_test = max(v for n, v in scores_test.items() if n.startswith("manual:"))
            gen_beats_manual += scores_test[winner] > best_manual_test
    optimism.sort()
    row = {"prompts": len(pred), "repeats": repeats,
           "optimism_mean": aps._r(statistics.mean(optimism)),
           "optimism_ci": f"[{aps._r(optimism[int(0.025 * repeats)])}, {aps._r(optimism[int(0.975 * repeats) - 1])}]",
           "winner_also_best_on_test": aps._r(winner_best_on_test / repeats),
           "generated_wins_selection": aps._r(gen_wins / repeats),
           "generated_winner_beats_best_manual_on_test": aps._r(gen_beats_manual / max(1, gen_wins))}
    aps._write_csv(STATS / "sensitivity_rq5_splits.csv", [row])
    md.append("## 4. RQ5 — selection bias under repeated stratified splits\n")
    md.append(f"The 200 selection samples (26 positive) are split {repeats} times into two stratified halves; "
              f"the best of the {len(pred)} prompts (3 manual, {len(generated)} generated) on one half is "
              "evaluated on the other.\n")
    md.append(f"- optimism of the winner (F1 on its selection half minus F1 on the other half): "
              f"mean {row['optimism_mean']}, 95% interval {row['optimism_ci']}")
    md.append(f"- the winner is also the best prompt on the other half in {row['winner_also_best_on_test']:.0%} of splits")
    md.append(f"- a generated prompt wins selection in {row['generated_wins_selection']:.0%} of splits; when it does, "
              f"it beats the best manual prompt on the other half in "
              f"{row['generated_winner_beats_best_manual_on_test']:.0%} of them\n")


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    md = ["# StaLLM paper — offline sensitivity analyses\n", "Generated by `sensitivity_analyses.py`; no model calls.\n"]
    chance_sensitivity(md)
    rq3_threshold_sensitivity(md)
    rq4_retrieval(md)
    rq5_split_simulation(md)
    (STATS / "sensitivity.md").write_text("\n".join(md))
    print(f"Wrote {STATS / 'sensitivity.md'}")


if __name__ == "__main__":
    main()
