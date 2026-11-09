"""Smell-classification analyses of RQ3 and RQ5 on the annotated DACOS sample,
computed from the stored outputs only (no model call).

    python smell_analyses.py

Reads output/paper_results_annot/ (DACOS, annotators' majority) and the MLCQ
runs of output/paper_results/. Writes CSV files and smell_analyses.md to
output/paper_results_annot/stats/:

1. Oracle shift per configuration: paired bootstrap of
   delta-kappa = kappa(metric oracle) - kappa(human labels), for the metric
   heuristic (DACOS, MLCQ) and PMD (DACOS, needs pmd_oracle.py first).
2. Paired bootstrap of delta-kappa between prompts and configurations, the
   kappa counterpart of the McNemar tests on per-sample correctness:
   RQ5 selected prompt vs each manual prompt (selection, held-out, MLCQ) and
   RQ3 best configuration vs the others.
3. RQ5 repeated splits by kappa, with two chance references: 1/8, and a null
   in which the eight prompts are exchangeable (their predictions are permuted
   among prompts within each sample), which keeps the dependence between the
   two halves. Also the run-to-run spread of kappa of the gpt-4o prompts.
4. Metric heuristic on the held-out split vs the LLM prompts.
5. MLCQ labels replaced by the reviewers' majority: effect on kappa and ranking.
6. DACOS inter-annotator agreement (Krippendorff's alpha) per smell.
"""
from __future__ import annotations

import csv
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import analyze_paper_stats as aps
from reviewer_analyses import _alpha_nominal, _q, _rho

MAIN = Path("output/paper_results")
ANNOT = Path("output/paper_results_annot")
STATS = ANNOT / "stats"
BOOT = 1000
MANUAL = ("binary_oracle_baseline", "conservative_evidence", "definition_driven")
SELECTED = "calibrated_confidence"


def _rows(path: Path) -> dict[str, tuple[bool, bool]]:
    return {str(r["sample_id"]): (bool(r["gold"]), bool(r["predicted"])) for r in aps._load(path)["rows"]}


def _kappa(gold: list[bool], pred: list[bool]) -> float:
    return aps._binary_scores(gold, pred)[1]


def _configs(dataset: str) -> dict[str, dict[str, tuple[bool, bool]]]:
    """Majority vote over the runs of each configuration, as in Table 3."""
    root = ANNOT if dataset == "DACOS" else MAIN
    groups: dict[str, list] = defaultdict(list)
    for f in sorted((root / "rq3").glob(f"rq3_AZ[12378]_*_{dataset}*.json")):
        d = aps._load(f)
        if d.get("dataset") == dataset:
            groups[f"{d['slot']}/{d['prompt']}"].append(d)
    return {k: {str(r["sample_id"]): (bool(r["gold"]), bool(r["predicted"]))
                for r in aps.majority_vote(v)["rows"]} for k, v in sorted(groups.items())}


def _ci(xs: list[float]) -> tuple[float, float]:
    return _q(xs, 0.025), _q(xs, 0.975)


def _boot_delta(ids: list[str], fa, fb, rng: random.Random) -> tuple[float, float, float]:
    """Observed fa - fb and its paired bootstrap 95% interval; fa, fb map a sample list to a score."""
    obs = fa(ids) - fb(ids)
    deltas = []
    for _ in range(BOOT):
        s = [rng.choice(ids) for _ in ids]
        deltas.append(fa(s) - fb(s))
    lo, hi = _ci(deltas)
    return obs, lo, hi


# ── 1. Oracle shift per configuration ────────────────────────────────────────

def oracle_shift(md: list[str]) -> None:
    rows = []
    oracles = {
        ("DACOS", "heuristic"): {r["sample_id"]: bool(r["predicted"]) for r in
                                 aps._load(ANNOT / "rq3" / "baselines" / "rq3b_metric_heuristic_DACOS.json")["rows"]},
        ("MLCQ", "heuristic"): {str(r["sample_id"]): bool(r["predicted"]) for r in
                                aps._load(ANNOT / "rq3" / "baselines" / "rq3b_metric_heuristic_MLCQ.json")["rows"]},
    }
    pmd_file = STATS / "pmd_labels_DACOS.json"
    if pmd_file.exists():
        oracles[("DACOS", "PMD")] = {str(r["sample_id"]): bool(r["pmd"]) for r in aps._load(pmd_file)}
    for (ds, oracle_name), oracle in oracles.items():
        rng = random.Random(42)
        for name, rows_ in _configs(ds).items():
            ids = sorted(i for i in rows_ if i in oracle)
            def k_metric(s, rows_=rows_, oracle=oracle):
                return _kappa([oracle[i] for i in s], [rows_[i][1] for i in s])
            def k_human(s, rows_=rows_):
                return _kappa([rows_[i][0] for i in s], [rows_[i][1] for i in s])
            obs, lo, hi = _boot_delta(ids, k_metric, k_human, rng)
            rows.append({"dataset": ds, "oracle": oracle_name, "config": name, "n": len(ids),
                         "kappa_human": aps._r(k_human(ids)), "kappa_metric": aps._r(k_metric(ids)),
                         "delta": aps._r(obs), "ci_lo": aps._r(lo), "ci_hi": aps._r(hi),
                         "ci_excludes_0": lo > 0 or hi < 0})
    aps._write_csv(STATS / "smell_oracle_shift.csv", rows)
    md.append("## 1. Oracle shift per configuration: Δκ = κ(metric oracle) − κ(human labels)\n")
    md.append(f"Paired bootstrap, {BOOT} resamples, seed 42, majority vote over the runs of each configuration.\n")
    md.append("| Dataset | Oracle | Configuration | κ human | κ metric | Δκ [95% CI] |\n|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['dataset']} | {r['oracle']} | {r['config']} | {r['kappa_human']} | {r['kappa_metric']} "
                  f"| {r['delta']} [{r['ci_lo']}, {r['ci_hi']}] |")
    md.append("")
    for key in sorted({(r["dataset"], r["oracle"]) for r in rows}):
        sel = [r for r in rows if (r["dataset"], r["oracle"]) == key]
        pos = sum(r["ci_excludes_0"] and r["delta"] > 0 for r in sel)
        neg = sum(r["ci_excludes_0"] and r["delta"] < 0 for r in sel)
        md.append(f"- {key[0]} / {key[1]}: {pos} of {len(sel)} configurations have Δκ > 0 with a CI excluding 0, "
                  f"{neg} have Δκ < 0; median Δκ {aps._r(statistics.median(r['delta'] for r in sel))}")
    md.append("")


# ── 2. Paired Δκ between prompts and configurations ─────────────────────────

def pairwise(md: list[str]) -> None:
    rows = []
    rng = random.Random(42)
    sets = {
        "DACOS selection": ({p: ANNOT / "rq3" / f"rq3_AZ1_{p}_DACOS.json" for p in MANUAL},
                            ANNOT / "rq5" / f"rq5_DACOS_{SELECTED}.json"),
        "DACOS held-out": ({p: ANNOT / "rq5" / "heldout" / f"rq5ho_AZ1_{p}.json" for p in MANUAL},
                           ANNOT / "rq5" / "heldout" / f"rq5ho_AZ1_{SELECTED}.json"),
        "MLCQ": ({p: MAIN / "rq3" / f"rq3_AZ1_{p}_MLCQ.json" for p in MANUAL},
                 ANNOT / "rq5" / f"rq5_MLCQ_{SELECTED}.json"),
    }
    for split, (manual, gen_file) in sets.items():
        gen = _rows(gen_file)
        for p, f in manual.items():
            man = _rows(f)
            ids = sorted(set(gen) & set(man))
            obs, lo, hi = _boot_delta(ids, lambda s: _kappa([gen[i][0] for i in s], [gen[i][1] for i in s]),
                                      lambda s, man=man: _kappa([man[i][0] for i in s], [man[i][1] for i in s]), rng)
            rows.append({"comparison": "RQ5 selected vs manual", "split": split, "a": SELECTED, "b": p,
                         "n": len(ids), "delta_kappa": aps._r(obs), "ci_lo": aps._r(lo), "ci_hi": aps._r(hi),
                         "ci_excludes_0": lo > 0 or hi < 0})
    for ds in ("DACOS", "MLCQ"):
        cfg = _configs(ds)
        k = {n: _kappa([v[0] for v in r.values()], [v[1] for v in r.values()]) for n, r in cfg.items()}
        best = max(k, key=k.get)
        for n, r in cfg.items():
            if n == best:
                continue
            b = cfg[best]
            ids = sorted(set(b) & set(r))
            obs, lo, hi = _boot_delta(ids, lambda s: _kappa([b[i][0] for i in s], [b[i][1] for i in s]),
                                      lambda s, r=r: _kappa([r[i][0] for i in s], [r[i][1] for i in s]), rng)
            rows.append({"comparison": "RQ3 best vs other", "split": ds, "a": best, "b": n, "n": len(ids),
                         "delta_kappa": aps._r(obs), "ci_lo": aps._r(lo), "ci_hi": aps._r(hi),
                         "ci_excludes_0": lo > 0 or hi < 0})
    aps._write_csv(STATS / "smell_pairwise_kappa.csv", rows)
    md.append("## 2. Paired bootstrap of Δκ between prompts and configurations\n")
    md.append("| Comparison | Split | A | B | Δκ (A − B) [95% CI] |\n|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['comparison']} | {r['split']} | {r['a']} | {r['b']} | {r['delta_kappa']} [{r['ci_lo']}, {r['ci_hi']}] |")
    md.append("")
    for comp in ("RQ5 selected vs manual", "RQ3 best vs other"):
        sel = [r for r in rows if r["comparison"] == comp]
        md.append(f"- {comp}: {sum(r['ci_excludes_0'] for r in sel)} of {len(sel)} intervals exclude 0")
    md.append("")


# ── 3. RQ5 repeated splits by kappa, with chance references ─────────────────

def rq5_splits(md: list[str], repeats: int = 1000) -> None:
    prompts = {f"manual:{p}": _rows(ANNOT / "rq3" / f"rq3_AZ1_{p}_DACOS.json") for p in MANUAL}
    prompts.update({f"generated:{f.stem.replace('rq5_DACOS_', '')}": _rows(f)
                    for f in sorted((ANNOT / "rq5").glob("rq5_DACOS_*.json"))})
    names = sorted(prompts)
    gold = {i: v[0] for i, v in prompts[names[0]].items()}
    pred = {n: {i: v[1] for i, v in prompts[n].items()} for n in names}
    pos = sorted(i for i, g in gold.items() if g)
    neg = sorted(i for i, g in gold.items() if not g)

    def simulate(pred_: dict[str, dict[str, bool]], seed: int) -> tuple[float, float]:
        rng = random.Random(seed)
        stay, optimism = 0, []
        p_, n_ = pos[:], neg[:]
        for _ in range(repeats):
            rng.shuffle(p_)
            rng.shuffle(n_)
            a = p_[: len(p_) // 2] + n_[: len(n_) // 2]
            b = p_[len(p_) // 2:] + n_[len(n_) // 2:]
            sa = {n: _kappa([gold[i] for i in a], [pred_[n][i] for i in a]) for n in names}
            sb = {n: _kappa([gold[i] for i in b], [pred_[n][i] for i in b]) for n in names}
            w = max(sa, key=sa.get)
            stay += sb[w] >= max(sb.values())
            optimism.append(sa[w] - sb[w])
        return stay / repeats, statistics.mean(optimism)

    observed_stay, observed_opt = simulate(pred, 42)
    null_stay, null_opt = [], []
    rng = random.Random(7)
    for b in range(100):
        permuted = {n: {} for n in names}
        for i in gold:
            vals = [pred[n][i] for n in names]
            rng.shuffle(vals)
            for n, v in zip(names, vals):
                permuted[n][i] = v
        s, o = _simulate_light(permuted, names, gold, pos, neg, 1000 + b)
        null_stay.append(s)
        null_opt.append(o)
    spread = []
    for p in MANUAL:
        ks = []
        for suffix in ("", "_r2", "_r3"):
            f = ANNOT / "rq3" / f"rq3_AZ1_{p}_DACOS{suffix}.json"
            if f.exists():
                ks.append(aps._load(f)["metrics"]["kappa"])
        spread.append({"prompt": p, "kappa_runs": [aps._r(k) for k in ks], "range": aps._r(max(ks) - min(ks))})
    gain = aps._load(ANNOT / "rq5" / f"rq5_DACOS_{SELECTED}.json")["metrics"]["kappa"] - max(
        aps._load(ANNOT / "rq3" / f"rq3_AZ1_{p}_DACOS.json")["metrics"]["kappa"] for p in MANUAL)
    row = {"prompts": len(names), "repeats": repeats, "winner_stays_best": aps._r(observed_stay),
           "chance_1_over_k": aps._r(1 / len(names)),
           "null_exchangeable_median": aps._r(statistics.median(null_stay)),
           "null_exchangeable_ci": f"[{aps._r(_q(null_stay, 0.025))}, {aps._r(_q(null_stay, 0.975))}]",
           "optimism_mean": aps._r(observed_opt), "null_optimism_median": aps._r(statistics.median(null_opt)),
           "selection_gain_kappa": aps._r(gain),
           "gpt4o_run_ranges": "; ".join(f"{s['prompt']} {s['kappa_runs']} range {s['range']}" for s in spread)}
    aps._write_csv(STATS / "smell_rq5_splits.csv", [row])
    md.append("## 3. RQ5 — repeated stratified splits of the selection sample, selected by κ\n")
    md.append(f"{repeats} splits into two halves; the winner by κ on one half is evaluated on the other. The halves "
              "resample the same 200 samples, so they are not independent experiments. Null: the predictions of "
              "the eight prompts are permuted among prompts within each sample (100 permutations, "
              f"{repeats // 10} splits each), so that the prompts are exchangeable and the dependence between "
              "halves is kept.\n")
    for k, v in row.items():
        md.append(f"- {k}: {v}")
    md.append("")


def _simulate_light(pred_, names, gold, pos, neg, seed, repeats: int = 100) -> tuple[float, float]:
    rng = random.Random(seed)
    stay, optimism = 0, []
    p_, n_ = pos[:], neg[:]
    for _ in range(repeats):
        rng.shuffle(p_)
        rng.shuffle(n_)
        a = p_[: len(p_) // 2] + n_[: len(n_) // 2]
        b = p_[len(p_) // 2:] + n_[len(n_) // 2:]
        sa = {n: _kappa([gold[i] for i in a], [pred_[n][i] for i in a]) for n in names}
        sb = {n: _kappa([gold[i] for i in b], [pred_[n][i] for i in b]) for n in names}
        w = max(sa, key=sa.get)
        stay += sb[w] >= max(sb.values())
        optimism.append(sa[w] - sb[w])
    return stay / repeats, statistics.mean(optimism)


# ── 4. Metric heuristic on the held-out split ───────────────────────────────

def heldout_heuristic(md: list[str]) -> None:
    rows = []
    with open(ANNOT / "summary_rq3_baselines.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset"] in ("DACOS", "DACOS_heldout") and r["method"] == "metric_heuristic":
                rows.append({"split": r["dataset"], "method": "metric heuristic", "kappa": r["kappa"], "f1": r["f1"]})
    for f in sorted((ANNOT / "rq5" / "heldout").glob("rq5ho_*.json")):
        m = aps._load(f)["metrics"]
        rows.append({"split": "DACOS_heldout", "method": f.stem.replace("rq5ho_", ""),
                     "kappa": aps._r(m["kappa"]), "f1": aps._r(m["f1"])})
    aps._write_csv(STATS / "smell_heldout_heuristic.csv", rows)
    md.append("## 4. Metric heuristic vs LLM prompts, selection and held-out DACOS\n")
    md.append("| Split | Method | κ | F1 |\n|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['split']} | {r['method']} | {r['kappa']} | {r['f1']} |")
    md.append("")


# ── 5. MLCQ with the reviewers' majority ────────────────────────────────────

def mlcq_majority(md: list[str]) -> None:
    with open(Path("data/apps/MLCQ/MLCQCodeSmellSamples.csv"), newline="", encoding="utf-8") as f:
        allrows = list(csv.DictReader(f, delimiter=";"))
    by_id = {r["id"]: r for r in allrows}
    votes: dict[tuple, list[int]] = defaultdict(list)
    for r in allrows:
        votes[(r["sample_id"], r["smell"])].append(0 if r["severity"].strip().lower() == "none" else 1)
    cfg = _configs("MLCQ")
    first = next(iter(cfg.values()))
    majority, changed, ties = {}, 0, 0
    for sid, (g, _) in first.items():
        r = by_id.get(sid)
        v = votes[(r["sample_id"], r["smell"])] if r else []
        if len(v) >= 2 and sum(v) * 2 == len(v):
            ties += 1
            majority[sid] = g
        elif len(v) >= 2:
            majority[sid] = sum(v) * 2 > len(v)
        else:
            majority[sid] = g
        changed += majority[sid] != g
    rows, k_orig, k_maj = [], [], []
    for n, r in cfg.items():
        ids = sorted(r)
        ko = _kappa([r[i][0] for i in ids], [r[i][1] for i in ids])
        km = _kappa([majority[i] for i in ids], [r[i][1] for i in ids])
        k_orig.append(ko)
        k_maj.append(km)
        rows.append({"config": n, "kappa_row_label": aps._r(ko), "kappa_majority": aps._r(km)})
    names = list(cfg)
    summary = {"labels_changed": changed, "ties_kept_row_label": ties,
               "positives_row_label": sum(v[0] for v in first.values()), "positives_majority": sum(majority.values()),
               "rho_rankings": aps._r(_rho(k_orig, k_maj)),
               "best_row_label": names[k_orig.index(max(k_orig))], "best_majority": names[k_maj.index(max(k_maj))]}
    aps._write_csv(STATS / "smell_mlcq_majority.csv", rows)
    md.append("## 5. MLCQ — row labels vs the reviewers' majority\n")
    md.append("Majority over all reviewers of the (fragment, smell) pair in the MLCQ export; ties keep the row label.\n")
    for k, v in summary.items():
        md.append(f"- {k}: {v}")
    md.append("\n| Configuration | κ (row label) | κ (majority) |\n|---|---|---|")
    for r in rows:
        md.append(f"| {r['config']} | {r['kappa_row_label']} | {r['kappa_majority']} |")
    md.append("")


# ── 6. DACOS inter-annotator agreement ──────────────────────────────────────

def dacos_alpha(md: list[str]) -> None:
    from StaLLM_benchmarks import _iter_sql_table
    sql = Path("data/apps/DACOS/DACOSMain.sql").read_text(encoding="utf-8", errors="ignore")
    samples = {r[0]: r for r in _iter_sql_table(sql, "sample") if len(r) >= 8}
    ann = defaultdict(list)
    for r in _iter_sql_table(sql, "annotation"):
        ann[r[6]].append(r)
    rows = []
    for sid_smell, col, name in (("1", 1, "complex_method"), ("2", 3, "long_parameter_list")):
        units, two = [], []
        for sid, rs in ann.items():
            tagged = re.findall(r"\d+", str(samples.get(sid, [None] * 8)[7] or ""))
            if sid_smell not in tagged or len(rs) < 2:
                continue
            u = [int(r[col]) for r in rs]
            units.append(u)
            if len(u) == 2:
                two.append(u)
        agree = sum(len(set(u)) == 1 for u in units) / len(units)
        rows.append({"smell": name, "units": len(units), "pct_agreement": aps._r(agree),
                     "krippendorff_alpha": aps._r(_alpha_nominal(units)),
                     "ties_two_annotators": sum(len(set(u)) > 1 for u in two)})
    aps._write_csv(STATS / "smell_dacos_agreement.csv", rows)
    md.append("## 6. DACOS inter-annotator agreement (samples selected for the smell, two or more annotators)\n")
    md.append("| Smell | Units | Agreement | Krippendorff's α | Two-annotator ties |\n|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['smell']} | {r['units']} | {r['pct_agreement']} | {r['krippendorff_alpha']} | {r['ties_two_annotators']} |")
    md.append("")


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    md = ["# Smell-classification analyses (RQ3, RQ5)\n", "Computed from stored outputs, no model call.\n"]
    for step in (oracle_shift, pairwise, rq5_splits, heldout_heuristic, mlcq_majority, dacos_alpha):
        step(md)
        print(f"done: {step.__name__}")
    (STATS / "smell_analyses.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"→ {STATS / 'smell_analyses.md'}")


if __name__ == "__main__":
    main()
