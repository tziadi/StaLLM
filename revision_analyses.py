"""Analyses added for the fourth review round, computed from stored outputs only
(no model call).

    python revision_analyses.py

Writes to output/paper_results/stats/ (span level) and
output/paper_results_annot/stats/ (smells): revision_*.csv and revision_analyses.md.

1. Random-placement test of every RQ1/RQ2 configuration with 10,000 draws.
   The statistic is the F1 averaged over the runs of a configuration; each draw
   places every span of every run at random (same file, same length) and the
   null value is the run-averaged random F1. Holm within each protocol and over
   all tests.
2. Direct test of the protocol effect. For each configuration run under both
   protocols, the excess over chance is F1 minus the expected random F1, with
   both computed from counts pooled over the runs. Delta = excess(numbered) -
   excess(original). 95% interval by bootstrap over the 20 files (the same
   resampled files for both protocols). Expected random counts per file come
   from 500 placements.
3. Structural non-LLM span baseline: one single-line prediction at every member
   signature and at every line with `throw` or `catch`, scored with the lenient
   preset, and its own random-placement test.
4. Matthews correlation coefficient and balanced accuracy for every RQ3
   configuration (mean over runs), against the human labels.
"""
from __future__ import annotations

import csv
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path

import analyze_paper_stats as aps
import structured_baselines as sb

SPAN_STATS = aps.OUT / "stats"
SMELL_STATS = Path("output/paper_results_annot/stats")
DRAWS = 10000
BOOT = 5000


def _counts(spans_one: dict, preds_one: dict, match) -> tuple[int, int, int]:
    return aps._span_counts(spans_one, preds_one, match)


def _f1(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def chance_tests(md: list[str], groups: dict, match) -> None:
    out = []
    for key in sorted(groups):
        runs = groups[key]
        observed = sb.llm_f1(runs, match)
        rng = random.Random(42)
        k = 0
        null_sum = 0.0
        for _ in range(DRAWS):
            v = statistics.mean(sb.placement_f1(s, None, rng, match) for s in runs)
            null_sum += v
            k += v >= observed
        p = (1 + k) / (1 + DRAWS)
        out.append({"protocol": key[0], "slot": key[1], "prompt": key[2], "runs": len(runs),
                    "f1": aps._r(observed), "random": aps._r(null_sum / DRAWS), "p": p})
        print(f"  chance {key}: p={p:.4f}")
    for scope, sel in (("protocol", lambda r: r["protocol"]), ("all", lambda r: "all")):
        by = defaultdict(list)
        for r in out:
            by[sel(r)].append(r)
        for rs in by.values():
            for r, a in zip(rs, aps.holm([r["p"] for r in rs])):
                r[f"p_holm_{scope}"] = aps._r(a, 4)
    for r in out:
        r["p"] = aps._r(r["p"], 4)
    aps._write_csv(SPAN_STATS / "revision_chance_10000.csv", out)
    md.append(f"## 1. Random-placement tests, {DRAWS:,} draws\n")
    md.append("| Protocol | Slot | Prompt | F1 | random | p | Holm (protocol) | Holm (all) |\n|---|---|---|---|---|---|---|---|")
    for r in out:
        md.append(f"| {r['protocol']} | {r['slot']} | {r['prompt']} | {r['f1']} | {r['random']} | {r['p']} "
                  f"| {r['p_holm_protocol']} | {r['p_holm_all']} |")
    md.append("")


def protocol_effect(md: list[str], groups: dict, match, placements: int = 500) -> None:
    def per_file(runs):
        files = sorted(runs[0])
        llm = {b: [0.0, 0.0, 0.0] for b in files}
        rnd = {b: [0.0, 0.0, 0.0] for b in files}
        rng = random.Random(7)
        for s in runs:
            for b in files:
                one = {b: s[b]}
                c = _counts(one, {b: [{"startLine": x, "endLine": y} for x, y in s[b]["pred"]]}, match)
                for i in range(3):
                    llm[b][i] += c[i]
                acc = [0, 0, 0]
                n_lines = s[b]["lines"]
                for _ in range(placements):
                    preds = []
                    for x, y in s[b]["pred"]:
                        length = min(max(0, y - x), n_lines - 1)
                        st = rng.randint(1, max(1, n_lines - length))
                        preds.append({"startLine": st, "endLine": st + length})
                    c = _counts(one, {b: preds}, match)
                    for i in range(3):
                        acc[i] += c[i]
                for i in range(3):
                    rnd[b][i] += acc[i] / placements
        return llm, rnd

    def excess(llm, rnd, files):
        a = [sum(llm[b][i] for b in files) for i in range(3)]
        r = [sum(rnd[b][i] for b in files) for i in range(3)]
        return _f1(*a) - _f1(*r)

    out = []
    pairs = sorted({k[1:] for k in groups if ("original",) + k[1:] in groups and ("numbered",) + k[1:] in groups})
    for slot, prompt in pairs:
        lo_, ro_ = per_file(groups[("original", slot, prompt)])
        ln_, rn_ = per_file(groups[("numbered", slot, prompt)])
        files = sorted(lo_)
        obs = excess(ln_, rn_, files) - excess(lo_, ro_, files)
        rng = random.Random(42)
        deltas = []
        for _ in range(BOOT):
            s = [rng.choice(files) for _ in files]
            deltas.append(excess(ln_, rn_, s) - excess(lo_, ro_, s))
        deltas.sort()
        lo, hi = deltas[int(0.025 * BOOT)], deltas[int(0.975 * BOOT) - 1]
        p_two = 2 * min(sum(d <= 0 for d in deltas), sum(d >= 0 for d in deltas)) / BOOT
        out.append({"slot": slot, "prompt": prompt, "excess_original": aps._r(excess(lo_, ro_, files)),
                    "excess_numbered": aps._r(excess(ln_, rn_, files)), "delta": aps._r(obs),
                    "ci_low": aps._r(lo), "ci_high": aps._r(hi), "p_boot": aps._r(min(1.0, p_two), 4)})
        print(f"  protocol effect {slot} {prompt}: delta={obs:.3f} [{lo:.3f}, {hi:.3f}]")
    for r, a in zip(out, aps.holm([r["p_boot"] for r in out])):
        r["p_holm"] = aps._r(a, 4)
    aps._write_csv(SPAN_STATS / "revision_protocol_effect.csv", out)
    md.append("## 2. Protocol effect on the excess over chance (numbered minus original)\n")
    md.append(f"Bootstrap over files, {BOOT:,} resamples; expected random counts from {placements} placements per file and run.\n")
    md.append("| Slot | Prompt | excess original | excess numbered | delta | 95% CI | p | Holm |\n|---|---|---|---|---|---|---|---|")
    for r in out:
        md.append(f"| {r['slot']} | {r['prompt']} | {r['excess_original']} | {r['excess_numbered']} | {r['delta']} "
                  f"| [{r['ci_low']}, {r['ci_high']}] | {r['p_boot']} | {r['p_holm']} |")
    md.append("")


_THROW_CATCH = re.compile(r"\b(throw|catch)\b")


def structural_baseline(md: list[str], groups: dict, match) -> None:
    sources = sb.load_sources()
    any_run = next(iter(groups.values()))[0]
    spans = {}
    for b, f in any_run.items():
        text = sources[b]
        lines = set()
        for i, raw in enumerate(text, start=1):
            s = raw.split("//", 1)[0].rstrip()
            if _THROW_CATCH.search(s):
                lines.add(i)
            elif (sb._SIGNATURE.search(s) and not sb._CONTROL.match(s) and not sb._TYPE_DECL.search(s)
                  and "(" in s):
                lines.add(i)
        spans[b] = {"lines": f["lines"], "gt": f["gt"], "pred": [(x, x) for x in sorted(lines)]}
    preds = {b: [{"startLine": x, "endLine": y} for x, y in v["pred"]] for b, v in spans.items()}
    tp, fp, fn = aps._span_counts(spans, preds, match)
    f1 = _f1(tp, fp, fn)
    rng = random.Random(42)
    null = [sb.placement_f1(spans, None, rng, match) for _ in range(DRAWS)]
    p = (1 + sum(x >= f1 for x in null)) / (1 + DRAWS)
    n = sum(len(v["pred"]) for v in spans.values())
    row = {"predictions": n, "tp": tp, "fp": fp, "fn": fn, "precision": aps._r(tp / (tp + fp) if tp + fp else 0),
           "recall": aps._r(tp / (tp + fn) if tp + fn else 0), "f1": aps._r(f1),
           "random": aps._r(statistics.mean(null)), "p": aps._r(p, 4)}
    aps._write_csv(SPAN_STATS / "revision_structural_baseline.csv", [row])
    print(f"  structural baseline: {row}")
    md.append("## 3. Structural non-LLM span baseline (signatures, throw, catch)\n")
    md.append("| predictions | P | R | F1 | random | p |\n|---|---|---|---|---|---|")
    md.append(f"| {n} | {row['precision']} | {row['recall']} | {row['f1']} | {row['random']} | {row['p']} |\n")


def _mcc_bacc(gold: list[bool], pred: list[bool]) -> tuple[float, float]:
    tp = sum(g and p for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    fp = sum(p and not g for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return mcc, (tpr + tnr) / 2


def smell_mcc(md: list[str]) -> None:
    out = []
    for dataset, root in (("DACOS", Path("output/paper_results_annot")), ("MLCQ", aps.OUT)):
        by = defaultdict(list)
        for f in sorted((root / "rq3").glob(f"rq3_*_{dataset}*.json")):
            d = aps._load(f)
            rows = d["rows"]
            by[(d["slot"], d["prompt"])].append(_mcc_bacc([bool(r["gold"]) for r in rows],
                                                          [bool(r["predicted"]) for r in rows]))
        for (slot, prompt), vals in sorted(by.items()):
            out.append({"dataset": dataset, "slot": slot, "prompt": prompt, "runs": len(vals),
                        "mcc": aps._r(statistics.mean(v[0] for v in vals)),
                        "balanced_accuracy": aps._r(statistics.mean(v[1] for v in vals))})
    aps._write_csv(SMELL_STATS / "revision_mcc.csv", out)
    md.append("## 4. MCC and balanced accuracy of the RQ3 configurations (mean over runs)\n")
    md.append("| Dataset | Slot | Prompt | runs | MCC | balanced accuracy |\n|---|---|---|---|---|---|")
    for r in out:
        md.append(f"| {r['dataset']} | {r['slot']} | {r['prompt']} | {r['runs']} | {r['mcc']} | {r['balanced_accuracy']} |")
    md.append("")


def main() -> None:
    md = ["# Revision analyses\n", "Computed from stored outputs, no model call.\n"]
    groups = sb.load_groups()
    match = sb.matcher("Lenient")
    smell_mcc(md)
    structural_baseline(md, groups, match)
    protocol_effect(md, groups, match)
    chance_tests(md, groups, match)
    (SPAN_STATS / "revision_analyses.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"→ {SPAN_STATS / 'revision_analyses.md'}")


if __name__ == "__main__":
    main()
