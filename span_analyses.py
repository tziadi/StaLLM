"""Multiplicity control and category-aware matching for the span-level runs
(RQ1/RQ2), computed from the stored outputs only (no model call).

    python span_analyses.py

Writes to output/paper_results/stats/: span_holm.csv, span_borderline.csv,
span_category_matching.csv and span_analyses.md.

The random-placement p of the paper (analyze_paper_stats.analyze_rq12) is
p = (1 + #draws whose F1, averaged over the runs of the configuration, reaches
the configuration's F1 averaged over its runs) / (1 + draws). Each draw moves
every predicted span of every run to a random line of the same file, keeping
its length. The test unit is therefore the mean F1 over the runs.

1. Holm adjustment of these p values within each protocol (9 configurations)
   and over all tests of both protocols.
2. Borderline configurations rerun with 10,000 draws.
3. Category-aware matching: a lenient match also requires that the keyword
   category of the prediction's description (structured_baselines.DESC_KEYWORDS)
   equals the category of the finding's rule (RULE_CATEGORY). Descriptions are
   recovered from llm_debug/ by matching the prediction sets. Variant A counts
   unclassified or unrecovered predictions as non-matching, variant B drops
   them. Chance baseline: random placement of the same spans, which keep
   their category.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import analyze_paper_stats as aps
import structured_baselines as sb

STATS = aps.OUT / "stats"
BORDERLINE = [("numbered", "AZ1", "baseline", "Lenient"), ("numbered", "AZ1", "hybrid", "Lenient"),
              ("numbered", "AZ2", "scanner", "Lenient"), ("numbered", "AZ4", "scanner", "Lenient"),
              ("numbered", "AZ8", "scanner", "Lenient"),
              ("numbered", "AZ1", "hybrid", "Strict")]


def holm(md: list[str]) -> None:
    with open(STATS / "rq12_rerun.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for scope, groups in (("per protocol", {p: [r for r in rows if r["protocol"] == p] for p in ("original", "numbered")}),
                          (f"all {len(rows)}", {"all": rows})):
        for _, rs in groups.items():
            adj = aps.holm([float(r["p_random"]) for r in rs])
            for r, a in zip(rs, adj):
                out.append({"scope": scope, "protocol": r["protocol"], "slot": r["slot"], "prompt": r["prompt"],
                            "f1": r["f1_mean"], "random": r["f1_random_mean"], "p": r["p_random"],
                            "p_holm": aps._r(a, 4), "significant_holm": a < 0.05})
    aps._write_csv(STATS / "span_holm.csv", out)
    md.append("## 1. Random-placement tests with Holm adjustment\n")
    md.append("p = (1 + #draws whose run-averaged random F1 ≥ the run-averaged F1) / (1 + 1,000 draws).\n")
    md.append("| Scope | Protocol | Slot | Prompt | F1 | random | p | p (Holm) |\n|---|---|---|---|---|---|---|---|")
    for r in out:
        md.append(f"| {r['scope']} | {r['protocol']} | {r['slot']} | {r['prompt']} | {r['f1']} | {r['random']} "
                  f"| {r['p']} | {r['p_holm']}{' *' if r['significant_holm'] else ''} |")
    md.append("")


def borderline(md: list[str], draws: int = 10000) -> None:
    groups = sb.load_groups()
    out = []
    for protocol, slot, prompt, preset in BORDERLINE:
        runs = groups.get((protocol, slot, prompt))
        if not runs:
            continue
        match = sb.matcher(preset)
        observed = sb.llm_f1(runs, match)
        rng = random.Random(42)
        vals = [statistics.mean(sb.placement_f1(s, None, rng, match) for s in runs) for _ in range(draws)]
        k = sum(x >= observed for x in vals)
        p = (1 + k) / (1 + draws)
        se = (p * (1 - p) / draws) ** 0.5
        out.append({"protocol": protocol, "slot": slot, "prompt": prompt, "preset": preset, "runs": len(runs),
                    "f1": aps._r(observed), "random": aps._r(statistics.mean(vals)), "draws": draws,
                    "p": aps._r(p, 4), "mc_se": aps._r(se, 4)})
        print(f"  borderline {protocol} {slot} {prompt} {preset}: p={p:.4f}")
    aps._write_csv(STATS / "span_borderline.csv", out)
    md.append(f"## 2. Borderline configurations with {draws:,} draws\n")
    md.append("| Protocol | Slot | Prompt | Preset | F1 | random | p | Monte Carlo SE |\n|---|---|---|---|---|---|---|---|")
    for r in out:
        md.append(f"| {r['protocol']} | {r['slot']} | {r['prompt']} | {r['preset']} | {r['f1']} | {r['random']} "
                  f"| {r['p']} | {r['mc_se']} |")
    md.append("")


def _typed_oracle() -> dict[str, list[tuple[int, int, str]]]:
    from StaLLM_core import _build_universe, load_ground_truth_spans
    gt = load_ground_truth_spans(str(aps.RQ12_CSV), allowed_exts=[".cs"])
    universe, _, aligned = _build_universe(str(aps.RQ12_ZIP), gt, [".cs"], 20, 1.0)
    keys = set(universe["basename"])
    oracle: dict[str, list] = {k: [] for k in keys}
    for _, r in aligned[aligned["basename"].isin(keys)].iterrows():
        oracle[r["basename"]].append((int(r["startLine"]), int(r["endLine"]),
                                      "rule:" + sb.RULE_CATEGORY.get(r["type"], "other")))
    return oracle


def _answers() -> dict[frozenset, list]:
    from StaLLM_core import _coerce_items_schema
    answers: dict[frozenset, list] = {}
    for f in Path("llm_debug").glob("*_sanitized.txt"):
        try:
            items = _coerce_items_schema(json.loads(f.read_text()))
        except Exception:
            continue
        spans = [(int(i.get("startLine") or i.get("line") or 0),
                  int(i.get("endLine") or i.get("startLine") or i.get("line") or 0),
                  str(i.get("description") or "")) for i in items]
        answers.setdefault(frozenset((s, e) for s, e, _ in spans), spans)
    return answers


def category_matching(md: list[str], draws: int = 1000) -> None:
    from StaLLM_core import _match_file

    def match(p, g):
        return _match_file([dict(x, startColumn=None, endColumn=None) for x in p],
                           [dict(x, startColumn=None, endColumn=None) for x in g],
                           require_type=True, use_cols_single=False, preset="Lenient")

    oracle = _typed_oracle()
    answers = _answers()
    groups: dict[tuple, list] = defaultdict(list)
    coverage = defaultdict(lambda: [0, 0, 0])   # predictions, recovered, classified
    for protocol, folder in aps.RQ12_DIRS.items():
        for f in sorted(folder.glob("rq2_*.json")):
            d = aps._load(f)
            run = {}
            for b, fs in d["metrics"]["spans_by_file"].items():
                preds = [tuple(x) for x in fs["pred"]]
                desc = answers.get(frozenset(preds)) if preds else []
                by_span = defaultdict(list)
                for s, e, t in desc or []:
                    by_span[(s, e)].append(sb.describe_category(t))
                typed = []
                for s, e in preds:
                    cats = by_span.get((s, e))
                    cat = cats.pop(0) if cats else None
                    typed.append((s, e, cat))
                key = (protocol, d["slot"], d["prompt"])
                coverage[key][0] += len(typed)
                coverage[key][1] += sum(c is not None for _, _, c in typed)
                coverage[key][2] += sum(c not in (None, "other") for _, _, c in typed)
                run[b] = {"lines": fs["lines"], "pred": typed, "gt": oracle[b]}
            groups[(protocol, d["slot"], d["prompt"])].append(run)

    def f1_of(runs, variant: str, rng: random.Random | None = None) -> float:
        vals = []
        for run in runs:
            tp = fp = fn = 0
            for b, fs in run.items():
                preds = []
                for s, e, cat in fs["pred"]:
                    if cat in (None, "other"):
                        if variant == "B":
                            continue
                        cat = "unclassified"
                    length = e - s
                    if rng is not None:
                        length = min(max(0, length), fs["lines"] - 1)
                        s = rng.randint(1, max(1, fs["lines"] - length))
                        e = s + length
                    preds.append({"startLine": s, "endLine": e, "type": "rule:" + cat})
                a, p_, n = match(preds, [{"startLine": s, "endLine": e, "type": t} for s, e, t in fs["gt"]])
                tp, fp, fn = tp + a, fp + p_, fn + n
            vals.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        return statistics.mean(vals)

    out = []
    for key, runs in sorted(groups.items()):
        n, rec, cls = coverage[key]
        row = {"protocol": key[0], "slot": key[1], "prompt": key[2], "runs": len(runs), "predictions": n,
               "recovered": aps._r(rec / n if n else 0.0), "classified": aps._r(cls / n if n else 0.0)}
        for variant in ("A", "B"):
            obs = f1_of(runs, variant)
            rng = random.Random(42)
            null = [f1_of(runs, variant, rng) for _ in range(draws)]
            p = (1 + sum(x >= obs for x in null)) / (1 + draws)
            row.update({f"f1_{variant}": aps._r(obs), f"random_{variant}": aps._r(statistics.mean(null)),
                        f"p_{variant}": aps._r(p, 4)})
        out.append(row)
        print(f"  category {key}: A p={row['p_A']} B p={row['p_B']}")
    aps._write_csv(STATS / "span_category_matching.csv", out)
    md.append("## 3. Category-aware lenient matching\n")
    md.append("A prediction matches a finding only if, in addition to the lenient line rule, the keyword category "
              "of its description equals the category of the finding's rule. Variant A: unclassified or unrecovered "
              "predictions never match. Variant B: they are dropped. Random placement keeps each span's category. "
              f"{draws} draws.\n")
    md.append("| Protocol | Slot | Prompt | recovered | classified | F1 A | random A | p A | F1 B | random B | p B |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    for r in out:
        md.append(f"| {r['protocol']} | {r['slot']} | {r['prompt']} | {r['recovered']} | {r['classified']} "
                  f"| {r['f1_A']} | {r['random_A']} | {r['p_A']} | {r['f1_B']} | {r['random_B']} | {r['p_B']} |")
    md.append("")


def main() -> None:
    md = ["# Span-level analyses (RQ1/RQ2)\n", "Computed from stored outputs, no model call.\n"]
    for step in (holm, borderline, category_matching):
        step(md)
        print(f"done: {step.__name__}")
    (STATS / "span_analyses.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"→ {STATS / 'span_analyses.md'}")


if __name__ == "__main__":
    main()
