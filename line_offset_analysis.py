"""
StaLLM — line offsets of span-level predictions (RQ1/RQ2)
==========================================================
Re-uses recorded predictions only; no model is called.

For every predicted span in a file that has oracle spans, the signed offset is
the predicted start line minus the start line of the nearest oracle span. If the
model locates the right code but miscounts lines, offsets concentrate near zero
without being zero (near misses) and may drift with the position in the file.
Each configuration is compared with random placement of its own spans.

Usage:
    python line_offset_analysis.py   # writes output/paper_results/stats/line_offsets*.csv and line_offsets.md
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict

from scipy.stats import binomtest, spearmanr

import analyze_paper_stats as aps

STATS = aps.STATS
DRAWS = 200
NEAR = 5  # near miss: 1 <= |offset| <= NEAR


def _offsets(spans_by_file: dict, preds_by_file: dict) -> list[tuple[int, float]]:
    """(signed offset, relative position in the file) of every prediction in a file with oracle spans."""
    out = []
    for b, f in spans_by_file.items():
        starts = [s for s, _ in f["gt"]]
        if not starts:
            continue
        for s, _ in preds_by_file.get(b, []):
            nearest = min(starts, key=lambda g: (abs(s - g), g))
            out.append((s - nearest, s / max(1, f["lines"])))
    return out


def _summary(offs: list[tuple[int, float]]) -> dict:
    if not offs:
        return {"n": 0}
    o = [x for x, _ in offs]
    return {
        "n": len(o),
        "exact": sum(x == 0 for x in o) / len(o),
        "near": sum(1 <= abs(x) <= NEAR for x in o) / len(o),
        "within_near": sum(abs(x) <= NEAR for x in o) / len(o),
        "median_abs": statistics.median(abs(x) for x in o),
        "median_signed_near": statistics.median([x for x in o if abs(x) <= 20] or [0]),
    }


def _random_preds(spans_by_file: dict, rng: random.Random) -> dict:
    preds = defaultdict(list)
    for b, f in spans_by_file.items():
        for s, e in f["pred"]:
            length = min(max(0, e - s), f["lines"] - 1)
            start = rng.randint(1, max(1, f["lines"] - length))
            preds[b].append((start, start + length))
    return preds


BOOT = 2000
TOLERANCES = (0, 1, 2)


def _per_file(spans: list[dict], rng: random.Random) -> list[dict]:
    """Per file, pooled over runs: model offsets and the expected hits of random placement.
    Files are the resampling unit, so the near-identical predictions of repeated runs
    at temperature 0 are not counted as independent observations."""
    files = []
    for b in sorted(spans[0]):
        starts = [s for s, _ in spans[0][b]["gt"]]
        if not starts:
            continue
        model, rand_hits = [], {k: 0.0 for k in TOLERANCES}
        for s in spans:
            f = s[b]
            model += [x for x, _ in _offsets({b: f}, {b: f["pred"]})]
            for _ in range(50):
                r = [x for x, _ in _offsets({b: f}, _random_preds({b: f}, rng))]
                for k in TOLERANCES:
                    rand_hits[k] += sum(abs(x) <= k for x in r) / 50
        if model:
            files.append({"model": model, "rand_hits": rand_hits})
    return files


def _bootstrap(files: list[dict], rng: random.Random) -> dict:
    """Point estimates and 95% file-level bootstrap intervals for (i) the excess of
    predictions within k lines of an oracle start over random placement and (ii) the
    share of misplaced predictions (0 < |offset| <= 20) that lie before the oracle line."""
    def stats(sample):
        n = sum(len(f["model"]) for f in sample)
        out = {f"excess_{k}": (sum(sum(abs(x) <= k for x in f["model"]) for f in sample)
                               - sum(f["rand_hits"][k] for f in sample)) / n for k in TOLERANCES}
        neg = sum(sum(-20 <= x < 0 for x in f["model"]) for f in sample)
        pos = sum(sum(0 < x <= 20 for x in f["model"]) for f in sample)
        out["before_share"] = neg / (neg + pos) if neg + pos else float("nan")
        return out
    point = stats(files)
    boots = [stats([rng.choice(files) for _ in files]) for _ in range(BOOT)]
    res = {"files": len(files)}
    for key, v in point.items():
        vals = sorted(b[key] for b in boots if b[key] == b[key])
        res[key] = aps._r(v)
        res[key + "_ci"] = f"[{aps._r(vals[int(0.025 * len(vals))])}, {aps._r(vals[int(0.975 * len(vals)) - 1])}]"
    return res


def main() -> None:
    oracle = aps._rq12_oracle()
    groups: dict[tuple, list] = defaultdict(list)
    for protocol, folder in aps.RQ12_DIRS.items():
        for f in sorted(folder.glob("rq2_*.json")):
            d = aps._load(f)
            groups[(protocol, d["slot"], d["prompt"])].append(d)

    rows, hist = [], []
    for (protocol, slot, prompt), runs in sorted(groups.items()):
        spans = [aps._rescored_spans(d["metrics"]["spans_by_file"], oracle) for d in runs]
        model = [x for s in spans for x in _offsets(s, {b: v["pred"] for b, v in s.items()})]
        rng = random.Random(42)
        rand = [_summary([x for s in spans for x in _offsets(s, _random_preds(s, rng))]) for _ in range(DRAWS)]
        m = _summary(model)
        # Drift: does the signed offset depend on where the prediction lies in the file?
        near = [(x, pos) for x, pos in model if abs(x) <= 20]
        rho, p_rho = spearmanr([pos for _, pos in near], [x for x, _ in near]) if len(near) > 5 else (float("nan"), float("nan"))
        # Direction of the error: among predictions within 20 lines of an oracle span
        # but not on it, are they before (negative) or after (positive) it?
        neg = sum(x < 0 for x, _ in near)
        pos = sum(x > 0 for x, _ in near)
        p_sign = binomtest(neg, neg + pos).pvalue if neg + pos else float("nan")
        rows.append({
            "protocol": protocol, "slot": slot, "prompt": prompt, "runs": len(runs), "n_pred": m["n"],
            "exact": aps._r(m["exact"]), "exact_random": aps._r(statistics.mean(r["exact"] for r in rand)),
            "near": aps._r(m["near"]), "near_random": aps._r(statistics.mean(r["near"] for r in rand)),
            "p_near": aps._r((1 + sum(r["near"] >= m["near"] for r in rand)) / (1 + DRAWS), 4),
            "median_abs": m["median_abs"],
            "median_abs_random": aps._r(statistics.mean(r["median_abs"] for r in rand), 1),
            "median_signed_near": statistics.median(x for x, _ in near) if near else None,
            "median_signed_near_random": aps._r(statistics.mean(r["median_signed_near"] for r in rand), 1),
            "before": neg, "after": pos, "p_sign": aps._r(p_sign, 4),
            "drift_rho": aps._r(rho), "drift_p": aps._r(p_rho, 4),
        })
        rows[-1].update(_bootstrap(_per_file(spans, random.Random(7)), random.Random(11)))
        for x, _ in model:
            if abs(x) <= 10:
                hist.append({"protocol": protocol, "slot": slot, "prompt": prompt, "offset": x})

    aps._write_csv(STATS / "line_offsets.csv", rows)
    aps._write_csv(STATS / "line_offsets_hist.csv", hist)

    md = ["# Line offsets of span-level predictions (RQ1/RQ2)\n",
          f"Offset = predicted start line - start of the nearest oracle span, for predictions in files "
          f"with oracle spans; pooled over runs. *near*: 1 <= |offset| <= {NEAR}. Random: same spans "
          f"placed at random in the same file ({DRAWS} draws). *drift*: Spearman rho between relative "
          f"position in the file and signed offset (|offset| <= 20).\n",
          "| Protocol | Slot | Prompt | n | exact | exact (rand) | near | near (rand) | p near | median abs | "
          "median abs (rand) | median signed (near) | median signed (rand) | before/after | p sign | drift rho | drift p |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['protocol']} | {r['slot']} | {r['prompt']} | {r['n_pred']} | {r['exact']} | "
                  f"{r['exact_random']} | {r['near']} | {r['near_random']} | {r['p_near']} | {r['median_abs']} | "
                  f"{r['median_abs_random']} | {r['median_signed_near']} | {r['median_signed_near_random']} | {r['before']}/{r['after']} | {r['p_sign']} | {r['drift_rho']} | {r['drift_p']} |")
    md += ["", "## File-level bootstrap (files resampled with all their runs; 95% intervals)\n",
           "Excess k: share of predictions within k lines of an oracle start minus the share expected "
           "under random placement in the same files. Before share: misplaced predictions "
           "(0 < |offset| <= 20) that lie before the oracle line (0.5 = no direction).\n",
           "| Protocol | Slot | Prompt | files | excess 0 | excess 1 | excess 2 | before share |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['protocol']} | {r['slot']} | {r['prompt']} | {r['files']} | "
                  + " | ".join(f"{r[k]} {r[k + '_ci']}" for k in ("excess_0", "excess_1", "excess_2", "before_share")) + " |")
    (STATS / "line_offsets.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
