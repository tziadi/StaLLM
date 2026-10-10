"""Tests for the chance baselines and the statistical helpers used in the paper.

Run with:  python -m unittest discover tests
"""

import csv
import glob
import json
import random
import unittest
from pathlib import Path

from scipy.stats import binomtest, spearmanr

import analyze_paper_stats as aps
from StaLLM_core import _match_file

ROOT = Path(__file__).resolve().parents[1]


def _lenient(p, g):
    strip = lambda xs: [dict(x, startColumn=None, endColumn=None, type=None) for x in xs]
    return _match_file(strip(p), strip(g), require_type=False, use_cols_single=False, preset="Lenient")


class RandomPlacementTest(unittest.TestCase):
    SPANS = {
        "a.cs": {"lines": 50, "pred": [[3, 5], [40, 40]], "gt": [[4, 4]]},
        "b.cs": {"lines": 8, "pred": [[1, 8]], "gt": []},
        "c.cs": {"lines": 30, "pred": [], "gt": [[10, 12]]},
    }

    def _capture(self, shuffle):
        seen = {}

        def match(p, g):
            return 0, len(p), len(g)

        def spy(p, g):
            seen.setdefault("calls", []).append(p)
            return match(p, g)

        orig = aps._span_f1
        try:
            aps._span_f1 = lambda spans, preds, m: seen.update(preds=preds) or 0.0
            aps.random_placement_f1(self.SPANS, random.Random(1), spy, shuffle)
        finally:
            aps._span_f1 = orig
        return seen["preds"]

    def test_same_file_keeps_count_length_and_bounds(self):
        preds = self._capture(shuffle=False)
        for name, f in self.SPANS.items():
            got = preds.get(name, [])
            self.assertEqual(len(got), len(f["pred"]))
            self.assertEqual(sorted(p["endLine"] - p["startLine"] for p in got),
                             sorted(e - s for s, e in f["pred"]))
            for p in got:
                self.assertGreaterEqual(p["startLine"], 1)
                self.assertLessEqual(p["endLine"], f["lines"])

    def test_random_file_keeps_total_count_and_bounds(self):
        preds = self._capture(shuffle=True)
        self.assertEqual(sum(len(v) for v in preds.values()), 3)
        for name, got in preds.items():
            for p in got:
                self.assertGreaterEqual(p["startLine"], 1)
                self.assertLessEqual(p["endLine"], self.SPANS[name]["lines"])

    def test_seeded_draws_are_reproducible(self):
        a = [aps.random_placement_f1(self.SPANS, random.Random(42), _lenient) for _ in range(1)]
        b = [aps.random_placement_f1(self.SPANS, random.Random(42), _lenient) for _ in range(1)]
        self.assertEqual(a, b)

    @unittest.skipUnless(glob.glob(str(ROOT / "output/paper_results/rq2/*.json")), "no RQ2 results")
    def test_offline_rescoring_reproduces_pipeline_f1(self):
        for f in sorted(glob.glob(str(ROOT / "output/paper_results/rq2/*.json"))):
            m = json.loads(Path(f).read_text())["metrics"]
            spans = m["spans_by_file"]
            preds = {b: [{"startLine": s, "endLine": e} for s, e in v["pred"]] for b, v in spans.items()}
            self.assertAlmostEqual(aps._span_f1(spans, preds, _lenient), m["f1"], places=9, msg=f)


class StatsTest(unittest.TestCase):
    def test_mcnemar_exact_matches_binomial(self):
        a = {i: i < 50 for i in range(100)}
        b = {i: 8 <= i < 94 for i in range(100)}
        a_only, b_only, p = aps.mcnemar_exact(a, b)
        self.assertEqual((a_only, b_only), (8, 44))
        self.assertAlmostEqual(p, binomtest(8, 52, 0.5).pvalue)
        self.assertEqual(aps.mcnemar_exact(a, a)[2], 1.0)

    def test_holm(self):
        self.assertEqual(aps.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(aps.holm([0.5]), [0.5])

    def test_binary_scores(self):
        gold = [True, True, False, False, False, True]
        pred = [True, False, False, True, False, True]
        f1, kappa = aps._binary_scores(gold, pred)
        self.assertAlmostEqual(f1, 2 * 2 / (2 * 2 + 1 + 1))
        po, pe = 4 / 6, 0.5 * 0.5 + 0.5 * 0.5
        self.assertAlmostEqual(kappa, (po - pe) / (1 - pe))

    def test_p_at_10(self):
        self.assertEqual(aps._p_at_10({"recall@10": 10 / 237}, 237), 1.0)
        self.assertEqual(aps._p_at_10({"recall@10": 3 / 9}, 9), 0.3)
        self.assertEqual(aps._p_at_10({"recall@10": 0.0}, 40), 0.0)

    @unittest.skipUnless((ROOT / "output/paper_results/stats/rq3_oracle_swap.csv").exists(), "no stats")
    def test_reported_spearman_values(self):
        rows = list(csv.DictReader((ROOT / "output/paper_results/stats/rq3_oracle_swap.csv").read_text().splitlines()))
        for ds, expected in (("DACOS", 0.567), ("MLCQ", 0.583)):
            r = [x for x in rows if x["dataset"] == ds]
            rho, _ = spearmanr([float(x["kappa_human"]) for x in r], [float(x["kappa_heuristic"]) for x in r])
            self.assertAlmostEqual(rho, expected, places=3)


if __name__ == "__main__":
    unittest.main()
