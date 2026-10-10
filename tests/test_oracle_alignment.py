"""Regression tests for matching static-analysis oracle records to source files.

Run with:  python -m unittest discover tests
"""

import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from StaLLM_core import _align_gt_to_zip, _build_universe, _detect_line_offset, load_ground_truth_spans

ROOT = Path(__file__).resolve().parents[1]
CATALOG_ZIP = ROOT / "data/apps/eShopOnWeb/Catalog.zip"
ESHOP_CSV = ROOT / "data/apps/eShopOnWeb/eShopOnContainers-sonarqube-quality-analysis.csv"


def _gt(rows):
    df = pd.DataFrame(rows, columns=["file", "startLine", "endLine"])
    df["basename"] = df["file"].map(lambda f: os.path.basename(f.replace("\\", "/")).lower())
    return df


def _zip(paths):
    return pd.DataFrame({"relpath": paths, "basename": [os.path.basename(p).lower() for p in paths]})


class AlignmentTest(unittest.TestCase):
    def test_homonymous_files_stay_distinct(self):
        gt = _gt([
            ("Services/Catalog/Catalog.API/Program.cs", 10, 10),
            ("Services/Basket/Basket.API/Program.cs", 20, 20),
            ("Services/Catalog/Catalog.API/Seed.cs", 5, 6),
        ])
        zdf = _zip(["Catalog/Catalog.API/Program.cs", "Catalog/Catalog.API/Seed.cs"])
        aligned, _, diag = _align_gt_to_zip(gt, zdf)
        self.assertEqual(diag["mode"], "path")
        self.assertEqual(diag["gt_spans_matched"], 2)
        self.assertEqual(diag["gt_spans_dropped"], 1)
        program = aligned[aligned["basename"] == "catalog/catalog.api/program.cs"]
        self.assertEqual(program["startLine"].tolist(), [10])

    def test_windows_separators_are_normalized(self):
        gt = _gt([("Services\\Catalog\\Catalog.API\\Seed.cs", 3, 3)])
        aligned, _, diag = _align_gt_to_zip(gt, _zip(["Catalog/Catalog.API/Seed.cs"]))
        self.assertEqual(diag["mode"], "path")
        self.assertEqual(len(aligned), 1)

    def test_falls_back_to_file_name_without_alignment(self):
        # Two oracle files share the name and suffix length: no unambiguous prefix.
        gt = _gt([("x/A.cs", 1, 1), ("y/A.cs", 2, 2)])
        aligned, _, diag = _align_gt_to_zip(gt, _zip(["z/A.cs"]))
        self.assertEqual(diag["mode"], "basename")
        self.assertEqual(len(aligned), 2)


class LineOffsetTest(unittest.TestCase):
    SOURCE = "using System;\n\npublic class A\n{\n    public int Field = 1;\n}\n"

    def _check(self, reported_lines):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.zip")
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("src/A.cs", self.SOURCE)
            gt = pd.DataFrame({"basename": ["src/a.cs"] * 2, "startLine": reported_lines,
                               "endLine": reported_lines, "startColumn": [13, 15], "endColumn": [14, 25]})
            return _detect_line_offset(gt, path)

    def test_zero_based_export_is_detected(self):
        # "class A" is on line 3 and the field on line 5; a 0-based export reports 2 and 4.
        self.assertEqual(self._check([2, 4])["line_offset"], 1)

    def test_correct_lines_are_kept(self):
        self.assertEqual(self._check([3, 5])["line_offset"], 0)


class LoaderTest(unittest.TestCase):
    def test_rows_without_line_are_dropped(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("file;rule;startLine;endLine\n"
                    "src/A.cs;S1;10;12\n"
                    "src/B.cs;S2;0;0\n"
                    "src/C.cs;S3;;\n")
            path = f.name
        try:
            gt = load_ground_truth_spans(path, allowed_exts=[".cs"])
        finally:
            os.unlink(path)
        self.assertEqual(gt["file"].tolist(), ["src/A.cs"])
        self.assertEqual((gt["startLine"].iloc[0], gt["endLine"].iloc[0]), (10, 12))


@unittest.skipUnless(CATALOG_ZIP.exists() and ESHOP_CSV.exists(), "eShop data not available")
class CatalogRegressionTest(unittest.TestCase):
    """Pins the RQ1-RQ2 universe reported in the paper."""

    def test_catalog_universe(self):
        gt = load_ground_truth_spans(str(ESHOP_CSV), allowed_exts=[".cs"])
        universe, diag, aligned = _build_universe(str(CATALOG_ZIP), gt, [".cs"], 20, 1.0)
        self.assertEqual(diag["gt_alignment"]["mode"], "path")
        self.assertEqual(diag["gt_alignment"]["gt_prefix"], "services")
        self.assertEqual(diag["gt_alignment"]["line_offset"], 1)
        self.assertEqual(len(aligned), 72)
        self.assertEqual(diag["positives"], 13)
        self.assertEqual(diag["negatives"], 7)
        self.assertEqual(int(universe["gt_lines"].sum()), 72)
        program = universe[universe["basename"] == "catalog/catalog.api/program.cs"]
        self.assertEqual(int(program["gt_lines"].iloc[0]), 10)
        lines = sorted(aligned[aligned["basename"] == "catalog/catalog.api/program.cs"]["startLine"])
        self.assertEqual(lines[0], 87)   # hard-coded URI, after the +1 correction


if __name__ == "__main__":
    unittest.main()
