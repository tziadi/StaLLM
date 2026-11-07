"""Tests for the DACOS loader that labels samples with the developers' annotations.

Run with:  python -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path

from StaLLM_benchmarks import load_dacos_annotated_splits, load_dacos_smell_samples

ONE, ZERO = "_binary '\x01'", "_binary '\\0'"
CODE = "class Sample {\n" + "    void work(int a, int b, int c) { System.out.println(a + b + c); }\n" * 3 + "}\n"


def _sample(sid, is_class, project, smells):
    return (f"({sid},{sid + 1000},{ONE if smells else ZERO},{ONE if is_class else ZERO},"
            f"'/split/{project}/S{sid}.code','{project}',2,'{smells}')")


def _annotation(aid, sid, cm=0, lp=0, ma=0):
    flag = lambda v: ONE if v else ZERO
    return f"({aid},{flag(cm)},{ZERO},{flag(lp)},{flag(ma)},{flag(cm or lp or ma)},{sid})"


class DacosAnnotatedSplitsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        samples, annotations, aid = [], [], 1
        # sample id: (is_class, project, tagged smells, votes for the tagged smell)
        spec = {}
        for i in range(1, 13):
            spec[i] = (False, f"p{i % 3}", "1", [1, 1] if i % 2 else [0, 0])      # complex method
        for i in range(13, 25):
            spec[i] = (False, f"p{i % 3}", "2", [1, 1] if i % 3 else [0, 0])      # long parameter list
        for i in range(25, 37):
            spec[i] = (True, f"p{i % 3}", "3", [1, 1] if i % 4 else [0, 0])       # multifaceted abstraction
        spec[37] = (False, "p0", "1", [1, 0])                                     # tie: excluded
        spec[38] = (False, "p1", "1,2", [1, 1])                                   # tagged twice
        spec[39] = (False, "p2", "", [1, 1])                                      # not tagged: excluded
        spec[40] = (False, "p0", "1", None)                                       # not annotated: excluded
        for sid, (is_class, project, smells, votes) in spec.items():
            samples.append(_sample(sid, is_class, project, smells))
            target = {"1": "cm", "1,2": "cm", "2": "lp", "3": "ma", "": "cm"}[smells]
            for vote in votes or []:
                annotations.append(_annotation(aid, sid, **{target: vote}))
                aid += 1
            path = root / "files" / "split" / project / f"S{sid}.code"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(CODE)
        self.sql = root / "DACOSMain.sql"
        self.sql.write_text(
            "CREATE TABLE `sample` (`path_to_file` varchar(255));\n"
            f"INSERT INTO `sample` VALUES {','.join(samples)};\n"
            f"INSERT INTO `annotation` VALUES {','.join(annotations)};\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_majority_labels_disjoint_splits(self):
        splits = load_dacos_annotated_splits(str(self.sql), split_size=12)
        ids = {name: {s.metadata["dacos_sample_id"] for s in items} for name, items in splits.items()}
        self.assertFalse(ids["selection"] & ids["heldout"])
        for items in splits.values():
            self.assertEqual(len(items), 12)
            self.assertEqual({s.smell for s in items},
                             {"complex_method", "long_parameter_list", "multifaceted_abstraction"})
            for s in items:
                sid = s.metadata["dacos_sample_id"]
                self.assertNotIn(sid, {37, 39, 40})
                self.assertEqual(s.label, s.metadata["annotations_yes"] > s.metadata["annotations_no"])
        all_ids = [s.metadata["dacos_sample_id"] for items in splits.values() for s in items]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_deterministic_and_default_unchanged(self):
        first = load_dacos_annotated_splits(str(self.sql), split_size=12)
        second = load_dacos_annotated_splits(str(self.sql), split_size=12)
        self.assertEqual([s.sample_id for s in first["selection"]], [s.sample_id for s in second["selection"]])
        heldout = load_dacos_smell_samples(str(self.sql), limit=12, label_source="annotations", split="heldout")
        self.assertEqual([s.sample_id for s in heldout], [s.sample_id for s in first["heldout"]])
        legacy = load_dacos_smell_samples(str(self.sql), limit=6)
        self.assertTrue(all(s.metadata.get("label_source") is None for s in legacy))


if __name__ == "__main__":
    unittest.main()
