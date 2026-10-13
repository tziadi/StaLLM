"""Smoke tests for the workbench pages that read output/paper_results."""
import pytest
from streamlit.testing.v1 import AppTest

from stallm_ui import paper_results as pr
from stallm_ui.reproducibility import build_manifest, text_hash

PAGES = ["render_overview", "render_benchmark_results", "render_controls", "render_about"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = AppTest.from_string(f"from stallm_ui import paper_results\npaper_results.{page}()", default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]


def test_run_history_renders_without_exception():
    at = AppTest.from_string("from StaLLM_models import init_db\ninit_db()\n"
                             "from stallm_ui.history import render_run_history\nrender_run_history()",
                             default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]


def test_ci_and_p_formatting():
    assert pr._ci("[0.146, 0.214]") == (0.146, 0.214)
    lo, hi = pr._ci("not an interval")
    assert lo != lo and hi != hi  # NaN
    assert pr._p(0.0004) == "< 0.001"
    assert pr._p(0.0266) == "0.027"
    assert pr._p(None) == "–"


def test_latex_export_escapes_and_formats():
    import pandas as pd

    tex = pr._latex(pd.DataFrame({"Model_name": ["GPT-4o"], "F1": [0.21734]}), "Cap & more", "tab:x")
    assert r"Model\_name" in tex and "0.217" in tex and r"Cap \& more" in tex
    assert tex.startswith(r"\begin{table}") and tex.rstrip().endswith(r"\end{table}")


def test_verdict_tones():
    assert pr._verdict(3, 3)[1] == "good"
    assert pr._verdict(0, 3)[1] == "bad"
    assert pr._verdict(1, 3)[1] == "warn"
    assert pr._verdict(0, 0)[1] == "neutral"


def test_manifest_hashes_prompts_and_records_code_version():
    m = build_manifest({"task": "x", "prompts": {"baseline": "Find smells in {code}"}}, {"f1": 0.2})
    assert m["prompts"]["baseline"]["hash"] == text_hash("Find smells in {code}")
    assert m["prompts"]["baseline"]["template"] == "Find smells in {code}"
    assert "prompts" not in m["experiment"]
    assert m["code"]["short"] and m["results"]["f1"] == 0.2
