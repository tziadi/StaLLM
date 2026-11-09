"""Tests for the fourth review round: build-copy exclusion and RQ3 majority vote."""
import zipfile

import analyze_paper_stats as aps
from StaLLM_tasks import list_repo_candidates


def test_build_copies_are_not_candidates(tmp_path):
    z = tmp_path / "repo.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name in ("app/src/org/a/A.java", "app/target/classes/org/a/A.java",
                     "app/target/test-classes/org/a/ATest.java", "app/src/org/a/targetmanager/T.java"):
            zf.writestr(name, "class X {}")
    got = list_repo_candidates(str(z), allowed_exts=[".java"])
    assert got == ["app/src/org/a/A.java", "app/src/org/a/targetmanager/T.java"]


def _run(run, preds):
    return {"slot": "AZ1", "prompt": "p", "dataset": "D", "run": run,
            "rows": [{"sample_id": str(i), "gold": g, "predicted": p}
                     for i, (g, p) in enumerate(zip([True, True, False, False], preds))]}


def test_majority_vote_over_three_runs():
    runs = [_run(1, [True, False, True, False]), _run(2, [True, True, False, False]),
            _run(3, [False, True, True, False])]
    voted = aps.majority_vote(runs)
    assert [r["predicted"] for r in voted["rows"]] == [True, True, True, False]
    assert voted["runs_voted"] == 3
    m = voted["metrics"]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 1)


def test_majority_vote_single_run_is_identity():
    run = _run(1, [True, False, True, False])
    voted = aps.majority_vote([run])
    assert [r["predicted"] for r in voted["rows"]] == [True, False, True, False]
