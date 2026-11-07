"""Non-LLM re-rankers on the same candidate lists as the LLMs, and duplicated
source files in the candidate lists (RQ4). No model call.

    python rq4_reranker_analyses.py

Writes output/paper_results/stats/rq4_rerankers.csv, rq4_duplicates.csv and
rq4_reranker_analyses.md.

Re-rankers, applied to the first 600 candidates of the BM25 list (unmasked or
masked query), i.e. the list the LLM receives:
- path_bm25: BM25 over the candidate paths only, which is the information the
  LLM receives;
- keyword_first: candidates whose path contains the feature name first, in BM25
  order (unmasked list only, since masking removes the name).
"""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import analyze_paper_stats as aps
import run_paper_experiments as rpe
from StaLLM_benchmarks import load_argouml_feature_tasks, load_feature_locations_csv_tasks
from StaLLM_tasks import RankedLocation, list_repo_candidates, paths_match, ranking_metrics

STATS = aps.OUT / "stats"


def _tasks(system: str):
    if system == "eshop":
        return load_feature_locations_csv_tasks(
            str(rpe.ESHOP_FL_CSV), repo_zip=str(rpe.ESHOP_FL_ZIP), project="eShopOnContainers",
            language="C#", descriptions=rpe.ESHOP_FEATURE_DESCRIPTIONS, extensions=[".cs"])
    return load_argouml_feature_tasks(str(rpe.ARGOUML_GT), repo_zip=str(rpe.ARGOUML_ZIP), single_only=True)


def _path_bm25(paths: list[str], query: str, k1: float = 1.2, b: float = 0.75) -> list[str]:
    tf = [Counter(rpe._bm25_tokens(p)) for p in paths]
    lens = [sum(c.values()) for c in tf]
    avg = sum(lens) / max(1, len(lens))
    df = Counter()
    for c in tf:
        df.update(c.keys())
    n = len(paths)
    idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    q = set(rpe._bm25_tokens(query))
    scored = []
    for i, c in enumerate(tf):
        norm = k1 * (1 - b + b * lens[i] / avg)
        scored.append((-sum(idf[t] * c[t] * (k1 + 1) / (c[t] + norm) for t in q if t in c), i, paths[i]))
    return [p for _, _, p in sorted(scored)]


def _keyword_first(paths: list[str], feature: str) -> list[str]:
    pattern = rpe.FEATURE_MASK_TERMS.get(feature, re.escape(feature.lower()))
    hit = [p for p in paths if re.search(pattern, p, flags=re.IGNORECASE)]
    return hit + [p for p in paths if p not in set(hit)]


def _score(ranked: list[str], task) -> tuple[float, float]:
    preds = [RankedLocation(file=f, rank=i) for i, f in enumerate(ranked[:rpe.FL_TOP_K], start=1)]
    m = ranking_metrics(preds, task.gold_locations, ks=(1, 5, 10))
    return m["mrr"], aps._p_at_10(m, len(task.gold_locations))


def _llm_reference(system: str) -> dict[str, dict[str, float]]:
    """Mean MRR and P@10 of each hosted model on the BM25 and masked lists (run 1, both prompts)."""
    root = aps.OUT / "eshop" if system == "eshop" else aps.OUT
    out = {}
    for cond, folder in (("bm25_sorted", "rq4_bm25"), ("bm25_masked", "rq4_masked")):
        vals = defaultdict(lambda: ([], []))
        for f in sorted((root / folder).glob("rq4_AZ[123]_*.json")):
            d = aps._load(f)
            if d.get("run", 1) != 1:
                continue
            m = d["metrics"]
            vals[d["slot"]][0].append(m["mrr"])
            vals[d["slot"]][1].append(aps._p_at_10(m, d["gold_count"]))
        for slot, (mrr, p10) in vals.items():
            out[f"{cond}/{slot}"] = {"mrr": statistics.mean(mrr), "p@10": statistics.mean(p10)}
    return out


def _llm_copies(md: list[str]) -> None:
    """How often an LLM top ten contains a target/classes copy or two copies of one source file."""
    def key(p: str) -> str:
        return re.sub(r".*/(src|target/classes)/", "", p.replace("\\", "/")).lower()
    md.append("\n## Copies in the LLM top ten (ArgoUML, all hosted runs)\n")
    md.append("| Condition | rankings | mean target/classes files in top 10 | mean duplicated slots in top 10 |\n|---|---|---|---|")
    for folder in ("rq4", "rq4_sorted", "rq4_bm25", "rq4_masked"):
        copies, dups = [], []
        for f in sorted((aps.OUT / folder).glob("rq4_AZ[123]_*.json")):
            files = [p.get("file") if isinstance(p, dict) else p for p in aps._load(f).get("predictions", [])][:10]
            if files:
                copies.append(sum("target/classes" in x for x in files))
                dups.append(len(files) - len({key(x) for x in files}))
        if copies:
            md.append(f"| {folder} | {len(copies)} | {aps._r(statistics.mean(copies), 2)} | {aps._r(statistics.mean(dups), 2)} |")


def main() -> None:
    rows, dup_rows, md = [], [], ["# RQ4 — non-LLM re-rankers and duplicated files\n",
                                  "Computed from the repository archives and stored outputs, no model call.\n"]
    for system in ("argouml", "eshop"):
        rpe.SYSTEM = system
        cfg = rpe.SYSTEMS[system]
        tasks = _tasks(system)
        allc = list_repo_candidates(str(cfg["zip"]), allowed_exts=[cfg["ext"]])
        copies = [c for c in allc if "/target/classes/" in c.replace("\\", "/")]
        per = defaultdict(lambda: defaultdict(list))
        for task in tasks:
            feature = task.metadata.get("feature", task.task_id)
            masked = rpe.mask_feature_task(task)
            lists = {"bm25_sorted": (task, rpe.rq4_candidates("bm25", task, allc)[:rpe.CANDIDATE_BUDGET]),
                     "bm25_masked": (masked, rpe.rq4_candidates("bm25", masked, allc)[:rpe.CANDIDATE_BUDGET])}
            for cond, (t, lst) in lists.items():
                rankers = {"list_order": lst, "path_bm25": _path_bm25(lst, t.query)}
                if cond == "bm25_sorted":
                    rankers["keyword_first"] = _keyword_first(lst, feature)
                for name, ranked in rankers.items():
                    mrr, p10 = _score(ranked, task)
                    per[(cond, name)]["mrr"].append(mrr)
                    per[(cond, name)]["p@10"].append(p10)
            for cond in ("keyword_sorted", "bm25_sorted"):
                lst = rpe.rq4_candidates("bm25" if cond == "bm25_sorted" else "keyword_sorted", task, allc)[:rpe.CANDIDATE_BUDGET]
                gold = [g.normalized_file for g in task.gold_locations]
                matching = [c for c in lst if any(paths_match(c, g) for g in gold)]
                dup_rows.append({"system": system, "condition": cond, "feature": feature, "gold": len(gold),
                                 "candidates_matching_gold": len(matching),
                                 "distinct_gold_in_list": len({g for g in gold if any(paths_match(c, g) for c in lst)}),
                                 "copies_in_list": sum("/target/classes/" in c for c in lst)})
        for (cond, name), v in sorted(per.items()):
            rows.append({"system": system, "condition": cond, "reranker": name, "features": len(v["mrr"]),
                         "mrr": aps._r(statistics.mean(v["mrr"])), "p@10": aps._r(statistics.mean(v["p@10"]))})
        for key, v in sorted(_llm_reference(system).items()):
            cond, slot = key.split("/")
            rows.append({"system": system, "condition": cond, "reranker": f"LLM {slot}", "features": "",
                         "mrr": aps._r(v["mrr"]), "p@10": aps._r(v["p@10"])})
        md.append(f"- {system}: {len(allc)} candidate files, of which {len(copies)} under target/classes/")
    aps._write_csv(STATS / "rq4_rerankers.csv", rows)
    aps._write_csv(STATS / "rq4_duplicates.csv", dup_rows)
    md.append("\n## Re-rankers on the first 600 candidates of the BM25 list (mean over features)\n")
    md.append("| System | List | Re-ranker | MRR | P@10 |\n|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['system']} | {r['condition']} | {r['reranker']} | {r['mrr']} | {r['p@10']} |")
    md.append("\n## Several candidates for the same gold file\n")
    md.append("| System | List | Feature | gold | candidates matching gold | distinct gold in list | target/classes copies in list |\n"
              "|---|---|---|---|---|---|---|")
    for r in dup_rows:
        md.append(f"| {r['system']} | {r['condition']} | {r['feature']} | {r['gold']} | {r['candidates_matching_gold']} "
                  f"| {r['distinct_gold_in_list']} | {r['copies_in_list']} |")
    _llm_copies(md)
    (STATS / "rq4_reranker_analyses.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:3] + md[-60:]))


if __name__ == "__main__":
    main()
