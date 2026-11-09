"""RQ5 on MLCQ: repeated, independent automated prompt searches with a held-out split.

    python rq5_mlcq_searches.py [--searches 5] [--workers 8] [--dry]

Design (gpt-4o, slot AZ1, as in RQ5 on DACOS):
- Selection split: the 200 MLCQ instances of RQ3 (first 200 reviews of the CSV).
- Held-out split: the next 200 reviews whose code sample (MLCQ `sample_id`) does
  not occur in the selection split.
- Each search asks gpt-4o (temperature 1.0, so that searches differ) for five
  prompt candidates, giving it the Cohen's kappa of the three manual prompts on
  the selection split. Each candidate is scored on the selection split, the
  best one by kappa is the search's winner, and every candidate and the manual
  prompts are also scored on the held-out split.

Outputs in output/paper_results/rq5_mlcq/: one JSON per (search, prompt,
split), the generated prompts, and summary.csv. The analysis
(rq5_mlcq_analysis in this file) writes stats/rq5_mlcq_*.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from pathlib import Path

import run_paper_experiments as rpe
from run_paper_experiments import DACOS_PROMPTS
from StaLLM_benchmarks import _load_rows, load_mlcq_smell_samples, run_dacos_smell_benchmark

OUT = rpe.OUT / "rq5_mlcq"
STATS = rpe.OUT / "stats"
SLOT = "AZ1"
N_CAND = 5
SPLIT = 200

META = rpe._RQ5_META_PROMPT.replace(
    "that may improve F1 beyond the three existing prompts shown below.",
    "that may improve Cohen's kappa against human labels beyond the three existing prompts shown below.",
).replace("=== Existing prompts and their DACOS F1 scores ===",
          "=== Existing prompts and their Cohen's kappa on the MLCQ selection split ===").replace(
    "(F1={f1_baseline:.3f})", "(kappa={f1_baseline:.3f})").replace(
    "(F1={f1_conservative:.3f})", "(kappa={f1_conservative:.3f})").replace(
    "(F1={f1_definition:.3f})", "(kappa={f1_definition:.3f})")


def splits() -> tuple[list, list]:
    rows = _load_rows(str(rpe.MLCQ_CSV))
    code_of = {str(r.get("id")): str(r.get("sample_id")) for r in rows}
    pool = load_mlcq_smell_samples(str(rpe.MLCQ_CSV), limit=2 * SPLIT + 300)
    selection = pool[:SPLIT]
    seen = {code_of.get(s.sample_id) for s in selection}
    heldout = []
    for s in pool[SPLIT:]:
        c = code_of.get(s.sample_id)
        if c in seen:
            continue
        seen.add(c)
        heldout.append(s)
        if len(heldout) == SPLIT:
            break
    assert len(heldout) == SPLIT, len(heldout)
    return selection, heldout


def _score(samples, llm, template: str, path: Path, workers: int, meta: dict) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    data = run_dacos_smell_benchmark(samples, llm, prompt_template=template, max_samples=SPLIT, workers=workers)
    data.update(meta, elapsed_s=round(time.time() - t0, 1), template=template)
    path.write_text(json.dumps(data, indent=2, default=str))
    m = data["metrics"]
    print(f"    {path.name}: kappa={m.get('kappa', 0):.3f} f1={m.get('f1', 0):.3f}")
    return data


def run(searches: int, workers: int, dry: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection, heldout = splits()
    pos = lambda xs: sum(s.label for s in xs)
    print(f"MLCQ selection {len(selection)} ({pos(selection)} positive), held-out {len(heldout)} ({pos(heldout)} positive)")
    (OUT / "splits.json").write_text(json.dumps({
        "selection": [s.sample_id for s in selection], "heldout": [s.sample_id for s in heldout]}, indent=2))
    if dry:
        return
    llm = rpe.build_llm_from_slot_lazy(SLOT)
    manual = {}
    for name in ("binary_oracle_baseline", "conservative_evidence", "definition_driven"):
        for split, samples in (("selection", selection), ("heldout", heldout)):
            d = _score(samples, llm, DACOS_PROMPTS[name], OUT / f"manual_{name}_{split}.json", workers,
                       {"prompt": name, "split": split, "search": "manual", "slot": SLOT})
            manual[(name, split)] = d["metrics"]["kappa"]
    for k in range(1, searches + 1):
        gen = OUT / f"search{k}_prompts.json"
        if gen.exists():
            cands = json.loads(gen.read_text())
        else:
            meta = META.format(
                n=N_CAND,
                f1_baseline=manual[("binary_oracle_baseline", "selection")],
                f1_conservative=manual[("conservative_evidence", "selection")],
                f1_definition=manual[("definition_driven", "selection")],
                tpl_baseline=DACOS_PROMPTS["binary_oracle_baseline"][:600],
                tpl_conservative=DACOS_PROMPTS["conservative_evidence"][:600],
                tpl_definition=DACOS_PROMPTS["definition_driven"][:600],
            )
            for attempt in range(4):
                try:
                    content = llm.chat([{"role": "user", "content": meta}], temperature=1.0, max_tokens=4096).strip()
                    if content.startswith("```"):
                        content = content.split("```")[1]
                        content = content[4:] if content.startswith("json") else content
                    cands = json.loads(content.strip())[:N_CAND]
                    break
                except Exception as exc:
                    print(f"  [retry generation] {type(exc).__name__}: {str(exc)[:100]}")
            gen.write_text(json.dumps(cands, indent=2))
        print(f"\nSearch {k}: {[c['name'] for c in cands]}")
        for i, c in enumerate(cands, start=1):
            for split, samples in (("selection", selection), ("heldout", heldout)):
                _score(samples, llm, c["template"], OUT / f"search{k}_c{i}_{split}.json", workers,
                       {"prompt": c["name"], "split": split, "search": k, "candidate": i, "slot": SLOT})
    summarize()
    analyze()


def summarize() -> None:
    rows = []
    for f in sorted(OUT.glob("*.json")):
        if f.name in ("splits.json",) or f.name.endswith("_prompts.json"):
            continue
        d = json.loads(f.read_text())
        rows.append({"file": f.name, "search": d.get("search"), "candidate": d.get("candidate", ""),
                     "prompt": d.get("prompt"), "split": d.get("split"),
                     "kappa": round(d["metrics"].get("kappa", 0), 4), "f1": round(d["metrics"].get("f1", 0), 4)})
    with open(OUT / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"→ {OUT / 'summary.csv'}")


def analyze() -> None:
    """Per search: winner's kappa on the selection and held-out splits (optimism),
    its held-out rank among the search's candidates, and a paired bootstrap of
    the held-out kappa difference with the manual prompt that is best on the
    selection split. Over all prompts: Spearman rho between selection and
    held-out kappa."""
    from scipy.stats import spearmanr

    def load(name):
        d = json.loads((OUT / name).read_text())
        return {str(r["sample_id"]): (bool(r["gold"]), r.get("predicted") is True) for r in d["rows"]}, d["metrics"]["kappa"]

    def kappa(rows, ids):
        return rpe_kappa([rows[i][0] for i in ids], [rows[i][1] for i in ids])

    manual = {n: {sp: load(f"manual_{n}_{sp}.json") for sp in ("selection", "heldout")}
              for n in ("binary_oracle_baseline", "conservative_evidence", "definition_driven")}
    best_manual = max(manual, key=lambda n: manual[n]["selection"][1])
    out, allpts = [], [(manual[n]["selection"][1], manual[n]["heldout"][1]) for n in manual]
    rng = random.Random(42)
    k = 1
    while (OUT / f"search{k}_prompts.json").exists():
        cands = []
        i = 1
        while (OUT / f"search{k}_c{i}_heldout.json").exists():
            cands.append({sp: load(f"search{k}_c{i}_{sp}.json") for sp in ("selection", "heldout")})
            i += 1
        if not cands:
            break
        allpts += [(c["selection"][1], c["heldout"][1]) for c in cands]
        w = max(range(len(cands)), key=lambda j: cands[j]["selection"][1])
        win = cands[w]
        ho_sorted = sorted((c["heldout"][1] for c in cands), reverse=True)
        mrows, wrows = manual[best_manual]["heldout"][0], win["heldout"][0]
        ids = sorted(wrows)
        deltas = []
        for _ in range(2000):
            sm = [rng.choice(ids) for _ in ids]
            deltas.append(kappa(wrows, sm) - kappa(mrows, sm))
        deltas.sort()
        out.append({"search": k, "candidates": len(cands), "winner": w + 1,
                    "winner_selection": round(win["selection"][1], 3), "winner_heldout": round(win["heldout"][1], 3),
                    "optimism": round(win["selection"][1] - win["heldout"][1], 3),
                    "winner_heldout_rank": ho_sorted.index(win["heldout"][1]) + 1,
                    "mean_candidate_heldout": round(statistics.mean(c["heldout"][1] for c in cands), 3),
                    "best_manual": best_manual, "best_manual_selection": round(manual[best_manual]["selection"][1], 3),
                    "best_manual_heldout": round(manual[best_manual]["heldout"][1], 3),
                    "delta_vs_manual": round(win["heldout"][1] - manual[best_manual]["heldout"][1], 3),
                    "delta_ci": f"[{deltas[50]:.3f}, {deltas[1949]:.3f}]"})
        k += 1
    rho, p = spearmanr([a for a, _ in allpts], [b for _, b in allpts])
    STATS.mkdir(parents=True, exist_ok=True)
    with open(STATS / "rq5_mlcq_searches.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    summary = {"searches": len(out), "prompts": len(allpts), "spearman_selection_vs_heldout": round(rho, 3),
               "p": round(p, 4), "mean_optimism": round(statistics.mean(r["optimism"] for r in out), 3),
               "manual_heldout": {n: round(manual[n]["heldout"][1], 3) for n in manual},
               "manual_selection": {n: round(manual[n]["selection"][1], 3) for n in manual}}
    (STATS / "rq5_mlcq_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for r in out:
        print(r)


def rpe_kappa(gold, pred):
    import analyze_paper_stats as aps
    return aps._binary_scores(gold, pred)[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--searches", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    elif a.summary:
        summarize()
    else:
        run(a.searches, a.workers, a.dry)
