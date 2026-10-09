"""
StaLLM — Paper experiment batch runner
=======================================
Collects data for RQ3 (human-oracle smells) and RQ4 (feature location).
RQ1/RQ2 results already exist from previous runs.

Usage:
    python run_paper_experiments.py           # run RQ3 + RQ4
    python run_paper_experiments.py --rq 3    # RQ3 only
    python run_paper_experiments.py --rq 4    # RQ4 only
    python run_paper_experiments.py --rq 4 --include-compound
    python run_paper_experiments.py --dry     # print plan, no API calls

Results are saved to:
    output/paper_results/rq3/  — DACOS/MLCQ binary smell classification
    output/paper_results/rq4/  — ArgoUML feature location

A summary CSV is written to output/paper_results/summary_rq3.csv
and output/paper_results/summary_rq4.csv after each RQ finishes.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── Load .env before importing anything from StaLLM ──────────────────────────
try:
    import dotenv
    dotenv.load_dotenv(Path(__file__).parent / ".env", override=True)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))

from StaLLM_benchmarks import (
    load_dacos_smell_samples,
    load_mlcq_smell_samples,
    run_dacos_smell_benchmark,
    load_argouml_feature_tasks,
    load_feature_locations_csv_tasks,
    DACOS_PROMPT_TEMPLATE,
)
from StaLLM_tasks import (
    LOCATION_PROMPT_TEMPLATES,
    RankedLocation,
    list_repo_candidates,
    paths_match,
    ranking_metrics,
    run_location_task,
    run_location_task_method,
    summarize_location_results,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA       = Path("data/apps")
OUT        = Path("output/paper_results")
OUT_3      = OUT / "rq3"
OUT_4      = OUT / "rq4"           # alphabetical candidate order (baseline)
OUT_4_KW   = OUT / "rq4_sorted"    # keyword pre-sorted candidates (main condition)
OUT_4_MTH  = OUT / "rq4_method"    # method-level evaluation (keyword-sorted)
OUT_4_BM25 = OUT / "rq4_bm25"      # BM25-ranked candidates (no feature-name heuristic)
OUT_4_MASK = OUT / "rq4_masked"    # feature name masked in the query, BM25-ranked candidates
OUT_5      = OUT / "rq5"           # AutoResearch: generated prompt variants
OUT_5_HO   = OUT_5 / "heldout"     # RQ5 held-out DACOS test split

DACOS_MAIN     = DATA / "DACOS" / "DACOSMain.sql"
DACOS_EXT      = DATA / "DACOS" / "DACOSExtended.sql"
DACOS_FILES    = DATA / "DACOS" / "files.zip"
MLCQ_CSV       = DATA / "MLCQ" / "MLCQCodeSmellSamples.csv"
ARGOUML_GT     = DATA / "Feature Location-ArgoUML"
ARGOUML_ZIP    = DATA / "ArgoUML" / "ArgoUML.zip"
ESHOP_FL_CSV   = DATA / "eShopFL" / "eShopOnContainers-feature-locations.csv"
ESHOP_FL_ZIP   = DATA / "eShopFL" / "eShopOnContainers.zip"

# One-sentence functional descriptions of the eShopOnContainers features (the
# benchmark only provides names); written from the project's architecture
# documentation and used verbatim as the feature-location queries.
ESHOP_FEATURE_DESCRIPTIONS = {
    "Catalog": "Manages the product catalog: catalog items, brands and types, product pictures, "
               "and the item information and stock shown in the storefront.",
    "Basket": "Keeps each buyer's shopping basket: adding and removing items, updating quantities, "
              "and checking out the basket to start an order.",
    "Order": "Handles the ordering process: creating an order at checkout, tracking its status through "
             "stock confirmation, payment and shipment, and cancelling orders.",
    "Identity": "Authenticates users and issues access tokens: login and logout, user registration and "
                "account management, and protecting the services and gateways with tokens.",
    "Payment": "Processes the payment of an order once its stock is confirmed and publishes whether "
               "the payment succeeded or failed.",
}

# Feature-location systems for RQ4 (--system); "argouml" keeps the historical paths.
SYSTEMS = {
    "argouml": {"zip": ARGOUML_ZIP, "ext": ".java", "out": OUT, "suffix": ""},
    "eshop": {"zip": ESHOP_FL_ZIP, "ext": ".cs", "out": OUT / "eshop", "suffix": "_eshop"},
}
SYSTEM = "argouml"

# ── Experiment config ─────────────────────────────────────────────────────────
RQ3_SLOTS   = ["AZ1"]          # models to run for RQ3
RQ4_SLOTS   = ["AZ1"]          # models to run for RQ4
MAX_SAMPLES = 200               # DACOS samples per prompt (cost control)
CANDIDATE_BUDGET = 600          # ArgoUML candidate files sent to LLM
FL_TOP_K    = 10                # ranking depth
RUNS        = 1                 # repeated runs per configuration (--runs)
WORKERS     = 4                 # concurrent requests for hosted slots (--workers)

DACOS_PROMPTS = {
    "binary_oracle_baseline": DACOS_PROMPT_TEMPLATE,
    "conservative_evidence": (
        "You are evaluating a human code-smell oracle.\n\n"
        "Question: is the target smell present in this snippet?\n"
        "Target smell: {smell}\nLanguage: {language}\n\n"
        "Use only visible evidence in the snippet. Do not report unrelated smells.\n"
        "Return JSON only:\n"
        '{{"present": true|false, "confidence": 0.0, "rationale": "one short evidence sentence"}}\n\n'
        "Code:\n{code}\n"
    ),
    "definition_driven": (
        "You are a software-maintenance researcher.\n\n"
        "Decide whether the code contains this exact smell: {smell}.\n\n"
        "Interpretation guide:\n"
        "- complex_method: a method is too complex because of branching, nested logic, mixed responsibilities, or difficult control flow.\n"
        "- long_method: a method is too long or does too much to be easily understood.\n"
        "- long_parameter_list: a method/constructor exposes too many parameters or parameter groups.\n"
        "- multifaceted_abstraction: a class mixes multiple responsibilities or represents an overly broad abstraction.\n"
        "- god_class: a class centralizes too many responsibilities and collaborators.\n"
        "- data_class: a class mostly stores data with little meaningful behavior.\n"
        "- feature_envy: a method relies more on data or behavior from another class than its own class.\n\n"
        "Return strict JSON only:\n"
        '{{"present": true|false, "confidence": 0.0-1.0, "rationale": "one evidence sentence"}}\n\n'
        "Language: {language}\nCode:\n{code}\n"
    ),
}

FL_PROMPTS = {k: v for k, v in LOCATION_PROMPT_TEMPLATES.items()
              if k in ("baseline", "feature_evidence")}

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_llm_from_slot_lazy(slot: str):
    """Import the LLM backend only for non-dry runs.

    This keeps planning/dry-run commands usable in lightweight environments that
    do not have hosted-LLM dependencies installed.
    """
    from StaLLM_llm import build_llm_from_slot

    return build_llm_from_slot(slot)

def ensure_dirs():
    OUT_3.mkdir(parents=True, exist_ok=True)
    OUT_4.mkdir(parents=True, exist_ok=True)
    OUT_4_KW.mkdir(parents=True, exist_ok=True)
    OUT_4_MTH.mkdir(parents=True, exist_ok=True)
    OUT_5.mkdir(parents=True, exist_ok=True)
    OUT_4_BM25.mkdir(parents=True, exist_ok=True)
    OUT_4_MASK.mkdir(parents=True, exist_ok=True)
    OUT_5_HO.mkdir(parents=True, exist_ok=True)


def run_key(rq: str, slot: str, prompt: str, dataset: str = "", run: int = 1) -> str:
    parts = [rq, slot, prompt]
    if dataset:
        parts.append(dataset)
    if run > 1:
        parts.append(f"r{run}")
    return "_".join(parts).replace(" ", "_")


def slot_workers(slot: str) -> int:
    """Local Ollama slots run sequentially; hosted slots use WORKERS."""
    provider = os.getenv(f"{slot}_PROVIDER", "").lower()
    return 1 if provider == "ollama" else WORKERS


def result_path(rq: str, key: str, out_dir: Optional[Path] = None) -> Path:
    if out_dir is not None:
        base = out_dir
    elif rq == "rq3":
        base = OUT_3
    else:
        base = OUT_4
    return base / f"{key}.json"


def already_done(rq: str, key: str, out_dir: Optional[Path] = None) -> bool:
    p = result_path(rq, key, out_dir)
    if p.exists():
        print(f"  [skip] {p.name} already exists")
        return True
    return False


def save_result(rq: str, key: str, data: dict, out_dir: Optional[Path] = None):
    p = result_path(rq, key, out_dir)
    p.write_text(json.dumps(data, indent=2, default=str))
    print(f"  [saved] {p}")


def fmt_metrics(m: dict) -> str:
    parts = []
    for k in ("f1", "precision", "recall", "accuracy", "hit@1", "hit@5", "mrr", "map"):
        if k in m:
            parts.append(f"{k}={m[k]:.3f}")
    return "  ".join(parts) or str(m)


# ── RQ3 : Human-oracle smell classification ───────────────────────────────────

def run_rq3(dry: bool = False):
    print("\n" + "="*60)
    print("RQ3 — Static vs Human Oracle (DACOS / MLCQ)")
    print("="*60)

    datasets = {}
    if DACOS_MAIN.exists():
        print(f"Loading DACOS from {DACOS_MAIN} …")
        files_root = str(DACOS_FILES) if DACOS_FILES.exists() else None
        samples_main = load_dacos_smell_samples(str(DACOS_MAIN), files_root=files_root, limit=MAX_SAMPLES)
        if DACOS_EXT.exists():
            samples_ext = load_dacos_smell_samples(str(DACOS_EXT), files_root=files_root, limit=MAX_SAMPLES)
            datasets["DACOS"] = samples_main + samples_ext
        else:
            datasets["DACOS"] = samples_main
        print(f"  {len(datasets['DACOS'])} DACOS samples loaded")
    else:
        print(f"[warn] {DACOS_MAIN} not found — skipping DACOS")

    if MLCQ_CSV.exists():
        print(f"Loading MLCQ from {MLCQ_CSV} …")
        mlcq = load_mlcq_smell_samples(str(MLCQ_CSV), limit=MAX_SAMPLES)
        datasets["MLCQ"] = mlcq
        print(f"  {len(mlcq)} MLCQ samples loaded")
    else:
        print(f"[warn] {MLCQ_CSV} not found — skipping MLCQ")

    if not datasets:
        print("[error] No datasets available for RQ3")
        return

    rows_summary = []
    for slot in RQ3_SLOTS:
        llm = None if dry else build_llm_from_slot_lazy(slot)
        for run, prompt_name, ds_name in [
            (r, p, d) for r in range(1, RUNS + 1) for p in DACOS_PROMPTS for d in datasets
        ]:
                prompt_tpl = DACOS_PROMPTS[prompt_name]
                samples = datasets[ds_name]
                key = run_key("rq3", slot, prompt_name, ds_name, run)
                print(f"\n  [{slot}] [{prompt_name}] [{ds_name}] run {run}  ({len(samples)} samples)")
                if already_done("rq3", key):
                    data = json.loads(result_path("rq3", key).read_text())
                    m = data.get("metrics", {})
                else:
                    if dry:
                        print("    [dry-run] would call LLM here")
                        continue
                    t0 = time.time()
                    data = run_dacos_smell_benchmark(
                        samples, llm,
                        prompt_template=prompt_tpl,
                        max_samples=MAX_SAMPLES,
                        workers=slot_workers(slot),
                    )
                    data["slot"] = slot
                    data["prompt"] = prompt_name
                    data["dataset"] = ds_name
                    data["run"] = run
                    data["elapsed_s"] = round(time.time() - t0, 1)
                    save_result("rq3", key, data)
                    m = data.get("metrics", {})
                print(f"    {fmt_metrics(m)}")
                rows_summary.append({
                    "slot": slot, "prompt": prompt_name, "dataset": ds_name,
                    **{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in m.items()},
                })

    if rows_summary:
        rebuild_summaries()


# ── RQ4 : Feature location (ArgoUML SPL) ─────────────────────────────────────

_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_BM25_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "in", "is", "are", "be", "by", "for",
    "with", "that", "this", "as", "on", "it", "they", "their", "can", "used", "use",
    "feature", "description", "java", "org", "argouml", "import", "public", "private",
    "protected", "return", "new", "void", "class", "static", "final", "if", "else",
    "null", "true", "false", "this", "string", "int", "boolean",
}


def _bm25_tokens(text: str) -> list[str]:
    out = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9]*", text):
        for part in _CAMEL_RE.findall(word):
            tok = part.lower()
            if len(tok) > 2 and tok not in _BM25_STOP:
                out.append(tok)
    return out


class _BM25Index:
    """Minimal Okapi BM25 over the Java files of a repository ZIP (path + content)."""

    def __init__(self, repo_zip: str, paths: list[str], k1: float = 1.2, b: float = 0.75):
        import zipfile
        from collections import Counter
        self.paths, self.k1, self.b = paths, k1, b
        self.tf: list[Counter] = []
        with zipfile.ZipFile(repo_zip) as zf:
            for path in paths:
                try:
                    text = zf.read(path).decode("utf-8", errors="replace")
                except KeyError:
                    text = ""
                self.tf.append(Counter(_bm25_tokens(path + "\n" + text)))
        self.lens = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(paths)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def rank(self, query: str) -> list[str]:
        q = set(_bm25_tokens(query))
        scores = []
        for i, c in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            score = sum(self.idf[t] * c[t] * (self.k1 + 1) / (c[t] + norm) for t in q if t in c)
            scores.append((score, self.paths[i]))
        scores.sort(key=lambda x: (-x[0], x[1]))
        return [p for _, p in scores]


_BM25_CACHE: dict[str, _BM25Index] = {}


def _bm25_index(paths: list[str]) -> _BM25Index:
    key = str(SYSTEMS[SYSTEM]["zip"])
    if key not in _BM25_CACHE:
        print(f"  Building BM25 index over {len(paths)} files …")
        _BM25_CACHE[key] = _BM25Index(key, paths)
    return _BM25_CACHE[key]


def rq4_candidates(policy: str, task, all_candidates: list[str]) -> list[str]:
    """Full ordered candidate list for a candidate-ordering policy."""
    feature = task.metadata.get("feature", task.task_id)
    if policy == "alphabetical":
        return list(all_candidates)
    if policy == "keyword_sorted":
        cfg = SYSTEMS[SYSTEM]
        return list_repo_candidates(str(cfg["zip"]), allowed_exts=[cfg["ext"]], feature_hint=feature)
    if policy == "bm25":
        return _bm25_index(all_candidates).rank(task.query)
    raise ValueError(f"Unknown candidate policy: {policy}")


RQ4_CONDITIONS = {
    # label: (candidate policy, output dir)
    "alphabetical": ("alphabetical", OUT_4),
    "keyword_sorted": ("keyword_sorted", OUT_4_KW),
    "bm25_sorted": ("bm25", OUT_4_BM25),
    # Same BM25 policy, but run on name-masked tasks (see mask_feature_task).
    "bm25_masked": ("bm25", OUT_4_MASK),
}

# Terms that name each feature; masked in the query so neither BM25 nor the LLM
# can rely on lexical overlap between the feature name and package/class names.
FEATURE_MASK_TERMS = {
    "CATALOG": r"catalog(?:ue)?s?",
    "BASKET": r"baskets?",
    "ORDER": r"order(?:s|ing)?",
    "IDENTITY": r"identit(?:y|ies)",
    "PAYMENT": r"payments?",
    "ACTIVITYDIAGRAM": r"activit(?:y|ies)",
    "COGNITIVE": r"critics?|cognitive",
    "COLLABORATIONDIAGRAM": r"collaborations?",
    "DEPLOYMENTDIAGRAM": r"deployments?",
    "LOGGING": r"log(?:s|ging|ger)?",
    "SEQUENCEDIAGRAM": r"sequences?",
    "STATEDIAGRAM": r"states?",
    "USECASEDIAGRAM": r"use[ -]?cases?",
}


def mask_feature_task(task):
    """Copy of a feature task whose query keeps only the masked description."""
    import dataclasses
    feature = task.metadata.get("feature", task.task_id)
    lines = [l for l in task.query.splitlines() if not l.lower().startswith("feature:")]
    query = "\n".join(lines)
    pattern = FEATURE_MASK_TERMS.get(feature)
    if pattern:
        query = re.sub(rf"\b(?:{pattern})\b", "[MASKED]", query, flags=re.IGNORECASE)
    return dataclasses.replace(task, query=query)


def _run_rq4_condition(
    tasks: list,
    all_candidates: list[str],
    slots: list[str],
    out_dir: Path,
    label: str,
    dry: bool,
    policy: str,
):
    """Run one RQ4 candidate-ordering condition for every slot, prompt and run."""

    for slot in slots:
        llm = None if dry else build_llm_from_slot_lazy(slot)
        for run in range(1, RUNS + 1):
            for prompt_style in FL_PROMPTS:
                for task in tasks:
                    feature = task.metadata.get("feature", task.task_id)
                    key = run_key("rq4", slot, prompt_style, feature, run)
                    print(f"\n  [{slot}] [{prompt_style}] [{label}] {feature} run {run}")
                    if already_done("rq4", key, out_dir):
                        continue
                    if dry:
                        print("    [dry-run] would call LLM here")
                        continue
                    candidates = rq4_candidates(policy, task, all_candidates)[:CANDIDATE_BUDGET]
                    gold = [g.normalized_file for g in task.gold_locations]
                    visible = sum(1 for g in gold if any(paths_match(c, g) for c in candidates))
                    t0 = time.time()
                    for attempt in range(5):
                        try:
                            data = run_location_task(
                                task, candidates, llm,
                                top_k=FL_TOP_K,
                                prompt_style=prompt_style,
                            )
                            break
                        except Exception as exc:
                            if attempt == 4:
                                raise
                            print(f"    [retry] {type(exc).__name__}: {str(exc)[:120]}")
                            time.sleep(2 ** (attempt + 2))
                    data["slot"] = slot
                    data["feature"] = feature
                    data["gold_count"] = len(task.gold_locations)
                    data["gold_visible"] = visible
                    data["candidate_order"] = label
                    data["run"] = run
                    data["elapsed_s"] = round(time.time() - t0, 1)
                    save_result("rq4", key, data, out_dir)
                    print(f"    {fmt_metrics(data.get('metrics', {}))}")


def run_rq4_baselines(tasks: list, all_candidates: list[str], include_compound: bool,
                      masked_tasks: Optional[list] = None) -> None:
    """Non-LLM rankings: the top-k of each candidate-ordering policy, scored directly."""

    rows = []
    runs = [("keyword_sorted", "keyword_sorted", tasks), ("bm25", "bm25", tasks)]
    if masked_tasks:
        runs.append(("bm25_masked", "bm25", masked_tasks))
    for name, policy, task_list in runs:
        for task in task_list:
            ranked = rq4_candidates(policy, task, all_candidates)[:FL_TOP_K]
            preds = [RankedLocation(file=f, rank=i) for i, f in enumerate(ranked, start=1)]
            m = ranking_metrics(preds, task.gold_locations, ks=(1, 5, 10))
            rows.append({
                "method": f"{name}_only",
                "feature": task.metadata.get("feature", task.task_id),
                "gold_count": len(task.gold_locations),
                **{k: round(v, 4) for k, v in m.items()},
            })
    name = "summary_rq4_baselines_all.csv" if include_compound else "summary_rq4_baselines.csv"
    out_root = SYSTEMS[SYSTEM]["out"]
    out_root.mkdir(parents=True, exist_ok=True)
    _write_csv(out_root / name, rows)
    print(f"\n[RQ4] Non-LLM baselines → {out_root / name}")
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        agg = {k: sum(r[k] for r in sub) / len(sub) for k in ("hit@1", "hit@5", "mrr", "map")}
        print(f"  {method:22s} {fmt_metrics(agg)}")


def run_rq4(dry: bool = False, *, include_compound: bool = False,
            conditions: Optional[list[str]] = None):
    cfg = SYSTEMS[SYSTEM]
    print("\n" + "="*60)
    scope = "all feature scenarios" if include_compound else "single features"
    print(f"RQ4 — Feature Location ({SYSTEM}, {scope})")
    print("="*60)

    if SYSTEM == "eshop":
        tasks = load_feature_locations_csv_tasks(
            str(ESHOP_FL_CSV), repo_zip=str(ESHOP_FL_ZIP), project="eShopOnContainers",
            language="C#", descriptions=ESHOP_FEATURE_DESCRIPTIONS, extensions=[".cs"],
        )
    else:
        if not ARGOUML_GT.exists() or not ARGOUML_ZIP.exists():
            print(f"[error] ArgoUML benchmark data not found under {DATA}")
            return
        tasks = load_argouml_feature_tasks(
            str(ARGOUML_GT),
            repo_zip=str(ARGOUML_ZIP),
            single_only=not include_compound,
        )
    print(f"Loaded {len(tasks)} feature tasks")

    all_candidates = list_repo_candidates(str(cfg["zip"]), allowed_exts=[cfg["ext"]])
    print(f"  {len(all_candidates)} candidate {cfg['ext']} files → budget = {CANDIDATE_BUDGET}")

    masked = [mask_feature_task(t) for t in tasks]
    run_rq4_baselines(tasks, all_candidates, include_compound, masked_tasks=masked)

    for label in conditions or list(RQ4_CONDITIONS):
        policy, out_dir = RQ4_CONDITIONS[label]
        out_dir = SYSTEMS[SYSTEM]["out"] / out_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Condition: {label} candidate ordering ---")
        _run_rq4_condition(masked if label == "bm25_masked" else tasks,
                           all_candidates, RQ4_SLOTS, out_dir,
                           label=label, dry=dry, policy=policy)

    rebuild_summaries()


# ── RQ4-method : Method-level feature location ────────────────────────────────

def run_rq4_method(dry: bool = False):
    print("\n" + "="*60)
    print("RQ4-Method — Feature Location at Method Level (ArgoUML SPL)")
    print("="*60)

    if not ARGOUML_GT.exists() or not ARGOUML_ZIP.exists():
        print("[error] ArgoUML benchmark data not found")
        return

    from StaLLM_benchmarks import load_argouml_feature_tasks as _load
    tasks = _load(str(ARGOUML_GT), repo_zip=str(ARGOUML_ZIP),
                  single_only=True, granularity="method")
    print(f"Loaded {len(tasks)} feature tasks (method-level gold)")
    all_candidates = list_repo_candidates(str(ARGOUML_ZIP), allowed_exts=[".java"])
    print(f"  {len(all_candidates)} Java files available")

    rows_summary = []
    for slot in RQ4_SLOTS:
        llm = None if dry else build_llm_from_slot_lazy(slot)
        task_results = []
        for task in tasks:
            feature = task.metadata.get("feature", task.task_id)
            key = run_key("rq4", slot, "method_level", feature)
            method_gold = [g for g in task.gold_locations if g.symbol]
            print(f"\n  [{slot}] {feature}  "
                  f"({len(method_gold)} method-gold / {len(task.gold_locations)} total gold)")
            if already_done("rq4", key, OUT_4_MTH):
                import json as _json
                data = _json.loads(result_path("rq4", key, OUT_4_MTH).read_text())
            else:
                if dry:
                    print("    [dry-run] would call LLM here")
                    continue
                t0 = time.time()
                data = run_location_task_method(
                    task, str(ARGOUML_ZIP), all_candidates, llm,
                    top_k=FL_TOP_K, file_budget=100, max_methods_per_file=20,
                )
                data["slot"] = slot
                data["feature"] = feature
                data["elapsed_s"] = round(time.time() - t0, 1)
                save_result("rq4", key, data, OUT_4_MTH)
            m = data.get("metrics_method") or data.get("metrics", {})
            mf = data.get("metrics_file", {})
            print(f"    method: {fmt_metrics(m)}")
            print(f"    file:   {fmt_metrics(mf)}")
            task_results.append(data)

        if task_results:
            agg = summarize_location_results([
                {"metrics": r.get("metrics_method", r.get("metrics", {}))} for r in task_results
            ])
            print(f"\n  [{slot}] METHOD AGGREGATE: {fmt_metrics(agg)}")
            for res in task_results:
                m = res.get("metrics_method") or res.get("metrics", {})
                rows_summary.append({
                    "slot": slot,
                    "feature": res.get("feature", ""),
                    "method_gold": res.get("method_gold_count", ""),
                    "file_gold": res.get("file_gold_count", ""),
                    "candidates": res.get("method_candidates_count", ""),
                    **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                })

    if rows_summary:
        _write_csv(OUT / "summary_rq4_method.csv", rows_summary)
        print(f"\n[RQ4-Method] Summary → {OUT / 'summary_rq4_method.csv'}")


# ── RQ5 : AutoResearch — LLM-generated prompt optimisation ───────────────────

RQ5_N_CANDIDATES = 5   # number of prompt variants to generate
RQ5_SLOT = "AZ1"       # model used for both generation and evaluation

_RQ5_META_PROMPT = """\
You are an expert prompt engineer specialising in code quality analysis with LLMs.

Your goal: generate {n} NEW and DIVERSE prompt templates for binary code-smell \
detection that may improve F1 beyond the three existing prompts shown below.

=== Task description ===
Given a code snippet and a smell name, the LLM must decide whether the smell \
is present. It must return strict JSON:
{{"present": true|false, "confidence": 0.0-1.0, "rationale": "one evidence sentence"}}

Template variables available (use ALL three):
  {{smell}}     — smell name, e.g. "complex_method", "god_class"
  {{language}}  — programming language, e.g. "Java"
  {{code}}      — the code snippet to analyse

Smell catalogue:
  complex_method, long_method, long_parameter_list, multifaceted_abstraction,
  god_class, data_class, feature_envy

=== Existing prompts and their DACOS F1 scores ===

--- baseline  (F1={f1_baseline:.3f}) ---
{tpl_baseline}

--- conservative_evidence  (F1={f1_conservative:.3f}) ---
{tpl_conservative}

--- definition_driven  (F1={f1_definition:.3f}) ---
{tpl_definition}

=== Instructions ===
Generate exactly {n} new prompt variants. Explore diverse strategies, for example:
  1. structured diagnostic checklist: ask the model to inspect evidence internally
  2. calibrated-confidence: ask the model to explicitly calibrate its certainty
  3. contrastive: ask the model to compare against what is NOT the smell
  4. threshold-anchored: include concrete measurable thresholds per smell
  5. diagnostic questions: guide the model through evidence checks

Each template MUST:
  - Use {{smell}}, {{language}}, {{code}} placeholders
  - End with an instruction to return the JSON object above
  - Return only the JSON object, without exposing step-by-step reasoning
  - Be substantially different from the three existing prompts

Return a JSON array of exactly {n} objects:
[
  {{"name": "short_snake_case_id", "template": "full prompt text..."}},
  ...
]
Return ONLY the JSON array, no prose.
"""


def _generate_prompt_candidates(llm, existing_f1: dict[str, float]) -> list[dict]:
    """Ask the LLM to generate RQ5_N_CANDIDATES new prompt templates."""
    meta = _RQ5_META_PROMPT.format(
        n=RQ5_N_CANDIDATES,
        f1_baseline=existing_f1.get("binary_oracle_baseline", 0.0),
        f1_conservative=existing_f1.get("conservative_evidence", 0.0),
        f1_definition=existing_f1.get("definition_driven", 0.0),
        tpl_baseline=DACOS_PROMPTS["binary_oracle_baseline"][:600],
        tpl_conservative=DACOS_PROMPTS["conservative_evidence"][:600],
        tpl_definition=DACOS_PROMPTS["definition_driven"][:600],
    )
    content = llm.chat([{"role": "user", "content": meta}], max_tokens=4096)
    # strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


def _read_rq3_baselines() -> list[dict[str, Any]]:
    summary_path = OUT / "summary_rq3.csv"
    rows: list[dict[str, Any]] = []
    if not summary_path.exists():
        return rows
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("prompt") in DACOS_PROMPTS and row.get("dataset") in {"DACOS", "MLCQ"}:
                rows.append({
                    "source": "RQ3_baseline",
                    "prompt": row.get("prompt", ""),
                    "dataset": row.get("dataset", ""),
                    "f1": _to_float(row.get("f1")),
                    "precision": _to_float(row.get("precision")),
                    "recall": _to_float(row.get("recall")),
                    "kappa": _to_float(row.get("kappa")),
                })
    return rows


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_rq5(dry: bool = False):
    print("\n" + "="*60)
    print("RQ5 — AutoResearch: LLM-generated prompt optimisation")
    print("="*60)

    # ── load datasets ─────────────────────────────────────────────────────────
    datasets: dict = {}
    if DACOS_MAIN.exists():
        files_root = str(DACOS_FILES) if DACOS_FILES.exists() else None
        samples = load_dacos_smell_samples(str(DACOS_MAIN), files_root=files_root, limit=MAX_SAMPLES)
        datasets["DACOS"] = samples
        print(f"DACOS: {len(samples)} samples")
    if MLCQ_CSV.exists():
        datasets["MLCQ"] = load_mlcq_smell_samples(str(MLCQ_CSV), limit=MAX_SAMPLES)
        print(f"MLCQ:  {len(datasets['MLCQ'])} samples")
    if not datasets:
        print("[error] No datasets found — run RQ3 first to verify data paths")
        return

    llm = None if dry else build_llm_from_slot_lazy(RQ5_SLOT)

    # ── Step 1: load or generate prompt candidates ────────────────────────────
    candidates_file = OUT_5 / "generated_prompts.json"
    if candidates_file.exists():
        candidates = json.loads(candidates_file.read_text())
        print(f"Loaded {len(candidates)} existing candidates from {candidates_file}")
    else:
        # read RQ3 DACOS F1 scores for context
        existing_f1: dict[str, float] = {}
        for p_name in DACOS_PROMPTS:
            key = run_key("rq3", RQ5_SLOT, p_name, "DACOS")
            p = result_path("rq3", key)
            if p.exists():
                d = json.loads(p.read_text())
                existing_f1[p_name] = d.get("metrics", {}).get("f1", 0.0)
        print(f"RQ3 baseline F1 (DACOS): {existing_f1}")

        if dry:
            print("[dry-run] would call LLM to generate prompt candidates")
            return
        print(f"Generating {RQ5_N_CANDIDATES} prompt candidates via {RQ5_SLOT} …")
        candidates = _generate_prompt_candidates(llm, existing_f1)
        candidates_file.write_text(json.dumps(candidates, indent=2))
        print(f"  Saved {len(candidates)} candidates → {candidates_file}")

    # ── Step 2: evaluate each candidate on DACOS (optimisation set) ───────────
    print(f"\nEvaluating {len(candidates)} candidates on DACOS …")
    dacos_results: list[dict] = []
    pending = 0
    for cand in candidates:
        name = cand["name"]
        template = cand["template"]
        result_f = OUT_5 / f"rq5_DACOS_{name}.json"
        if result_f.exists():
            data = json.loads(result_f.read_text())
            print(f"  [skip] {name}: f1={data['metrics'].get('f1', 0):.3f}")
        else:
            if dry:
                print(f"  [dry-run] would evaluate {name}")
                pending += 1
                continue
            print(f"  Evaluating {name} …", end=" ", flush=True)
            t0 = time.time()
            data = run_dacos_smell_benchmark(
                datasets["DACOS"], llm,
                prompt_template=template,
                max_samples=MAX_SAMPLES,
            )
            data.update({"prompt": name, "dataset": "DACOS", "slot": RQ5_SLOT,
                         "elapsed_s": round(time.time() - t0, 1), "template": template})
            result_f.write_text(json.dumps(data, indent=2, default=str))
            m = data.get("metrics", {})
            print(f"f1={m.get('f1', 0):.3f}  prec={m.get('precision', 0):.3f}  rec={m.get('recall', 0):.3f}")
        dacos_results.append(data)

    if dry and pending:
        print(f"\n[RQ5] {pending} DACOS candidate(s) still pending; rerun without --dry to continue.")

    if not dacos_results:
        return

    # ── Step 3: find best on DACOS → validate on MLCQ ────────────────────────
    best = max(dacos_results, key=lambda r: r.get("metrics", {}).get("f1", 0))
    best_name = best["prompt"]
    best_f1_dacos = best.get("metrics", {}).get("f1", 0)
    print(f"\nBest on DACOS: '{best_name}' (F1={best_f1_dacos:.3f})")

    mlcq_result = None
    if "MLCQ" in datasets:
        result_f = OUT_5 / f"rq5_MLCQ_{best_name}.json"
        if result_f.exists():
            mlcq_result = json.loads(result_f.read_text())
            print(f"[skip] MLCQ validation: f1={mlcq_result['metrics'].get('f1', 0):.3f}")
        else:
            if not dry:
                print(f"Validating best prompt on MLCQ …", end=" ", flush=True)
                best_template = next(c["template"] for c in candidates if c["name"] == best_name)
                t0 = time.time()
                mlcq_result = run_dacos_smell_benchmark(
                    datasets["MLCQ"], llm,
                    prompt_template=best_template,
                    max_samples=MAX_SAMPLES,
                )
                mlcq_result.update({"prompt": best_name, "dataset": "MLCQ", "slot": RQ5_SLOT,
                                    "elapsed_s": round(time.time() - t0, 1)})
                result_f.write_text(json.dumps(mlcq_result, indent=2, default=str))
                m = mlcq_result.get("metrics", {})
                print(f"f1={m.get('f1', 0):.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    rows = _read_rq3_baselines()
    for r in dacos_results:
        m = r.get("metrics", {})
        rows.append({"source": "RQ5_generated", "prompt": r["prompt"], "dataset": "DACOS",
                     "f1": round(m.get("f1", 0), 4), "precision": round(m.get("precision", 0), 4),
                     "recall": round(m.get("recall", 0), 4), "kappa": round(m.get("kappa", 0), 4)})
    if mlcq_result:
        m = mlcq_result.get("metrics", {})
        rows.append({"source": "RQ5_generated", "prompt": best_name, "dataset": "MLCQ",
                     "f1": round(m.get("f1", 0), 4), "precision": round(m.get("precision", 0), 4),
                     "recall": round(m.get("recall", 0), 4), "kappa": round(m.get("kappa", 0), 4)})
    _write_csv(OUT / "summary_rq5.csv", rows)
    print(f"\n[RQ5] Summary → {OUT / 'summary_rq5.csv'}")


# ── RQ3 baselines : non-LLM smell classifiers ────────────────────────────────

_SIG_RE = re.compile(
    r"(?:public|protected|private|static|final|abstract|synchronized|\s)+[\w<>\[\],.?\s]+?\s+(\w+)\s*\(([^)]*)\)\s*(?:throws[^{;]+)?\{")
_FIELD_RE = re.compile(r"^\s*(?:private|protected|public)\s+(?:static\s+)?(?:final\s+)?[\w<>\[\],.?\s]+\s+\w+\s*(?:=[^;]*)?;", re.M)
_DECISION_RE = re.compile(r"\b(?:if|for|while|case|catch)\b|&&|\|\||\?(?!\s*>)")


def smell_heuristic(smell: str, code: str) -> bool:
    """Designite-inspired thresholds, approximated with regexes, fixed a priori.

    long_parameter_list: a method with >= 5 parameters; complex_method: >= 8
    decision points (cyclomatic proxy); long_method: > 100 non-blank lines;
    multifaceted_abstraction: >= 7 fields and >= 7 methods; god_class: >= 20
    methods or > 500 non-blank lines; data_class: >= 3 fields and >= 70% of the
    methods are accessors; feature_envy: >= 5 calls on other objects and more
    than twice the accesses to `this`.
    """
    sigs = _SIG_RE.findall(code)
    loc = sum(1 for line in code.splitlines() if line.strip())
    params = max((len([p for p in a.split(",") if p.strip()]) for _, a in sigs), default=0)
    fields = len(_FIELD_RE.findall(code))
    methods = len(sigs)
    if smell == "long_parameter_list":
        return params >= 5
    if smell == "complex_method":
        return len(_DECISION_RE.findall(code)) >= 8
    if smell == "long_method":
        return loc > 100
    if smell == "multifaceted_abstraction":
        return fields >= 7 and methods >= 7
    if smell == "god_class":
        return methods >= 20 or loc > 500
    if smell == "data_class":
        accessors = sum(1 for name, _ in sigs if re.match(r"(?:get|set|is)[A-Z]", name))
        return fields >= 3 and methods > 0 and accessors / methods >= 0.7
    if smell == "feature_envy":
        external = len(re.findall(r"\b(?!this\b|super\b)[a-z]\w*\.\w+\s*\(", code))
        own = len(re.findall(r"\bthis\.", code))
        return external >= 5 and external > 2 * own
    return False


def run_rq3_baselines():
    from StaLLM_benchmarks import _binary_metrics
    print("\n" + "="*60)
    print("RQ3 baselines — non-LLM smell classifiers")
    print("="*60)
    files_root = str(DACOS_FILES) if DACOS_FILES.exists() else None
    sets = {
        "DACOS": load_dacos_smell_samples(str(DACOS_MAIN), files_root=files_root, limit=MAX_SAMPLES),
        "MLCQ": load_mlcq_smell_samples(str(MLCQ_CSV), limit=MAX_SAMPLES),
        "DACOS_heldout": load_dacos_heldout(),
    }
    rows = []
    for ds, samples in sets.items():
        for name, fn in (("always_present", lambda s: True),
                         ("metric_heuristic", lambda s: smell_heuristic(s.smell, s.code))):
            preds = [{"sample_id": s.sample_id, "smell": s.smell, "gold": bool(s.label),
                      "predicted": bool(fn(s))} for s in samples]
            m = _binary_metrics(preds)
            rows.append({"method": name, "dataset": ds, **_rounded(m, _METRIC_KEYS_BIN)})
            print(f"  {ds:14s} {name:17s} {fmt_metrics(m)}  kappa={m['kappa']:.3f}")
            (OUT_3 / "baselines").mkdir(exist_ok=True)
            (OUT_3 / "baselines" / f"rq3b_{name}_{ds}.json").write_text(
                json.dumps({"method": name, "dataset": ds, "rows": preds, "metrics": m}, indent=2))
    _write_csv(OUT / "summary_rq3_baselines.csv", rows)
    print(f"[RQ3] Baselines → {OUT / 'summary_rq3_baselines.csv'}")


# ── RQ2 : span-level model sensitivity (rerun with correct Ollama context) ───

ESHOP_ZIP = DATA / "eShopOnWeb" / "Catalog.zip"
ESHOP_CSV = DATA / "eShopOnWeb" / "eShopOnContainers-sonarqube-quality-analysis.csv"
OUT_2 = OUT / "rq2"


class _CountingLLM:
    """Wraps a ChatModel and counts failed calls.

    StaLLM_core.analyze_with_llm silently falls back to heuristics when a call
    raises, so failures must be counted here to keep the rerun auditable.
    """

    def __init__(self, inner):
        self.inner, self.calls, self.errors = inner, 0, 0
        self.slot_key = getattr(inner, "slot_key", None)
        self.cfg = inner.cfg

    def chat(self, *args, **kwargs):
        self.calls += 1
        try:
            return self.inner.chat(*args, **kwargs)
        except Exception:
            self.errors += 1
            raise

    def __getattr__(self, name):
        return getattr(self.inner, name)


def run_rq2(dry: bool = False, prompts: tuple[str, ...] = ("scanner",)):
    """RQ1/RQ2 span-level runs with the ENASE E1/E2 setup on the current pipeline:
    eShop Catalog, Top-k=20, positive ratio 1.0, Lenient (RQ1 = prompts on AZ1,
    RQ2 = Scanner across slots)."""
    from StaLLM_core import run_experiment

    print("\n" + "="*60)
    print(f"RQ1/RQ2 — span-level runs (prompts={','.join(prompts)}, {ESHOP_ZIP.name})")
    print("="*60)
    OUT_2.mkdir(parents=True, exist_ok=True)
    for slot, prompt, run in [(sl, p, r) for sl in RQ3_SLOTS for r in range(1, RUNS + 1) for p in prompts]:
        suffix = f"_r{run}" if run > 1 else ""
        out = OUT_2 / f"rq2_{slot}_{prompt}{suffix}.json"
        if out.exists():
            print(f"  [skip] {out.name}")
            continue
        if dry:
            print(f"  [dry-run] would run {slot}")
            continue
        llm = _CountingLLM(build_llm_from_slot_lazy(slot))
        t0 = time.time()
        _, summary_u, metrics, usage = run_experiment(
            str(ESHOP_ZIP), str(ESHOP_CSV), mode=prompt, top_k=20, pos_ratio=1.0,
            preset="Lenient", user_require_type=False, user_use_line_span=True,
            user_use_cols_single=False, llm=llm,
        )
        data = {
            "slot": slot, "prompt": prompt, "run": run, "zip": ESHOP_ZIP.name,
            "model": llm.model_label(), "num_ctx": getattr(llm.cfg, "num_ctx", None),
            "files": len(summary_u), "llm_calls": llm.calls, "llm_errors": llm.errors,
            "elapsed_s": round(time.time() - t0, 1), "usage": usage,
            "metrics": {k: v for k, v in metrics.items() if k not in ("file_diagnostics",)},
        }
        out.write_text(json.dumps(data, indent=2, default=str))
        m = metrics
        print(f"  {slot} {prompt} r{run}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"files={len(summary_u)} calls={llm.calls} errors={llm.errors} time={data['elapsed_s']}s")


# ── RQ5 held-out : selection/test split on DACOS ─────────────────────────────

def _rq5_best_generated() -> Optional[dict]:
    """Best generated prompt on the DACOS selection split (as chosen in RQ5)."""
    best = None
    for f in sorted(OUT_5.glob("rq5_DACOS_*.json")):
        d = json.loads(f.read_text())
        if best is None or d["metrics"]["f1"] > best["metrics"]["f1"]:
            best = d
    return best


def load_dacos_heldout() -> list:
    """DACOS samples disjoint (by DACOS sample id) from the RQ3/RQ5 selection split."""
    files_root = str(DACOS_FILES) if DACOS_FILES.exists() else None
    selection = load_dacos_smell_samples(str(DACOS_MAIN), files_root=files_root, limit=MAX_SAMPLES)
    used = {s.sample_id.split(":")[0] for s in selection}
    pool = load_dacos_smell_samples(str(DACOS_MAIN), files_root=files_root, limit=MAX_SAMPLES * 3)
    test = [s for s in pool if s.sample_id.split(":")[0] not in used]
    return test[:MAX_SAMPLES]


def run_rq5_heldout(dry: bool = False):
    print("\n" + "="*60)
    print("RQ5-HO — Manual vs. selected generated prompt on held-out DACOS")
    print("="*60)
    best = _rq5_best_generated()
    if best is None:
        print("[error] No RQ5 selection results found — run --rq 5 first")
        return
    test = load_dacos_heldout()
    pos = sum(1 for s in test if s.label)
    print(f"Held-out DACOS: {len(test)} samples ({pos} positive); best generated = {best['prompt']}")

    prompts = dict(DACOS_PROMPTS)
    prompts[best["prompt"]] = best["template"]
    for slot in RQ3_SLOTS:
        llm = None if dry else build_llm_from_slot_lazy(slot)
        for name, template in prompts.items():
            out = OUT_5_HO / f"rq5ho_{slot}_{name}.json"
            if out.exists():
                print(f"  [skip] {out.name}")
                continue
            if dry:
                print(f"  [dry-run] would evaluate {slot} {name}")
                continue
            t0 = time.time()
            data = run_dacos_smell_benchmark(test, llm, prompt_template=template,
                                             workers=slot_workers(slot))
            data.update({"slot": slot, "prompt": name, "dataset": "DACOS_heldout",
                         "source": "RQ5_generated" if name == best["prompt"] else "RQ3_manual",
                         "elapsed_s": round(time.time() - t0, 1)})
            out.write_text(json.dumps(data, indent=2, default=str))
            print(f"  {slot} {name}: {fmt_metrics(data['metrics'])}")
    rebuild_summaries()


# ── Summaries (rebuilt from all saved JSON results) ──────────────────────────

_METRIC_KEYS_BIN = ("precision", "recall", "f1", "accuracy", "kappa", "f1_ci_lo", "f1_ci_hi",
                    "kappa_ci_lo", "kappa_ci_hi", "tp", "fp", "fn", "tn", "samples")
_METRIC_KEYS_RANK = ("hit@1", "recall@1", "hit@5", "recall@5", "hit@10", "recall@10", "mrr", "map")


def _rounded(m: dict, keys: tuple) -> dict:
    return {k: round(m[k], 4) if isinstance(m.get(k), float) else m.get(k) for k in keys if k in m}


def _rebuild_rq4_summary(out_root: Path, conditions: dict) -> list[dict]:
    rq4_all = []
    for label, (_, out_dir) in conditions.items():
        for f in sorted((out_root / out_dir.name).glob("rq4_*.json")):
            d = json.loads(f.read_text())
            rq4_all.append({"slot": d.get("slot"), "prompt": d.get("prompt_style"),
                            "candidate_order": d.get("candidate_order", label),
                            "feature": d.get("feature"), "gold_count": d.get("gold_count"),
                            "gold_visible": d.get("gold_visible", ""), "run": d.get("run", 1),
                            **_rounded(d.get("metrics", {}), _METRIC_KEYS_RANK),
                            "total_tokens": d.get("usage", {}).get("total_tokens")})
    if rq4_all:
        _write_csv(out_root / "summary_rq4_all.csv", rq4_all)
        _write_csv(out_root / "summary_rq4.csv", [r for r in rq4_all if "_and_" not in str(r["feature"])])
    return rq4_all


def rebuild_summaries() -> None:
    """Rewrite summary CSVs from every result file, so parallel/partial runs stay consistent."""
    rq3 = []
    for f in sorted(OUT_3.glob("rq3_*.json")):
        d = json.loads(f.read_text())
        rq3.append({"slot": d.get("slot"), "prompt": d.get("prompt"), "dataset": d.get("dataset"),
                    "run": d.get("run", 1), **_rounded(d.get("metrics", {}), _METRIC_KEYS_BIN),
                    "total_tokens": d.get("usage", {}).get("total_tokens"),
                    "elapsed_s": d.get("elapsed_s")})
    _write_csv(OUT / "summary_rq3.csv", rq3)

    for system, cfg in SYSTEMS.items():
        _rebuild_rq4_summary(cfg["out"], RQ4_CONDITIONS)

    ho = []
    for f in sorted(OUT_5_HO.glob("rq5ho_*.json")):
        d = json.loads(f.read_text())
        ho.append({"slot": d.get("slot"), "source": d.get("source"), "prompt": d.get("prompt"),
                   "dataset": d.get("dataset"), **_rounded(d.get("metrics", {}), _METRIC_KEYS_BIN)})
    if ho:
        _write_csv(OUT / "summary_rq5_heldout.csv", ho)
    print(f"[summaries] rq3={len(rq3)} rq5_heldout={len(ho)} rows → {OUT}")


# ── CSV writer ────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global RQ3_SLOTS, RQ4_SLOTS, RUNS, WORKERS, SYSTEM
    parser = argparse.ArgumentParser(description="StaLLM paper experiment runner")
    parser.add_argument("--rq", choices=["2", "3", "3b", "4", "4m", "5", "5h", "summary"],
                        help="Run only this RQ. 4m=method-level, 5=AutoResearch, "
                             "5h=RQ5 held-out DACOS test, summary=rebuild CSVs only.")
    parser.add_argument("--dry", action="store_true", help="Dry run — print plan without calling the LLM")
    parser.add_argument(
        "--include-compound",
        action="store_true",
        help="For RQ4, include compound ArgoUML scenarios and write summary_rq4_all.csv.",
    )
    parser.add_argument("--slots", help="Comma-separated model slots (default: AZ1).")
    parser.add_argument("--runs", type=int, default=RUNS, help="Repeated runs per configuration.")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Concurrent requests for hosted slots.")
    parser.add_argument("--system", choices=list(SYSTEMS), default="argouml",
                        help="RQ4 feature-location system.")
    parser.add_argument("--prompts", help="RQ3: comma-separated prompt names (default: all).")
    parser.add_argument("--conditions", help="RQ4 candidate orderings, comma-separated "
                                             f"(default: {','.join(RQ4_CONDITIONS)}).")
    args = parser.parse_args()

    if args.slots:
        RQ3_SLOTS = RQ4_SLOTS = [s.strip() for s in args.slots.split(",") if s.strip()]
    RUNS, WORKERS, SYSTEM = args.runs, args.workers, args.system
    conditions = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
    if args.prompts:
        keep = {p.strip() for p in args.prompts.split(",")}
        for name in list(DACOS_PROMPTS):
            if name not in keep:
                del DACOS_PROMPTS[name]

    ensure_dirs()

    if args.dry:
        print("[DRY RUN] No API calls will be made.\n")

    rq = args.rq
    if rq == "summary":
        rebuild_summaries()
    if rq == "2":
        run_rq2(dry=args.dry, prompts=tuple(p.strip() for p in (args.prompts or "scanner").split(",")))
    if rq == "3b":
        run_rq3_baselines()
    if rq is None or rq == "3":
        run_rq3(dry=args.dry)
    if rq is None or rq == "4":
        run_rq4(dry=args.dry, include_compound=args.include_compound, conditions=conditions)
    if rq is None or rq == "4m":
        run_rq4_method(dry=args.dry)
    if rq == "5":
        run_rq5(dry=args.dry)
    if rq == "5h":
        run_rq5_heldout(dry=args.dry)

    print("\nDone.")


if __name__ == "__main__":
    main()
