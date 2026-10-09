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
import os
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
    DACOS_PROMPT_TEMPLATE,
)
from StaLLM_tasks import (
    LOCATION_PROMPT_TEMPLATES,
    list_repo_candidates,
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
OUT_5      = OUT / "rq5"           # AutoResearch: generated prompt variants

DACOS_MAIN     = DATA / "DACOS" / "DACOSMain.sql"
DACOS_EXT      = DATA / "DACOS" / "DACOSExtended.sql"
DACOS_FILES    = DATA / "DACOS" / "files.zip"
MLCQ_CSV       = DATA / "MLCQ" / "MLCQCodeSmellSamples.csv"
ARGOUML_GT     = DATA / "Feature Location-ArgoUML"
ARGOUML_ZIP    = DATA / "ArgoUML" / "ArgoUML.zip"

# ── Experiment config ─────────────────────────────────────────────────────────
RQ3_SLOTS   = ["AZ1"]          # models to run for RQ3
RQ4_SLOTS   = ["AZ1"]          # models to run for RQ4
MAX_SAMPLES = 200               # DACOS samples per prompt (cost control)
CANDIDATE_BUDGET = 600          # ArgoUML candidate files sent to LLM
FL_TOP_K    = 10                # ranking depth

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


def run_key(rq: str, slot: str, prompt: str, dataset: str = "") -> str:
    parts = [rq, slot, prompt]
    if dataset:
        parts.append(dataset)
    return "_".join(parts).replace(" ", "_")


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
        for prompt_name, prompt_tpl in DACOS_PROMPTS.items():
            for ds_name, samples in datasets.items():
                key = run_key("rq3", slot, prompt_name, ds_name)
                print(f"\n  [{slot}] [{prompt_name}] [{ds_name}]  ({len(samples)} samples)")
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
                    )
                    data["slot"] = slot
                    data["prompt"] = prompt_name
                    data["dataset"] = ds_name
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
        _write_csv(OUT / "summary_rq3.csv", rows_summary)
        print(f"\n[RQ3] Summary → {OUT / 'summary_rq3.csv'}")


# ── RQ4 : Feature location (ArgoUML SPL) ─────────────────────────────────────

def _run_rq4_condition(
    tasks: list,
    all_candidates: list[str],
    slots: list[str],
    out_dir: Path,
    label: str,
    dry: bool,
    use_feature_hint: bool,
):
    """Run one RQ4 condition (alphabetical vs. keyword-sorted candidates)."""

    rows_summary = []
    for slot in slots:
        llm = None if dry else build_llm_from_slot_lazy(slot)
        for prompt_style, _ in FL_PROMPTS.items():
            task_results = []
            for task in tasks:
                feature = task.metadata.get("feature", task.task_id)
                key = run_key("rq4", slot, prompt_style, feature)
                print(f"\n  [{slot}] [{prompt_style}] {feature}  ({len(task.gold_locations)} gold files)")
                if already_done("rq4", key, out_dir):
                    data = json.loads(result_path("rq4", key, out_dir).read_text())
                else:
                    if dry:
                        print("    [dry-run] would call LLM here")
                        continue
                    if use_feature_hint:
                        from StaLLM_tasks import list_repo_candidates as _lrc
                        sorted_cands = _lrc(str(ARGOUML_ZIP), allowed_exts=[".java"], feature_hint=feature)
                        candidates = sorted_cands[:CANDIDATE_BUDGET]
                    else:
                        candidates = all_candidates[:CANDIDATE_BUDGET]
                    t0 = time.time()
                    data = run_location_task(
                        task, candidates, llm,
                        top_k=FL_TOP_K,
                        prompt_style=prompt_style,
                    )
                    data["slot"] = slot
                    data["feature"] = feature
                    data["gold_count"] = len(task.gold_locations)
                    data["candidate_order"] = label
                    data["elapsed_s"] = round(time.time() - t0, 1)
                    save_result("rq4", key, data, out_dir)
                m = data.get("metrics", {})
                print(f"    {fmt_metrics(m)}")
                task_results.append(data)

            if task_results:
                agg = summarize_location_results(task_results)
                print(f"\n  [{slot}] [{prompt_style}] AGGREGATE: {fmt_metrics(agg)}")
                for res in task_results:
                    m = res.get("metrics", {})
                    rows_summary.append({
                        "slot": slot,
                        "prompt": prompt_style,
                        "candidate_order": label,
                        "feature": res.get("feature", ""),
                        "gold_count": res.get("gold_count", ""),
                        **{k: round(v, 4) if isinstance(v, float) else v
                           for k, v in m.items()},
                    })
    return rows_summary


def run_rq4(dry: bool = False, *, include_compound: bool = False):
    print("\n" + "="*60)
    scope = "all feature scenarios" if include_compound else "8 core single features"
    print(f"RQ4 — Feature Location (ArgoUML SPL, {scope})")
    print("="*60)

    if not ARGOUML_GT.exists():
        print(f"[error] Ground truth not found: {ARGOUML_GT}")
        return
    if not ARGOUML_ZIP.exists():
        print(f"[error] Repository ZIP not found: {ARGOUML_ZIP}")
        return

    tasks = load_argouml_feature_tasks(
        str(ARGOUML_GT),
        repo_zip=str(ARGOUML_ZIP),
        single_only=not include_compound,
    )
    print(f"Loaded {len(tasks)} feature tasks")

    print(f"Building alphabetical candidate list from {ARGOUML_ZIP.name} …")
    all_candidates = list_repo_candidates(str(ARGOUML_ZIP), allowed_exts=[".java"])
    print(f"  {len(all_candidates)} total Java files → budget = {CANDIDATE_BUDGET}")

    rows_all = []

    print("\n--- Condition A: alphabetical order (no-guidance baseline) ---")
    rows_all += _run_rq4_condition(
        tasks, all_candidates, RQ4_SLOTS, OUT_4,
        label="alphabetical", dry=dry, use_feature_hint=False,
    )

    print("\n--- Condition B: keyword pre-sorted candidates ---")
    rows_all += _run_rq4_condition(
        tasks, all_candidates, RQ4_SLOTS, OUT_4_KW,
        label="keyword_sorted", dry=dry, use_feature_hint=True,
    )

    if rows_all:
        summary_name = "summary_rq4_all.csv" if include_compound else "summary_rq4.csv"
        _write_csv(OUT / summary_name, rows_all)
        print(f"\n[RQ4] Combined summary → {OUT / summary_name}")


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
    parser = argparse.ArgumentParser(description="StaLLM paper experiment runner")
    parser.add_argument("--rq", choices=["3", "4", "4m", "5"], help="Run only this RQ. 4m=method-level, 5=AutoResearch.")
    parser.add_argument("--dry", action="store_true", help="Dry run — print plan without calling the LLM")
    parser.add_argument(
        "--include-compound",
        action="store_true",
        help="For RQ4, include compound ArgoUML scenarios and write summary_rq4_all.csv.",
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.dry:
        print("[DRY RUN] No API calls will be made.\n")

    rq = args.rq
    if rq is None or rq == "3":
        run_rq3(dry=args.dry)
    if rq is None or rq == "4":
        run_rq4(dry=args.dry, include_compound=args.include_compound)
    if rq is None or rq == "4m":
        run_rq4_method(dry=args.dry)
    if rq == "5":
        run_rq5(dry=args.dry)

    print("\nDone.")


if __name__ == "__main__":
    main()
