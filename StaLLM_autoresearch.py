"""AutoResearch-style benchmark runner for StarLLM.

This module implements the first safe slice of an autonomous research loop:
evaluate candidate prompts against a fixed StarLLM benchmark and keep a
reproducible run log. It deliberately does not modify prompts or source files;
that ratchet step can be layered on top once dev/holdout splits are stable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from StaLLM_benchmarks import load_argouml_feature_tasks
from StaLLM_core import load_strategies, run_experiment
from StaLLM_llm import build_llm_from_slot, default_slot_key
from StaLLM_tasks import LOCATION_PROMPT_TEMPLATES, list_repo_candidates, run_location_task


OUTPUT_DIR = Path("output") / "autoresearch"


@dataclass(frozen=True)
class PromptCandidate:
    name: str
    template: str
    source: str = "candidate"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return slug or "candidate"


def _load_candidate_prompts(path: str | None, task: str) -> list[PromptCandidate]:
    if not path:
        if task == "code_smell":
            return [
                PromptCandidate(name=name, template=template, source="strategies.json")
                for name, template in load_strategies().items()
            ]
        names = ["baseline", "feature_evidence", "terse_ranking"]
        return [
            PromptCandidate(name=name, template=LOCATION_PROMPT_TEMPLATES[name], source="built-in")
            for name in names
            if name in LOCATION_PROMPT_TEMPLATES
        ]

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "prompts" in data:
        data = data["prompts"]
    if not isinstance(data, dict):
        raise ValueError("Prompt candidate file must be a JSON object or contain a top-level 'prompts' object.")

    candidates: list[PromptCandidate] = []
    for name, entry in data.items():
        if isinstance(entry, dict):
            template = str(entry.get("template", ""))
            source = str(entry.get("source", path))
        else:
            template = str(entry)
            source = path
        if template.strip():
            candidates.append(PromptCandidate(name=_slug(str(name)), template=template, source=source))
    return candidates


def _select_prompt_subset(candidates: list[PromptCandidate], names: Iterable[str] | None) -> list[PromptCandidate]:
    requested = [name for name in (names or []) if name]
    if not requested:
        return candidates
    by_name = {candidate.name: candidate for candidate in candidates}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise ValueError(f"Unknown prompt candidate(s): {', '.join(missing)}")
    return [by_name[name] for name in requested]


def _rank_feature_candidates(query: str, candidates: list[str]) -> list[str]:
    terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", query)
        if len(token) >= 4 and token.lower() not in {"feature", "description", "diagram", "diagrams"}
    }

    def score(path: str) -> tuple[int, int, str]:
        low = path.lower()
        hits = sum(1 for term in terms if term in low)
        package_bonus = 2 if "org/argouml" in low else 0
        return (hits + package_bonus, -len(path), path)

    return sorted(candidates, key=score, reverse=True)


def _sum_usage(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    items = list(rows)
    return {
        "prompt_tokens": sum(int((row.get("usage") or {}).get("prompt_tokens", 0)) for row in items),
        "completion_tokens": sum(int((row.get("usage") or {}).get("completion_tokens", 0)) for row in items),
        "total_tokens": sum(int((row.get("usage") or {}).get("total_tokens", 0)) for row in items),
    }


def evaluate_feature_location(args: argparse.Namespace, prompts: list[PromptCandidate], llm: Any) -> list[dict[str, Any]]:
    tasks = load_argouml_feature_tasks(args.gt_path, repo_zip=args.repo_zip)
    tasks = tasks[: args.max_tasks] if args.max_tasks else tasks
    candidates = list_repo_candidates(args.repo_zip, allowed_exts=[".java"])

    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        started = time.time()
        task_results = []
        prompt_templates = {prompt.name: prompt.template, "baseline": prompt.template}
        for task in tasks:
            focused = _rank_feature_candidates(task.query, candidates)[: args.candidate_budget]
            task_results.append(
                run_location_task(
                    task,
                    focused,
                    llm,
                    top_k=args.top_k,
                    prompt_style=prompt.name,
                    prompt_templates=prompt_templates,
                )
            )
        metric_keys = sorted({key for row in task_results for key in (row.get("metrics") or {})})
        metrics = {
            key: sum(float((row.get("metrics") or {}).get(key, 0.0)) for row in task_results) / max(1, len(task_results))
            for key in metric_keys
        }
        usage = _sum_usage(task_results)
        rows.append(
            {
                "rank_order": index,
                "task": "feature_location",
                "prompt": prompt.name,
                "source": prompt.source,
                "primary_metric": args.primary_metric,
                "score": metrics.get(args.primary_metric, 0.0),
                "task_count": len(task_results),
                "elapsed_s": round(time.time() - started, 2),
                **metrics,
                **usage,
            }
        )
        print(f"[{index}/{len(prompts)}] {prompt.name}: {args.primary_metric}={rows[-1]['score']:.4f}")
    return rows


def evaluate_code_smell(args: argparse.Namespace, prompts: list[PromptCandidate], llm: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        started = time.time()
        prompt_templates = {prompt.name: prompt.template, "baseline": prompt.template}
        _, _, metrics, usage = run_experiment(
            args.repo_zip,
            args.static_csv,
            prompt.name,
            top_k=args.top_k,
            pos_ratio=args.pos_ratio,
            preset=args.preset,
            user_require_type=args.require_type,
            user_use_line_span=True,
            user_use_cols_single=True,
            llm=llm,
            prompt_templates=prompt_templates,
        )
        rows.append(
            {
                "rank_order": index,
                "task": "code_smell",
                "prompt": prompt.name,
                "source": prompt.source,
                "primary_metric": args.primary_metric,
                "score": metrics.get(args.primary_metric, 0.0),
                "task_count": int(metrics.get("top_k_total_files", args.top_k)),
                "elapsed_s": round(time.time() - started, 2),
                **metrics,
                **usage,
            }
        )
        print(f"[{index}/{len(prompts)}] {prompt.name}: {args.primary_metric}={rows[-1]['score']:.4f}")
    return rows


def write_autoresearch_outputs(rows: list[dict[str, Any]], prompts: list[PromptCandidate], args: argparse.Namespace) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(rows, key=lambda row: float(row.get("score", 0.0)), reverse=True)
    fields = sorted({key for row in ordered for key in row})
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    prompt_payload = {
        candidate.name: {"template": candidate.template, "source": candidate.source}
        for candidate in prompts
    }
    (out_dir / "candidate_prompts.json").write_text(json.dumps(prompt_payload, indent=2), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    if ordered:
        best = ordered[0]
        best_prompt = next((p for p in prompts if p.name == best["prompt"]), None)
        report = {
            "best_prompt": best["prompt"],
            "score": best["score"],
            "primary_metric": best["primary_metric"],
            "task": best["task"],
            "template": best_prompt.template if best_prompt else "",
        }
        (out_dir / "best_prompt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate prompt candidates with an AutoResearch-style StarLLM benchmark.")
    parser.add_argument("--task", choices=["feature_location", "code_smell"], default="feature_location")
    parser.add_argument("--slot", default=None, help="LLM slot key from .env. Defaults to STALLM_DEFAULT_SLOT.")
    parser.add_argument("--prompt-file", default=None, help="JSON file containing prompt candidates.")
    parser.add_argument("--prompts", nargs="*", default=None, help="Optional subset of prompt ids to evaluate.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--primary-metric", default=None, help="Metric used to rank candidates.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=3, help="Feature-location task cap for quick dev runs. Use 0 for all.")
    parser.add_argument("--candidate-budget", type=int, default=300)
    parser.add_argument("--gt-path", default="data/apps/Feature Location-ArgoUML")
    parser.add_argument("--repo-zip", default="data/apps/ArgoUML/ArgoUML.zip")
    parser.add_argument("--static-csv", default="data/apps/ArgoUML/ArgoUml-sonarqube-quality-analysis.csv")
    parser.add_argument("--pos-ratio", type=float, default=0.6)
    parser.add_argument("--preset", default="Balanced")
    parser.add_argument("--require-type", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.primary_metric = args.primary_metric or ("mrr" if args.task == "feature_location" else "f1")

    slot = args.slot or default_slot_key()
    if not slot:
        raise SystemExit("No LLM slot configured. Set STALLM_DEFAULT_SLOT or pass --slot.")
    llm = build_llm_from_slot(slot)

    prompts = _select_prompt_subset(_load_candidate_prompts(args.prompt_file, args.task), args.prompts)
    if not prompts:
        raise SystemExit("No prompt candidates to evaluate.")

    print(f"StarLLM AutoResearch run: task={args.task}, slot={slot}, prompts={len(prompts)}")
    if args.task == "feature_location":
        rows = evaluate_feature_location(args, prompts, llm)
    else:
        rows = evaluate_code_smell(args, prompts, llm)

    out_dir = write_autoresearch_outputs(rows, prompts, args)
    best = max(rows, key=lambda row: float(row.get("score", 0.0))) if rows else None
    if best:
        print(f"Best: {best['prompt']} ({args.primary_metric}={float(best['score']):.4f})")
    print(f"Wrote AutoResearch run to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
