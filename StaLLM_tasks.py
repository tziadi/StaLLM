# StaLLM_tasks.py
"""Task abstractions for extending StarLLM beyond static-analysis spans.

This module is intentionally independent from the current static-analysis
pipeline. It defines a small interchange format for localization tasks such as
feature location and bug location, plus ranking metrics shared by both tasks.
"""

from __future__ import annotations

import csv
import json
import os
import zipfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Optional


class MaintenanceTaskType(StrEnum):
    STATIC_ANALYSIS = "static_analysis"
    FEATURE_LOCATION = "feature_location"
    BUG_LOCATION = "bug_location"


@dataclass(frozen=True)
class LocalizationGold:
    """A ground-truth code location for feature/bug localization."""

    file: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    symbol: Optional[str] = None
    score: float = 1.0

    @property
    def normalized_file(self) -> str:
        return normalize_path(self.file)


@dataclass(frozen=True)
class MaintenanceTask:
    """Common task instance consumed by future StaLLM adapters/runners."""

    task_id: str
    task_type: MaintenanceTaskType
    project: str
    query: str
    gold_locations: tuple[LocalizationGold, ...]
    repo_zip: Optional[str] = None
    language: Optional[str] = None
    candidate_level: str = "file"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "project": self.project,
            "query": self.query,
            "repo_zip": self.repo_zip,
            "language": self.language,
            "candidate_level": self.candidate_level,
            "gold_locations": [g.__dict__ for g in self.gold_locations],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RankedLocation:
    """A ranked prediction returned by an LLM or baseline."""

    file: str
    rank: int
    confidence: Optional[float] = None
    rationale: str = ""

    @property
    def normalized_file(self) -> str:
        return normalize_path(self.file)


LOCATION_PROMPT_TEMPLATES: dict[str, str] = {
    "baseline": """You are a software maintenance assistant performing {task_name}.

Project: {project}
Language: {language}
Candidate level: {candidate_level}

Query:
{query}

Candidate files:
{candidate_text}

Return JSON only, no markdown. Rank the top {top_k} files most likely to be relevant.
Schema:
[
  {{"rank": 1, "file": "path/to/File.ext", "confidence": 0.0, "rationale": "short reason"}}
]
""",
    "feature_evidence": """You are performing feature location in a software product line.

Goal: identify source files that implement, refine, or directly support the requested feature.
Prefer files whose package/class names, diagram concepts, UI actions, graph models, renderers, parsers, or property panels semantically match the feature.

Project: {project}
Language: {language}

Feature query:
{query}

Candidate files:
{candidate_text}

Return strict JSON only. Rank the top {top_k} files.
Each item must include: rank, file, confidence from 0 to 1, and a short rationale.
""",
    "bug_report": """You are performing bug localization from a bug report.

Goal: rank the source files most likely to require a fix.
Use report terms, stack/API clues, component names, symptoms, and likely control/data flow.

Project: {project}
Language: {language}

Bug report:
{query}

Candidate files:
{candidate_text}

Return strict JSON only. Rank the top {top_k} files.
Each item must include: rank, file, confidence from 0 to 1, and a short rationale.
""",
    "terse_ranking": """Rank files for {task_name}.

Project: {project}
Query:
{query}

Files:
{candidate_text}

Return only JSON array with top {top_k}: rank, file, confidence, rationale.
""",
}


def normalize_path(path: Any) -> str:
    """Normalize paths for file-level matching across datasets."""

    if path is None:
        return ""
    return str(path).replace("\\", "/").strip().lstrip("./").lower()


def parse_ranked_locations(raw: str) -> list[RankedLocation]:
    """Parse an LLM JSON response into ranked file predictions.

    Expected shape:
      [{"file": "src/Foo.java", "rank": 1, "confidence": 0.83, "rationale": "..."}]
    """

    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("locations") or data.get("files") or data.get("ranked_files") or []
    if not isinstance(data, list):
        return []

    out: list[RankedLocation] = []
    seen: set[str] = set()
    for idx, item in enumerate(data, start=1):
        if isinstance(item, str):
            file_name = item
            confidence = None
            rationale = ""
            rank = idx
        elif isinstance(item, dict):
            file_name = item.get("file") or item.get("path") or item.get("filename")
            confidence = _to_float_or_none(item.get("confidence") or item.get("score"))
            rationale = str(item.get("rationale") or item.get("reason") or "")
            rank = _to_int_or_none(item.get("rank")) or idx
        else:
            continue
        norm = normalize_path(file_name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(RankedLocation(file=str(file_name), rank=int(rank), confidence=confidence, rationale=rationale))
    return sorted(out, key=lambda p: p.rank)


def ranking_metrics(predictions: Iterable[RankedLocation], gold_locations: Iterable[LocalizationGold],
                    ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Compute standard file-level ranking metrics for localization tasks."""

    preds = list(predictions)
    gold = {g.normalized_file for g in gold_locations if g.normalized_file}
    if not gold:
        return {**{f"hit@{k}": 0.0 for k in ks}, **{f"recall@{k}": 0.0 for k in ks}, "mrr": 0.0, "map": 0.0}

    pred_files = [p.normalized_file for p in sorted(preds, key=lambda p: p.rank) if p.normalized_file]
    metrics: dict[str, float] = {}
    for k in ks:
        top = pred_files[:k]
        hits = len({g for g in gold if any(paths_match(p, g) for p in top)})
        metrics[f"hit@{k}"] = 1.0 if hits > 0 else 0.0
        metrics[f"recall@{k}"] = hits / len(gold)

    rr = 0.0
    precisions = []
    found_gold: set[str] = set()
    for rank, file_name in enumerate(pred_files, start=1):
        matched = next((g for g in gold if g not in found_gold and paths_match(file_name, g)), None)
        if matched:
            if rr == 0.0:
                rr = 1.0 / rank
            found_gold.add(matched)
            precisions.append(len(found_gold) / rank)
    metrics["mrr"] = rr
    metrics["map"] = sum(precisions) / len(gold) if precisions else 0.0
    return metrics


def paths_match(predicted: Any, gold: Any) -> bool:
    """Return True when two code paths match exactly or by source-root suffix."""

    p = normalize_path(predicted)
    g = normalize_path(gold)
    if not p or not g:
        return False
    return p == g or p.endswith(f"/{g}") or g.endswith(f"/{p}")


def build_location_prompt(
    task: MaintenanceTask,
    candidates: Iterable[str],
    top_k: int = 10,
    prompt_style: str = "baseline",
    prompt_templates: Optional[dict[str, str]] = None,
) -> str:
    """Build a task-specific prompt for file-level bug/feature localization."""

    task_name = {
        MaintenanceTaskType.FEATURE_LOCATION: "feature location",
        MaintenanceTaskType.BUG_LOCATION: "bug localization",
        MaintenanceTaskType.STATIC_ANALYSIS: "static analysis",
    }.get(task.task_type, "code localization")
    candidate_text = "\n".join(f"- {c}" for c in candidates)
    templates = prompt_templates or LOCATION_PROMPT_TEMPLATES
    template = templates.get(prompt_style) or templates.get("baseline") or LOCATION_PROMPT_TEMPLATES["baseline"]
    return template.format(
        task_name=task_name,
        project=task.project,
        language=task.language or "unknown",
        candidate_level=task.candidate_level,
        query=task.query,
        candidate_text=candidate_text,
        top_k=top_k,
    )


def run_location_task(
    task: MaintenanceTask,
    candidates: Iterable[str],
    llm: Any,
    top_k: int = 10,
    prompt_style: str = "baseline",
    prompt_templates: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Run one feature/bug localization task with a ChatModel-like object.

    The `llm` object is expected to provide `chat(..., return_meta=True)`, like
    `StaLLM_llm.ChatModel`. Keeping the type loose avoids coupling this module
    to one concrete LLM implementation.
    """

    prompt = build_location_prompt(
        task,
        candidates,
        top_k=top_k,
        prompt_style=prompt_style,
        prompt_templates=prompt_templates,
    )
    content, meta = llm.chat(
        messages=[
            {"role": "system", "content": "You rank source files for software maintenance localization. Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1200,
        temperature=0,
        return_meta=True,
    )
    predictions = parse_ranked_locations(content)[:top_k]
    metrics = ranking_metrics(predictions, task.gold_locations, ks=(1, 5, 10))
    return {
        "task_id": task.task_id,
        "task_type": task.task_type.value,
        "project": task.project,
        "predictions": [p.__dict__ for p in predictions],
        "metrics": metrics,
        "prompt_style": prompt_style,
        "raw_response": content,
        "usage": {
            "prompt_tokens": int((meta or {}).get("prompt_tokens", 0)),
            "completion_tokens": int((meta or {}).get("completion_tokens", 0)),
            "total_tokens": int((meta or {}).get("total_tokens", 0)),
        },
    }


def summarize_location_results(results: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Average ranking metrics over several localization task results."""

    rows = list(results)
    if not rows:
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in (row.get("metrics") or {}).items():
            try:
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
            except Exception:
                continue
    return {key: totals[key] / counts[key] for key in sorted(totals) if counts.get(key)}


def list_repo_candidates(repo_zip: str, allowed_exts: Optional[Iterable[str]] = None) -> list[str]:
    """List source-code candidate paths from a repository ZIP."""

    allowed = tuple(ext.lower() for ext in allowed_exts or [])
    candidates: list[str] = []
    with zipfile.ZipFile(repo_zip, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/") or name.startswith("__MACOSX/"):
                continue
            base = os.path.basename(name)
            if base.startswith("."):
                continue
            if allowed and not name.lower().endswith(allowed):
                continue
            candidates.append(name)
    return sorted(candidates)


def load_localization_tasks_csv(csv_path: str, task_type: MaintenanceTaskType | str,
                                *, repo_zip: Optional[str] = None,
                                project: Optional[str] = None,
                                language: Optional[str] = None) -> list[MaintenanceTask]:
    """Load a simple feature/bug localization benchmark CSV.

    Supported columns are intentionally permissive:
    - id/task_id/bug_id/feature_id
    - project
    - query OR title/body/description/summary
    - files/gold_files/changed_files/locations, separated by ;, comma, or newline
    """

    ttype = MaintenanceTaskType(str(task_type))
    tasks: list[MaintenanceTask] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            low = {str(k).lower(): v for k, v in row.items() if k is not None}
            task_id = str(_pick(low, "task_id", "id", "bug_id", "issue_id", "feature_id") or row_idx)
            task_project = str(_pick(low, "project", "repo", "repository") or project or Path(csv_path).stem)
            query = _build_query_from_row(low)
            gold_files = _split_locations(_pick(low, "gold_files", "changed_files", "files", "locations", "file"))
            gold = tuple(LocalizationGold(file=f) for f in gold_files)
            if not query or not gold:
                continue
            tasks.append(MaintenanceTask(
                task_id=task_id,
                task_type=ttype,
                project=task_project,
                query=query,
                gold_locations=gold,
                repo_zip=repo_zip,
                language=language,
                metadata={"source_csv": csv_path, "row": row_idx},
            ))
    return tasks


def _pick(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _build_query_from_row(row: dict[str, Any]) -> str:
    direct = _pick(row, "query", "feature", "bug_report")
    if direct:
        return str(direct).strip()
    parts = []
    for key in ("title", "summary", "description", "body", "steps", "expected", "actual"):
        val = _pick(row, key)
        if val:
            parts.append(f"{key.capitalize()}: {val}")
    return "\n".join(parts).strip()


def _split_locations(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            raw_items = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            raw_items = []
            for chunk in text.replace("\n", ";").split(";"):
                raw_items.extend(part for part in chunk.split(",") if part.strip())
    files = []
    for item in raw_items:
        if isinstance(item, dict):
            item = item.get("file") or item.get("path") or item.get("filename")
        if item:
            files.append(str(item).strip())
    return [f for f in files if f]


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None
