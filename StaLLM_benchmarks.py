# StaLLM_benchmarks.py
"""Benchmark adapters for feature and bug localization datasets."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from StaLLM_tasks import (
    LocalizationGold,
    MaintenanceTask,
    MaintenanceTaskType,
    load_localization_tasks_csv,
)


def load_argouml_feature_tasks(path: str, *, repo_zip: Optional[str] = None) -> list[MaintenanceTask]:
    """Load ArgoUML SPL feature-location mappings.

    The public ArgoUML SPL benchmark has been reused in several formats. This
    adapter accepts CSV/TSV/JSON exports with permissive column names and groups
    rows by feature/scenario. Commonly supported columns include:

    - feature, feature_name, scenario, concern
    - file, path, class, class_name, method, method_name
    - description, query
    """

    path_obj = Path(path)
    if path_obj.is_dir():
        return _load_argouml_directory(path_obj, repo_zip=repo_zip)

    rows = _load_rows(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        feature = _pick(row, "feature", "feature_name", "scenario", "concern", "name")
        if not feature:
            continue
        grouped[str(feature)].append(row)

    tasks: list[MaintenanceTask] = []
    for feature, items in grouped.items():
        gold = []
        descriptions = []
        for row in items:
            file_name = _location_from_row(row)
            if file_name:
                gold.append(LocalizationGold(
                    file=file_name,
                    symbol=_pick(row, "method", "method_name", "class", "class_name"),
                ))
            desc = _pick(row, "description", "query", "summary")
            if desc:
                descriptions.append(str(desc))
        unique_gold = _dedupe_gold(gold)
        if not unique_gold:
            continue
        query = _feature_query(feature, descriptions)
        tasks.append(MaintenanceTask(
            task_id=f"argouml-feature-{_slug(feature)}",
            task_type=MaintenanceTaskType.FEATURE_LOCATION,
            project="ArgoUML",
            query=query,
            gold_locations=tuple(unique_gold),
            repo_zip=repo_zip,
            language="Java",
            candidate_level="file",
            metadata={
                "benchmark": "ArgoUML SPL feature location",
                "feature": feature,
                "source": path,
                "rows": len(items),
            },
        ))
    return tasks


def _load_argouml_directory(path: Path, *, repo_zip: Optional[str] = None) -> list[MaintenanceTask]:
    """Load the ArgoUML SPL benchmark directory format.

    Expected layout:
      featuresInfo/features.txt
      groundTruth/SEQUENCEDIAGRAM.txt

    Ground-truth rows look like Java elements:
      org.argouml.foo.Bar
      org.argouml.foo.Bar method(Type) Refinement

    We evaluate file-level localization, so both forms map to:
      org/argouml/foo/Bar.java
    """

    root = path
    gt_dir = root / "groundTruth"
    if not gt_dir.exists() and root.name.lower() == "groundtruth":
        gt_dir = root
        root = root.parent
    if not gt_dir.exists():
        raise FileNotFoundError(f"ArgoUML groundTruth directory not found under {path}")

    feature_info = _load_argouml_feature_info(root / "featuresInfo" / "features.txt")
    tasks: list[MaintenanceTask] = []
    for gt_file in sorted(gt_dir.glob("*.txt")):
        feature_key = gt_file.stem
        gold = []
        for raw in gt_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            file_name = _argouml_gt_line_to_file(raw)
            if file_name:
                gold.append(LocalizationGold(file=file_name, symbol=_argouml_gt_line_to_symbol(raw)))
        unique_gold = _dedupe_gold(gold)
        if not unique_gold:
            continue
        query = _argouml_query_from_feature_key(feature_key, feature_info)
        tasks.append(MaintenanceTask(
            task_id=f"argouml-feature-{_slug(feature_key)}",
            task_type=MaintenanceTaskType.FEATURE_LOCATION,
            project="ArgoUML",
            query=query,
            gold_locations=tuple(unique_gold),
            repo_zip=repo_zip,
            language="Java",
            candidate_level="file",
            metadata={
                "benchmark": "ArgoUML SPL feature location",
                "feature": feature_key,
                "source": str(gt_file),
                "raw_locations": len(gold),
            },
        ))
    return tasks


def load_bench4bl_bug_tasks(path: str, *, repo_zip_by_project: Optional[dict[str, str]] = None,
                            default_repo_zip: Optional[str] = None) -> list[MaintenanceTask]:
    """Load Bench4BL-style bug localization records.

    Bench4BL-like exports usually contain a bug report identifier, a title/body,
    the project name, and fixed/changed files. This adapter accepts CSV/TSV/JSON
    with permissive columns:

    - bug_id, issue_id, id
    - project, repo, repository
    - title, summary, description, body
    - fixed_files, changed_files, files, gold_files
    """

    path_obj = Path(path)
    if path_obj.suffix.lower() in {".csv", ".tsv"}:
        return load_localization_tasks_csv(
            path,
            MaintenanceTaskType.BUG_LOCATION,
            repo_zip=default_repo_zip,
            project=None,
            language=None,
        )

    rows = _load_rows(path)
    tasks: list[MaintenanceTask] = []
    for idx, row in enumerate(rows, start=1):
        project = str(_pick(row, "project", "repo", "repository") or "unknown")
        bug_id = str(_pick(row, "bug_id", "issue_id", "id", "task_id") or idx)
        query = _bug_query(row)
        gold_files = _split_locations(_pick(row, "fixed_files", "changed_files", "gold_files", "files", "locations", "file"))
        if not query or not gold_files:
            continue
        tasks.append(MaintenanceTask(
            task_id=f"{project}-{bug_id}",
            task_type=MaintenanceTaskType.BUG_LOCATION,
            project=project,
            query=query,
            gold_locations=tuple(LocalizationGold(file=f) for f in gold_files),
            repo_zip=(repo_zip_by_project or {}).get(project) or default_repo_zip,
            candidate_level="file",
            metadata={
                "benchmark": "Bench4BL-style bug localization",
                "bug_id": bug_id,
                "source": path,
                "row": idx,
            },
        ))
    return tasks


def load_location_benchmark(path: str, benchmark: str, **kwargs: Any) -> list[MaintenanceTask]:
    """Dispatch to a known localization benchmark adapter."""

    key = benchmark.strip().lower()
    if key in {"argouml", "argouml-spl", "argouml_feature", "argouml_feature_location"}:
        return load_argouml_feature_tasks(path, repo_zip=kwargs.get("repo_zip"))
    if key in {"bench4bl", "bench4bl_bug", "bug_location"}:
        return load_bench4bl_bug_tasks(
            path,
            repo_zip_by_project=kwargs.get("repo_zip_by_project"),
            default_repo_zip=kwargs.get("repo_zip"),
        )
    raise ValueError(f"Unknown localization benchmark adapter: {benchmark}")


def _load_rows(path: str) -> list[dict[str, Any]]:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix in {".json", ".jsonl", ".ndjson"}:
        text = path_obj.read_text(encoding="utf-8")
        if suffix == ".json":
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("tasks") or data.get("items") or data.get("records") or []
            return [_normalize_keys(row) for row in data if isinstance(row, dict)]
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(_normalize_keys(item))
        return rows

    delimiter = "\t" if suffix == ".tsv" else None
    with open(path_obj, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except Exception:
                delimiter = ","
        return [_normalize_keys(row) for row in csv.DictReader(f, delimiter=delimiter)]


def _normalize_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k).strip().lower(): v for k, v in row.items() if k is not None}


def _pick(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _location_from_row(row: dict[str, Any]) -> Optional[str]:
    file_name = _pick(row, "file", "path", "filepath", "file_path", "class_file", "source_file")
    if file_name:
        return str(file_name)
    class_name = _pick(row, "class", "class_name", "classname")
    if class_name:
        text = str(class_name).replace(".", "/").strip()
        if not text.endswith(".java"):
            text = f"{text}.java"
        return text
    return None


def _load_argouml_feature_info(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    info = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in raw.split(";")]
        if len(parts) < 2 or not parts[0]:
            continue
        key = parts[0]
        aliases = parts[1] if len(parts) > 1 else key
        label = aliases.split(",")[-1].strip() or aliases
        description = parts[2] if len(parts) > 2 else ""
        if len(parts) > 3:
            label = parts[2] or label
            description = parts[3]
        info[key] = {"aliases": aliases, "label": label, "description": description}
    return info


def _argouml_gt_line_to_file(raw: str) -> Optional[str]:
    line = raw.strip()
    if not line:
        return None
    line = line.removesuffix(" Refinement").strip()
    java_element = line.split()[0]
    if not java_element.startswith("org.argouml."):
        return None
    return java_element.replace(".", "/") + ".java"


def _argouml_gt_line_to_symbol(raw: str) -> Optional[str]:
    line = raw.strip().removesuffix(" Refinement").strip()
    parts = line.split(maxsplit=1)
    return parts[1] if len(parts) > 1 else None


def _argouml_query_from_feature_key(feature_key: str, feature_info: dict[str, dict[str, str]]) -> str:
    parts = feature_key.split("_and_")
    query_parts = []
    for part in parts:
        if part.startswith("not_"):
            base = part[4:]
            label = feature_info.get(base, {}).get("label", base)
            query_parts.append(f"Exclude feature: {label}")
            continue
        meta = feature_info.get(part, {})
        label = meta.get("label", part)
        desc = meta.get("description", "")
        if desc:
            query_parts.append(f"Feature: {label}\nDescription: {desc}")
        else:
            query_parts.append(f"Feature: {label}")
    return "\n\n".join(query_parts)


def _feature_query(feature: str, descriptions: Iterable[str]) -> str:
    desc = next((d for d in descriptions if d), "")
    if desc:
        return f"Feature: {feature}\nDescription: {desc}"
    return f"Locate the implementation of the ArgoUML feature: {feature}"


def _bug_query(row: dict[str, Any]) -> str:
    parts = []
    for key in ("title", "summary", "description", "body", "steps", "expected", "actual"):
        value = _pick(row, key)
        if value:
            parts.append(f"{key.capitalize()}: {value}")
    direct = _pick(row, "query", "bug_report")
    if direct:
        parts.insert(0, str(direct))
    return "\n".join(parts).strip()


def _split_locations(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            items = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            items = []
            for chunk in text.replace("\n", ";").split(";"):
                items.extend(part for part in chunk.split(",") if part.strip())
    out = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("file") or item.get("path") or item.get("filename")
        if item:
            out.append(str(item).strip())
    return [x for x in out if x]


def _dedupe_gold(items: Iterable[LocalizationGold]) -> list[LocalizationGold]:
    seen = set()
    out = []
    for item in items:
        key = item.normalized_file
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _slug(value: Any) -> str:
    raw = str(value).strip().lower()
    keep = []
    last_dash = False
    for ch in raw:
        if ch.isalnum():
            keep.append(ch)
            last_dash = False
        elif not last_dash:
            keep.append("-")
            last_dash = True
    return "".join(keep).strip("-") or "feature"
