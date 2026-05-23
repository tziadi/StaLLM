# StaLLM_benchmarks.py
"""Benchmark adapters for feature and bug localization datasets."""

from __future__ import annotations

import csv
import json
import os
import re
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from StaLLM_tasks import (
    LocalizationGold,
    MaintenanceTask,
    MaintenanceTaskType,
    load_localization_tasks_csv,
)

DACOS_SMELL_LABELS = {
    1: ("multifaceted_abstraction", True),
    4: ("multifaceted_abstraction", False),
    2: ("long_parameter_list", True),
    5: ("long_parameter_list", False),
    3: ("complex_method", True),
    6: ("complex_method", False),
}

DACOS_SAMPLE_SMELL_IDS = {
    1: "complex_method",
    2: "long_parameter_list",
    3: "multifaceted_abstraction",
}

MLCQ_SEVERITY_ORDER = {
    "none": 0,
    "no": 0,
    "absent": 0,
    "false": 0,
    "0": 0,
    "minor": 1,
    "low": 1,
    "1": 1,
    "major": 2,
    "medium": 2,
    "2": 2,
    "critical": 3,
    "high": 3,
    "3": 3,
}


@dataclass(frozen=True)
class SmellSample:
    """A human-oracle code-smell sample such as DACOS/DACOSX."""

    sample_id: str
    smell: str
    label: bool
    code: str
    language: str = "Java"
    source: str = ""
    metadata: dict[str, Any] | None = None


def load_dacos_smell_samples(path: str, *, files_root: Optional[str] = None, limit: Optional[int] = None) -> list[SmellSample]:
    """Load DACOS/DACOSX human smell samples.

    DACOS is published as MySQL dumps plus a source-code archive. For practical
    use inside StarLLM, this adapter accepts either:

    - CSV/TSV/JSON exports with permissive columns (`code`, `smell`, `label`),
    - SQL dumps where rows contain smell ids and either inline code or file ids.

    Smell ids follow the DACOS documentation:
      1/4 multifaceted abstraction present/absent
      2/5 long parameter list present/absent
      3/6 complex method present/absent
    """

    path_obj = Path(path)
    if path_obj.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl", ".ndjson"}:
        rows = _load_rows(path)
    elif path_obj.suffix.lower() == ".sql":
        sql_text = path_obj.read_text(encoding="utf-8", errors="ignore")
        if "`path_to_file`" in sql_text and "INSERT INTO `sample`" in sql_text:
            return _load_dacos_sample_table(sql_text, path_obj=path_obj, files_root=files_root, limit=limit)
        rows = _load_dacos_sql_rows(path_obj)
    else:
        raise ValueError(f"Unsupported DACOS input format: {path_obj.suffix or path_obj}")

    samples: list[SmellSample] = []
    for idx, row in enumerate(rows, start=1):
        sample = _dacos_sample_from_row(row, idx=idx, files_root=files_root, source=str(path_obj))
        if sample:
            samples.append(sample)
            if limit and len(samples) >= limit:
                break
    return samples


def load_mlcq_smell_samples(
    path: str,
    *,
    limit: Optional[int] = None,
    positive_threshold: str = "minor",
    fetch_remote: bool = True,
) -> list[SmellSample]:
    """Load MLCQ human smell samples from the official CSV/XLSX export converted to CSV.

    MLCQ labels Java snippets with smell type and severity. Official records vary
    slightly across derived exports, so this adapter accepts permissive column
    names for code, smell and severity.
    """

    rows = _load_rows(path)
    threshold = MLCQ_SEVERITY_ORDER.get(str(positive_threshold).strip().lower(), 1)
    samples: list[SmellSample] = []
    for idx, row in enumerate(rows, start=1):
        smell = _normalize_smell_name(str(_pick(row, "smell", "code smell", "codesmell", "code_smell", "type", "smell type", "smell_type") or ""))
        if not smell:
            continue
        severity_value = _pick(row, "severity", "label", "class", "classification", "smell severity", "smell_severity")
        severity = _mlcq_severity_score(severity_value)
        if severity is None:
            label_value = _pick(row, "is_smelly", "present", "has_smell", "is_smell")
            label = _to_bool_or_none(label_value)
            severity = 1 if label else 0 if label is not None else None
        if severity is None:
            continue
        code = str(_pick(row, "code", "snippet", "source", "source_code", "reviewed code", "reviewed_code", "method", "class_body") or "")
        if len(code.strip()) < 40 and fetch_remote:
            code = _read_mlcq_remote_snippet(row, cache_root=Path(path).parent / ".mlcq_cache")
        if len(code.strip()) < 40:
            continue
        sample_id = str(_pick(row, "id", "sample_id", "instance_id", "review_id", "snippet_id") or f"mlcq-{idx}")
        samples.append(
            SmellSample(
                sample_id=sample_id,
                smell=smell,
                label=severity >= threshold,
                code=code,
                language="Java",
                source=str(path),
                metadata={
                    **{k: v for k, v in row.items() if k not in {"code", "snippet", "source_code", "reviewed_code"}},
                    "severity_score": severity,
                    "positive_threshold": threshold,
                },
            )
        )
        if limit and len(samples) >= limit:
            break
    return samples


def _load_dacos_sample_table(
    sql_text: str,
    *,
    path_obj: Path,
    files_root: Optional[str],
    limit: Optional[int],
) -> list[SmellSample]:
    source_root = _resolve_dacos_files_root(path_obj, files_root)
    if not source_root:
        return []

    samples: list[SmellSample] = []
    buckets: dict[str, list[SmellSample]] = defaultdict(list)
    quota = max(1, (int(limit) + 2) // 3) if limit else None
    sample_re = re.compile(r"INSERT\s+INTO\s+`sample`\s+VALUES\s*(?P<values>.*?);", re.IGNORECASE | re.DOTALL)
    for match in sample_re.finditer(sql_text):
        for values in _iter_sql_value_groups(match.group("values")):
            parsed = _parse_sql_values(values)
            if len(parsed) < 8:
                continue
            sample_id, designite_id, has_smell, is_class, file_ref, project_name, constraints, smells = parsed[:8]
            smell_ids = {
                int(value)
                for value in re.findall(r"\d+", str(smells or ""))
                if int(value) in DACOS_SAMPLE_SMELL_IDS
            }
            class_level = bool(_to_bool_or_none(is_class))
            targets = ["multifaceted_abstraction"] if class_level else ["complex_method", "long_parameter_list"]
            if quota and all(len(buckets[target]) >= quota for target in targets):
                continue
            code = _read_dacos_file(str(source_root), str(file_ref or ""))
            if len(code.strip()) < 80:
                continue
            for smell in targets:
                if quota and len(buckets[smell]) >= quota:
                    continue
                positive_id = next((sid for sid, name in DACOS_SAMPLE_SMELL_IDS.items() if name == smell), None)
                label = positive_id in smell_ids if positive_id is not None else bool(_to_bool_or_none(has_smell))
                sample = SmellSample(
                    sample_id=f"{sample_id}:{smell}",
                    smell=smell,
                    label=label,
                    code=code,
                    language="Java",
                    source=str(path_obj),
                    metadata={
                        "dacos_sample_id": sample_id,
                        "designite_id": designite_id,
                        "project": project_name,
                        "path": file_ref,
                        "sample_constraints": constraints,
                        "smell_ids": sorted(smell_ids),
                        "source_archive": str(source_root),
                    },
                )
                if quota:
                    buckets[smell].append(sample)
                else:
                    samples.append(sample)
            if quota and all(len(buckets[smell]) >= quota for smell in DACOS_SAMPLE_SMELL_IDS.values()):
                return _flatten_dacos_sample_buckets(buckets, limit)
    return _flatten_dacos_sample_buckets(buckets, limit) if quota else samples


def _flatten_dacos_sample_buckets(buckets: dict[str, list[SmellSample]], limit: Optional[int]) -> list[SmellSample]:
    ordered: list[SmellSample] = []
    smells = ["complex_method", "long_parameter_list", "multifaceted_abstraction"]
    max_len = max((len(items) for items in buckets.values()), default=0)
    for idx in range(max_len):
        for smell in smells:
            if idx < len(buckets[smell]):
                ordered.append(buckets[smell][idx])
                if limit and len(ordered) >= limit:
                    return ordered
    return ordered


def _dacos_sample_from_row(row: dict[str, Any], *, idx: int, files_root: Optional[str], source: str) -> Optional[SmellSample]:
    smell_id = _to_int_or_none(_pick(row, "smell_id", "smellid", "condition", "condition_id", "code_smell", "label_id", "id_smell"))
    smell_name = str(_pick(row, "smell", "smell_name", "type", "code_smell_name") or "").strip()
    label_value = _pick(row, "label", "is_smelly", "present", "oracle", "annotation", "class")

    if smell_id in DACOS_SMELL_LABELS:
        smell, label = DACOS_SMELL_LABELS[int(smell_id)]
    else:
        smell = _normalize_smell_name(smell_name)
        label = _to_bool_or_none(label_value)

    if not smell or label is None:
        return None

    code = str(_pick(row, "code", "snippet", "source", "source_code", "method", "class_body") or "")
    file_ref = _pick(row, "file", "path", "file_path", "filepath", "filename", "source_file", "fileid", "file_id")
    if not code and file_ref and files_root:
        code = _read_dacos_file(files_root, str(file_ref))
    if not code.strip():
        return None

    sample_id = str(_pick(row, "sample_id", "snippet_id", "code_id", "instance_id", "id") or f"dacos-{idx}")
    language = str(_pick(row, "language", "lang") or "Java")
    return SmellSample(
        sample_id=sample_id,
        smell=smell,
        label=bool(label),
        code=code,
        language=language,
        source=source,
        metadata={k: v for k, v in row.items() if k not in {"code", "snippet", "source_code"}},
    )


def _load_dacos_sql_rows(path: Path) -> list[dict[str, Any]]:
    """Best-effort MySQL INSERT parser for DACOS exports.

    The public dumps can be imported into MySQL for full fidelity. This parser is
    intentionally permissive so users can still inspect/evaluate subsets without
    a database server.
    """

    text = path.read_text(encoding="utf-8", errors="ignore")
    rows: list[dict[str, Any]] = []
    insert_re = re.compile(r"INSERT\s+INTO\s+`?(?P<table>\w+)`?\s*(?:\((?P<cols>[^)]*)\))?\s*VALUES\s*(?P<values>.*?);", re.IGNORECASE | re.DOTALL)
    for match in insert_re.finditer(text):
        table = match.group("table")
        cols = _parse_sql_columns(match.group("cols") or "")
        if not _dacos_table_looks_relevant(table, cols):
            continue
        for values in _split_sql_value_groups(match.group("values")):
            parsed = _parse_sql_values(values)
            if not parsed:
                continue
            if cols and len(cols) == len(parsed):
                row = dict(zip(cols, parsed))
            else:
                row = {f"col_{i}": value for i, value in enumerate(parsed)}
                row["table"] = table
            row["table"] = table
            rows.append(_normalize_keys(row))
    return rows


def _dacos_table_looks_relevant(table: str, cols: list[str]) -> bool:
    low_table = table.lower()
    haystack = " ".join([low_table, *cols]).lower()
    return any(token in haystack for token in ("annotation", "snippet", "sample", "smell", "code", "file"))


def _parse_sql_columns(raw: str) -> list[str]:
    if not raw:
        return []
    return [part.strip().strip("`").strip().lower() for part in raw.split(",") if part.strip()]


def _split_sql_value_groups(raw: str) -> list[str]:
    return list(_iter_sql_value_groups(raw))


def _iter_sql_value_groups(raw: str) -> Iterable[str]:
    groups: list[str] = []
    depth = 0
    in_quote = False
    escape = False
    start = None
    for idx, ch in enumerate(raw):
        if in_quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_quote = False
            continue
        if ch == "'":
            in_quote = True
        elif ch == "(":
            if depth == 0:
                start = idx + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                yield raw[start:idx]
                start = None


def _parse_sql_values(raw: str) -> list[Any]:
    values: list[Any] = []
    current: list[str] = []
    in_quote = False
    escape = False
    for ch in raw:
        if in_quote:
            if escape:
                current.append(_sql_escape(ch))
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_quote = False
            else:
                current.append(ch)
            continue
        if ch == "'":
            in_quote = True
        elif ch == ",":
            values.append(_coerce_sql_value("".join(current).strip()))
            current = []
        else:
            current.append(ch)
    values.append(_coerce_sql_value("".join(current).strip()))
    return values


def _sql_escape(ch: str) -> str:
    return {"n": "\n", "r": "\r", "t": "\t", "0": "\0"}.get(ch, ch)


def _coerce_sql_value(value: str) -> Any:
    if value.lower().startswith("_binary"):
        if "\x01" in value or "\\x01" in value or "'1'" in value:
            return 1
        return 0
    if value.upper() == "NULL":
        return None
    if re.fullmatch(r"-?\d+", value or ""):
        try:
            return int(value)
        except Exception:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value or ""):
        try:
            return float(value)
        except Exception:
            return value
    return value


def _normalize_smell_name(value: str) -> str:
    raw = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    aliases = {
        "data_class": "data_class",
        "dataclass": "data_class",
        "feature_envy": "feature_envy",
        "god_class": "god_class",
        "godclass": "god_class",
        "blob": "god_class",
        "long_parameter_list": "long_parameter_list",
        "long_param_list": "long_parameter_list",
        "complex_method": "complex_method",
        "long_method": "long_method",
        "longmethod": "long_method",
        "multifaceted_abstraction": "multifaceted_abstraction",
        "multi_faceted_abstraction": "multifaceted_abstraction",
    }
    return aliases.get(raw, raw)


def _mlcq_severity_score(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in MLCQ_SEVERITY_ORDER:
        return MLCQ_SEVERITY_ORDER[text]
    try:
        number = int(float(text))
        return max(0, min(3, number))
    except Exception:
        return None


def _to_bool_or_none(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present", "smelly", "positive"}:
        return True
    if text in {"0", "false", "no", "n", "absent", "benign", "negative", "not_detected"}:
        return False
    return None


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(str(value)))
    except Exception:
        return None


def _read_dacos_file(files_root: str, file_ref: str) -> str:
    root = Path(files_root)
    normalized_ref = str(file_ref).replace("\\", "/").lstrip("/")
    if root.is_file() and root.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(root) as archive:
                names = archive.namelist()
                member = normalized_ref if normalized_ref in names else next(
                    (name for name in names if name.endswith(normalized_ref) or normalized_ref.endswith(name)),
                    "",
                )
                if member:
                    with archive.open(member) as handle:
                        return handle.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""
    direct = root / file_ref
    if direct.exists() and direct.is_file():
        return direct.read_text(encoding="utf-8", errors="ignore")
    direct = root / normalized_ref
    if direct.exists() and direct.is_file():
        return direct.read_text(encoding="utf-8", errors="ignore")
    target = normalized_ref.split("/")[-1]
    for path in root.rglob(target):
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _resolve_dacos_files_root(sql_path: Path, files_root: Optional[str]) -> Optional[Path]:
    candidates = []
    if files_root:
        candidates.append(Path(files_root))
    candidates.extend([
        sql_path.parent / "files",
        sql_path.parent / "files.zip",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_mlcq_remote_snippet(row: dict[str, Any], *, cache_root: Path) -> str:
    url = str(_pick(row, "link", "url", "github_url") or "").strip()
    if not url:
        repo = str(_pick(row, "repository") or "").strip()
        commit = str(_pick(row, "commit_hash", "commit", "sha") or "").strip()
        path = str(_pick(row, "path", "file", "file_path") or "").strip().lstrip("/")
        url = _mlcq_raw_url_from_parts(repo, commit, path)
    else:
        url = _github_blob_to_raw_url(url)
    if not url:
        return ""

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", url)[-220:]
    cache_path = cache_root / cache_name
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "StaLLM-MLCQ-loader"})
            with urllib.request.urlopen(request, timeout=12) as response:
                text = response.read().decode("utf-8", errors="ignore")
            cache_path.write_text(text, encoding="utf-8")
        except Exception:
            return ""

    start = _to_int_or_none(_pick(row, "start_line", "startline", "line_start"))
    end = _to_int_or_none(_pick(row, "end_line", "endline", "line_end"))
    if start and end and end >= start:
        lines = text.splitlines()
        pad = 8
        lo = max(1, start - pad)
        hi = min(len(lines), end + pad)
        return "\n".join(lines[lo - 1:hi])
    return text


def _github_blob_to_raw_url(url: str) -> str:
    clean = str(url).split("#", 1)[0].rstrip("/")
    match = re.match(r"https://github\.com/([^/]+/[^/]+)/blob/([^/]+)/(.+)", clean)
    if match:
        repo, commit, path = match.groups()
        return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    return clean if "raw.githubusercontent.com" in clean else ""


def _mlcq_raw_url_from_parts(repo: str, commit: str, path: str) -> str:
    if not repo or not commit or not path:
        return ""
    clean_repo = repo.strip()
    if clean_repo.startswith("git@github.com:"):
        clean_repo = clean_repo.removeprefix("git@github.com:").removesuffix(".git")
    elif clean_repo.startswith("https://github.com/"):
        clean_repo = clean_repo.removeprefix("https://github.com/").removesuffix(".git")
    else:
        return ""
    return f"https://raw.githubusercontent.com/{clean_repo}/{commit}/{path}"


DACOS_PROMPT_TEMPLATE = """You are evaluating a human-annotated code smell benchmark.

Task: decide whether the following code snippet contains the target smell.

Target smell: {smell}
Language: {language}

Decision rules:
- Return JSON only, no markdown.
- Schema: {{"present": true|false, "confidence": 0.0, "rationale": "short evidence"}}
- Focus on the target smell only.
- Prefer a conservative decision when evidence is weak.

Code:
{code}
"""


def run_dacos_smell_benchmark(
    samples: Iterable[SmellSample],
    llm: Any,
    *,
    prompt_template: str = DACOS_PROMPT_TEMPLATE,
    max_samples: Optional[int] = None,
) -> dict[str, Any]:
    """Run a binary LLM smell classifier over DACOS/DACOSX samples."""

    rows = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for idx, sample in enumerate(samples, start=1):
        if max_samples and idx > max_samples:
            break
        prompt = _safe_dacos_format(prompt_template, {
            "smell": sample.smell,
            "language": sample.language,
            "code": sample.code[:12000],
        })
        content, meta = llm.chat(
            messages=[
                {"role": "system", "content": "You classify code smells against a human oracle. Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=700,
            return_meta=True,
        )
        pred, confidence, rationale = _parse_dacos_prediction(content)
        rows.append({
            "sample_id": sample.sample_id,
            "smell": sample.smell,
            "gold": bool(sample.label),
            "predicted": pred,
            "confidence": confidence,
            "rationale": rationale,
            "raw_response": content,
        })
        for key in usage:
            usage[key] += int((meta or {}).get(key, 0))

    metrics = _binary_metrics(rows)
    return {"rows": rows, "metrics": metrics, "usage": usage}


def _parse_dacos_prediction(raw: Any) -> tuple[bool, Optional[float], str]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    present = _to_bool_or_none((data or {}).get("present") if isinstance(data, dict) else None)
    confidence = None
    if isinstance(data, dict):
        try:
            confidence = float(data.get("confidence")) if data.get("confidence") is not None else None
        except Exception:
            confidence = None
    rationale = str(data.get("rationale") or data.get("reason") or "") if isinstance(data, dict) else ""
    return bool(present), confidence, rationale


def _binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row.get("gold") is True and row.get("predicted") is True)
    fp = sum(1 for row in rows if row.get("gold") is False and row.get("predicted") is True)
    fn = sum(1 for row in rows if row.get("gold") is True and row.get("predicted") is False)
    tn = sum(1 for row in rows if row.get("gold") is False and row.get("predicted") is False)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "samples": len(rows),
    }


def _safe_dacos_format(template: str, values: dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except (KeyError, ValueError):
        protected = str(template)
        tokens: dict[str, str] = {}
        for key in sorted(values, key=len, reverse=True):
            token = f"@@DACOS_{key.upper()}@@"
            tokens[key] = token
            protected = protected.replace("{" + key + "}", token)
        protected = protected.replace("{", "{{").replace("}", "}}")
        for key, token in tokens.items():
            protected = protected.replace(token, "{" + key + "}")
        return protected.format(**values)


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
            header = sample.splitlines()[0] if sample.splitlines() else ""
            if ";" in header and "," not in header:
                delimiter = ";"
            else:
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
