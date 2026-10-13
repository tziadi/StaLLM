# StaLLM_v2_core.py
import os, re, json, zipfile, tempfile, time, random
from typing import Dict, List, Any, Tuple, Optional, NamedTuple
from pathlib import Path

import pandas as pd

try:
    from StaLLM_models import init_db
except Exception:
    def init_db():  # no-op if models layer is not present
        pass

from StaLLM_llm import ChatModel, LLMConfig, build_llm_from_slot, default_slot_key


# ---------- .env loader (robust) ----------
def _load_env_fallback(dotenv_path: Path, override: bool = False) -> bool:
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                if key and (override or key not in os.environ):
                    os.environ[key] = val
        return True
    except FileNotFoundError:
        return False

def _load_dotenv_robust(path: Path, override: bool = False) -> None:
    try:
        from dotenv import load_dotenv as _ld  # type: ignore
        _ld(dotenv_path=path, override=override)
    except Exception:
        _load_env_fallback(path, override=override)

_load_dotenv_robust(Path(__file__).with_name(".env"), override=False)
init_db()

# =========================
# Supported Languages
# =========================
LANGUAGE_EXTS: Dict[str, List[str]] = {
    "C#": [".cs"],
    "Java": [".java"],
    "PHP": [".php"],
    "Python": [".py"],
    "JavaScript": [".js", ".jsx"],
    "TypeScript": [".ts", ".tsx"],
    "Go": [".go"],
    "C++": [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h"],
}
def find_exts_for_language(language: str) -> List[str]:
    return LANGUAGE_EXTS.get(language, [])

# =========================
# Strategies (prompts)
# =========================
STRATEGY_FILE = "strategies.json"

# Built-in fallback strategies to avoid silent failure when strategies.json is missing
DEFAULT_STRATEGIES: Dict[str, str] = {
    "baseline": """You are a static-analysis annotator.
Language: {language}

TASK: Given the source code below, find concrete maintainability/code-smell issues.
Return findings as JSON ONLY (array). No prose. No markdown. No code fences.
Each finding MUST be an object with: "type" (string), "description" (string),
and either "line" (int) OR ("startLine" (int), "endLine" (int)).
Prefer spans; for single-line you may use {"line": N}.
Provide at least 3 plausible findings when possible.

CODE START
{code}
CODE END""",

    "java_strict": """You are a static-analysis annotator for Java.
Return a JSON ARRAY ONLY. No prose, no fences.
Each item: {"type": string, "description": string, and either "line": int
OR ("startLine": int, "endLine": int)}.
Prefer common Java smells: Long Method, Deeply Nested Ifs, Magic Number,
Long Parameter List, God Class, Tight Coupling, Unused Variable,
Null Dereference Risk, Resource Leak, Duplicated Code.
Return at least 3 plausible findings when reasonable.

{code}""",

    "php_strict": """You are a static-analysis annotator for PHP.
Return a JSON ARRAY ONLY. No prose, no fences.
Each item: {"type": string, "description": string, and either "line": int
OR ("startLine": int, "endLine": int)}.
Prefer: Long Method, Deeply Nested Ifs, Magic Number, Long Parameter List,
God Class, Tight Coupling, Unused Variable, SQL Injection Risk, XSS Risk,
Null Dereference Risk, Resource Leak.
Return at least 3 plausible findings when reasonable.

{code}"""
}

def load_strategies(file_path: str = STRATEGY_FILE) -> Dict[str, str]:
    """Merge built-in defaults with on-disk strategies.json if present."""
    strategies = dict(DEFAULT_STRATEGIES)
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
            if isinstance(on_disk, dict):
                strategies.update({str(k): str(v) for k, v in on_disk.items()})
    except Exception:
        # Keep defaults on parse error
        pass
    return strategies

def save_strategies(strategies: Dict[str, str], file_path: str = STRATEGY_FILE):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(strategies, f, indent=2)

# =========================
# Language detection
# =========================
def detect_language_from_zip(zip_path: str) -> str:
    counts: Dict[str, int] = {k: 0 for k in LANGUAGE_EXTS}
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith("/") or name.startswith("__MACOSX/"):
                continue
            ext = os.path.splitext(name)[1].lower()
            for lang, exts in LANGUAGE_EXTS.items():
                if ext in exts:
                    counts[lang] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if any(counts.values()) else "Unknown"

# =========================
# JSON parsing (robust) + normalization helpers
# =========================
_JSON_ARRAY_FINDER = re.compile(r"\[\s*[\s\S]*\s*\]")

def _to_int_or_none(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, int):
            return x
        s = str(x).strip()
        if s == "" or s.lower() == "null":
            return None
        return int(float(s))
    except Exception:
        return None

def _sanitize_to_json_array_text(raw: str) -> str:
    """
    Always return text that is a valid JSON array.
    - If raw is an array -> return as-is (if valid).
    - If raw is a single object -> wrap into [ ... ].
    - If raw is any JSON scalar (e.g., "type") -> return "[]".
    - Otherwise try to extract the first [...] block; else "[]".
    """
    if not raw:
        return "[]"
    s = raw.strip()

    if s.startswith("[") and s.endswith("]"):
        try:
            json.loads(s)
            return s
        except Exception:
            pass

    if s.startswith("{") and s.endswith("}"):
        try:
            json.loads(s)
            return f"[{s}]"
        except Exception:
            pass

    m = re.search(r"\[\s*[\s\S]*\s*\]", s)
    if m:
        cand = m.group(0)
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass

    try:
        v = json.loads(s)
        if isinstance(v, list):
            return s
        if isinstance(v, dict):
            return f"[{s}]"
        return "[]"
    except Exception:
        return "[]"

def _coerce_items_schema(arr: Any) -> List[Dict[str, Any]]:
    """Normalize a loosely-valid array into our expected item schema (span-first)."""
    def to_int(x):
        try:
            if x is None:
                return None
            if isinstance(x, int):
                return x
            s = str(x).strip()
            if s == "" or s.lower() == "null":
                return None
            return int(float(s))
        except Exception:
            return None

    if not isinstance(arr, list):
        return []

    out = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        t = str(it.get("type", "") or "").strip()
        d = str(it.get("description", "") or "").strip()
        if not t or not d:
            continue

        # Accept either single "line" or a span ("startLine","endLine")
        line = to_int(it.get("line"))
        sline = to_int(it.get("startLine", it.get("start_line")))
        eline = to_int(it.get("endLine", it.get("end_line")))
        if line is not None:
            sline = line if sline is None else sline
            eline = line if eline is None else eline

        # If still missing, skip (no usable location)
        if sline is None and eline is None:
            continue
        if sline is None:
            sline = eline
        if eline is None:
            eline = sline

        sc = to_int(it.get("startColumn", it.get("start_column")))
        ec = to_int(it.get("endColumn", it.get("end_column")))
        out.append({
            "type": t,
            "description": d,
            "startLine": int(sline),
            "endLine": int(eline),
            "startColumn": sc,
            "endColumn": ec,
        })
    return out

def _write_debug_dump(kind: str, text: str, suffix: str = ""):
    """Optional debug dumps if DEBUG_LLM=1."""
    try:
        if not os.getenv("DEBUG_LLM"):
            return
        os.makedirs("llm_debug", exist_ok=True)
        ts = int(time.time() * 1000)
        name = f"llm_debug/{ts}_{kind}{suffix}.txt"
        with open(name, "w", encoding="utf-8") as f:
            f.write(text or "")
    except Exception:
        pass

# =========================
# Heuristic fallbacks (language-agnostic)
# =========================
_METHOD_SIG = re.compile(
    r"""(?ix)
    ^\s*
    (public|private|protected|internal|static|sealed|abstract|virtual|override|\w+\s+)?   # modifiers (optional)
    ([A-Za-z_][A-Za-z0-9_<>,\[\]\.?]*)\s+                                                 # return/type (optional-ish)
    ([A-Za-z_][A-Za-z0-9_]*)\s*                                                            # name
    \([^)]*\)\s*\{                                                                         # body open
    """
)

def _heuristic_long_methods(lines: List[str], max_len: int = 50):
    res = []
    n = len(lines)
    i = 0
    while i < n:
        m = _METHOD_SIG.match(lines[i])
        if not m:
            i += 1
            continue
        start = i + 1  # 1-based lines
        depth = 0
        j = i
        opened = False
        while j < n:
            depth += lines[j].count("{")
            depth -= lines[j].count("}")
            if depth > 0:
                opened = True
            if opened and depth == 0:
                break
            j += 1
        end = j + 1
        length = end - start + 1
        if opened and length >= max_len:
            res.append({
                "type": "Long Method",
                "description": f"Method spans ~{length} lines (>{max_len}).",
                "startLine": start,
                "endLine": end
            })
        i = j + 1 if j > i else i + 1
    return res

def _heuristic_nested_ifs(lines: List[str], min_depth: int = 3):
    res = []
    stack = []
    for idx, line in enumerate(lines, start=1):
        s = line.strip()
        if re.search(r"\bif\s*\(", s):
            stack.append(idx)
            if len(stack) >= min_depth:
                res.append({
                    "type": "Deeply Nested Ifs",
                    "description": f"Nested if depth ≥{min_depth}.",
                    "startLine": max(1, idx-1),
                    "endLine": idx
                })
        if "}" in s and stack:
            stack.pop()
    return res

def _heuristic_magic_numbers(lines: List[str]):
    res = []
    for idx, line in enumerate(lines, start=1):
        if re.match(r"\s*(//|#)", line):
            continue
        for m in re.finditer(r"(?<![A-Za-z0-9_])(-?\d+)(?![A-Za-z0-9_])", line):
            try:
                val = int(m.group(1))
            except Exception:
                continue
            if val in (-1, 0, 1):
                continue
            res.append({
                "type": "Magic Number",
                "description": f"Literal {val} used inline; consider a named constant.",
                "startLine": idx,
                "endLine": idx
            })
            break
    return res[:10]

def _heuristic_long_params(lines: List[str], max_params: int = 5):
    res = []
    for idx, line in enumerate(lines, start=1):
        if "(" in line and ")" in line and "=>" not in line:
            inside = re.search(r"\((.*)\)", line)
            if inside:
                params = [p for p in inside.group(1).split(",") if p.strip()]
                if len(params) > max_params:
                    res.append({
                        "type": "Long Parameter List",
                        "description": f"{len(params)} parameters (> {max_params}).",
                        "startLine": idx,
                        "endLine": idx
                    })
    return res

def _heuristic_todos(lines: List[str]):
    res = []
    for idx, line in enumerate(lines, start=1):
        if re.search(r"\b(TODO|FIXME|HACK)\b", line, re.IGNORECASE):
            res.append({
                "type": "Maintainability Note",
                "description": "TODO/FIXME/HACK found.",
                "startLine": idx,
                "endLine": idx
            })
    return res[:10]

def _fallback_heuristics(code: str, min_findings: int = 3) -> List[Dict[str, Any]]:
    lines = code.splitlines()
    findings = []
    findings.extend(_heuristic_long_methods(lines, max_len=50))
    findings.extend(_heuristic_nested_ifs(lines, min_depth=3))
    findings.extend(_heuristic_long_params(lines, max_params=5))
    findings.extend(_heuristic_magic_numbers(lines))
    findings.extend(_heuristic_todos(lines))
    return findings[: max(min_findings, len(findings))]

def _safe_prompt_format(template: str, values: Dict[str, Any]) -> str:
    """Format known placeholders while tolerating literal JSON braces."""

    try:
        return template.format(**values)
    except (KeyError, ValueError):
        protected = str(template)
        tokens: Dict[str, str] = {}
        for key in sorted(values, key=len, reverse=True):
            token = f"@@STALLM_{key.upper()}@@"
            tokens[key] = token
            protected = protected.replace("{" + key + "}", token)
        protected = protected.replace("{", "{{").replace("}", "}}")
        for key, token in tokens.items():
            protected = protected.replace(token, "{" + key + "}")
        return protected.format(**values)

# =========================
# LLM call (sanitizer + retry + heuristics; never bubble exceptions)
# =========================
def analyze_with_llm(
    code: str,
    mode: str,
    language: str,
    llm: ChatModel,
    prompt_templates: Optional[Dict[str, str]] = None,
    number_lines: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    strategies = prompt_templates or load_strategies()
    template = strategies.get(mode) or strategies.get("baseline") or "Language: {language}\nCode:\n{code}\nReturn JSON array."
    code_snippet = code[:20000]
    # With number_lines, each line is prefixed with its 1-based number so that the
    # model does not have to count lines to report a location.
    shown_code = ("\n".join(f"{i:>4}| {line}" for i, line in enumerate(code_snippet.split("\n"), start=1))
                  if number_lines else code_snippet)
    base_prompt = _safe_prompt_format(template, {"language": language, "code": shown_code})

    rules_footer = (
        "\n\nOutput rules (MANDATORY):\n"
        "1) Output MUST be a valid JSON array starting with '[' and ending with ']'.\n"
        "2) Each element MUST have \"type\" (string), \"description\" (string), and a location: "
        "\"line\" (int) OR (\"startLine\" (int), \"endLine\" (int)).\n"
        "3) No prose, no markdown, no code fences. If no findings: return []."
    )
    if number_lines:
        rules_footer += ("\n4) Each code line above is prefixed with its line number and '| '; "
                         "report locations with those numbers.")
    prompt = base_prompt + rules_footer

    meta = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # First attempt (catch errors)
    try:
        content, meta1 = llm.chat(
            messages=[
                {"role": "system", "content": "You are a precise static-analysis annotator. Return STRICT JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1400,
            temperature=0,
            return_meta=True,
        )
        meta = {k: int(meta1.get(k, 0)) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        _write_debug_dump("first_raw", content)
        array_text = _sanitize_to_json_array_text(content)
        _write_debug_dump("first_sanitized", array_text)
        try:
            raw_list = json.loads(array_text)
        except Exception:
            raw_list = []
        smells = _coerce_items_schema(raw_list)
    except Exception as exc:
        meta["source"] = "llm_error"
        meta["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        if _heuristic_fallback_enabled():
            meta["source"] = "heuristic_fallback"
            return _fallback_heuristics(code_snippet, min_findings=int(os.getenv("STALLM_MIN_FINDINGS", "3"))), meta
        return [], meta

    meta["source"] = "llm"
    # Retry with a nudge if empty (also catch errors)
    if not smells:
        try:
            nudge = (
                "\n\nFollow this schema (example), do not repeat it:\n"
                "[{\"type\":\"Long Method\",\"description\":\"Method too long\",\"line\":42}]\n"
                f"Return AT LEAST {int(os.getenv('STALLM_MIN_FINDINGS','3'))} findings if plausible. "
                "Prefer: Long Method, Deeply Nested Ifs, Magic Number, Long Parameter List, Duplicated Code, "
                "Unused Variable, God Class, Tight Coupling."
            )
            content2, meta2 = llm.chat(
                messages=[
                    {"role": "system", "content": "You are a static-analysis annotator. Return STRICT JSON only."},
                    {"role": "user", "content": base_prompt + rules_footer + nudge},
                ],
                max_tokens=2000,
                temperature=0.25,
                return_meta=True,
            )
            _write_debug_dump("retry_raw", content2)
            array_text2 = _sanitize_to_json_array_text(content2)
            _write_debug_dump("retry_sanitized", array_text2)
            try:
                raw_list2 = json.loads(array_text2)
            except Exception:
                raw_list2 = []
            smells2 = _coerce_items_schema(raw_list2)
            meta = {
                "prompt_tokens": meta.get("prompt_tokens", 0) + int(meta2.get("prompt_tokens", 0)),
                "completion_tokens": meta.get("completion_tokens", 0) + int(meta2.get("completion_tokens", 0)),
                "total_tokens": meta.get("total_tokens", 0) + int(meta2.get("total_tokens", 0)),
                "source": "llm_retry" if smells2 else "llm",
            }
            smells = smells2 or smells
        except Exception:
            pass

    # Heuristic findings are not LLM output: injecting them silently would mix
    # two methods in one score, so they are opt-in (STALLM_HEURISTIC_FALLBACK=1).
    if not smells and _heuristic_fallback_enabled():
        meta["source"] = "heuristic_fallback"
        smells = _fallback_heuristics(code_snippet, min_findings=int(os.getenv("STALLM_MIN_FINDINGS", "3")))
    return smells, meta


def _heuristic_fallback_enabled() -> bool:
    return os.getenv("STALLM_HEURISTIC_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}

# =========================
# Type normalization (aliases) for robust matching
# =========================
_TYPE_ALIASES = {
    "longmethod": "long_method",
    "longfunction": "long_method",
    "largemethod": "long_method",
    "bigmethod": "long_method",
    "methodtoolong": "long_method",
    "functiontoolong": "long_method",
    "nestedifs": "deeply_nested_ifs",
    "deeplynestedifs": "deeply_nested_ifs",
    "magicnumber": "magic_number",
    "longparameterlist": "long_parameter_list",
    "longparamlist": "long_parameter_list",
    "godclass": "god_class",
    "largeclass": "large_class",
    "unusedvariable": "unused_variable",
    "deadcode": "dead_code",
    "resourceleak": "resource_leak",
    "missingnullcheck": "missing_null_check",
}

# SonarQube rule IDs (java:S138, squid:S138, csharpsquid:S138, php:S138, …)
# S-numbers are consistent across languages for equivalent rules.
_SONAR_RULE_TO_SMELL: Dict[str, str] = {
    "s138":   "long_method",           # Method too long
    "s3776":  "complex_method",        # Cognitive complexity
    "s1541":  "complex_method",        # Cyclomatic complexity
    "s6541":  "complex_method",        # Brain method
    "s107":   "long_parameter_list",   # Too many parameters
    "s1448":  "god_class",             # Too many methods in class
    "s1200":  "god_class",             # Too many dependencies
    "s2436":  "god_class",             # Too many fields
    "s1135":  "maintainability_note",  # TODO/FIXME
    "s1854":  "unused_variable",       # Unnecessary assignment
    "s1481":  "unused_variable",       # Unused local variable
    "s1068":  "unused_variable",       # Unused private field
    "s2095":  "resource_leak",         # Resources not closed
    "s1143":  "resource_leak",         # Return/throw inside finally
    "s2259":  "missing_null_check",    # Null dereference
    "s1066":  "deeply_nested_ifs",     # Collapsible if
    "s1479":  "deeply_nested_ifs",     # Switch cases too many
    "s109":   "magic_number",          # Magic number
    "s1192":  "magic_number",          # String literal duplicated
    "s1301":  "deeply_nested_ifs",     # Switch with single case
}

# Pattern: optional prefix (java, squid, csharpsquid, php, …) + colon + S + digits
_SONAR_RULE_RE = re.compile(r"^(?:[a-z]+:)?[Ss](\d+)$", re.IGNORECASE)

def _norm_type(x: Any) -> Optional[str]:
    if x is None:
        return None
    raw = str(x).strip()
    if not raw:
        return None
    # SonarQube rule ID takes priority before stripping special chars
    m = _SONAR_RULE_RE.match(raw)
    if m:
        key = f"s{m.group(1)}"
        return _SONAR_RULE_TO_SMELL.get(key, key)
    t = re.sub(r"[^a-z0-9]+", "", raw.lower())
    if not t:
        return None
    return _TYPE_ALIASES.get(t, t)

# =========================
# Matching (span-level)
# =========================
def _iou_1d(a0: int, a1: int, b0: int, b1: int) -> float:
    if a0 > a1:
        a0, a1 = a1, a0
    if b0 > b1:
        b0, b1 = b1, b0
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union > 0 else 0.0

def _dist_1d(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0

def _preset_cfg(name: str) -> Dict[str, Any]:
    name = (name or "Balanced").lower()
    if name.startswith("len"):
        return {"iou_thr": 0.10, "delta": 3}
    if name.startswith("str"):
        return {"iou_thr": 0.50, "delta": 1}
    return {"iou_thr": 0.25, "delta": 2}

# =========================
# Robust CSV + GT utils (pandas 2.x safe)
# =========================
def _read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "utf-16", "utf-16le", "utf-16be"]
    seps = [None, ",", ";", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, engine="python", encoding=enc, on_bad_lines="skip")
                if isinstance(df, pd.DataFrame) and len(df.columns) >= 1:
                    return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Robust CSV read failed for {path}: {last_err}")

def _norm_basename(p: str) -> str:
    if p is None:
        return ""
    s = str(p).replace("\\", "/")
    return os.path.basename(s).lower()

def load_ground_truth_spans(static_csv: str, allowed_exts: Optional[List[str]] = None) -> pd.DataFrame:
    df = _read_csv_robust(static_csv)
    low = {c.lower(): c for c in df.columns}

    file_col = None
    for k in ["file", "component", "path", "file_path", "filename"]:
        if k in low:
            file_col = low[k]
            break
    if not file_col:
        raise ValueError(f"No file column found. Available: {list(df.columns)}")

    def _pick(*cands, default=None):
        for k in cands:
            if k in low:
                return low[k]
        return default

    c_start = _pick("startline", "start_line", "line", "line_number")
    c_end   = _pick("endline", "end_line", default=c_start)
    c_sc    = _pick("startcolumn", "start_column")
    c_ec    = _pick("endcolumn", "end_column")
    c_type  = _pick("rule", "type", "issue", "code", "rulekey", "rules")
    c_desc  = _pick("description", "desc", "message")

    out = pd.DataFrame()
    out["file"] = df[file_col].astype(str)
    out["basename"] = out["file"].map(_norm_basename)

    def _to_int_or_pos_none(v):
        try:
            iv = int(float(v))
            return iv if iv > 0 else None
        except Exception:
            return None

    out["startLine"] = df[c_start].apply(_to_int_or_pos_none) if c_start else None
    out["endLine"]   = df[c_end].apply(_to_int_or_pos_none) if c_end else None
    # File-level findings without a line cannot be matched to a span; they are
    # dropped below (dropna on startLine) rather than given the previous row's line.
    out["endLine"]   = out["endLine"].fillna(out["startLine"])
    out["startLine"] = out["startLine"].fillna(out["endLine"])

    def _to_pos_int(v):
        try:
            iv = int(float(v))
            return iv if iv >= 0 else None
        except Exception:
            return None

    out["startColumn"] = df[c_sc].apply(_to_pos_int) if c_sc else None
    out["endColumn"]   = df[c_ec].apply(_to_pos_int) if c_ec else None
    out["type"] = df[c_type].astype(str) if c_type else None
    out["description"] = df[c_desc].astype(str) if c_desc else None

    out = out.dropna(subset=["basename", "startLine"]).copy()
    out["startLine"] = out["startLine"].astype(int)
    out["endLine"]   = out["endLine"].fillna(out["startLine"]).astype(int)

    if allowed_exts:
        allowed = tuple(ext.lower() for ext in allowed_exts)
        out = out[out["basename"].str.endswith(allowed)]

    return out.reset_index(drop=True)

def detect_gt_capabilities(df: pd.DataFrame) -> Dict[str, Any]:
    low = {c.lower(): c for c in df.columns}
    caps = {
        "has_type": any(x in low for x in ["rule", "type", "issue", "code", "rulekey", "rules"]),
        "has_line_span": any(x in low for x in ["endline", "end_line"]) or (
            any(x in low for x in ["startline", "start_line"]) and any(x in low for x in ["line", "line_number"])
        ),
        "has_col_span": any(x in low for x in ["startcolumn", "endcolumn", "start_column", "end_column"]),
    }
    sample = {}
    for k in ["rule", "type", "issue", "code", "rulekey", "rules"]:
        if k in low:
            sample["rule/type"] = str(df[low[k]].dropna().astype(str).head(1).tolist() or ["…"][-1])
    for k in ["startline", "start_line", "line", "line_number"]:
        if k in low:
            sample["startLine"] = str(df[low[k]].dropna().head(1).tolist() or ["?"][-1])
    for k in ["endline", "end_line"]:
        if k in low:
            sample["endLine"] = str(df[low[k]].dropna().head(1).tolist() or ["?"][-1])
    caps["sample"] = sample
    return caps

# =========================
# Extract predicted spans (normalized)
# =========================
def _extract_pred_spans(smells: List[Dict[str, Any]], basename: str) -> List[Dict[str, Any]]:
    out = []
    for s in smells or []:
        sline = _to_int_or_none(s.get("startLine"))
        eline = _to_int_or_none(s.get("endLine"))
        if sline is None or eline is None:
            continue
        sc = _to_int_or_none(s.get("startColumn"))
        ec = _to_int_or_none(s.get("endColumn"))
        out.append({
            "basename": basename,
            "startLine": int(sline),
            "endLine": int(eline),
            "startColumn": sc,
            "endColumn": ec,
            "type": s.get("type"),
            "description": s.get("description"),
        })
    return out

# =========================
# Statistical utilities
# =========================
def _bootstrap_ci_spanwise(
    file_diagnostics: List[Dict[str, Any]],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    rng = random.Random(seed)
    k = len(file_diagnostics)
    if k == 0:
        return {"precision_ci": (0.0, 0.0), "recall_ci": (0.0, 0.0), "f1_ci": (0.0, 0.0)}
    p_vals, r_vals, f_vals = [], [], []
    for _ in range(n):
        sample = rng.choices(file_diagnostics, k=k)
        tp = sum(r["tp"] for r in sample)
        fp = sum(r["fp"] for r in sample)
        fn = sum(r["fn"] for r in sample)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        p_vals.append(p); r_vals.append(r); f_vals.append(f)
    lo_idx = max(0, min(n - 1, int(alpha / 2 * n)))
    hi_idx = max(0, min(n - 1, int((1 - alpha / 2) * n)))
    def _ci(vals: List[float]) -> Tuple[float, float]:
        s = sorted(vals)
        return (s[lo_idx], s[hi_idx])
    return {"precision_ci": _ci(p_vals), "recall_ci": _ci(r_vals), "f1_ci": _ci(f_vals)}


def _type_breakdown(
    pred_by_file: Dict[str, List[Dict[str, Any]]],
    gt_by_file: Dict[str, List[Dict[str, Any]]],
    U_bases: List[str],
    *,
    require_type: bool,
    use_cols_single: bool,
    preset: str,
) -> List[Dict[str, Any]]:
    """Per-type TP/FP/FN across all files in U."""
    from collections import defaultdict
    cfg = _preset_cfg(preset)
    iou_thr, delta = cfg["iou_thr"], cfg["delta"]
    tp_by_type: Dict[str, int] = defaultdict(int)
    fp_by_type: Dict[str, int] = defaultdict(int)
    fn_by_type: Dict[str, int] = defaultdict(int)

    for b in U_bases:
        preds = pred_by_file.get(b, [])
        gts = gt_by_file.get(b, [])
        pairs = []
        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                iou_line = _iou_1d(p["startLine"], p["endLine"], g["startLine"], g["endLine"])
                dist_line = _dist_1d(p["startLine"], p["endLine"], g["startLine"], g["endLine"])
                ok = (iou_line >= iou_thr) or (dist_line <= delta)
                if ok and use_cols_single:
                    if (p["startLine"] == p["endLine"] == g["startLine"] == g["endLine"]
                            and all(x is not None for x in [p.get("startColumn"), p.get("endColumn"),
                                                             g.get("startColumn"), g.get("endColumn")])):
                        iou_col = _iou_1d(int(p["startColumn"]), int(p["endColumn"]),
                                          int(g["startColumn"]), int(g["endColumn"]))
                        ok = (min(iou_line, iou_col) >= iou_thr) or (dist_line <= delta)
                if ok:
                    pairs.append((max(iou_line, 1.0 / (1.0 + dist_line)), i, j))
        pairs.sort(key=lambda x: x[0], reverse=True)
        used_p, used_g = set(), set()
        for _, i, j in pairs:
            if i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j)
            gt_type = _norm_type(gts[j].get("type")) or "unknown"
            tp_by_type[gt_type] += 1
        for i, p in enumerate(preds):
            if i not in used_p:
                pt = _norm_type(p.get("type")) or "unknown"
                fp_by_type[pt] += 1
        for j, g in enumerate(gts):
            if j not in used_g:
                gt_type = _norm_type(g.get("type")) or "unknown"
                fn_by_type[gt_type] += 1

    all_types = sorted(set(list(tp_by_type) + list(fp_by_type) + list(fn_by_type)))
    rows = []
    for t in all_types:
        tp = tp_by_type[t]; fp = fp_by_type[t]; fn = fn_by_type[t]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({"type": t, "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                     "gt_count": tp + fn})
    return sorted(rows, key=lambda x: x["gt_count"], reverse=True)


# =========================
# Matching (span-level) with type normalization
# =========================
def _match_file(preds: List[Dict[str, Any]], gts: List[Dict[str, Any]],
                require_type: bool, use_cols_single: bool, preset: str) -> Tuple[int, int, int]:
    cfg = _preset_cfg(preset)
    iou_thr, delta = cfg["iou_thr"], cfg["delta"]
    pairs = []
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            if require_type:
                pt = _norm_type(p.get("type"))
                gt = _norm_type(g.get("type"))
                if pt and gt:
                    if pt != gt and not (pt in gt or gt in pt):
                        continue
            iou_line = _iou_1d(p["startLine"], p["endLine"], g["startLine"], g["endLine"])
            dist_line = _dist_1d(p["startLine"], p["endLine"], g["startLine"], g["endLine"])
            ok = (iou_line >= iou_thr) or (dist_line <= delta)

            if ok and use_cols_single:
                if (p["startLine"] == p["endLine"] == g["startLine"] == g["endLine"]
                    and p.get("startColumn") is not None and p.get("endColumn") is not None
                    and g.get("startColumn") is not None and g.get("endColumn") is not None):
                    iou_col = _iou_1d(int(p["startColumn"]), int(p["endColumn"]),
                                      int(g["startColumn"]), int(g["endColumn"]))
                    ok = (min(iou_line, iou_col) >= iou_thr) or (dist_line <= delta)
            if ok:
                score = max(iou_line, 1.0 / (1.0 + dist_line))
                pairs.append((score, i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)
    used_p, used_g = set(), set()
    tp = 0
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn

# =========================
# Universe U (sampling positives + negatives)
# =========================
def _zip_listing(zip_path: str, allowed_exts: Optional[List[str]]) -> pd.DataFrame:
    rows = []
    exts = tuple((allowed_exts or []))
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith("/") or name.startswith("__MACOSX/"):
                continue
            base = os.path.basename(name).lower()
            if allowed_exts and not base.endswith(exts):
                continue
            rows.append({"relpath": name, "basename": base})
    return pd.DataFrame(rows)


def _norm_path(p: str) -> str:
    return re.sub(r"^(\./|/)+", "", str(p).replace("\\", "/")).lower()


def _common_suffix_len(a: List[str], b: List[str]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def _align_gt_to_zip(gt_spans: pd.DataFrame, zip_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Match oracle rows to archive files by path rather than by file name.

    Matching on the file name alone attaches to e.g. Catalog/Program.cs the
    findings of every Program.cs in the system. We infer the prefix that maps
    archive paths to oracle paths (e.g. "catalog/" -> "services/catalog/") from
    files whose best path match is unambiguous, then require an exact path
    match. Both frames get their "basename" column replaced by the matched
    archive path, which is the key used downstream. Falls back to file-name
    matching when no prefix can be inferred.
    """
    gt = gt_spans.copy()
    zdf = zip_df.copy()
    gt_paths = {f: _norm_path(f) for f in gt["file"].unique()}
    by_base: Dict[str, List[str]] = {}
    for norm in set(gt_paths.values()):
        by_base.setdefault(norm.rsplit("/", 1)[-1], []).append(norm)

    votes: Dict[Tuple[str, str], int] = {}
    for rel in zdf["relpath"]:
        z = _norm_path(rel)
        zc = z.split("/")
        scored = sorted(((_common_suffix_len(zc, g.split("/")), g) for g in by_base.get(zc[-1], [])), reverse=True)
        if not scored or (len(scored) > 1 and scored[0][0] == scored[1][0]):
            continue
        n, g = scored[0]
        gc = g.split("/")
        key = ("/".join(zc[:len(zc) - n]), "/".join(gc[:len(gc) - n]))
        votes[key] = votes.get(key, 0) + 1

    if not votes:
        diag = {"mode": "basename", "gt_spans_matched": int(len(gt)), "gt_spans_dropped": 0}
        return gt, zdf, diag

    (zp, gp), _ = max(votes.items(), key=lambda kv: kv[1])

    def _to_gt(z: str) -> Optional[str]:
        if zp and not z.startswith(zp + "/"):
            return None
        rest = z[len(zp) + 1:] if zp else z
        return f"{gp}/{rest}" if gp else rest

    zip_key = {_to_gt(_norm_path(r)): _norm_path(r) for r in zdf["relpath"]}
    zip_key.pop(None, None)
    zdf["basename"] = zdf["relpath"].map(_norm_path)
    gt["basename"] = gt["file"].map(lambda f: zip_key.get(gt_paths[f]))
    n_all = len(gt)
    gt = gt.dropna(subset=["basename"]).reset_index(drop=True)
    diag = {"mode": "path", "zip_prefix": zp, "gt_prefix": gp,
            "gt_spans_matched": int(len(gt)), "gt_spans_dropped": int(n_all - len(gt))}
    return gt, zdf, diag

def _detect_line_offset(gt_spans: pd.DataFrame, zip_path: str, max_offset: int = 2,
                        sample: int = 600) -> Dict[str, Any]:
    """Check oracle line numbers against the archived source.

    For single-line records with columns, an offset k "fits" when line
    startLine+k is long enough to contain the reported column range and the
    range is not blank. Some SonarQube exports write 0-based lines; the
    offset is applied only when one non-zero offset fits at least 95% of the
    records while the reported lines fit fewer than half of them.
    """
    rows = gt_spans[(gt_spans["startLine"] == gt_spans["endLine"])
                    & gt_spans["startColumn"].notna() & gt_spans["endColumn"].notna()]
    if len(rows) > sample:
        rows = rows.sample(n=sample, random_state=1)
    if rows.empty:
        return {"line_offset": 0, "checked": 0}
    fits = {k: 0 for k in range(-max_offset, max_offset + 1)}
    with zipfile.ZipFile(zip_path, "r") as z:
        names = {n.lower(): n for n in z.namelist()}
        cache: Dict[str, List[str]] = {}
        for _, r in rows.iterrows():
            key = r["basename"]
            if key not in names:
                continue
            if key not in cache:
                cache[key] = z.read(names[key]).decode("utf-8-sig", "ignore").splitlines()
            lines, sc, ec = cache[key], int(r["startColumn"]), int(r["endColumn"])
            for k in fits:
                i = int(r["startLine"]) + k
                if 1 <= i <= len(lines) and len(lines[i - 1]) >= ec and lines[i - 1][sc:ec].strip():
                    fits[k] += 1
    n = len(rows)
    rate = {k: v / n for k, v in fits.items()}
    best = max((k for k in rate if k != 0), key=lambda k: rate[k])
    offset = best if rate[best] >= 0.95 and rate[0] < 0.5 else 0
    return {"line_offset": offset, "checked": n, "fit_rate": {str(k): round(v, 3) for k, v in rate.items()}}


def _build_universe(zip_path: str, gt_spans: pd.DataFrame, allowed_exts: Optional[List[str]],
                    top_k: int, pos_ratio: float) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    gt_spans, zip_df, align = _align_gt_to_zip(gt_spans, _zip_listing(zip_path, allowed_exts))
    if align.get("mode") == "path":
        lines_check = _detect_line_offset(gt_spans, zip_path)
        align.update(lines_check)
        if lines_check["line_offset"]:
            gt_spans = gt_spans.copy()
            gt_spans["startLine"] += lines_check["line_offset"]
            gt_spans["endLine"] += lines_check["line_offset"]
    gt_counts = gt_spans.groupby("basename").size().rename("gt_lines").reset_index()

    summary = zip_df.merge(gt_counts, on="basename", how="left")
    summary["gt_lines"] = summary["gt_lines"].fillna(0).astype(int)
    summary["is_positive"] = summary["gt_lines"] > 0

    pos = summary[summary["is_positive"]]
    neg = summary[~summary["is_positive"]]
    want_pos = max(0, min(len(pos), int(round(top_k * pos_ratio))))
    want_neg = max(0, top_k - want_pos)

    pos_pick = pos.sort_values("gt_lines", ascending=False).head(want_pos)
    neg_pick = neg.sample(n=min(want_neg, len(neg)), random_state=42) if want_neg and len(neg) else neg.head(0)
    U = pd.concat([pos_pick, neg_pick], ignore_index=True)

    mapping_cov = float(gt_counts["basename"].isin(summary["basename"]).mean()) if len(gt_counts) else 0.0
    diag = {
        "mapping_coverage": mapping_cov,
        "mapped_gt_files": int((summary["gt_lines"] > 0).sum()),
        "top_k_total_files": int(len(U)),
        "positives": int(U["is_positive"].sum()),
        "negatives": int((~U['is_positive']).sum()),
        "gt_alignment": align,
    }
    return U, diag, gt_spans

# =========================
# Cost helpers
# =========================
def _read_rates(llm: ChatModel) -> Tuple[float, float]:
    slot = getattr(llm, "slot_key", None)
    if slot:
        try_in = os.getenv(f"{slot}_PRICE_IN_PER_1K")
        try_out = os.getenv(f"{slot}_PRICE_OUT_PER_1K")
        if try_in or try_out:
            return float(try_in or 0), float(try_out or 0)
    return float(os.getenv("STALLM_PRICE_IN_PER_1K", 0)), float(os.getenv("STALLM_PRICE_OUT_PER_1K", 0))

def _cost_usd(pt: int, ct: int, llm: ChatModel) -> float:
    in_rate, out_rate = _read_rates(llm)
    return (pt / 1000.0) * in_rate + (ct / 1000.0) * out_rate

# =========================
# Experiment context (shared universe across runs)
# =========================
class ExperimentContext(NamedTuple):
    language: str
    exts: Optional[List[str]]
    gt_spans: Any          # pd.DataFrame
    summary_U: Any         # pd.DataFrame
    diag: Dict[str, Any]

def _prepare_context(
    zip_path: str,
    static_csv: str,
    top_k: int,
    pos_ratio: float,
) -> ExperimentContext:
    language = detect_language_from_zip(zip_path)
    exts = find_exts_for_language(language) or None
    gt_spans = load_ground_truth_spans(static_csv, allowed_exts=exts)
    summary_U, diag, gt_spans = _build_universe(zip_path, gt_spans, exts, top_k, pos_ratio)
    return ExperimentContext(language=language, exts=exts, gt_spans=gt_spans,
                             summary_U=summary_U, diag=diag)

# =========================
# Default LLM
# =========================
def _default_llm() -> ChatModel:
    slot = default_slot_key()
    if slot:
        return build_llm_from_slot(slot)
    provider = os.getenv("LLM_PROVIDER", "azure-openai")
    if provider == "azure-openai":
        return ChatModel(LLMConfig(
            provider="azure-openai",
            model=os.getenv("OPENAI_DEPLOYMENT_NAME", ""),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("OPENAI_API_BASE", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
        ))
    return ChatModel(LLMConfig(provider=provider, model=os.getenv("OPENAI_MODEL", "gpt-4o")))

# =========================
# Core run (span-level only)
# =========================
def run_experiment(
    zip_path: str,
    static_csv: str,
    mode: str = "baseline",
    progress=None,
    top_k: int = 20,
    pos_ratio: float = 0.5,
    preset: str = "Balanced",
    user_require_type: Optional[bool] = True,
    user_use_line_span: Optional[bool] = True,   # compatibility with UI switch
    user_use_cols_single: Optional[bool] = True,
    llm: Optional[ChatModel] = None,
    prompt_templates: Optional[Dict[str, str]] = None,
    _ctx: Optional[ExperimentContext] = None,
    number_lines: bool = False,
):
    if _ctx is not None:
        language, exts, gt_spans, summary_U, diag = _ctx
    else:
        language = detect_language_from_zip(zip_path)
        exts = find_exts_for_language(language) or None
        gt_spans = load_ground_truth_spans(static_csv, allowed_exts=exts)
        summary_U, diag, gt_spans = _build_universe(zip_path, gt_spans, exts, top_k, pos_ratio)

    llm = llm or _default_llm()
    pt_sum = ct_sum = tt_sum = 0
    source_counts: Dict[str, int] = {}
    preds_all: List[Dict[str, Any]] = []
    results_all: List[Dict[str, Any]] = []
    code_by_file: Dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)
        for i, row in summary_U.iterrows():
            rel = row["relpath"]
            base = row["basename"]
            file_path = os.path.join(tmp, rel)
            if progress:
                progress.progress((i + 1) / max(1, len(summary_U)), text=f"🔍 {mode} analyzing {os.path.basename(rel)}")
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except Exception:
                code = ""
            code_by_file[base] = code

            # analyze_with_llm handles its own errors; usage["source"] records provenance
            smells, usage = analyze_with_llm(code, mode, language, llm, prompt_templates=prompt_templates,
                                             number_lines=number_lines)
            source_counts[usage.get("source", "llm")] = source_counts.get(usage.get("source", "llm"), 0) + 1

            for smell in smells or []:
                item = dict(smell)
                item["basename"] = base
                item["relpath"] = rel
                results_all.append(item)
            preds_all.extend(_extract_pred_spans(smells, basename=base))
            pt_sum += usage.get("prompt_tokens", 0)
            ct_sum += usage.get("completion_tokens", 0)
            tt_sum += usage.get("total_tokens", 0)

    require_type = bool(user_require_type)
    use_cols_single = bool(user_use_cols_single)

    # Group GT and predictions by file key (archive path, limited to U)
    U_bases = set(summary_U["basename"])
    gt_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for _, g in gt_spans.iterrows():
        b = g["basename"]
        if b not in U_bases:
            continue
        gt_by_file.setdefault(b, []).append({
            "basename": b,
            "startLine": int(g["startLine"]),
            "endLine": int(g["endLine"]),
            "startColumn": g.get("startColumn"),
            "endColumn": g.get("endColumn"),
            "type": g.get("type"),
            "description": g.get("description"),
        })
    pred_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for p in preds_all:
        b = p["basename"]
        if b not in U_bases:
            continue
        pred_by_file.setdefault(b, []).append(p)

    def _span_preview(items: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        preview = []
        for item in (items or [])[:limit]:
            preview.append({
                "type": item.get("type"),
                "description": item.get("description"),
                "startLine": item.get("startLine"),
                "endLine": item.get("endLine"),
            })
        return preview

    def _code_excerpt(basename: str, gt_items: List[Dict[str, Any]], pred_items: List[Dict[str, Any]],
                      radius: int = 4, max_lines: int = 36, focus: str = "both") -> Dict[str, Any]:
        code = code_by_file.get(basename, "")
        lines = code.splitlines()
        if not lines:
            return {"start_line": 0, "lines": [], "gt_lines": [], "llm_lines": []}

        gt_lines = set()
        llm_lines = set()
        gt_anchors = []
        llm_anchors = []
        for item in gt_items or []:
            s = _to_int_or_none(item.get("startLine"))
            e = _to_int_or_none(item.get("endLine")) or s
            if s:
                gt_anchors.append(s)
                for n in range(max(1, s), min(len(lines), e or s) + 1):
                    gt_lines.add(n)
        for item in pred_items or []:
            s = _to_int_or_none(item.get("startLine"))
            e = _to_int_or_none(item.get("endLine")) or s
            if s:
                llm_anchors.append(s)
                for n in range(max(1, s), min(len(lines), e or s) + 1):
                    llm_lines.add(n)

        if focus == "gt":
            anchors = gt_anchors
            visible_gt_lines = gt_lines
            visible_llm_lines = set()
        elif focus == "llm":
            anchors = llm_anchors
            visible_gt_lines = set()
            visible_llm_lines = llm_lines
        else:
            anchors = gt_anchors + llm_anchors
            visible_gt_lines = gt_lines
            visible_llm_lines = llm_lines

        if not anchors:
            return {"start_line": 0, "lines": [], "gt_lines": [], "llm_lines": []}

        center = min(anchors)
        start = max(1, center - radius)
        end = min(len(lines), start + max_lines - 1)
        return {
            "start_line": int(start),
            "lines": lines[start - 1:end],
            "gt_lines": sorted(n for n in visible_gt_lines if start <= n <= end),
            "llm_lines": sorted(n for n in visible_llm_lines if start <= n <= end),
        }

    TP = FP = FN = 0
    file_diagnostics: List[Dict[str, Any]] = []
    for b in U_bases:
        tp, fp, fn = _match_file(pred_by_file.get(b, []), gt_by_file.get(b, []),
                                 require_type=require_type, use_cols_single=use_cols_single, preset=preset)
        TP += tp
        FP += fp
        FN += fn
        row = summary_U[summary_U["basename"] == b].head(1)
        relpath = str(row["relpath"].iloc[0]) if len(row) else b
        gt_count = len(gt_by_file.get(b, []))
        pred_count = len(pred_by_file.get(b, []))
        if tp and not fp and not fn:
            bucket = "matched"
        elif tp:
            bucket = "partial"
        elif fn and not fp:
            bucket = "missed"
        elif fp and not fn:
            bucket = "extra"
        elif fp and fn:
            bucket = "mismatch"
        else:
            bucket = "true_negative"
        file_diagnostics.append({
            "file": relpath,
            "basename": b,
            "bucket": bucket,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "gt_spans": int(gt_count),
            "llm_spans": int(pred_count),
            "is_positive": bool(gt_count > 0),
            "gt_examples": _span_preview(gt_by_file.get(b, [])),
            "llm_examples": _span_preview(pred_by_file.get(b, [])),
            "code_excerpt": _code_excerpt(b, gt_by_file.get(b, []), pred_by_file.get(b, [])),
            "gt_code_excerpt": _code_excerpt(b, gt_by_file.get(b, []), pred_by_file.get(b, []), focus="gt"),
            "llm_code_excerpt": _code_excerpt(b, gt_by_file.get(b, []), pred_by_file.get(b, []), focus="llm"),
        })

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    ci        = _bootstrap_ci_spanwise(file_diagnostics)
    type_breakdown = _type_breakdown(
        pred_by_file, gt_by_file, list(U_bases),
        require_type=require_type, use_cols_single=use_cols_single, preset=preset,
    )

    usage_totals = {
        "prompt_tokens": int(pt_sum),
        "completion_tokens": int(ct_sum),
        "total_tokens": int(tt_sum),
        "usd_cost": _cost_usd(pt_sum, ct_sum, llm),
    }

    metrics = {
        "precision": precision, "recall": recall, "f1": f1,
        "precision_ci_lo": ci["precision_ci"][0], "precision_ci_hi": ci["precision_ci"][1],
        "recall_ci_lo": ci["recall_ci"][0], "recall_ci_hi": ci["recall_ci"][1],
        "f1_ci_lo": ci["f1_ci"][0], "f1_ci_hi": ci["f1_ci"][1],
        "tp": TP, "fp": FP, "fn": FN,
        "language": language,
        "diagnostics": diag,
        "top_k_total_files": diag.get("top_k_total_files", len(summary_U)),
        "preset": preset,
        "require_type": require_type,
        "use_cols_single": use_cols_single,
        "file_diagnostics": sorted(file_diagnostics, key=lambda x: (x["bucket"], x["file"])),
        "type_breakdown": type_breakdown,
        "file_sources": source_counts,
        "number_lines": bool(number_lines),
        # Per-file spans and sizes, so that span-level results can be re-scored
        # offline (e.g. against a random-placement baseline) without new calls.
        "spans_by_file": {
            b: {
                "lines": code_by_file.get(b, "").count("\n") + 1,
                "pred": [[p["startLine"], p["endLine"]] for p in pred_by_file.get(b, [])],
                "gt": [[g["startLine"], g["endLine"]] for g in gt_by_file.get(b, [])],
            }
            for b in sorted(U_bases)
        },
    }

    return results_all, summary_U, metrics, usage_totals

# =========================
# Multi-runs
# =========================
def run_selected_experiments(
    zip_path: str,
    static_csv: str,
    selected_modes: List[str],
    progress=None,
    status=None,
    timer=None,
    top_k: int = 20,
    pos_ratio: float = 0.5,
    preset: str = "Balanced",
    user_require_type: Optional[bool] = True,
    user_use_line_span: Optional[bool] = True,
    user_use_cols_single: Optional[bool] = True,
    llm: Optional[ChatModel] = None,
):
    all_metrics = []
    samples_dict = {}
    llm = llm or _default_llm()

    ctx = _prepare_context(zip_path, static_csv, top_k, pos_ratio)

    for idx, mode in enumerate(selected_modes):
        if status:
            status.markdown(f"🔍 Running `{mode}` strategy…")
        start = time.time()
        results, _, metrics, usage_totals = run_experiment(
            zip_path, static_csv, mode, top_k=top_k, pos_ratio=pos_ratio, preset=preset,
            user_require_type=user_require_type, user_use_line_span=user_use_line_span,
            user_use_cols_single=user_use_cols_single, llm=llm, _ctx=ctx
        )
        elapsed = time.time() - start
        row = dict(metrics)
        row.update({
            "strategy": f"{mode}@span",
            "time_s": round(elapsed, 2),
            "prompt_tokens": usage_totals["prompt_tokens"],
            "completion_tokens": usage_totals["completion_tokens"],
            "total_tokens": usage_totals["total_tokens"],
            "usd_cost": usage_totals["usd_cost"],
        })
        all_metrics.append(row)
        samples_dict[mode] = results[:5]
        if progress:
            progress.progress((idx + 1) / max(1, len(selected_modes)))
        if timer:
            timer.markdown(f"🕒 Elapsed: `{elapsed:.2f}` seconds for `{mode}`")

    metrics_df = pd.DataFrame(all_metrics).set_index("strategy")
    return ctx.summary_U, metrics_df, samples_dict

def run_selected_models_experiments(
    zip_path: str,
    static_csv: str,
    strategy: str,
    llms: List[ChatModel],
    progress=None,
    status=None,
    timer=None,
    top_k: int = 20,
    pos_ratio: float = 0.5,
    preset: str = "Balanced",
    user_require_type: Optional[bool] = True,
    user_use_line_span: Optional[bool] = True,
    user_use_cols_single: Optional[bool] = True,
):
    rows = []
    samples = {}
    ctx = _prepare_context(zip_path, static_csv, top_k, pos_ratio)

    for i, llm in enumerate(llms, 1):
        label = llm.model_label()
        if status:
            status.markdown(f"🔍 `{label}` with strategy `{strategy}`…")
        start = time.time()
        results, _, metrics, usage_totals = run_experiment(
            zip_path, static_csv, strategy, top_k=top_k, pos_ratio=pos_ratio, preset=preset,
            user_require_type=user_require_type, user_use_line_span=user_use_line_span,
            user_use_cols_single=user_use_cols_single, llm=llm, _ctx=ctx
        )
        elapsed = time.time() - start
        rows.append({
            "model": label,
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "tp": metrics.get("tp", 0),
            "fp": metrics.get("fp", 0),
            "fn": metrics.get("fn", 0),
            "time_s": round(elapsed, 2),
            "language": ctx.language,
            "top_k_total_files": metrics.get("top_k_total_files", 0),
            "prompt_tokens": usage_totals.get("prompt_tokens", 0),
            "completion_tokens": usage_totals.get("completion_tokens", 0),
            "total_tokens": usage_totals.get("total_tokens", 0),
            "usd_cost": usage_totals.get("usd_cost", 0.0),
        })
        samples[label] = results[:5]
        if progress:
            progress.progress(i / max(1, len(llms)))
        if timer:
            timer.markdown(f"🕒 Elapsed: `{elapsed:.2f}` seconds • {label}")

    metrics_df = pd.DataFrame(rows).set_index("model")
    return ctx.summary_U, metrics_df, samples
