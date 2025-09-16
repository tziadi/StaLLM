# StaLLM_core.py
import os, re, json, zipfile, tempfile, time, random
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

import pandas as pd

from StaLLM_models import init_db, save_run_result
from StaLLM_llm import ChatModel, LLMConfig, build_llm_from_slot, default_slot_key


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
                key = key.strip(); val = val.strip()
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
# Supported Langages 
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
# Strategies
# =========================
STRATEGY_FILE = "strategies.json"

def load_strategies(file_path: str = STRATEGY_FILE) -> Dict[str, str]:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

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
            ext = os.path.splitext(name)[1].lower()
            for lang, exts in LANGUAGE_EXTS.items():
                if ext in exts:
                    counts[lang] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if any(counts.values()) else "Unknown"

# =========================
# Helpers
# =========================
def _extract_json(text: str) -> str:
    if not text:
        return "[]"
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
        if m:
            return m.group(1)
    m = re.search(r"(\[.*\])", text, re.S)
    return m.group(0) if m else "[]"

def analyze_with_llm(code: str, mode: str, language: str, llm: ChatModel) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    strategies = load_strategies()
    if mode not in strategies:
        raise ValueError(f"Strategy '{mode}' not found in strategies.json")
    prompt = strategies[mode].format(language=language, code=code[:8000])

    content, meta = llm.chat(
        messages=[{"role": "system", "content": "Return only STRICT JSON."},
                  {"role": "user", "content": prompt}],
        max_tokens=1400,
        temperature=0,
        return_meta=True,
    )

    try:
        data = json.loads(_extract_json(content))
        if not isinstance(data, list):
            data = []
    except Exception:
        data = []
    return data, {
        "prompt_tokens": int(meta.get("prompt_tokens", 0)),
        "completion_tokens": int(meta.get("completion_tokens", 0)),
        "total_tokens": int(meta.get("total_tokens", 0)),
    }

# =========================
# Ground truth loader
# =========================
def load_ground_truth(static_csv: str, allowed_exts: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(static_csv, sep=None, engine="python", on_bad_lines="skip")
    low = {c.lower(): c for c in df.columns}
    file_col = None
    for k in ["file", "component", "path"]:
        if k in low:
            file_col = low[k]
            break
    if not file_col:
        raise ValueError(f"No file column found. Available: {list(df.columns)}")

    files = df[file_col].astype(str).value_counts().rename("static_count")
    issues = pd.DataFrame(files)
    issues["total"] = issues["static_count"]

    if allowed_exts:
        allowed = tuple(ext.lower() for ext in allowed_exts)
        # robust index filtering (mixed types safe)
        mask = issues.index.to_series().astype(str).str.lower().str.endswith(allowed)
        issues = issues[mask]

    return issues.sort_values("total", ascending=False)

# =========================
# Metrics (file-level)
# =========================
def _extract_file_from_item(item: Dict[str, Any]) -> Optional[str]:
    for k in ("file", "filename", "path", "component"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

def _canonize_path(s: str) -> str:
    return s.replace("\\", "/")

def _suffix_match(pred: str, pool: set[str]) -> bool:
    return any(pred.endswith("/"+g) or pred.endswith(g) or g.endswith("/"+pred) or g.endswith(pred) for g in pool)

def compute_metrics_file_level(results: List[dict], issues_top: pd.DataFrame) -> dict:
    gt_files = {_canonize_path(str(idx)) for idx in issues_top.index}

    pred_files: set[str] = set()
    for it in results or []:
        f = _extract_file_from_item(it)
        if f:
            pred_files.add(_canonize_path(f))

    TP = {p for p in pred_files if _suffix_match(p, gt_files)}
    FP = {p for p in pred_files if p not in TP}
    FN = {g for g in gt_files if not _suffix_match(g, pred_files)}

    tp, fp, fn = len(TP), len(FP), len(FN)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0

    return {
        "llm_issues": len(results or []),
        "gt_issues": int(issues_top["static_count"].sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp_files": tp, "fp_files": fp, "fn_files": fn,
    }

# =========================
# Pricing helpers (fix per-1K vs per-1M)
# =========================
def _safe_float(x: Optional[str]) -> Optional[float]:
    try:
        return float(x) if x is not None and x != "" else None
    except Exception:
        return None

def _read_rates_verbose(llm: ChatModel) -> Tuple[float, float, str]:
    """
    Returns (in_per_1k, out_per_1k, note).
    Reads slot-scoped env if available, else global.
    Accepts *_PER_1K or *_PER_1M (the latter converted /1000).
    Also auto-normalizes common mistake: values >1 that look like $/1M
    accidentally put into *_PER_1K.
    """
    slot = getattr(llm, "slot_key", None)

    def _pull(prefix: str):
        _in1k  = _safe_float(os.getenv(f"{prefix}_PRICE_IN_PER_1K"))
        _out1k = _safe_float(os.getenv(f"{prefix}_PRICE_OUT_PER_1K"))
        _in1m  = _safe_float(os.getenv(f"{prefix}_PRICE_IN_PER_1M"))
        _out1m = _safe_float(os.getenv(f"{prefix}_PRICE_OUT_PER_1M"))
        return _in1k, _out1k, _in1m, _out1m

    in1k = out1k = None
    note = ""

    if slot:
        a, b, c, d = _pull(slot)
        in1k, out1k = a, b
        if (in1k is None or out1k is None) and (c is not None or d is not None):
            # use per-1M if provided
            in1k = in1k if in1k is not None else (c or 0.0) / 1000.0
            out1k = out1k if out1k is not None else (d or 0.0) / 1000.0
            note = "normalized from per-1M (slot)"
    if in1k is None and out1k is None:
        # global fallback
        a, b, c, d = _pull("STALLM")
        in1k, out1k = a, b
        if (in1k is None or out1k is None) and (c is not None or d is not None):
            in1k = in1k if in1k is not None else (c or 0.0) / 1000.0
            out1k = out1k if out1k is not None else (d or 0.0) / 1000.0
            note = "normalized from per-1M (global)"

    # defaults if still None
    in1k = float(in1k or 0.0)
    out1k = float(out1k or 0.0)

    # common mistake: put $/1M into *_PER_1K (e.g., 5 and 15)
    # heuristics: both in/out > 1 and <= 200 → likely per-1M
    if note == "" and ((in1k > 1.0 and in1k <= 200.0) or (out1k > 1.0 and out1k <= 200.0)):
        in1k /= 1000.0
        out1k /= 1000.0
        note = "auto-normalized: values looked like per-1M placed in per-1K"

    return in1k, out1k, note

def _read_rates(llm: ChatModel) -> Tuple[float, float]:
    inp, outp, _ = _read_rates_verbose(llm)
    return inp, outp

def _cost_usd(prompt_tokens: int, completion_tokens: int, llm: ChatModel) -> Tuple[float, float, float, str]:
    """
    Returns (cost_usd, price_in_per_1k, price_out_per_1k, note)
    """
    in_rate, out_rate, note = _read_rates_verbose(llm)
    cost = (prompt_tokens / 1000.0) * in_rate + (completion_tokens / 1000.0) * out_rate
    return cost, in_rate, out_rate, note

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
# Pipelines
# =========================
def run_experiment(
    zip_path: str,
    static_csv: str,
    mode: str = "baseline",
    progress=None,
    top_k: int = 5,
    llm: Optional[ChatModel] = None,
    neg_files_ratio: float = 1.0,
):
    """
    Run one prompt strategy on:
      - top_k "positive" files (from CSV)
      - plus a sample of "negative" files
    Metrics: FILE-LEVEL (realistic).
    """
    language = detect_language_from_zip(zip_path)
    exts = find_exts_for_language(language) or None

    issues = load_ground_truth(static_csv, allowed_exts=exts)
    top_files = issues.head(top_k).index.tolist()
    results: List[Dict[str, Any]] = []
    llm = llm or _default_llm()

    pt_sum = ct_sum = tt_sum = 0

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        def _is_target(fname: str) -> bool:
            if exts:
                return any(fname.lower().endswith(e) for e in exts)
            return True

        all_src = [
            os.path.join(root, f)
            for root, _, files in os.walk(tmp)
            for f in files
            if _is_target(f)
        ]

        # === positive selection (legacy behaviour, robust to slashes/case) ===
        def canon(s: str) -> str:
            return (s or "").replace("\\", "/").lower()

        top_basenames = {os.path.basename(t).lower() for t in top_files}
        top_suffixes  = [canon(t) for t in top_files]

        pos_files = [p for p in all_src if os.path.basename(p).lower() in top_basenames]
        if not pos_files:
            pos_files = [p for p in all_src if any(canon(p).endswith(ts) for ts in top_suffixes)]
        if not pos_files:
            pos_files = all_src[:min(top_k, len(all_src))]

        neg_pool = [p for p in all_src if p not in pos_files]
        n_neg = min(int(len(pos_files) * max(0.0, neg_files_ratio)), len(neg_pool)) if pos_files else 0
        neg_files = random.sample(neg_pool, n_neg) if n_neg > 0 else []

        all_files = pos_files + neg_files
        total = len(all_files) or 1

        for i, file_path in enumerate(all_files, 1):
            if progress:
                progress.progress(i / total, text=f"🔍 {mode} analyzing {os.path.basename(file_path)} ({i}/{total})")
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    code = f.read()
            except Exception:
                code = ""

            smells, usage = analyze_with_llm(code, mode, language, llm)

            for it in (smells or []):
                it.setdefault("file", os.path.basename(file_path))
                it.setdefault("path", file_path)

            results.extend(smells or [])

            pt_sum += usage.get("prompt_tokens", 0)
            ct_sum += usage.get("completion_tokens", 0)
            tt_sum += usage.get("total_tokens", 0)

    cost_usd, price_in_k, price_out_k, price_note = _cost_usd(pt_sum, ct_sum, llm)
    usage_totals = {
        "prompt_tokens": pt_sum,
        "completion_tokens": ct_sum,
        "total_tokens": tt_sum,
        "usd_cost": cost_usd,
        # expose pricing to UI (for transparency)
        "price_in_per_1k": price_in_k,
        "price_out_per_1k": price_out_k,
        "pricing_note": price_note,
    }

    metrics = compute_metrics_file_level(results, issues.head(top_k))
    return results, issues.head(top_k), metrics, usage_totals

def run_selected_experiments(
    zip_path,
    static_csv,
    selected_modes,
    progress=None,
    status=None,
    timer=None,
    top_k=5,
    llm: Optional[ChatModel] = None,
    neg_files_ratio: float = 1.0,
):
    language = detect_language_from_zip(zip_path)
    exts = find_exts_for_language(language) or None

    issues_all = load_ground_truth(static_csv, allowed_exts=exts)
    issues_top = issues_all.head(top_k)

    all_metrics = []
    samples_dict = {}
    llm = llm or _default_llm()

    for idx, mode in enumerate(selected_modes):
        if status:
            status.markdown(f"🔍 Running `{mode}` strategy…")

        start = time.time()
        results, _, metrics, usage_totals = run_experiment(
            zip_path, static_csv, mode, top_k=top_k, llm=llm, neg_files_ratio=neg_files_ratio
        )
        elapsed = time.time() - start

        metrics["strategy"] = mode
        metrics["time_s"] = round(elapsed, 2)
        metrics["language"] = language
        metrics["top_k"] = top_k
        metrics["prompt_tokens"] = usage_totals["prompt_tokens"]
        metrics["completion_tokens"] = usage_totals["completion_tokens"]
        metrics["total_tokens"] = usage_totals["total_tokens"]
        metrics["usd_cost"] = usage_totals["usd_cost"]
        all_metrics.append(metrics)

        samples_dict[mode] = results[:5]

        save_run_result(
            project=os.path.basename(zip_path),
            filename="ALL",
            strategy=mode,
            language=language,
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            top_k=top_k,
            issues_detected=results,
            time_elapsed=elapsed,
            llm_used=llm.model_label(),
            sonar_detected_smells=issues_top.to_dict(orient="records"),
            prompt_tokens=usage_totals["prompt_tokens"],
            completion_tokens=usage_totals["completion_tokens"],
            total_tokens=usage_totals["total_tokens"],
            usd_cost=usage_totals["usd_cost"],
        )

        if progress:
            progress.progress((idx + 1) / len(selected_modes))
        if timer:
            timer.markdown(f"🕒 Elapsed: `{elapsed:.2f}` seconds for `{mode}`")

    metrics_df = pd.DataFrame(all_metrics).set_index("strategy")
    return issues_top, metrics_df, samples_dict


def run_selected_models_experiments(
    zip_path: str,
    static_csv: str,
    strategy: str,
    llms: List[ChatModel],
    progress=None,
    status=None,
    timer=None,
    top_k: int = 5,
    neg_files_ratio: float = 1.0,
):
    language = detect_language_from_zip(zip_path)
    exts = find_exts_for_language(language) or None

    issues_all = load_ground_truth(static_csv, allowed_exts=exts)
    issues_top = issues_all.head(top_k)

    all_metrics = []
    samples_dict: Dict[str, List[Dict[str, Any]]] = {}

    total = len(llms) or 1
    for i, llm in enumerate(llms, 1):
        label = llm.model_label()
        if status:
            status.markdown(f"🔍 `{label}` with strategy `{strategy}`…")

        start = time.time()
        results, _, metrics, usage_totals = run_experiment(
            zip_path, static_csv, strategy, top_k=top_k, llm=llm, neg_files_ratio=neg_files_ratio
        )
        elapsed = time.time() - start

        row = {
            "model": label,
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "time_s": round(elapsed, 2),
            "language": language,
            "top_k": top_k,
            "prompt_tokens": usage_totals.get("prompt_tokens", 0),
            "completion_tokens": usage_totals.get("completion_tokens", 0),
            "total_tokens": usage_totals.get("total_tokens", 0),
            "usd_cost": usage_totals.get("usd_cost", 0.0),
            "tp_files": metrics.get("tp_files", 0),
            "fp_files": metrics.get("fp_files", 0),
            "fn_files": metrics.get("fn_files", 0),
        }
        all_metrics.append(row)
        samples_dict[label] = results[:5]

        save_run_result(
            project=os.path.basename(zip_path),
            filename="ALL",
            strategy=strategy,
            language=language,
            f1=float(row["f1"]),
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            top_k=top_k,
            issues_detected=results,
            time_elapsed=elapsed,
            llm_used=label,
            sonar_detected_smells=issues_top.to_dict(orient="records"),
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            total_tokens=row["total_tokens"],
            usd_cost=row["usd_cost"],
        )

        if progress:
            progress.progress(i / total)
        if timer:
            timer.markdown(f"🕒 Elapsed: `{elapsed:.2f}` seconds • {label}")

    metrics_df = pd.DataFrame(all_metrics).set_index("model")
    return issues_top, metrics_df, samples_dict
