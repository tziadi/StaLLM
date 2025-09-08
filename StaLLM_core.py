# StaLLM_core.py
import os, re, json, zipfile, tempfile, time
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
        issues = issues[issues.index.str.lower().str.endswith(allowed)]

    return issues.sort_values("total", ascending=False)

# =========================
# Metrics
# =========================
def _prf(llm_count, gt_count) -> Tuple[float, float, float]:
    if llm_count == gt_count == 0:
        return (0, 0, 0)
    inter = min(llm_count, gt_count)
    p = inter / llm_count if llm_count > 0 else 0
    r = inter / gt_count if gt_count > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1

def compute_metrics(results: List[Dict[str, Any]], issues: pd.DataFrame) -> Dict[str, Any]:
    llm_count = len(results)
    gt_count = int(issues["static_count"].sum())
    p, r, f1 = _prf(llm_count, gt_count)
    return {"llm_issues": llm_count, "gt_issues": gt_count, "precision": p, "recall": r, "f1": f1}

# =========================
# Costs (USD) from tokens
# =========================
def _read_rates(llm: ChatModel) -> Tuple[float, float]:
    slot = getattr(llm, "slot_key", None)
    if slot:
        try_in = os.getenv(f"{slot}_PRICE_IN_PER_1K")
        try_out = os.getenv(f"{slot}_PRICE_OUT_PER_1K")
        if try_in or try_out:
            return float(try_in or 0), float(try_out or 0)
    return float(os.getenv("STALLM_PRICE_IN_PER_1K", 0)), float(os.getenv("STALLM_PRICE_OUT_PER_1K", 0))

def _cost_usd(prompt_tokens: int, completion_tokens: int, llm: ChatModel) -> float:
    in_rate, out_rate = _read_rates(llm)
    return (prompt_tokens / 1000.0) * in_rate + (completion_tokens / 1000.0) * out_rate

# =========================
# Default LLM (from env slots or manual)
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
    mode="baseline",
    progress=None,
    top_k: int = 5,
    llm: Optional[ChatModel] = None
):
    language = detect_language_from_zip(zip_path)
    exts = find_exts_for_language(language) or None

    issues = load_ground_truth(static_csv, allowed_exts=exts)
    top_files = issues.head(top_k).index.tolist()
    results: List[Dict[str, Any]] = []
    llm = llm or _default_llm()

    # Accumulate tokens
    pt_sum = ct_sum = tt_sum = 0

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        def _is_target(fname: str) -> bool:
            if exts:
                return any(fname.lower().endswith(e) for e in exts)
            return True

        all_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(tmp)
            for f in files
            if _is_target(f) and any(t.endswith(f) for t in top_files)
        ]

        total = len(all_files) or 1
        for i, file_path in enumerate(all_files, 1):
            if progress:
                progress.progress(i / total, text=f"🔍 {mode} analyzing {os.path.basename(file_path)} ({i}/{total})")
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                code = f.read()
            smells, usage = analyze_with_llm(code, mode, language, llm)
            results.extend(smells)
            pt_sum += usage.get("prompt_tokens", 0)
            ct_sum += usage.get("completion_tokens", 0)
            tt_sum += usage.get("total_tokens", 0)

    usage_totals = {
        "prompt_tokens": pt_sum,
        "completion_tokens": ct_sum,
        "total_tokens": tt_sum,
        "usd_cost": _cost_usd(pt_sum, ct_sum, llm),
    }

    metrics = compute_metrics(results, issues.head(top_k))
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
        results, _, metrics, usage_totals = run_experiment(zip_path, static_csv, mode, top_k=top_k, llm=llm)
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

        # Persist
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
