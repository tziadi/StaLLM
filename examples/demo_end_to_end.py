"""
StaLLM — end-to-end demonstration: configuration -> evaluation report
======================================================================
Runs one span-level evaluation through the benchmark pipeline (oracle
alignment and line check, universe sampling, model calls, matching,
metrics) and applies the built-in chance controls, then writes a Markdown
report with the full provenance.

The model is either a configured slot (e.g. "AZ1", needs credentials) or
"replay:<run.json>", which answers with the spans recorded in a previous run
and therefore needs no credentials and reproduces that run exactly.

Usage:
    python examples/demo_end_to_end.py examples/demo_config.json
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_paper_stats as aps  # noqa: E402
from StaLLM_core import _match_file, run_experiment  # noqa: E402


class ReplayModel:
    """ChatModel stand-in that answers with the spans recorded for each file."""

    def __init__(self, run_json: Path, archive: Path):
        recorded = json.loads(run_json.read_text())["metrics"]["spans_by_file"]
        self.spans = {k: v["pred"] for k, v in recorded.items()}
        with zipfile.ZipFile(archive) as z:
            names = {n.lower(): n for n in z.namelist()}
            # the pipeline reads files in text mode, which turns CRLF into LF
            self.code = {k: z.read(names[k]).decode("utf-8", "ignore").replace("\r\n", "\n")[:20000]
                         for k in self.spans}
        self.slot_key, self.calls = None, 0
        self.cfg = type("Cfg", (), {"model": f"replay:{run_json.name}", "provider": "replay"})()

    def model_label(self) -> str:
        return self.cfg.model

    def chat(self, messages, temperature=None, max_tokens=None, return_meta=False, **_):
        self.calls += 1
        prompt = messages[-1]["content"]
        # The file being analysed is the one whose code appears in the prompt
        # (longest match first, so that a short file contained in another is not chosen).
        key = next((k for k, c in sorted(self.code.items(), key=lambda kv: -len(kv[1]))
                    if c.strip() and c.strip() in prompt), None)
        items = [{"type": "replay", "description": "recorded span", "startLine": s, "endLine": e}
                 for s, e in self.spans.get(key, [])]
        content = json.dumps(items)
        meta = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return (content, meta) if return_meta else content


def build_model(spec: str, archive: Path):
    if spec.startswith("replay:"):
        return ReplayModel(Path(spec.split(":", 1)[1]), archive)
    from StaLLM_llm import build_llm_from_slot
    return build_llm_from_slot(spec)


def main(config_path: str) -> None:
    cfg = json.loads(Path(config_path).read_text())
    if cfg["task"] != "span_findings":
        raise SystemExit("this demo covers the span-level task; see run_paper_experiments.py for the others")
    archive, oracle = Path(cfg["archive"]), Path(cfg["oracle"])
    model = build_model(cfg["model"], archive)

    _, universe, metrics, usage = run_experiment(
        str(archive), str(oracle), mode=cfg["prompt"], top_k=cfg["top_k"], pos_ratio=cfg["positive_ratio"],
        preset=cfg["preset"], user_require_type=False, user_use_line_span=True, user_use_cols_single=False,
        llm=model, number_lines=cfg.get("number_lines", False))

    def match(p, g):
        strip = lambda xs: [dict(x, startColumn=None, endColumn=None, type=None) for x in xs]
        return _match_file(strip(p), strip(g), require_type=False, use_cols_single=False, preset=cfg["preset"])

    spans = metrics["spans_by_file"]
    controls = []
    c = cfg.get("controls", {})
    for name, shuffle in (("random_placement", False), ("random_file", True)):
        if not c.get(name):
            continue
        rng = random.Random(c.get("seed", 42))
        draws = sorted(aps.random_placement_f1(spans, rng, match, shuffle) for _ in range(c.get("draws", 1000)))
        n = len(draws)
        controls.append((name, statistics.mean(draws), draws[int(0.025 * n)], draws[int(0.975 * n) - 1],
                         (1 + sum(x >= metrics["f1"] for x in draws)) / (1 + n)))

    diag = metrics["diagnostics"]
    al = diag["gt_alignment"]
    lines = [
        "# StaLLM evaluation report\n",
        f"Configuration: `{config_path}`\n",
        "## Setup\n",
        f"- task: span-level findings; archive `{archive}`; oracle `{oracle}`",
        f"- model: `{model.model_label() if hasattr(model, 'model_label') else cfg['model']}`; prompt `{cfg['prompt']}`; "
        f"line numbers shown: {metrics['number_lines']}",
        f"- matching preset: {cfg['preset']} (location only)",
        f"- universe: {diag['top_k_total_files']} files, {diag['positives']} with oracle spans "
        f"(requested positive ratio {cfg['positive_ratio']}, achieved {diag['positives'] / diag['top_k_total_files']:.2f})",
        f"- oracle alignment: {al['mode']} matching, {al['gt_spans_matched']} spans kept, "
        f"{al['gt_spans_dropped']} outside the archive; line offset applied: {al.get('line_offset', 0)}\n",
        "## Model\n",
        f"- precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}, F1 {metrics['f1']:.3f} "
        f"[95% CI {metrics['f1_ci_lo']:.3f}, {metrics['f1_ci_hi']:.3f}]",
        f"- answers per source: {metrics['file_sources']}; tokens: {usage['total_tokens']}\n",
        "## Built-in controls\n",
        "| Control | F1 without localization | 95% interval | share of draws reaching the model |",
        "|---|---|---|---|",
    ]
    lines += [f"| {n} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {p:.3f} |" for n, m, lo, hi, p in controls]
    out = Path(cfg.get("report", "output/demo/demo_report.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    (out.with_suffix(".json")).write_text(json.dumps({"config": cfg, "metrics": {
        k: v for k, v in metrics.items() if k not in ("file_diagnostics", "type_breakdown")}, "controls": [
        dict(zip(("control", "mean", "lo", "hi", "p"), x)) for x in controls]}, indent=2, default=str))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/demo_config.json")
