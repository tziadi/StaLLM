"""Structured chance baselines and matching sensitivity for the span-level runs
(RQ1/RQ2), computed offline from the stored outputs. No model call.

    python structured_baselines.py

Writes output/paper_results/stats/structured_baselines.csv,
matching_sensitivity.csv, lenient_category_pairs.csv and
structured_baselines.md.

Null models (same number and length of every predicted span, same file,
1,000 draws, p = (1 + #draws reaching the model F1) / (1 + draws)):
- uniform: start line uniform over the file (the baseline of the paper);
- code_lines: start line uniform over the non-blank, non-comment lines;
- method_body: start line uniform over the code lines inside method,
  constructor, accessor, or local-function bodies, detected by a brace-matching heuristic
  (a line with a parameter list that is not a control statement, followed by
  `{` or `=>`). Files without detected bodies fall back to code_lines.
"""
from __future__ import annotations

import json
import random
import re
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import analyze_paper_stats as aps

STATS = aps.STATS
DRAWS = 1000

_CONTROL = re.compile(r"^\s*(if|else|for|foreach|while|switch|catch|using|lock|return|throw|new|await|var)\b")
_SIGNATURE = re.compile(r"[\w>\]?]\s+[\w<>,]+\s*\([^;]*\)?\s*(where\b.*)?(\{|=>)?\s*$|^\s*(get|set|init|add|remove)\s*(\{|=>)")
_TYPE_DECL = re.compile(r"\b(class|struct|interface|enum|record|namespace)\b")


def code_lines(text: list[str]) -> list[int]:
    lines, in_block = [], False
    for i, raw in enumerate(text, start=1):
        s = raw.strip()
        if in_block:
            if "*/" in s:
                in_block = False
                s = s.split("*/", 1)[1].strip()
            else:
                continue
        if s.startswith("/*"):
            if "*/" not in s:
                in_block = True
            continue
        if not s or s.startswith("//"):
            continue
        lines.append(i)
    return lines


def method_body_lines(text: list[str]) -> list[int]:
    """Lines inside the bodies of members with a parameter list or accessors."""
    body: set[int] = set()
    n = len(text)
    i = 0
    while i < n:
        s = text[i].split("//", 1)[0].rstrip()
        if (_SIGNATURE.search(s) and not _CONTROL.match(s) and not _TYPE_DECL.search(s)
                and ("(" in s or re.match(r"^\s*(get|set|init|add|remove)\b", s))):
            # find the opening of the body: `{` or `=>` on this or the next lines
            j, opened = i, None
            while j < n and j <= i + 5:
                t = text[j].split("//", 1)[0]
                if "=>" in t and "{" not in t.split("=>", 1)[0]:
                    opened = ("expr", j)
                    break
                if "{" in t:
                    opened = ("block", j)
                    break
                if ";" in t:
                    break
                j += 1
            if opened and opened[0] == "block":
                depth, k = 0, opened[1]
                while k < n:
                    t = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text[k].split("//", 1)[0])
                    depth += t.count("{") - t.count("}")
                    if k > i:
                        body.add(k + 1)
                    if depth <= 0 and k >= opened[1] and ("}" in t):
                        break
                    k += 1
                body.add(i + 1)
                i = k + 1
                continue
            if opened and opened[0] == "expr":
                k = opened[1]
                while k < n and ";" not in text[k]:
                    k += 1
                body.update(range(i + 1, k + 2))
                i = k + 1
                continue
        i += 1
    return sorted(x for x in body if x <= n)


def load_sources() -> dict[str, list[str]]:
    from StaLLM_core import _build_universe, load_ground_truth_spans
    gt = load_ground_truth_spans(str(aps.RQ12_CSV), allowed_exts=[".cs"])
    universe, _, _ = _build_universe(str(aps.RQ12_ZIP), gt, [".cs"], 20, 1.0)
    out = {}
    with zipfile.ZipFile(aps.RQ12_ZIP) as z:
        names = {n.lower(): n for n in z.namelist()}
        for rel, base in zip(universe["relpath"], universe["basename"]):
            out[base] = z.read(names[rel.lower()]).decode("utf-8", "replace").splitlines()
    return out


def placement_f1(spans: dict[str, Any], eligible: dict[str, list[int]] | None, rng: random.Random, match) -> float:
    preds: dict[str, list] = defaultdict(list)
    for b, f in spans.items():
        n_lines = f["lines"]
        pool = eligible[b] if eligible else None
        for s, e in f["pred"]:
            length = min(max(0, e - s), n_lines - 1)
            if pool:
                start = rng.choice(pool)
                start = min(start, max(1, n_lines - length))
            else:
                start = rng.randint(1, max(1, n_lines - length))
            preds[b].append({"startLine": start, "endLine": start + length})
    return aps._span_f1(spans, preds, match)


def matcher(preset: str):
    from StaLLM_core import _match_file

    def match(p, g):
        return _match_file([dict(x, startColumn=None, endColumn=None, type=None) for x in p],
                           [dict(x, startColumn=None, endColumn=None, type=None) for x in g],
                           require_type=False, use_cols_single=False, preset=preset)
    return match


def load_groups() -> dict[tuple, list]:
    oracle = aps._rq12_oracle()
    groups: dict[tuple, list] = defaultdict(list)
    for protocol, folder in aps.RQ12_DIRS.items():
        for f in sorted(folder.glob("rq2_*.json")):
            d = aps._load(f)
            groups[(protocol, d["slot"], d["prompt"])].append(
                aps._rescored_spans(d["metrics"]["spans_by_file"], oracle))
    return groups


def llm_f1(runs: list, match) -> float:
    return statistics.mean(
        aps._span_f1(s, {b: [{"startLine": x, "endLine": y} for x, y in v["pred"]] for b, v in s.items()}, match)
        for s in runs)


def null_test(runs: list, eligible, match, observed: float, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    draws = [statistics.mean(placement_f1(s, eligible, rng, match) for s in runs) for _ in range(DRAWS)]
    return statistics.mean(draws), (1 + sum(x >= observed for x in draws)) / (1 + DRAWS)


# ── type categories for the lenient-pair audit ──────────────────────────────

RULE_CATEGORY = {
    "csharpsquid:S125": "dead_code", "csharpsquid:S1481": "dead_code", "csharpsquid:S1450": "design",
    "csharpsquid:S1186": "dead_code", "external_roslyn:CA1822": "design", "external_roslyn:CA2211": "design",
    "csharpsquid:S1104": "design", "csharpsquid:S2223": "design", "csharpsquid:S3903": "design",
    "external_roslyn:CA1050": "design", "csharpsquid:S927": "naming", "csharpsquid:S112": "exception",
    "csharpsquid:S2699": "testing", "external_roslyn:CA1829": "performance", "csharpsquid:S1075": "hardcoded",
    "csharpsquid:S1066": "complexity", "csharpsquid:S3776": "complexity", "csharpsquid:S3925": "exception",
    "external_roslyn:CS0618": "deprecated",
}
DESC_KEYWORDS = [
    ("exception", r"exception|throw|catch|try\b|error handling"),
    ("hardcoded", r"hard-?coded|magic (number|string)|literal|url|connection string"),
    ("complexity", r"complex|nested|long method|cognitive|refactor"),
    ("dead_code", r"unused|dead code|commented|empty (method|block)|never used|redundant"),
    ("naming", r"naming|name|parameter name"),
    ("performance", r"performance|count\(\)|inefficient|allocation|linq|async|blocking"),
    ("testing", r"assert|test"),
    ("deprecated", r"obsolete|deprecated"),
    ("design", r"static|public field|encapsulat|mutable|visibility|readonly|namespace|accessib"),
]


def describe_category(text: str) -> str:
    t = (text or "").lower()
    for cat, pat in DESC_KEYWORDS:
        if re.search(pat, t):
            return cat
    return "other"


def lenient_pair_audit(md: list[str]) -> None:
    """Pairs a lenient match forms between a model prediction and an oracle finding,
    for the predictions whose description can be recovered from llm_debug/: share of
    pairs whose description falls in another rule category than the finding."""
    from StaLLM_core import _build_universe, load_ground_truth_spans, _coerce_items_schema
    gt = load_ground_truth_spans(str(aps.RQ12_CSV), allowed_exts=[".cs"])
    universe, _, aligned = _build_universe(str(aps.RQ12_ZIP), gt, [".cs"], 20, 1.0)
    findings = defaultdict(list)
    for _, r in aligned[aligned["basename"].isin(set(universe["basename"]))].iterrows():
        findings[r["basename"]].append((int(r["startLine"]), int(r["endLine"]), r["type"]))
    # index of saved answers: prediction set -> list of (start, end, description)
    answers: dict[frozenset, list] = {}
    for f in Path("llm_debug").glob("*_sanitized.txt"):
        try:
            items = _coerce_items_schema(json.loads(f.read_text()))
        except Exception:
            continue
        spans = [(int(i.get("startLine") or i.get("line") or 0),
                  int(i.get("endLine") or i.get("startLine") or i.get("line") or 0),
                  str(i.get("description") or "")) for i in items]
        answers.setdefault(frozenset((s, e) for s, e, _ in spans), spans)
    delta = 3  # lenient line tolerance
    pairs, seen = [], set()
    for protocol, folder in aps.RQ12_DIRS.items():
        for jf in sorted(folder.glob("rq2_*.json")):
            d = aps._load(jf)
            for b, f in d["metrics"]["spans_by_file"].items():
                key = frozenset(map(tuple, f["pred"]))
                if not f["pred"] or key not in answers or b not in findings:
                    continue
                for s, e, desc in answers[key]:
                    for gs, ge, rule in findings[b]:
                        if s <= ge + delta and gs <= e + delta:
                            k = (b, s, e, desc, gs, ge, rule)
                            if k in seen:
                                continue
                            seen.add(k)
                            pairs.append({"file": b, "pred_start": s, "pred_end": e, "description": desc[:200],
                                          "rule": rule, "rule_category": RULE_CATEGORY.get(rule, "other"),
                                          "description_category": describe_category(desc)})
    aps._write_csv(STATS / "lenient_category_pairs.csv", pairs)
    known = [p for p in pairs if p["description_category"] != "other"]
    same = sum(p["description_category"] == p["rule_category"] for p in known)
    md.append("## Lenient pairs: category of the prediction vs category of the matched finding\n")
    md.append(f"- distinct (prediction, finding) pairs within ±{delta} lines, with a recoverable description: {len(pairs)}")
    md.append(f"- descriptions assigned to a category by keywords: {len(known)} "
              f"({len(known) / max(1, len(pairs)):.0%}); same category as the finding's rule: {same} "
              f"({same / max(1, len(known)):.0%}), other category: {len(known) - same} "
              f"({(len(known) - same) / max(1, len(known)):.0%})")
    by_rule = defaultdict(lambda: [0, 0])
    for p in known:
        by_rule[p["rule_category"]][0] += 1
        by_rule[p["rule_category"]][1] += p["description_category"] == p["rule_category"]
    md.append("\n| Finding category | pairs | same category |\n|---|---|---|")
    for cat, (n, k) in sorted(by_rule.items(), key=lambda x: -x[1][0]):
        md.append(f"| {cat} | {n} | {k} |")
    md.append("")


def main() -> None:
    STATS.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    elig_code = {b: code_lines(t) for b, t in sources.items()}
    elig_meth, fallback = {}, []
    for b, t in sources.items():
        m = sorted(set(method_body_lines(t)) & set(elig_code[b]))
        if not m:
            fallback.append(b)
        elig_meth[b] = m or elig_code[b]
    md = ["# Structured chance baselines and matching sensitivity (RQ1/RQ2)\n",
          f"{DRAWS} draws per null; p = (1 + #draws ≥ model F1) / (1 + draws).\n",
          "Eligible lines per file (total / code / method body): "
          + ", ".join(f"{Path(b).name} {len(sources[b])}/{len(elig_code[b])}/{len(elig_meth[b])}"
                      for b in sorted(sources)) + "\n",
          f"Files without a detected method body (method_body falls back to code lines): "
          f"{', '.join(Path(b).name for b in fallback) or 'none'}\n"]
    groups = load_groups()
    rows, mrows = [], []
    lenient = matcher("Lenient")
    for (protocol, slot, prompt), runs in sorted(groups.items()):
        f1 = llm_f1(runs, lenient)
        row = {"protocol": protocol, "slot": slot, "prompt": prompt, "runs": len(runs), "f1": aps._r(f1)}
        for name, elig in (("uniform", None), ("code_lines", elig_code), ("method_body", elig_meth)):
            mean, p = null_test(runs, elig, lenient, f1)
            row[f"random_{name}"], row[f"p_{name}"] = aps._r(mean), aps._r(p, 4)
        rows.append(row)
        for preset in ("Balanced", "Strict"):
            m = matcher(preset)
            pf1 = llm_f1(runs, m)
            mean, p = null_test(runs, None, m, pf1)
            mrows.append({"protocol": protocol, "slot": slot, "prompt": prompt, "preset": preset,
                          "f1": aps._r(pf1), "random_uniform": aps._r(mean), "p_uniform": aps._r(p, 4)})
        print(row)
    aps._write_csv(STATS / "structured_baselines.csv", rows)
    aps._write_csv(STATS / "matching_sensitivity.csv", mrows)
    md.append("## Lenient matching, three null models\n")
    md.append("| Protocol | Slot | Prompt | runs | F1 | uniform | p | code lines | p | method body | p |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['protocol']} | {r['slot']} | {r['prompt']} | {r['runs']} | {r['f1']} | "
                  f"{r['random_uniform']} | {r['p_uniform']} | {r['random_code_lines']} | {r['p_code_lines']} | "
                  f"{r['random_method_body']} | {r['p_method_body']} |")
    md.append("\n## Balanced and Strict presets, uniform null\n")
    md.append("| Protocol | Slot | Prompt | preset | F1 | random | p |\n|---|---|---|---|---|---|---|")
    for r in mrows:
        md.append(f"| {r['protocol']} | {r['slot']} | {r['prompt']} | {r['preset']} | {r['f1']} | "
                  f"{r['random_uniform']} | {r['p_uniform']} |")
    md.append("")
    lenient_pair_audit(md)
    (STATS / "structured_baselines.md").write_text("\n".join(md))
    print(f"Wrote {STATS / 'structured_baselines.md'}")


if __name__ == "__main__":
    main()
