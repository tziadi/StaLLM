"""PMD as an established second oracle for RQ3 (no model call).

    python pmd_oracle.py --pmd /path/to/pmd-bin-7.7.0/bin/pmd

Runs PMD 7 with its default thresholds on the code fragments of the annotated
DACOS selection split and of the 200 MLCQ samples, labels each (fragment, smell)
pair from the PMD rules mapped to the smell, and scores the RQ3 predictions
(run 1) against these labels. Writes output/paper_results_annot/stats/pmd_*.csv
and pmd_oracle.md.

Rule mapping (PMD category/java/design, default thresholds):
- complex method: CyclomaticComplexity (method >= 10), CognitiveComplexity (>= 15),
  or NPathComplexity (>= 200) on a method;
- long parameter list: ExcessiveParameterList (>= 10 parameters);
- long method: NcssCount on a method (>= 60);
- god class: GodClass; data class: DataClass;
- feature envy: no PMD rule, so MLCQ feature-envy samples are excluded.
A fragment is parsed as is (wrapped in a class when it has no type declaration),
then repaired (orphan comment tail removed, braces balanced), then wrapped in a
class, then in a method; fragments that still fail are excluded and counted.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import analyze_paper_stats as aps
import run_paper_experiments as rpe
from reviewer_analyses import _rho

ANNOT = Path("output/paper_results_annot")
STATS = ANNOT / "stats"
MAIN = Path("output/paper_results")

RULES = {
    "CyclomaticComplexity": '<properties><property name="methodReportLevel" value="10"/>'
                            '<property name="classReportLevel" value="80"/></properties>',
    "CognitiveComplexity": '<properties><property name="reportLevel" value="15"/></properties>',
    "NPathComplexity": '<properties><property name="reportLevel" value="200"/></properties>',
    "ExcessiveParameterList": '<properties><property name="minimum" value="10"/></properties>',
    "NcssCount": '<properties><property name="methodReportLevel" value="60"/>'
                 '<property name="classReportLevel" value="1500"/></properties>',
    "GodClass": "",
    "DataClass": "",
}
SMELL_RULES = {
    "complex_method": {"CyclomaticComplexity", "CognitiveComplexity", "NPathComplexity"},
    "long_parameter_list": {"ExcessiveParameterList"},
    "long_method": {"NcssCount"},
    "god_class": {"GodClass"},
    "data_class": {"DataClass"},
}
METHOD_ONLY = {"CyclomaticComplexity", "NcssCount"}  # these rules also report at class level
_UNIT = re.compile(r"^\s*(package|import)\s|^\s*(@\w+(\([^)]*\))?\s+)*((public|protected|private|abstract|final|static)\s+)*"
                   r"(class|interface|enum|@interface|record)\s+\w+", re.M)


def ruleset() -> str:
    rules = "\n".join(f'  <rule ref="category/java/design.xml/{name}">{props}</rule>' for name, props in RULES.items())
    return ('<?xml version="1.0"?>\n<ruleset name="stallm-smells" xmlns="http://pmd.sourceforge.net/ruleset/2.0.0" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://pmd.sourceforge.net/ruleset/2.0.0 https://pmd.sourceforge.io/ruleset_2_0_0.xsd">\n'
            f'  <description>StaLLM RQ3 second oracle</description>\n{rules}\n</ruleset>\n')


def run_pmd(pmd: str, files: dict[str, str]) -> tuple[dict[str, set[str]], set[str]]:
    """Return {key: rules violated (method-level for METHOD_ONLY rules)} and the keys that failed to parse."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "src"
        src.mkdir()
        names = {}
        for i, (key, code) in enumerate(files.items()):
            p = src / f"F{i}.java"
            p.write_text(code)
            names[str(p.resolve())] = key
        (root / "rules.xml").write_text(ruleset())
        out = root / "report.json"
        subprocess.run([pmd, "check", "-d", str(src), "-R", str(root / "rules.xml"), "-f", "json",
                        "-r", str(out), "--no-cache", "--no-progress"], capture_output=True, text=True)
        report = json.loads(out.read_text())
    hits: dict[str, set[str]] = {k: set() for k in files}
    for f in report.get("files", []):
        key = names.get(str(Path(f["filename"]).resolve()))
        if key is None:
            continue
        for v in f.get("violations", []):
            rule = v["rule"]
            if rule in METHOD_ONLY and ("class" in v["description"].lower() and "method" not in v["description"].lower()):
                continue
            hits[key].add(rule)
    failed = {names[str(Path(e["filename"]).resolve())] for e in report.get("processingErrors", [])
              if str(Path(e["filename"]).resolve()) in names}
    return hits, failed


def wrap(code: str) -> str:
    return f"class StallmFragment {{\n{code}\n}}\n"


def wrap_method(code: str) -> str:
    return f"class StallmFragment {{\nvoid stallmFragment() {{\n{code}\n}}\n}}\n"


def repair(code: str) -> str:
    """MLCQ fragments are line ranges cut from files: drop the tail of a comment that
    starts before the range, and balance the braces at the end of the fragment."""
    lines = code.splitlines()
    first = next((i for i, l in enumerate(lines) if l.strip()), 0)
    if lines and lines[first].lstrip().startswith("*"):
        end = next((i for i in range(first, len(lines)) if "*/" in lines[i]), first)
        lines = lines[end + 1:]
    text = "\n".join(lines)
    depth = sum(re.sub(r'"(?:[^"\\]|\\.)*"', "", l.split("//", 1)[0]).count("{")
                - re.sub(r'"(?:[^"\\]|\\.)*"', "", l.split("//", 1)[0]).count("}") for l in lines)
    if depth < 0:
        for _ in range(-depth):
            i = text.rfind("}")
            text = text[:i] + text[i + 1:]
    elif depth > 0:
        text += "\n" + "}" * depth
    return text


def pmd_labels(pmd: str, samples: list) -> tuple[dict[str, bool], dict[str, int]]:
    """Try each fragment as a compilation unit (or wrapped in a class when it has no type
    declaration), then repaired, then wrapped in a class, then in a method of a class."""
    pending = {s.sample_id: s.code for s in samples if s.smell in SMELL_RULES}
    attempts = [
        ("as_is", lambda c: c if _UNIT.search(c) else wrap(c)),
        ("repaired", lambda c: (lambda r: r if _UNIT.search(r) else wrap(r))(repair(c))),
        ("class_wrapped", lambda c: wrap(repair(c))),
        ("method_wrapped", lambda c: wrap_method(repair(c))),
    ]
    hits: dict[str, set[str]] = {}
    used: dict[str, str] = {}
    for name, fn in attempts:
        if not pending:
            break
        h, failed = run_pmd(pmd, {k: fn(c) for k, c in pending.items()})
        for k in pending:
            if k not in failed:
                hits[k], used[k] = h[k], name
        pending = {k: c for k, c in pending.items() if k in failed}
    labels = {s.sample_id: bool(hits[s.sample_id] & SMELL_RULES[s.smell]) for s in samples if s.sample_id in hits}
    stats = {"samples": len(samples), "no_pmd_rule": sum(s.smell not in SMELL_RULES for s in samples),
             **{f"parsed_{name}": sum(v == name for v in used.values()) for name, _ in attempts},
             "parse_failed": len(pending), "scored": len(labels)}
    return labels, stats


def configs(dataset: str) -> dict[str, dict[str, bool]]:
    root = ANNOT if dataset == "DACOS" else MAIN
    out = {}
    for f in sorted((root / "rq3").glob(f"rq3_AZ[12378]_*_{dataset}.json")):
        d = aps._load(f)
        out[f"{d['slot']}/{d['prompt']}"] = {r["sample_id"]: bool(r["predicted"]) for r in d["rows"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmd", required=True, help="Path to the PMD 7 launcher (bin/pmd).")
    parser.add_argument("--lpl-minimum", type=int, default=10,
                        help="ExcessiveParameterList threshold (PMD default 10; Designite uses 5).")
    args = parser.parse_args()
    pmd = args.pmd
    tag = "" if args.lpl_minimum == 10 else f"_lpl{args.lpl_minimum}"
    RULES["ExcessiveParameterList"] = f'<properties><property name="minimum" value="{args.lpl_minimum}"/></properties>'

    STATS.mkdir(parents=True, exist_ok=True)
    rpe.DACOS_LABELS = "annotations"
    sets = {"DACOS": rpe.load_dacos("selection"),
            "MLCQ": rpe.load_mlcq_smell_samples(str(rpe.MLCQ_CSV), limit=rpe.MAX_SAMPLES)}
    md = ["# PMD as a second oracle for RQ3\n", f"PMD 7.7.0, default thresholds except ExcessiveParameterList minimum = {args.lpl_minimum}; see the module docstring for the rule mapping.\n"]
    rows, summary = [], []
    for ds, samples in sets.items():
        labels, st = pmd_labels(pmd, samples)
        (STATS / f"pmd_labels_{ds}{tag}.json").write_text(json.dumps(
            [{"sample_id": k, "pmd": v} for k, v in sorted(labels.items())], indent=1))
        human = {s.sample_id: bool(s.label) for s in samples if s.sample_id in labels}
        ids = sorted(labels)
        pmd_pos = sum(labels[i] for i in ids)
        f1_ph, k_ph = aps._binary_scores([human[i] for i in ids], [labels[i] for i in ids])
        by_smell = {}
        for s in samples:
            if s.sample_id in labels:
                c = by_smell.setdefault(s.smell, [0, 0, 0])
                c[0] += 1
                c[1] += labels[s.sample_id]
                c[2] += human[s.sample_id]
        k_h, k_p, names = [], [], []
        for name, pred in configs(ds).items():
            ids_c = [i for i in ids if i in pred]
            _, kh = aps._binary_scores([human[i] for i in ids_c], [pred[i] for i in ids_c])
            _, kp = aps._binary_scores([labels[i] for i in ids_c], [pred[i] for i in ids_c])
            names.append(name)
            k_h.append(kh)
            k_p.append(kp)
            rows.append({"dataset": ds, "config": name, "n": len(ids_c), "kappa_human": aps._r(kh), "kappa_pmd": aps._r(kp)})
        rho = _rho(k_h, k_p)
        best_h = names[max(range(len(names)), key=lambda i: k_h[i])]
        best_p = names[max(range(len(names)), key=lambda i: k_p[i])]
        summary.append({"dataset": ds, **st, "pmd_positive": pmd_pos, "human_positive": sum(human.values()),
                        "kappa_pmd_vs_human": aps._r(k_ph), "f1_pmd_vs_human": aps._r(f1_ph),
                        "rho_human_pmd": aps._r(rho), "best_human": best_h, "best_pmd": best_p,
                        "by_smell": json.dumps(by_smell)})
        md.append(f"## {ds}\n")
        md.append(f"- samples {st['samples']}, without a PMD rule {st['no_pmd_rule']}, parsed as is {st['parsed_as_is']}, "
                  f"after repair {st['parsed_repaired']}, wrapped in a class {st['parsed_class_wrapped']}, "
                  f"wrapped in a method {st['parsed_method_wrapped']}, not parsed {st['parse_failed']}, scored {st['scored']}")
        md.append(f"- PMD positives {pmd_pos}, human positives {sum(human.values())}; "
                  f"κ(PMD, human) = {aps._r(k_ph)}, F1 = {aps._r(f1_ph)}")
        md.append(f"- per smell (n, PMD positive, human positive): {by_smell}")
        md.append(f"- Spearman ρ between the κ rankings of the {len(names)} configurations (human vs PMD) = {aps._r(rho)}; "
                  f"best vs human: {best_h}; best vs PMD: {best_p}\n")
        md.append("| Configuration | n | κ human | κ PMD |\n|---|---|---|---|")
        for r in rows:
            if r["dataset"] == ds:
                md.append(f"| {r['config']} | {r['n']} | {r['kappa_human']} | {r['kappa_pmd']} |")
        md.append("")
    aps._write_csv(STATS / f"pmd_oracle_configs{tag}.csv", rows)
    aps._write_csv(STATS / f"pmd_oracle_summary{tag}.csv", summary)
    (STATS / f"pmd_oracle{tag}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
