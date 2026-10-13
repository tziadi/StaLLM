"""Provenance for runs launched from the workbench.

A *run manifest* is a small JSON document that records everything needed to
re-run or audit an experiment: code version, environment, model slot, prompt
(with its hash), oracle and evaluation settings.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st

REPO_DIR = Path(__file__).resolve().parent.parent
MANIFEST_SCHEMA = "stallm.run-manifest/1"

CITATION_BIBTEX = """@inproceedings{Ziadi2026StaLLM,
  author    = {Ziadi, Tewfik and Bouallegue, Seifeddine and Bendraou, Reda},
  title     = {{StaLLM}: Benchmarking Prompts and {LLMs} in Software Maintenance},
  booktitle = {Proceedings of the 21st International Conference on Evaluation of
               Novel Approaches to Software Engineering (ENASE)},
  year      = {2026},
  publisher = {SciTePress}
}"""

CITATION_TEXT = (
    "T. Ziadi, S. Bouallegue, R. Bendraou. StaLLM: Benchmarking Prompts and LLMs "
    "in Software Maintenance. In Proc. ENASE 2026, SciTePress."
)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True, timeout=3)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


@lru_cache(maxsize=1)
def git_info() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    return {
        "commit": commit,
        "short": commit[:7] if commit else "unknown",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "date": _git("log", "-1", "--format=%cs"),
    }


def environment_info() -> dict[str, str]:
    import pandas as pd

    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "streamlit": st.__version__,
        "pandas": pd.__version__,
        "executable": Path(sys.executable).name,
    }


def text_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_manifest(experiment: dict[str, Any], results: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble a manifest; prompt templates are stored together with their hash."""
    exp = dict(experiment)
    prompts = exp.pop("prompts", None) or {}
    return {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": git_info(),
        "environment": environment_info(),
        "experiment": exp,
        "prompts": {name: {"hash": text_hash(tpl), "template": tpl} for name, tpl in prompts.items()},
        "results": results or {},
    }


def manifest_download(experiment: dict[str, Any], results: dict[str, Any] | None, *, key: str,
                      file_stem: str = "stallm_run") -> None:
    manifest = build_manifest(experiment, results)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    st.download_button(
        "Download run manifest (JSON)",
        data=json.dumps(manifest, indent=2, default=str),
        file_name=f"{file_stem}_{stamp}.json",
        mime="application/json",
        key=key,
        help="Code version, environment, model, prompt hashes, oracle and evaluation settings: "
             "everything needed to re-run or audit this experiment.",
    )
