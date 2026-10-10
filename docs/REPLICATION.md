# StaLLM journal paper — replication package

This package regenerates every table and figure of the IST journal paper
*StaLLM: A Benchmark with Built-in Controls for Evaluating Large Language
Models on Software Maintenance Tasks*, and re-runs the experiments if model access is
available.

## Contents

| Path | What it is |
|---|---|
| `StaLLM_*.py` | The framework (LLM slots, span pipeline, adapters, localization tasks) |
| `run_paper_experiments.py` | Runs RQ1–RQ5 and the non-LLM baselines; writes `output/paper_results/` |
| `analyze_paper_stats.py` | Significance tests and stability measures; writes `output/paper_results/stats/` |
| `generate_paper_figures.py` | Paper figures from the summaries |
| `output/paper_results/` | Raw per-sample / per-feature model outputs (JSON) and summary CSVs used in the paper |
| `data/apps/` | ArgoUML (code + SPL feature oracle), eShopOnContainers (code, SonarQube report, feature oracle) |
| `strategies.json`, `prompt_repository.json` | Span-level prompts (Baseline, Scanner, Hybrid) |

DACOS and MLCQ are not redistributed; see *Datasets* below.

## Reproducing the paper without calling any model

All model outputs are included, so the statistics and figures can be
regenerated offline:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt scipy
.venv/bin/python run_paper_experiments.py --rq summary   # summary CSVs from raw JSON
.venv/bin/python analyze_paper_stats.py                  # stats/report.md + CSVs
.venv/bin/python generate_paper_figures.py --out-dir figures/
```

`output/paper_results/stats/report.md` contains every number reported in
Section 9.

## Re-running the experiments

Configure model slots in `.env` (see `env.example`). The paper uses:

| Slot | Model | Provider |
|---|---|---|
| AZ1 | gpt-4o (2024-11-20) | Azure OpenAI |
| AZ2 | gpt-4.1 | Azure OpenAI |
| AZ3 | o3 (2025-04-16) | Azure OpenAI |
| AZ4 | qwen2.5-coder:14b-instruct, `AZ4_NUM_CTX=16384` | Ollama |
| AZ6 | deepseek-coder:6.7b-instruct, `AZ6_NUM_CTX=16384` | Ollama (RQ2 only) |

Commands (existing result files are skipped, so runs can be resumed):

```bash
# RQ1 (prompts on AZ1) and RQ2 (Scanner across models), span level, eShop Catalog
python run_paper_experiments.py --rq 2 --slots AZ1 --runs 3 --prompts baseline,scanner,hybrid
python run_paper_experiments.py --rq 2 --slots AZ2,AZ3 --runs 3
python run_paper_experiments.py --rq 2 --slots AZ4,AZ6

# RQ3 — human-oracle smell classification, and non-LLM baselines
python run_paper_experiments.py --rq 3 --slots AZ1 --runs 3
python run_paper_experiments.py --rq 3 --slots AZ2,AZ3
python run_paper_experiments.py --rq 3 --slots AZ4 --prompts binary_oracle_baseline
python run_paper_experiments.py --rq 3b

# RQ4 — feature location on both systems (all orderings, incl. name-masked)
python run_paper_experiments.py --rq 4 --slots AZ1 --runs 3
python run_paper_experiments.py --rq 4 --slots AZ2,AZ3,AZ4
python run_paper_experiments.py --rq 4 --system eshop --slots AZ1 --runs 3
python run_paper_experiments.py --rq 4 --system eshop --slots AZ2,AZ3

# RQ5 — prompt search (selection split) and held-out evaluation
python run_paper_experiments.py --rq 5
python run_paper_experiments.py --rq 5h --slots AZ1
```

Span-level runs use `STALLM_HEURISTIC_FALLBACK=0` (the default): files for
which the model returns no finding contribute no prediction, and each result
records per-file provenance in `metrics.file_sources`.

## Datasets

- **ArgoUML SPL feature location** — included under `data/apps/Feature Location-ArgoUML/`.
- **eShopOnContainers** — code, SonarQube report and feature-location oracle
  included under `data/apps/eShopOnWeb/` and `data/apps/eShopFL/`. The feature
  descriptions used as queries are in `ESHOP_FEATURE_DESCRIPTIONS`
  (`run_paper_experiments.py`).
- **DACOS / DACOSX** — download the SQL dumps and `files.zip` from the DACOS
  release and place them under `data/apps/DACOS/` (`DACOSMain.sql`,
  `DACOSExtended.sql`, `files.zip`).
- **MLCQ** — place `MLCQCodeSmellSamples.csv` under `data/apps/MLCQ/`; snippets
  are fetched from GitHub at the recorded commits and cached locally.

The DACOS selection split is the first 200 samples produced by the loader; the
held-out test split is the next 200 samples whose DACOS sample ids do not
occur in the selection split (`load_dacos_heldout`).
