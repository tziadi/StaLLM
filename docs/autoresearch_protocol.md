# StarLLM AutoResearch Protocol

This protocol adapts the AutoResearch idea to StarLLM benchmarks. The primary
experience lives inside the StarLLM Streamlit app on the **AutoResearch** page;
the CLI runner exists for repeatable terminal/batch execution of the same
benchmark loop.

## Goal

Automatically improve prompt strategies against a fixed benchmark using a
ratchet loop:

- code smell detection: `f1` by default
- feature location: `mrr` by default

The implementation is intentionally conservative. It can generate and benchmark
prompt mutations, but it does not edit Python source files or overwrite the
prompt repository unless the user explicitly promotes the best prompt.

## Loop

1. Choose a benchmark task and LLM slot.
2. Benchmark the starting prompt to establish the seed score.
3. Generate competing mutations from the best prompt so far.
4. Benchmark each mutation on the same dataset split and settings.
5. Accept only the winning mutation when it improves the primary metric.
6. Repeat for the configured number of attempts.
7. Write `results.csv`, `results.json`, `candidate_prompts.json`, and
   `best_prompt.json` under `output/autoresearch/<run-id>/`.

## Guardrails

- Keep a dev split for accepting candidates.
- Keep a holdout split for final reporting only.
- Start by optimizing prompts, not Python pipeline code.
- Track token usage and elapsed time with every score.
- Do not compare prompts across different model slots unless the model is part
  of the explicit experiment design.

## Example

```bash
./.venv/bin/python StaLLM_autoresearch.py \
  --task feature_location \
  --slot OL5 \
  --max-tasks 3 \
  --top-k 10 \
  --candidate-budget 300
```

For code-smell detection:

```bash
./.venv/bin/python StaLLM_autoresearch.py \
  --task code_smell \
  --slot OL5 \
  --repo-zip data/apps/ArgoUML/ArgoUML.zip \
  --static-csv data/apps/ArgoUML/ArgoUml-sonarqube-quality-analysis.csv \
  --top-k 5
```

Prompt candidate files can be supplied as JSON:

```json
{
  "careful_feature_ranker": {
    "template": "Rank files for {task_name}...\n{candidate_text}",
    "source": "manual"
  }
}
```
