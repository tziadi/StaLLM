# StarLLM Maintenance Localization Extension

This branch extends StarLLM from static-analysis benchmarking toward a common
localization benchmark for software maintenance tasks.

## Target Tasks

| Task | Query | Gold locations | Main metrics |
| --- | --- | --- | --- |
| Static analysis | Source code + smell prompt | Analyzer spans/files | Precision, recall, F1 |
| Feature location | Feature description | Feature-related files/classes/methods | Hit@K, Recall@K, MRR, MAP |
| Bug location | Bug report / issue | Files changed by the fixing commit | Hit@K, Recall@K, MRR, MAP |

## Common Format

`StaLLM_tasks.py` introduces a shared task representation:

```json
{
  "task_id": "LANG-1",
  "task_type": "bug_location",
  "project": "commons-lang",
  "query": "Bug report title and body",
  "repo_zip": "path/to/repo.zip",
  "candidate_level": "file",
  "gold_locations": [
    {"file": "src/main/java/.../Foo.java"}
  ]
}
```

## Recommended Benchmark Roadmap

1. Feature location with ArgoUML SPL benchmark.
2. Bug location with Bench4BL for file-level bug report localization.
3. LLM-native localization with Long Code Arena or LOC-BENCH.
4. Optional fault localization with Defects4J using failing tests and stack traces.

## First Implementation Scope

- Keep the existing static-analysis pipeline stable.
- Add adapters that normalize feature/bug datasets into `MaintenanceTask`.
- Add file-level ranking metrics: `Hit@K`, `Recall@K`, `MRR`, and `MAP`.
- Add task-specific prompts that ask the LLM to return ranked file predictions.

## Adapter Entry Points

`StaLLM_benchmarks.py` provides benchmark-specific loaders:

```python
from StaLLM_benchmarks import load_argouml_feature_tasks, load_bench4bl_bug_tasks

feature_tasks = load_argouml_feature_tasks("argouml_feature_mappings.csv", repo_zip="ArgoUML.zip")
bug_tasks = load_bench4bl_bug_tasks("bench4bl_records.jsonl")
```

The adapters are intentionally permissive about column names so we can support
multiple public exports without changing the core task model.
