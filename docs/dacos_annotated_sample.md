# DACOS sample labeled by the annotations

`load_dacos_smell_samples(..., label_source="annotations", split=...)` and
`load_dacos_annotated_splits(...)` in `StaLLM_benchmarks.py` build the DACOS
samples from the developers' judgements. The selected ids are in
`data/apps/DACOS/stallm_annotated_splits.json`.

## What the DACOS dump contains

| Table / column | Meaning |
|---|---|
| `sample.smells` | Smell(s) for which the snippet falls in the "potentially subjective" metric range and was therefore sent to annotators (1 complex method, 2 long parameter list, 3 multifaceted abstraction). Empty for the other 94,199 snippets. |
| `sample.has_smell` | 1 exactly when `smells` is not empty. |
| `sample.sample_constraints` | Number of metric constraints the snippet meets. 0 for nearly all snippets without `smells`. |
| `annotation` | One row per annotator and snippet: `iscm`, `isim`, `islp`, `isma`, `is_smell`, `sample_id`. `isim` is never set. |
| DACOSX `entry` | Snippets classified by the metric thresholds alone (`smellPresented` uses the ids 1–6 for present/absent). No human judgement. |

DACOSMain has 99,387 snippets, 10,267 annotations on 5,115 snippets
(5,106 of them with two annotators), from 10 projects.

The default loader (`label_source="sample_table"`, used by the runs of the
IST manuscript) reads `smells` as the label. A "positive" therefore means
"selected for annotation of that smell" and a "negative" means a snippet that
was never selected and was never annotated.

## Agreement between `smells` and the annotators

Strict majority of the annotators for the smell, on annotated snippets:

| Smell | Tagged, majority yes | Tagged, majority no | Tagged, tie | Not tagged, majority yes |
|---|---|---|---|---|
| Complex method | 1,301 | 567 | 407 | 0 |
| Long parameter list | 731 | 460 | 366 | 205 |
| Multifaceted abstraction | 1,138 | 12 | 191 | 0 |

## Rule of the annotated sample

1. A (snippet, smell) pair is eligible when the snippet was selected for that
   smell (`smells`) and has at least one annotation.
2. Label: strict majority of the annotators' flag for that smell. Ties (in
   practice, two annotators who disagree) are excluded.
3. Each snippet and each source file is used once, for one smell and in one
   split.
4. Each split of 200 has 67 complex method, 67 long parameter list, and 66
   multifaceted abstraction pairs. The positive share of each smell follows its
   share among eligible pairs. Within each (smell, label) stratum, pairs are
   shuffled with seed 42 and drawn round-robin over the 10 projects.
5. Snippets whose source is missing or shorter than 80 characters are skipped.

## Resulting splits

| Split | Complex method (+/−) | Long parameter list (+/−) | Multifaceted abstraction (+/−) | Positives | Projects |
|---|---|---|---|---|---|
| selection | 47 / 20 | 41 / 26 | 65 / 1 | 153 / 200 | 10 |
| heldout | 47 / 20 | 41 / 26 | 65 / 1 | 153 / 200 | 10 |

The two splits share no snippet and no source file. Every pair has two
annotators, except one with five.

Multifaceted abstraction has only 12 negative pairs in the whole dataset, so
it is almost constant in this sample. Positives are the majority class (77%),
whereas they were 13% in the earlier sample.
