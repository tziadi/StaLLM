# RQ4 — non-LLM re-rankers and duplicated files

Computed from the repository archives and stored outputs, no model call.

- argouml: 1884 candidate files, of which 0 under target/classes/
- eshop: 523 candidate files, of which 0 under target/classes/

## Re-rankers on the first 600 candidates of the BM25 list (mean over features)

| System | List | Re-ranker | MRR | P@10 |
|---|---|---|---|---|
| argouml | bm25_masked | list_order | 0.583 | 0.325 |
| argouml | bm25_masked | path_bm25 | 0.275 | 0.075 |
| argouml | bm25_sorted | keyword_first | 0.713 | 0.525 |
| argouml | bm25_sorted | list_order | 0.646 | 0.412 |
| argouml | bm25_sorted | path_bm25 | 0.418 | 0.25 |
| argouml | bm25_masked | LLM AZ1 | 0.844 | 0.569 |
| argouml | bm25_masked | LLM AZ2 | 0.927 | 0.625 |
| argouml | bm25_masked | LLM AZ3 | 0.969 | 0.606 |
| argouml | bm25_sorted | LLM AZ1 | 0.896 | 0.662 |
| argouml | bm25_sorted | LLM AZ2 | 0.938 | 0.731 |
| argouml | bm25_sorted | LLM AZ3 | 0.906 | 0.756 |
| eshop | bm25_masked | list_order | 0.867 | 0.54 |
| eshop | bm25_masked | path_bm25 | 0.454 | 0.3 |
| eshop | bm25_sorted | keyword_first | 1.0 | 0.84 |
| eshop | bm25_sorted | list_order | 0.867 | 0.76 |
| eshop | bm25_sorted | path_bm25 | 0.687 | 0.62 |
| eshop | bm25_masked | LLM AZ1 | 0.8 | 0.77 |
| eshop | bm25_masked | LLM AZ2 | 0.82 | 0.82 |
| eshop | bm25_masked | LLM AZ3 | 1.0 | 0.86 |
| eshop | bm25_sorted | LLM AZ1 | 1.0 | 0.88 |
| eshop | bm25_sorted | LLM AZ2 | 1.0 | 0.88 |
| eshop | bm25_sorted | LLM AZ3 | 1.0 | 0.91 |

## Several candidates for the same gold file

| System | List | Feature | gold | candidates matching gold | distinct gold in list | target/classes copies in list |
|---|---|---|---|---|---|---|
| argouml | keyword_sorted | ACTIVITYDIAGRAM | 94 | 22 | 22 | 0 |
| argouml | bm25_sorted | ACTIVITYDIAGRAM | 94 | 67 | 67 | 0 |
| argouml | keyword_sorted | COGNITIVE | 237 | 216 | 216 | 0 |
| argouml | bm25_sorted | COGNITIVE | 237 | 191 | 191 | 0 |
| argouml | keyword_sorted | COLLABORATIONDIAGRAM | 48 | 20 | 20 | 0 |
| argouml | bm25_sorted | COLLABORATIONDIAGRAM | 48 | 33 | 33 | 0 |
| argouml | keyword_sorted | DEPLOYMENTDIAGRAM | 32 | 11 | 11 | 0 |
| argouml | bm25_sorted | DEPLOYMENTDIAGRAM | 32 | 30 | 30 | 0 |
| argouml | keyword_sorted | LOGGING | 214 | 82 | 82 | 0 |
| argouml | bm25_sorted | LOGGING | 214 | 167 | 167 | 0 |
| argouml | keyword_sorted | SEQUENCEDIAGRAM | 79 | 19 | 19 | 0 |
| argouml | bm25_sorted | SEQUENCEDIAGRAM | 79 | 35 | 35 | 0 |
| argouml | keyword_sorted | STATEDIAGRAM | 98 | 30 | 30 | 0 |
| argouml | bm25_sorted | STATEDIAGRAM | 98 | 67 | 67 | 0 |
| argouml | keyword_sorted | USECASEDIAGRAM | 52 | 13 | 13 | 0 |
| argouml | bm25_sorted | USECASEDIAGRAM | 52 | 33 | 33 | 0 |
| eshop | keyword_sorted | BASKET | 63 | 63 | 63 | 0 |
| eshop | bm25_sorted | BASKET | 63 | 63 | 63 | 0 |
| eshop | keyword_sorted | CATALOG | 76 | 76 | 76 | 0 |
| eshop | bm25_sorted | CATALOG | 76 | 76 | 76 | 0 |
| eshop | keyword_sorted | IDENTITY | 56 | 56 | 56 | 0 |
| eshop | bm25_sorted | IDENTITY | 56 | 56 | 56 | 0 |
| eshop | keyword_sorted | ORDER | 195 | 195 | 195 | 0 |
| eshop | bm25_sorted | ORDER | 195 | 195 | 195 | 0 |
| eshop | keyword_sorted | PAYMENT | 9 | 9 | 9 | 0 |
| eshop | bm25_sorted | PAYMENT | 9 | 9 | 9 | 0 |

## Copies in the LLM top ten (ArgoUML, all hosted runs)

| Condition | rankings | mean target/classes files in top 10 | mean duplicated slots in top 10 |
|---|---|---|---|
| rq4 | 79 | 0.0 | 0.0 |
| rq4_sorted | 80 | 0.0 | 0.0 |
| rq4_bm25 | 80 | 0.0 | 0.0 |
| rq4_masked | 80 | 0.0 | 0.0 |
