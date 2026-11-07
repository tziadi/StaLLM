# RQ4 — non-LLM re-rankers and duplicated files

Computed from the repository archives and stored outputs, no model call.

- argouml: 3753 candidate files, of which 1675 under target/classes/
- eshop: 523 candidate files, of which 0 under target/classes/

## Re-rankers on the first 600 candidates of the BM25 list (mean over features)

| System | List | Re-ranker | MRR | P@10 |
|---|---|---|---|---|
| argouml | bm25_masked | list_order | 0.473 | 0.163 |
| argouml | bm25_masked | path_bm25 | 0.201 | 0.075 |
| argouml | bm25_sorted | keyword_first | 0.675 | 0.275 |
| argouml | bm25_sorted | list_order | 0.592 | 0.237 |
| argouml | bm25_sorted | path_bm25 | 0.5 | 0.175 |
| argouml | bm25_masked | LLM AZ1 | 0.797 | 0.531 |
| argouml | bm25_masked | LLM AZ2 | 0.938 | 0.637 |
| argouml | bm25_masked | LLM AZ3 | 0.958 | 0.637 |
| argouml | bm25_sorted | LLM AZ1 | 0.969 | 0.787 |
| argouml | bm25_sorted | LLM AZ2 | 0.969 | 0.819 |
| argouml | bm25_sorted | LLM AZ3 | 1.0 | 0.794 |
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
| argouml | keyword_sorted | ACTIVITYDIAGRAM | 94 | 30 | 22 | 15 |
| argouml | bm25_sorted | ACTIVITYDIAGRAM | 94 | 103 | 53 | 256 |
| argouml | keyword_sorted | COGNITIVE | 237 | 376 | 191 | 199 |
| argouml | bm25_sorted | COGNITIVE | 237 | 286 | 143 | 273 |
| argouml | keyword_sorted | COLLABORATIONDIAGRAM | 48 | 26 | 20 | 7 |
| argouml | bm25_sorted | COLLABORATIONDIAGRAM | 48 | 48 | 24 | 256 |
| argouml | keyword_sorted | DEPLOYMENTDIAGRAM | 32 | 18 | 11 | 13 |
| argouml | bm25_sorted | DEPLOYMENTDIAGRAM | 32 | 54 | 27 | 269 |
| argouml | keyword_sorted | LOGGING | 214 | 83 | 82 | 2 |
| argouml | bm25_sorted | LOGGING | 214 | 244 | 122 | 286 |
| argouml | keyword_sorted | SEQUENCEDIAGRAM | 79 | 28 | 19 | 11 |
| argouml | bm25_sorted | SEQUENCEDIAGRAM | 79 | 66 | 33 | 265 |
| argouml | keyword_sorted | STATEDIAGRAM | 98 | 38 | 30 | 14 |
| argouml | bm25_sorted | STATEDIAGRAM | 98 | 80 | 40 | 271 |
| argouml | keyword_sorted | USECASEDIAGRAM | 52 | 20 | 13 | 8 |
| argouml | bm25_sorted | USECASEDIAGRAM | 52 | 48 | 24 | 238 |
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
| rq4 | 97 | 0.0 | 0.0 |
| rq4_sorted | 98 | 0.02 | 0.02 |
| rq4_bm25 | 80 | 0.0 | 0.0 |
| rq4_masked | 80 | 0.06 | 0.07 |
