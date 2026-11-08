# Revision analyses

Computed from stored outputs, no model call.

## 4. MCC and balanced accuracy of the RQ3 configurations (mean over runs)

| Dataset | Slot | Prompt | runs | MCC | balanced accuracy |
|---|---|---|---|---|---|
| DACOS | AZ1 | binary_oracle_baseline | 3 | 0.024 | 0.508 |
| DACOS | AZ1 | conservative_evidence | 3 | 0.074 | 0.522 |
| DACOS | AZ1 | definition_driven | 3 | 0.129 | 0.524 |
| DACOS | AZ2 | binary_oracle_baseline | 1 | 0.092 | 0.548 |
| DACOS | AZ2 | conservative_evidence | 1 | 0.025 | 0.513 |
| DACOS | AZ2 | definition_driven | 1 | 0.122 | 0.553 |
| DACOS | AZ3 | binary_oracle_baseline | 1 | 0.193 | 0.592 |
| DACOS | AZ3 | conservative_evidence | 1 | 0.081 | 0.543 |
| DACOS | AZ3 | definition_driven | 1 | 0.135 | 0.569 |
| DACOS | AZ7 | binary_oracle_baseline | 3 | 0.081 | 0.542 |
| DACOS | AZ7 | conservative_evidence | 3 | 0.09 | 0.546 |
| DACOS | AZ7 | definition_driven | 3 | 0.112 | 0.556 |
| DACOS | AZ8 | binary_oracle_baseline | 3 | 0.086 | 0.531 |
| DACOS | AZ8 | conservative_evidence | 3 | 0.071 | 0.518 |
| DACOS | AZ8 | definition_driven | 3 | 0.018 | 0.507 |
| MLCQ | AZ1 | binary_oracle_baseline | 3 | 0.36 | 0.655 |
| MLCQ | AZ1 | conservative_evidence | 3 | 0.429 | 0.718 |
| MLCQ | AZ1 | definition_driven | 3 | 0.376 | 0.727 |
| MLCQ | AZ2 | binary_oracle_baseline | 1 | 0.191 | 0.554 |
| MLCQ | AZ2 | conservative_evidence | 1 | 0.342 | 0.621 |
| MLCQ | AZ2 | definition_driven | 1 | 0.388 | 0.664 |
| MLCQ | AZ3 | binary_oracle_baseline | 1 | 0.46 | 0.658 |
| MLCQ | AZ3 | conservative_evidence | 1 | 0.565 | 0.735 |
| MLCQ | AZ3 | definition_driven | 1 | 0.402 | 0.667 |
| MLCQ | AZ7 | binary_oracle_baseline | 3 | 0.247 | 0.551 |
| MLCQ | AZ7 | conservative_evidence | 3 | 0.197 | 0.544 |
| MLCQ | AZ7 | definition_driven | 3 | 0.23 | 0.59 |
| MLCQ | AZ8 | binary_oracle_baseline | 3 | 0.476 | 0.702 |
| MLCQ | AZ8 | conservative_evidence | 3 | 0.557 | 0.79 |
| MLCQ | AZ8 | definition_driven | 3 | 0.444 | 0.737 |

## 3. Structural non-LLM span baseline (signatures, throw, catch)

| predictions | P | R | F1 | random | p |
|---|---|---|---|---|---|
| 128 | 0.305 | 0.542 | 0.39 | 0.199 | 0.0001 |

## 2. Protocol effect on the excess over chance (numbered minus original)

Bootstrap over files, 5,000 resamples; expected random counts from 500 placements per file and run.

| Slot | Prompt | excess original | excess numbered | delta | 95% CI | p | Holm |
|---|---|---|---|---|---|---|---|
| AZ1 | baseline | -0.015 | 0.039 | 0.055 | [-0.033, 0.118] | 0.2056 | 1.0 |
| AZ1 | hybrid | -0.038 | -0.001 | 0.038 | [-0.018, 0.08] | 0.1832 | 1.0 |
| AZ1 | scanner | -0.063 | -0.049 | 0.014 | [-0.055, 0.101] | 0.6724 | 1.0 |
| AZ2 | scanner | -0.025 | 0.034 | 0.059 | [-0.045, 0.122] | 0.2368 | 1.0 |
| AZ3 | scanner | 0.06 | 0.057 | -0.002 | [-0.06, 0.038] | 0.918 | 1.0 |
| AZ4 | scanner | -0.01 | 0.03 | 0.039 | [-0.115, 0.138] | 0.6032 | 1.0 |
| AZ6 | scanner | -0.099 | 0.066 | 0.165 | [0.03, 0.282] | 0.0028 | 0.0252 |
| AZ7 | scanner | 0.006 | 0.068 | 0.062 | [-0.04, 0.138] | 0.264 | 1.0 |
| AZ8 | scanner | -0.026 | 0.019 | 0.046 | [0.006, 0.081] | 0.0244 | 0.1952 |

## 1. Random-placement tests, 10,000 draws

| Protocol | Slot | Prompt | F1 | random | p | Holm (protocol) | Holm (all) |
|---|---|---|---|---|---|---|---|
| numbered | AZ1 | baseline | 0.217 | 0.179 | 0.0144 | 0.0864 | 0.2016 |
| numbered | AZ1 | hybrid | 0.156 | 0.157 | 0.5304 | 1.0 | 1.0 |
| numbered | AZ1 | scanner | 0.105 | 0.154 | 0.9999 | 1.0 | 1.0 |
| numbered | AZ2 | scanner | 0.163 | 0.13 | 0.0262 | 0.131 | 0.3406 |
| numbered | AZ3 | scanner | 0.224 | 0.167 | 0.0004 | 0.0028 | 0.006 |
| numbered | AZ4 | scanner | 0.179 | 0.149 | 0.0543 | 0.2172 | 0.6515 |
| numbered | AZ6 | scanner | 0.202 | 0.137 | 0.0001 | 0.0009 | 0.0018 |
| numbered | AZ7 | scanner | 0.239 | 0.172 | 0.0001 | 0.0009 | 0.0018 |
| numbered | AZ8 | scanner | 0.194 | 0.175 | 0.0547 | 0.2172 | 0.6515 |
| original | AZ1 | baseline | 0.138 | 0.154 | 0.8311 | 1.0 | 1.0 |
| original | AZ1 | hybrid | 0.11 | 0.15 | 0.9999 | 1.0 | 1.0 |
| original | AZ1 | scanner | 0.102 | 0.165 | 1.0 | 1.0 | 1.0 |
| original | AZ2 | scanner | 0.127 | 0.151 | 0.9008 | 1.0 | 1.0 |
| original | AZ3 | scanner | 0.233 | 0.173 | 0.0001 | 0.0009 | 0.0018 |
| original | AZ4 | scanner | 0.105 | 0.116 | 0.7642 | 1.0 | 1.0 |
| original | AZ6 | scanner | 0.047 | 0.146 | 1.0 | 1.0 | 1.0 |
| original | AZ7 | scanner | 0.167 | 0.16 | 0.3334 | 1.0 | 1.0 |
| original | AZ8 | scanner | 0.152 | 0.179 | 0.9838 | 1.0 | 1.0 |

