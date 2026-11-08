# Span-level analyses (RQ1/RQ2)

Computed from stored outputs, no model call.

## 1. Random-placement tests with Holm adjustment

p = (1 + #draws whose run-averaged random F1 ≥ the run-averaged F1) / (1 + 1,000 draws).

| Scope | Protocol | Slot | Prompt | F1 | random | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| per protocol | original | AZ1 | baseline | 0.138 | 0.155 | 0.8342 | 1.0 |
| per protocol | original | AZ1 | hybrid | 0.11 | 0.15 | 1.0 | 1.0 |
| per protocol | original | AZ1 | scanner | 0.102 | 0.165 | 1.0 | 1.0 |
| per protocol | original | AZ2 | scanner | 0.127 | 0.151 | 0.9001 | 1.0 |
| per protocol | original | AZ3 | scanner | 0.233 | 0.174 | 0.001 | 0.009 * |
| per protocol | original | AZ4 | scanner | 0.105 | 0.116 | 0.7552 | 1.0 |
| per protocol | original | AZ6 | scanner | 0.047 | 0.147 | 1.0 | 1.0 |
| per protocol | original | AZ7 | scanner | 0.167 | 0.16 | 0.3367 | 1.0 |
| per protocol | original | AZ8 | scanner | 0.152 | 0.179 | 0.988 | 1.0 |
| per protocol | numbered | AZ1 | baseline | 0.217 | 0.179 | 0.017 | 0.102 |
| per protocol | numbered | AZ1 | hybrid | 0.156 | 0.158 | 0.5744 | 1.0 |
| per protocol | numbered | AZ1 | scanner | 0.105 | 0.154 | 1.0 | 1.0 |
| per protocol | numbered | AZ2 | scanner | 0.163 | 0.13 | 0.03 | 0.15 |
| per protocol | numbered | AZ3 | scanner | 0.224 | 0.168 | 0.001 | 0.009 * |
| per protocol | numbered | AZ4 | scanner | 0.179 | 0.149 | 0.0609 | 0.2276 |
| per protocol | numbered | AZ6 | scanner | 0.202 | 0.137 | 0.001 | 0.009 * |
| per protocol | numbered | AZ7 | scanner | 0.239 | 0.172 | 0.001 | 0.009 * |
| per protocol | numbered | AZ8 | scanner | 0.194 | 0.175 | 0.0569 | 0.2276 |
| all 18 | numbered | AZ1 | baseline | 0.217 | 0.179 | 0.017 | 0.238 |
| all 18 | numbered | AZ1 | hybrid | 0.156 | 0.158 | 0.5744 | 1.0 |
| all 18 | numbered | AZ1 | scanner | 0.105 | 0.154 | 1.0 | 1.0 |
| all 18 | numbered | AZ2 | scanner | 0.163 | 0.13 | 0.03 | 0.39 |
| all 18 | numbered | AZ3 | scanner | 0.224 | 0.168 | 0.001 | 0.018 * |
| all 18 | numbered | AZ4 | scanner | 0.179 | 0.149 | 0.0609 | 0.6828 |
| all 18 | numbered | AZ6 | scanner | 0.202 | 0.137 | 0.001 | 0.018 * |
| all 18 | numbered | AZ7 | scanner | 0.239 | 0.172 | 0.001 | 0.018 * |
| all 18 | numbered | AZ8 | scanner | 0.194 | 0.175 | 0.0569 | 0.6828 |
| all 18 | original | AZ1 | baseline | 0.138 | 0.155 | 0.8342 | 1.0 |
| all 18 | original | AZ1 | hybrid | 0.11 | 0.15 | 1.0 | 1.0 |
| all 18 | original | AZ1 | scanner | 0.102 | 0.165 | 1.0 | 1.0 |
| all 18 | original | AZ2 | scanner | 0.127 | 0.151 | 0.9001 | 1.0 |
| all 18 | original | AZ3 | scanner | 0.233 | 0.174 | 0.001 | 0.018 * |
| all 18 | original | AZ4 | scanner | 0.105 | 0.116 | 0.7552 | 1.0 |
| all 18 | original | AZ6 | scanner | 0.047 | 0.147 | 1.0 | 1.0 |
| all 18 | original | AZ7 | scanner | 0.167 | 0.16 | 0.3367 | 1.0 |
| all 18 | original | AZ8 | scanner | 0.152 | 0.179 | 0.988 | 1.0 |

## 2. Borderline configurations with 10,000 draws

| Protocol | Slot | Prompt | Preset | F1 | random | p | Monte Carlo SE |
|---|---|---|---|---|---|---|---|
| numbered | AZ1 | baseline | Lenient | 0.217 | 0.179 | 0.0144 | 0.0012 |
| numbered | AZ1 | hybrid | Lenient | 0.156 | 0.157 | 0.5304 | 0.005 |
| numbered | AZ2 | scanner | Lenient | 0.163 | 0.13 | 0.0262 | 0.0016 |
| numbered | AZ4 | scanner | Lenient | 0.179 | 0.149 | 0.0543 | 0.0023 |
| numbered | AZ8 | scanner | Lenient | 0.194 | 0.175 | 0.0547 | 0.0023 |
| numbered | AZ1 | hybrid | Strict | 0.132 | 0.112 | 0.0279 | 0.0016 |

## 3. Category-aware lenient matching

A prediction matches a finding only if, in addition to the lenient line rule, the keyword category of its description equals the category of the finding's rule. Variant A: unclassified or unrecovered predictions never match. Variant B: they are dropped. Random placement keeps each span's category. 1000 draws.

| Protocol | Slot | Prompt | recovered | classified | F1 A | random A | p A | F1 B | random B | p B |
|---|---|---|---|---|---|---|---|---|---|---|
| numbered | AZ1 | baseline | 1.0 | 0.718 | 0.112 | 0.034 | 0.001 | 0.137 | 0.042 | 0.001 |
| numbered | AZ1 | hybrid | 0.999 | 0.734 | 0.078 | 0.048 | 0.001 | 0.097 | 0.059 | 0.001 |
| numbered | AZ1 | scanner | 0.998 | 0.796 | 0.062 | 0.042 | 0.008 | 0.072 | 0.049 | 0.011 |
| numbered | AZ2 | scanner | 1.0 | 0.785 | 0.068 | 0.03 | 0.001 | 0.079 | 0.034 | 0.001 |
| numbered | AZ3 | scanner | 0.985 | 0.7 | 0.078 | 0.042 | 0.001 | 0.095 | 0.05 | 0.001 |
| numbered | AZ4 | scanner | 0.913 | 0.509 | 0.101 | 0.025 | 0.001 | 0.14 | 0.035 | 0.001 |
| numbered | AZ6 | scanner | 1.0 | 0.369 | 0.009 | 0.003 | 0.0709 | 0.016 | 0.006 | 0.0809 |
| numbered | AZ7 | scanner | 0.997 | 0.663 | 0.101 | 0.033 | 0.001 | 0.133 | 0.045 | 0.001 |
| numbered | AZ8 | scanner | 0.995 | 0.725 | 0.077 | 0.051 | 0.002 | 0.096 | 0.064 | 0.001 |
| original | AZ1 | baseline | 1.0 | 0.742 | 0.033 | 0.043 | 0.8681 | 0.039 | 0.05 | 0.8821 |
| original | AZ1 | hybrid | 1.0 | 0.747 | 0.034 | 0.03 | 0.2737 | 0.042 | 0.037 | 0.2597 |
| original | AZ1 | scanner | 1.0 | 0.817 | 0.044 | 0.047 | 0.6933 | 0.05 | 0.055 | 0.7093 |
| original | AZ2 | scanner | 1.0 | 0.701 | 0.041 | 0.055 | 0.9161 | 0.049 | 0.065 | 0.8901 |
| original | AZ3 | scanner | 0.993 | 0.493 | 0.034 | 0.018 | 0.016 | 0.042 | 0.021 | 0.009 |
| original | AZ4 | scanner | 1.0 | 0.705 | 0.045 | 0.024 | 0.01 | 0.055 | 0.029 | 0.011 |
| original | AZ6 | scanner | 0.914 | 0.562 | 0.028 | 0.042 | 0.956 | 0.04 | 0.059 | 0.956 |
| original | AZ7 | scanner | 0.998 | 0.665 | 0.056 | 0.038 | 0.011 | 0.073 | 0.05 | 0.012 |
| original | AZ8 | scanner | 0.996 | 0.7 | 0.042 | 0.033 | 0.1139 | 0.053 | 0.042 | 0.0829 |

