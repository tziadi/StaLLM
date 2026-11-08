# PMD as a second oracle for RQ3

PMD 7.7.0, default thresholds except ExcessiveParameterList minimum = 10; see the module docstring for the rule mapping.

## DACOS

- samples 200, without a PMD rule 0, parsed as is 200, after repair 0, wrapped in a class 0, wrapped in a method 0, not parsed 0, scored 200
- PMD positives 41, human positives 131; κ(PMD, human) = 0.087, F1 = 0.372
- per smell (n, PMD positive, human positive): {'complex_method': [100, 41, 70], 'long_parameter_list': [100, 0, 61]}
- Spearman ρ between the κ rankings of the 15 configurations (human vs PMD) = 0.436; best vs human: AZ3/binary_oracle_baseline; best vs PMD: AZ3/definition_driven

| Configuration | n | κ human | κ PMD |
|---|---|---|---|
| AZ1/binary_oracle_baseline | 200 | 0.028 | 0.037 |
| AZ1/conservative_evidence | 200 | 0.046 | 0.045 |
| AZ1/definition_driven | 200 | 0.054 | 0.003 |
| AZ2/binary_oracle_baseline | 200 | 0.084 | 0.128 |
| AZ2/conservative_evidence | 200 | 0.025 | 0.099 |
| AZ2/definition_driven | 200 | 0.117 | 0.106 |
| AZ3/binary_oracle_baseline | 200 | 0.148 | 0.209 |
| AZ3/conservative_evidence | 200 | 0.077 | 0.225 |
| AZ3/definition_driven | 200 | 0.117 | 0.315 |
| AZ7/binary_oracle_baseline | 200 | 0.07 | -0.014 |
| AZ7/conservative_evidence | 200 | 0.109 | 0.035 |
| AZ7/definition_driven | 200 | 0.115 | 0.07 |
| AZ8/binary_oracle_baseline | 200 | 0.095 | 0.035 |
| AZ8/conservative_evidence | 200 | 0.041 | 0.022 |
| AZ8/definition_driven | 200 | 0.016 | 0.068 |

## MLCQ

- samples 200, without a PMD rule 54, parsed as is 63, after repair 26, wrapped in a class 5, wrapped in a method 16, not parsed 36, scored 110
- PMD positives 5, human positives 19; κ(PMD, human) = 0.282, F1 = 0.333
- per smell (n, PMD positive, human positive): {'long_method': [27, 1, 2], 'god_class': [41, 2, 9], 'data_class': [42, 2, 8]}
- Spearman ρ between the κ rankings of the 15 configurations (human vs PMD) = -0.524; best vs human: AZ8/conservative_evidence; best vs PMD: AZ7/conservative_evidence

| Configuration | n | κ human | κ PMD |
|---|---|---|---|
| AZ1/binary_oracle_baseline | 110 | 0.355 | 0.335 |
| AZ1/conservative_evidence | 110 | 0.355 | 0.335 |
| AZ1/definition_driven | 110 | 0.389 | 0.216 |
| AZ2/binary_oracle_baseline | 110 | 0.134 | 0.43 |
| AZ2/conservative_evidence | 110 | 0.277 | 0.393 |
| AZ2/definition_driven | 110 | 0.367 | 0.434 |
| AZ3/binary_oracle_baseline | 110 | 0.466 | 0.6 |
| AZ3/conservative_evidence | 110 | 0.538 | 0.492 |
| AZ3/definition_driven | 110 | 0.535 | 0.316 |
| AZ7/binary_oracle_baseline | 110 | 0.141 | 0.482 |
| AZ7/conservative_evidence | 110 | 0.121 | 0.653 |
| AZ7/definition_driven | 110 | 0.22 | 0.335 |
| AZ8/binary_oracle_baseline | 110 | 0.634 | 0.253 |
| AZ8/conservative_evidence | 110 | 0.639 | 0.197 |
| AZ8/definition_driven | 110 | 0.508 | 0.225 |
