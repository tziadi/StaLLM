# Structured chance baselines and matching sensitivity (RQ1/RQ2)

1000 draws per null; p = (1 + #draws ≥ model F1) / (1 + draws).

Eligible lines per file (total / code / method body): catalogcontroller.cs 315/231/171, piccontroller.cs 96/81/60, linqselectextensions.cs 50/45/6, catalogservice.cs 188/150/128, internalservererrorobjectresult.cs 14/13/4, catalogcontext.cs 36/31/15, catalogcontextseed.cs 391/340/318, 20170316120022_refactoringeventbusnamespaces.cs 17/13/6, 20170322124244_removeintegrationeventlogs.cs 33/31/23, 20170509130025_addstockproductitem.cs 53/45/38, 20170509130025_addstockproductitem.designer.cs 107/81/67, 20170530133114_addpicturefilename.designer.cs 99/77/63, catalogbrandentitytypeconfiguration.cs 25/21/11, catalogdomainexception.cs 21/15/6, catalogintegrationeventservice.cs 86/74/48, program.cs 124/108/37, startup.cs 370/313/270, catalogscenariobase.cs 93/78/52, catalogscenarios.cs 145/123/102, catalogcontrollertest.cs 132/116/85

Files without a detected method body (method_body falls back to code lines): none

## Lenient matching, three null models

| Protocol | Slot | Prompt | runs | F1 | uniform | p | code lines | p | method body | p |
|---|---|---|---|---|---|---|---|---|---|---|
| numbered | AZ1 | baseline | 3 | 0.217 | 0.179 | 0.017 | 0.17 | 0.003 | 0.181 | 0.021 |
| numbered | AZ1 | hybrid | 3 | 0.156 | 0.158 | 0.5744 | 0.15 | 0.2947 | 0.156 | 0.4835 |
| numbered | AZ1 | scanner | 3 | 0.105 | 0.154 | 1.0 | 0.146 | 1.0 | 0.151 | 1.0 |
| numbered | AZ2 | scanner | 3 | 0.163 | 0.13 | 0.03 | 0.121 | 0.007 | 0.133 | 0.032 |
| numbered | AZ3 | scanner | 3 | 0.224 | 0.168 | 0.001 | 0.161 | 0.002 | 0.172 | 0.002 |
| numbered | AZ4 | scanner | 3 | 0.179 | 0.149 | 0.0609 | 0.143 | 0.021 | 0.155 | 0.0769 |
| numbered | AZ6 | scanner | 3 | 0.202 | 0.137 | 0.001 | 0.132 | 0.001 | 0.142 | 0.001 |
| numbered | AZ7 | scanner | 3 | 0.239 | 0.172 | 0.001 | 0.165 | 0.001 | 0.175 | 0.001 |
| numbered | AZ8 | scanner | 3 | 0.194 | 0.175 | 0.0569 | 0.172 | 0.018 | 0.179 | 0.0869 |
| original | AZ1 | baseline | 3 | 0.138 | 0.155 | 0.8342 | 0.148 | 0.7403 | 0.157 | 0.8731 |
| original | AZ1 | hybrid | 3 | 0.11 | 0.15 | 1.0 | 0.143 | 0.999 | 0.149 | 1.0 |
| original | AZ1 | scanner | 3 | 0.102 | 0.165 | 1.0 | 0.16 | 1.0 | 0.163 | 1.0 |
| original | AZ2 | scanner | 3 | 0.127 | 0.151 | 0.9001 | 0.143 | 0.8312 | 0.161 | 0.97 |
| original | AZ3 | scanner | 3 | 0.233 | 0.174 | 0.001 | 0.166 | 0.001 | 0.18 | 0.001 |
| original | AZ4 | scanner | 3 | 0.105 | 0.116 | 0.7552 | 0.11 | 0.6214 | 0.122 | 0.8681 |
| original | AZ6 | scanner | 3 | 0.047 | 0.147 | 1.0 | 0.14 | 1.0 | 0.15 | 1.0 |
| original | AZ7 | scanner | 3 | 0.167 | 0.16 | 0.3367 | 0.154 | 0.1768 | 0.164 | 0.4156 |
| original | AZ8 | scanner | 3 | 0.152 | 0.179 | 0.988 | 0.174 | 0.969 | 0.184 | 0.993 |

## Balanced and Strict presets, uniform null

| Protocol | Slot | Prompt | preset | F1 | random | p |
|---|---|---|---|---|---|---|
| numbered | AZ1 | baseline | Balanced | 0.214 | 0.145 | 0.001 |
| numbered | AZ1 | baseline | Strict | 0.136 | 0.104 | 0.02 |
| numbered | AZ1 | hybrid | Balanced | 0.141 | 0.137 | 0.3876 |
| numbered | AZ1 | hybrid | Strict | 0.132 | 0.112 | 0.032 |
| numbered | AZ1 | scanner | Balanced | 0.105 | 0.124 | 0.9271 |
| numbered | AZ1 | scanner | Strict | 0.093 | 0.086 | 0.2857 |
| numbered | AZ2 | scanner | Balanced | 0.163 | 0.108 | 0.003 |
| numbered | AZ2 | scanner | Strict | 0.15 | 0.082 | 0.001 |
| numbered | AZ3 | scanner | Balanced | 0.217 | 0.138 | 0.001 |
| numbered | AZ3 | scanner | Strict | 0.202 | 0.101 | 0.001 |
| numbered | AZ4 | scanner | Balanced | 0.179 | 0.115 | 0.001 |
| numbered | AZ4 | scanner | Strict | 0.154 | 0.075 | 0.001 |
| numbered | AZ6 | scanner | Balanced | 0.19 | 0.107 | 0.001 |
| numbered | AZ6 | scanner | Strict | 0.127 | 0.072 | 0.001 |
| numbered | AZ7 | scanner | Balanced | 0.225 | 0.143 | 0.001 |
| numbered | AZ7 | scanner | Strict | 0.215 | 0.107 | 0.001 |
| numbered | AZ8 | scanner | Balanced | 0.173 | 0.156 | 0.0959 |
| numbered | AZ8 | scanner | Strict | 0.16 | 0.132 | 0.005 |
| original | AZ1 | baseline | Balanced | 0.112 | 0.126 | 0.8072 |
| original | AZ1 | baseline | Strict | 0.086 | 0.092 | 0.6454 |
| original | AZ1 | hybrid | Balanced | 0.083 | 0.126 | 1.0 |
| original | AZ1 | hybrid | Strict | 0.058 | 0.095 | 1.0 |
| original | AZ1 | scanner | Balanced | 0.086 | 0.144 | 1.0 |
| original | AZ1 | scanner | Strict | 0.073 | 0.118 | 1.0 |
| original | AZ2 | scanner | Balanced | 0.111 | 0.12 | 0.6843 |
| original | AZ2 | scanner | Strict | 0.086 | 0.083 | 0.3976 |
| original | AZ3 | scanner | Balanced | 0.229 | 0.147 | 0.001 |
| original | AZ3 | scanner | Strict | 0.209 | 0.115 | 0.001 |
| original | AZ4 | scanner | Balanced | 0.09 | 0.093 | 0.5804 |
| original | AZ4 | scanner | Strict | 0.079 | 0.067 | 0.1738 |
| original | AZ6 | scanner | Balanced | 0.047 | 0.115 | 1.0 |
| original | AZ6 | scanner | Strict | 0.009 | 0.076 | 1.0 |
| original | AZ7 | scanner | Balanced | 0.13 | 0.133 | 0.6014 |
| original | AZ7 | scanner | Strict | 0.111 | 0.1 | 0.1868 |
| original | AZ8 | scanner | Balanced | 0.134 | 0.158 | 0.966 |
| original | AZ8 | scanner | Strict | 0.113 | 0.134 | 0.957 |

## Lenient pairs: category of the prediction vs category of the matched finding

- distinct (prediction, finding) pairs within ±3 lines, with a recoverable description: 1752
- descriptions assigned to a category by keywords: 1396 (80%); same category as the finding's rule: 513 (37%), other category: 883 (63%)

| Finding category | pairs | same category |
|---|---|---|
| design | 443 | 63 |
| exception | 327 | 238 |
| testing | 321 | 113 |
| performance | 130 | 14 |
| dead_code | 72 | 38 |
| complexity | 39 | 13 |
| hardcoded | 38 | 24 |
| deprecated | 15 | 10 |
| naming | 11 | 0 |
