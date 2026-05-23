# DACOS / DACOSX Integration

StarLLM can use DACOS/DACOSX as a human-oracle benchmark for code-smell
detection. This complements the existing SonarQube CSV workflow, which measures
agreement with an analyzer rather than agreement with human annotations.

## Dataset

DACOS is distributed on Zenodo:

https://zenodo.org/records/7570428

The archive contains:

- `DACOSMain.sql`
- `DACOSExtended.sql`
- `files.zip`

DACOS smell ids:

| Smell id | Meaning |
| --- | --- |
| 1 | Multifaceted Abstraction present |
| 4 | Multifaceted Abstraction absent |
| 2 | Long Parameter List present |
| 5 | Long Parameter List absent |
| 3 | Complex Method present |
| 6 | Complex Method absent |

## Recommended Layout

```text
data/apps/DACOS/
  DACOSMain.sql
  DACOSExtended.sql
  files/
```

Extract `files.zip` into the `files/` folder.

## Supported Inputs

The adapter accepts:

- CSV/TSV/JSON exports with columns such as `code`, `snippet`, `smell`,
  `smell_id`, `label`, `file`, or `file_id`;
- the public SQL dumps using a permissive best-effort parser.

For the most reliable experiments, import the SQL dump into MySQL and export a
flat CSV with:

```text
sample_id,smell_id,code,label
```

or:

```text
sample_id,smell,label,code
```

## UI

Open:

```text
Maintenance Tasks -> Code smell detection -> Human smell oracle (DACOS/DACOSX)
```

StarLLM will ask the LLM whether the target smell is present for each sample and
compare the answer against the human label.
