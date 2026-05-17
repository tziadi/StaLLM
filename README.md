# StaLLM: Static Analysis meets LLMs

**StaLLM** is a research-grade Proof of Concept (PoC) exploring how **Large Language Models (LLMs)** can complement or challenge **static code analyzers** (e.g., SonarQube, SpotBugs, ESLint) for software maintenance and code quality assessment.

## Key capabilities

- **Provider-agnostic LLM configuration** via `.env` **slots** (Azure OpenAI, OpenAI, or **multiple Ollama deployments**) — reproducible experiments without code changes.
- **LLM comparison mode** — fix one prompt strategy and **compare multiple LLM models side-by-side** (precision/recall/F1, tokens, cost, time).
- **Multi-language dataset** (Java, C#, PHP) with paired static analysis reports for quantitative benchmarking.
- **Unified evaluation pipeline** (file-level precision/recall/F1) + **token and cost accounting**.
- **Streamlit UI** for interactive runs, preflight validation, review-board diagnostics, code evidence inspection, and database-backed history.
- **Demo-first workflow** with bundled datasets, one-click demo configuration, and HTML report export.
- **Batch mode** to run large experiment sweeps across projects/strategies/top-K.

---

## ✨ Objectives

- Compare **LLM-based findings** with **traditional static analyzers**.  
- Benchmark multiple **prompt strategies** and **LLM models** under top-K file selection and large inputs.  
- Support **Java, C#, PHP** projects out-of-the-box.  
- Provide a clean **Streamlit UI** and **SQLite** persistence for runs.  
- Track **LLM usage** (prompt/completion tokens, total tokens, USD cost).  
- Ship a **reproducible dataset** (apps + static analyzer CSVs).
- Make individual results explainable with file-level verdicts, span examples, and side-by-side code context.

---

## 📂 Dataset Provided

Curated dataset under `data/apps/`:

```
data/apps/
  ArgoUML/            # Java
    ArgoUML.zip
    ArgoUML.csv       # SonarQube export
  eShopOnWeb/         # C#
    eShopOnWeb.zip
    eShopOnWeb.csv
  Magento/            # PHP
    Magento.zip
    Magento.csv
```

Each folder contains the **source ZIP** and the **static analyzer CSV** (typically SonarQube). This enables **reproducible** LLM vs. static analysis comparisons and independent evaluation of new approaches.

---

## 🗂 Project Structure

```
StaLLM/
│── StatLLM_app.py          # Streamlit interface (Experiments, DB, Prompts, Batch)
│── StaLLM_core.py          # Core pipeline (LLM calls, metrics, usage/cost aggregation)
│── StaLLM_llm.py           # LLM wrapper + .env slot registry (Azure/OpenAI/Ollama)
│── StaLLM_models.py        # SQLAlchemy models + SQLite persistence
│── strategies.json         # Prompt strategies (editable from the UI)
│── StaLLM.db               # SQLite DB (auto-generated; path configurable)
│── .env                    # LLM slots and optional fallbacks
│
├── data/
│   └── apps/               # Dataset (apps + static analyzer CSVs)
├── output/                 # Batch results (CSV exports)
└── requirements.txt
```

---

## ⚙️ Installation

### 1) Clone & setup
```bash
git clone https://github.com/username/StaLLM.git
cd StaLLM
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2) (Recommended) Configure LLM **slots** in `.env`

Define one or more slots (e.g., `AZ1`, `OA1`, `OL1`) and list them in `STALLM_SLOTS`.  
Slots make experiments **reproducible and switchable** from the UI.

```env
# Which slots are available
STALLM_SLOTS=AZ1,OA1,OL1
STALLM_DEFAULT_SLOT=AZ1

# --- Azure OpenAI slot
AZ1_PROVIDER=azure-openai
AZ1_LABEL=Azure gpt-4o
AZ1_API_BASE=https://<your-azure-resource>.openai.azure.com/
AZ1_API_VERSION=2024-05-01-preview
AZ1_API_KEY=xxxxxxxxxxxxxxxx
AZ1_DEPLOYMENT=gpt-4o-<your-deployment-name>

# --- OpenAI slot
OA1_PROVIDER=openai
OA1_LABEL=OpenAI gpt-4o-mini
OA1_API_KEY=xxxxxxxxxxxxxxxx
OA1_MODEL=gpt-4o-mini

# --- Ollama slot (local)
OL1_PROVIDER=ollama
OL1_LABEL=Ollama llama3 (local)
OL1_HOST=http://localhost:11434
OL1_MODEL=llama3

# --- Ollama slot (remote server)
OL2_PROVIDER=ollama
OL2_LABEL=Ollama phi3 (server1)
OL2_HOST=http://192.168.1.100:11434
OL2_MODEL=phi3
```

> If **no slots** are found, the UI falls back to **manual** provider/model inputs.  
> Legacy env vars like `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_API_VERSION`, `OPENAI_DEPLOYMENT_NAME` are still supported but **slots are preferred**.

#### Pricing (per-1K tokens) + per-1M convenience
StaLLM expects **USD per 1,000 tokens**. You can set prices globally or per slot:
```env
# Global fallback
STALLM_PRICE_IN_PER_1K=0.0025
STALLM_PRICE_OUT_PER_1K=0.0100

# Slot-specific (recommended)
AZ1_PRICE_IN_PER_1K=0.0025
AZ1_PRICE_OUT_PER_1K=0.0100
```
If you only have **per-1M** prices, you can set:
```env
AZ1_PRICE_IN_PER_1M=2.5
AZ1_PRICE_OUT_PER_1M=10
```
They are normalized internally to **$/1K** for cost calculation.

---

## 🚀 Usage

### Launch the app
```bash
streamlit run StaLLM_app.py
```

### Workflow (Run Experiments tab)

**Execution modes:**
1. **Single prompt** — run one prompt strategy on one LLM.  
2. **Compare selected prompts** — compare several prompt strategies on the **same LLM**.  
3. **Compare LLM models** — fix **one prompt strategy** and select **multiple LLMs** to compare side-by-side.

**Fast demo path:**
1. Click **Load demo config** in the sidebar.
2. Keep **Use bundled demo dataset** enabled, or choose another bundled dataset.
3. Confirm the **Preflight checks** are green.
4. Run the analysis and inspect the **Review board** plus **Code evidence viewer**.

**Steps:**
1. Choose **Execution mode**.  
2. Select one **LLM slot** (modes 1–2) **or multiple slots/models** (mode 3).  
3. Upload the **project ZIP** and the matching **CSV** (static analyzer output), or use a bundled demo dataset.  
4. Choose a **prompt strategy** (from `strategies.json`) and **Top-K** files.  
5. Review **Preflight checks**:
   - ZIP validity,
   - CSV readability,
   - detected language,
   - experiment universe,
   - LLM configuration.
6. Run. The UI displays:
   - **Precision / Recall / F1** and **span-level TP/FP/FN**.  
   - A **Review board** with per-file verdicts: matched, partial, missed, extra, mismatch, true negative.
   - A **Code evidence viewer** with side-by-side **Ground Truth focus** and **LLM focus** snippets.
   - **Tokens** (prompt/completion/total) and **estimated cost (USD)**.  
   - For comparison modes, **tables, charts, and review boards per strategy or per model**.
   - An **Export HTML report** button for sharing standalone run/comparison reports.

### Other tabs
- **Stored Results (DB)**: browse historical runs (includes *LLM used*, tokens, cost).  
- **Manage Prompts**: edit/create strategies, persisted to `strategies.json`.  
- **Batch Experiments**: sweep projects/strategies/top-K; CSV exports under `output/apps/<project>/` + DB logging.

---

## 🔬 Metrics (span-level) — TP / FP / FN, Precision, Recall, F1

StaLLM evaluates LLM findings against static-analyzer spans from the CSV, restricted to the sampled universe `U` of Top-K files.

### Definitions
- **Universe U**: sampled files used for a run. It mixes positive files (with GT spans) and negative files (without GT spans) according to Top-K and positive ratio.
- **Ground-truth spans**: static-analyzer findings with file + line/span metadata.
- **LLM spans**: structured findings returned by the LLM with `line` or `startLine` / `endLine`.

For each file in `U`, StaLLM matches LLM spans to GT spans using line overlap / tolerance and optional rule/type matching.

- **TP (True Positive)**: an LLM span matched to a GT span.
- **FP (False Positive)**: an LLM span that did not match a GT span.
- **FN (False Negative)**: a GT span not matched by any LLM span.

### Formulas
Let `TP`, `FP`, `FN` be the counts defined above.

- **Precision** = `TP / (TP + FP)`  
  *Of the spans the LLM reported, how many matched the static analyzer?*

- **Recall** = `TP / (TP + FN)`  
  *Of the GT spans, how many did the LLM catch?*

- **F1** = `2 * Precision * Recall / (Precision + Recall)`  
  *Harmonic mean—penalizes imbalance between P and R.*

The Review board also groups files into qualitative verdicts:

- **Matched**: LLM spans align cleanly with GT spans.
- **Partial / mismatch**: some overlap exists, but FP/FN remain.
- **Missed by LLM**: GT spans exist, but the LLM found none.
- **LLM-only findings**: the LLM reported spans where the sampled GT has none.
- **True negative**: neither GT nor LLM reports findings for the file.

### Edge cases
- If `TP + FP = 0` ⇒ **Precision = 0** (model flagged nothing).  
- If `TP + FN = 0` ⇒ **Recall = 0** (no GT spans in scope).  
- If both P and R are 0 ⇒ **F1 = 0**.

### Qualitative inspection

Each Review board includes:

- per-file TP/FP/FN,
- file precision / recall / F1,
- an automatic summary of dominant error type,
- GT and LLM findings tables,
- side-by-side code evidence:
  - **Ground Truth focus**,
  - **LLM focus**,
  - colored line highlights for GT, LLM, and overlap.

---

## 📊 Results & Persistence

- Per-run details are visible in the UI and saved to **`StaLLM.db`** (SQLite):  
  project, language, strategy, **LLM used**, **TP/FP/FN**, precision, recall, F1, top-K, time, tokens, USD cost, timestamp.  
- The DB tab shows a global **F1 comparison** and groups results by **strategy** and **LLM model**.  
- Batch runs additionally export **CSV** summaries under `output/apps/<project>/`.
- Single runs and comparisons can export standalone **HTML reports** containing configuration, metrics, review boards, and code evidence snippets.

---

## 🧰 Troubleshooting

- **Costs look too high**: set **$/1K** prices (or provide **$/1M**; they're auto-normalized).  
- **Missing credentials**: check `.env` slots (e.g., `AZ1_API_KEY`) and Azure deployment/endpoint.  
- **Azure vs OpenAI**: for Azure, `model` = **deployment name**; for OpenAI, `model` = **model id**.  
- **No slots shown**: ensure `STALLM_SLOTS` is set and each `<SLOT>_PROVIDER` exists.  
- **Language filtering**: CSV paths must match detected language extensions.  
- **Large ZIPs**: raise Streamlit upload limits or split the project.
- **Ollama connection issues**: see [Ollama Multi-Deployment Guide](OLLAMA_MULTI_DEPLOYMENT.md) for detailed troubleshooting.
- **Changing a selected file/prompt seems to reset the UI**: results are cached in the current Streamlit session after a run; rerun the analysis if you changed code or need newly added diagnostics.
- **No code context shown**: rerun the analysis so the latest diagnostics include code excerpts.

---

## 📌 Roadmap

- More languages & projects.  
- Issue-level matching and severity/confidence thresholds.  
- Deeper cost modeling and caching for large batch runs.  
- CLI for reproducible experiment scripts.

---

## ☕ Citation

If you use StaLLM or its dataset, please cite the paper (TBD).  
Feedback and PRs are welcome!
