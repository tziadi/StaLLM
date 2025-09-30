# StaLLM: Static Analysis meets LLMs

**StaLLM** is a research-grade Proof of Concept (PoC) exploring how **Large Language Models (LLMs)** can complement or challenge **static code analyzers** (e.g., SonarQube, SpotBugs, ESLint) for software maintenance and code quality assessment.

## Key capabilities

- **Provider-agnostic LLM configuration** via `.env` **slots** (Azure OpenAI, OpenAI, or **multiple Ollama deployments**) — reproducible experiments without code changes.
- **LLM comparison mode** — fix one prompt strategy and **compare multiple LLM models side-by-side** (precision/recall/F1, tokens, cost, time).
- **Multi-language dataset** (Java, C#, PHP) with paired static analysis reports for quantitative benchmarking.
- **Unified evaluation pipeline** (file-level precision/recall/F1) + **token and cost accounting**.
- **Streamlit UI** for interactive runs, visual comparison, and database-backed history.
- **Batch mode** to run large experiment sweeps across projects/strategies/top-K.

---

## ✨ Objectives

- Compare **LLM-based findings** with **traditional static analyzers**.  
- Benchmark multiple **prompt strategies** and **LLM models** under top-K file selection and large inputs.  
- Support **Java, C#, PHP** projects out-of-the-box.  
- Provide a clean **Streamlit UI** and **SQLite** persistence for runs.  
- Track **LLM usage** (prompt/completion tokens, total tokens, USD cost).  
- Ship a **reproducible dataset** (apps + static analyzer CSVs).

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

**Steps:**
1. Choose **Execution mode**.  
2. Select one **LLM slot** (modes 1–2) **or multiple slots/models** (mode 3).  
3. Upload the **project ZIP** and the matching **CSV** (static analyzer output).  
4. Choose a **prompt strategy** (from `strategies.json`) and **Top-K** files.  
5. Run. The UI displays:
   - **Precision / Recall / F1** and a **file-level confusion summary (TP/FP/FN)**.  
   - **Tokens** (prompt/completion/total) and **estimated cost (USD)**.  
   - For comparison modes, **tables & charts per strategy or per model**.

### Other tabs
- **Stored Results (DB)**: browse historical runs (includes *LLM used*, tokens, cost).  
- **Manage Prompts**: edit/create strategies, persisted to `strategies.json`.  
- **Batch Experiments**: sweep projects/strategies/top-K; CSV exports under `output/apps/<project>/` + DB logging.

---

## 🔬 Metrics (file-level) — TP / FP / TN / FN, Precision, Recall, F1

We evaluate at the **file level** against the static analyzer CSV, restricted to the **Top-K ground-truth files** for the selected language.

### Definitions
- **Ground-truth (GT files)**: the Top-K files (from the CSV) with the highest issue counts for the detected language.
- **Predicted files**: files for which the LLM reports **at least one issue**.

We compare **file identities by suffix** (robust to relative paths). For each run:

- **TP (True Positive)**: a GT file that the LLM flagged (≥1 issue reported).
- **FN (False Negative)**: a GT file that the LLM did **not** flag.
- **FP (False Positive)**: a non-GT file that the LLM flagged.
- **TN (True Negative)**: a non-GT file that the LLM did **not** flag.  
  > TN is not used in precision/recall/F1, but it matters for accuracy/specificity if you ever need them.

### Formulas
Let `TP`, `FP`, `FN` be the counts defined above.

- **Precision** = `TP / (TP + FP)`  
  *Of the files the LLM flagged, how many were truly among the Top-K GT files?*

- **Recall** = `TP / (TP + FN)`  
  *Of the Top-K GT files, how many did the LLM catch?*

- **F1** = `2 * Precision * Recall / (Precision + Recall)`  
  *Harmonic mean—penalizes imbalance between P and R.*

> If different strategies have the **same recall**, it means they covered the **same number of GT files**; precision may still differ via different FPs.

### Edge cases
- If `TP + FP = 0` ⇒ **Precision = 0** (model flagged nothing).  
- If `TP + FN = 0` ⇒ **Recall = 0** (no GT files in scope—shouldn’t happen when Top-K > 0).  
- If both P and R are 0 ⇒ **F1 = 0**.

### What’s **not** measured here
- Issue-level matching (per smell type/severity) — possible future extension.  
- Confidence thresholds — you can add a policy like “count file as TP only if ≥ N issues or confidence ≥ τ” to make recall more discriminative.

---

## 📊 Results & Persistence

- Per-run details are visible in the UI and saved to **`StaLLM.db`** (SQLite):  
  project, language, strategy, **LLM used**, **TP/FP/FN**, precision, recall, F1, top-K, time, tokens, USD cost, timestamp.  
- The DB tab shows a global **F1 comparison** and groups results by **strategy** and **LLM model**.  
- Batch runs additionally export **CSV** summaries under `output/apps/<project>/`.

---

## 🧰 Troubleshooting

- **Costs look too high**: set **$/1K** prices (or provide **$/1M**; they're auto-normalized).  
- **Missing credentials**: check `.env` slots (e.g., `AZ1_API_KEY`) and Azure deployment/endpoint.  
- **Azure vs OpenAI**: for Azure, `model` = **deployment name**; for OpenAI, `model` = **model id**.  
- **No slots shown**: ensure `STALLM_SLOTS` is set and each `<SLOT>_PROVIDER` exists.  
- **Language filtering**: CSV paths must match detected language extensions.  
- **Large ZIPs**: raise Streamlit upload limits or split the project.
- **Ollama connection issues**: see [Ollama Multi-Deployment Guide](OLLAMA_MULTI_DEPLOYMENT.md) for detailed troubleshooting.

---

## 📌 Roadmap

- More languages & projects.  
- Issue-level matching and severity/confidence thresholds.  
- Deeper cost modeling and caching for large batch runs.  
- CLI for reproducible experiment scripts and report export.

---

## ☕ Citation

If you use StaLLM or its dataset, please cite the paper (TBD).  
Feedback and PRs are welcome!
