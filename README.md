# StaLLM: Static Analysis meets LLMs

**StaLLM** is a research-grade Proof of Concept (PoC) exploring how **Large Language Models (LLMs)** can complement or challenge **static code analyzers** (e.g., SonarQube, SpotBugs, ESLint) for software maintenance and code quality assessment.

Key capabilities:

- **Provider‑agnostic LLM configuration** via `.env` **slots** (Azure OpenAI, OpenAI, or local **Ollama**) — reproducible experiments without code changes.
- **Multi-language dataset** (Java, C#, PHP) with paired static analysis reports for quantitative benchmarking.
- **Unified evaluation pipeline** (precision/recall/F1) + **token and cost accounting**.
- **Streamlit UI** for interactive runs, visual comparison, and database-backed history.
- **Batch mode** to run large experiment sweeps across projects/strategies/top‑K.

---

## ✨ Objectives

- Compare **LLM-based findings** with **traditional static analyzers**.  
- Benchmark multiple **prompt strategies** under top‑K file selection and large inputs.  
- Support **Java, C#, PHP** projects out‑of‑the‑box.  
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
│── StaLLM_UI.py            # Streamlit interface (Experiments, DB, Prompts, Batch)
│── StaLLM_core.py          # Core pipeline (LLM calls, metrics, usage/cost aggregation)
│── StaLLM_llm.py           # LLM wrapper + .env slot registry (Azure/OpenAI/Ollama)
│── StaLLM_models.py        # SQLAlchemy models + SQLite persistence
│── strategies.json         # Prompt strategies (editable from the UI)
│── smell_results.db        # SQLite DB (auto-generated)
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
# .venv\Scripts\activate       # Windows
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
AZ1_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
AZ1_DEPLOYMENT=gpt-4o-<your-deployment-name>

# --- OpenAI slot (public OpenAI or proxy via OPENAI_BASE_URL)
OA1_PROVIDER=openai
OA1_LABEL=OpenAI gpt-4o-mini
OA1_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
OA1_MODEL=gpt-4o-mini

# --- Ollama slot (local)
OL1_PROVIDER=ollama
OL1_LABEL=Ollama llama3
OL1_HOST=http://localhost:11434
OL1_MODEL=llama3
```

> 🔎 If **no slots** are found, the UI falls back to **manual** provider/model inputs.  
> Legacy env vars like `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_API_VERSION`, `OPENAI_DEPLOYMENT_NAME` are still supported but **slots are preferred**.

---

## 🚀 Usage

### Launch the app
```bash
streamlit run StaLLM_UI.py
```

### Workflow (Run Experiments tab)
1. Select **Execution mode**: *Single prompt* or *Compare selected prompts*.
2. Pick an **LLM Slot** from your `.env` (or configure manually).
3. Upload the **project ZIP** and the corresponding **CSV** (static analyzer output).
4. Choose a **prompt strategy** (from `strategies.json`) and **Top‑K** files.
5. Run the experiment.  
   - The app shows selected files, **precision/recall/F1**, charts, and **token/cost**.  
   - **All runs are stored** in the SQLite DB with the **model used** and **usage stats**.

### Other tabs
- **Stored Results (DB)**: browse historical runs; includes LLM used, tokens, and cost.  
- **Manage Prompts**: edit/create strategies from the UI; persisted to `strategies.json`.  
- **Batch Experiments**: run sweeps across projects/strategies/top‑K; CSV exports under `output/apps/<project>/` + DB logging.

---

## 🔬 Metrics

- **Precision**: overlap of LLM‑detected items vs static analyzer.  
- **Recall**: how much of the static analyzer’s items are captured by the LLM.  
- **F1**: harmonic mean of precision and recall.  
- **Usage**: `prompt_tokens`, `completion_tokens`, `total_tokens`, and estimated `usd_cost` (per‑provider pricing table in code; treat as indicative).

---

## 📊 Results & Persistence

- Per‑run details are visible in the UI and saved to **`smell_results.db`** (SQLite):  
  project, language, strategy, precision, recall, F1, top‑K, time, **LLM used**, tokens, cost, timestamp.  
- Batch runs additionally export **CSV** summaries under `output/apps/<project>/`.

---

## 🧰 Troubleshooting

- **Missing credentials**: ensure `.env` slots contain the right keys (e.g., `AZ1_API_KEY`) and a valid endpoint/deployment for Azure.  
- **Azure vs OpenAI confusion**: for Azure, `model` = **deployment name**; for OpenAI, `model` = **model id**.  
- **No slots shown**: check `STALLM_SLOTS` and that each `<SLOT>_PROVIDER` is set.  
- **PHP/JS/TS/Python support**: the app filters CSV rows by file extension per detected language; ensure your CSV paths match those extensions.  
- **Large ZIPs**: increase Streamlit upload limits or split the project.  

---

## 📌 Roadmap

- Add more languages and projects to the dataset.  
- Expand metrics (false positives/negatives categorization, redundancy).  
- Deeper cost modeling and caching for large batch runs.  
- Reproducible experiment scripts (CLI) and exportable reports.

---

## ☕ Citation

If you use StaLLM or its dataset, please cite the paper (TBD).  
Feedback and PRs are welcome!
