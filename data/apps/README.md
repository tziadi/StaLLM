# StaLLM Dataset (`data/apps/`)

This folder holds test projects and their paired static analysis reports.  
It is used by the **Streamlit UI** (tabs *Run Experiments* and *Batch Experiments*).

## 📁 Expected Layout

```
data/apps/
  ArgoUML/
    ArgoUML.zip           # project source (repo root)
    ArgoUML.csv           # static analysis export (e.g., SonarQube)
  eShopOnWeb/
    eShopOnWeb.zip        # C#
    eShopOnWeb.csv
  Magento/
    Magento.zip           # PHP
    Magento.csv
```

- **One folder per project** (stable, readable name).
- At least **one `.zip` + one `.csv`** per folder. Multiple `.zip` files are allowed (batch); the single `.csv` is the **ground truth**.
- The **`.zip` must contain the repo files at the root** (avoid an extra top-level directory inside the zip).

## 🔤 Language Detection & CSV Filtering

StaLLM detects the dominant language of the project by counting file extensions in the `.zip`, then **filters the CSV** to keep only files of that language.

Supported extensions (by default):
- **C#**: `.cs`
- **Java**: `.java`
- **PHP**: `.php`
- **Python**: `.py`
- **JavaScript**: `.js`, `.jsx`
- **TypeScript**: `.ts`, `.tsx`
- **Go**: `.go`
- **C/C++**: `.c`, `.h`, `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hh`, `.hxx`

> You can extend/adjust this list in `StaLLM_core.py` (`LANGUAGE_EXTS`).

## 🧾 CSV Structure (Minimum Requirements)

- The loader **auto-detects** the delimiter (`,` or `;`).  
- **One of these columns must exist** (case-insensitive): `file`, `component`, or `path`.  
  This column must hold a **file path** (relative or absolute) that matches the source file in the zip.

Conventions:
- **1 row = 1 issue** reported by the static analyzer.  
- If a file has multiple issues, it appears multiple times → contributes to **Top-K** via frequency.  
- Paths can be relative (recommended). Use forward slashes `/` (Windows `\` tolerated).  
- Matching between `.zip` and `.csv` is done via **path suffix** (`endswith`). Try to avoid extra leading directories in the zip.

### Valid CSV Examples

**SonarQube-like CSV (`;` separated):**
```csv
Component;Type;Severity;Message;Rule
src/main/java/org/argouml/ui/Main.java;CODE_SMELL;MAJOR;"Avoid static accessors";java:S1234
src/main/java/org/argouml/core/Model.java;BUG;CRITICAL;"Null dereference";java:S2259
```
Here, `Component` serves as the required file column.

**Minimal CSV (`,` separated):**
```csv
file,rule,severity,message
app/code/Magento/Catalog/Model/Product.php,php:S100,MAJOR,"Magic number"
app/code/Magento/Checkout/Controller/Index.php,php:S125,MINOR,"Remove commented code"
```
Here, `file` serves as the required file column.

## 🎯 How Top-K Is Computed

1. The CSV is aggregated by **file** (`value_counts`).  
2. The **Top-K** files with the highest number of issues are selected.  
3. The `.zip` is scanned, and only those files (paths ending with the Top-K filenames) are sent to the LLM.

## ✅ Best Practices

- Export issues with **stable, relative paths** from the repo root.  
- Ensure file **extensions in the CSV** match the language primarily detected from the zip.  
- Avoid zips that introduce an additional root directory (e.g., `Project/Project/...`).  
- Keep the CSV as **one row per issue** (no manual dedup needed — frequency is part of the pipeline).

## 🛠️ Troubleshooting

- **“No file column found”** → ensure the CSV includes `file` or `component` or `path`.  
- **0 LLM results** →
  - detected language doesn’t match extensions present in the CSV; or
  - CSV paths don’t match the zip layout (extra root dir); or
  - Top-K too small and/or CSV nearly empty.  
- **PHP/JS/TS/Python projects** → verify `LANGUAGE_EXTS` covers your extensions.

## 🧪 Complete Example (Magento, PHP)

```
data/apps/Magento/
  Magento.zip
  Magento.csv
```

**Magento.csv:**
```csv
file,rule,severity,message
app/code/Magento/Catalog/Model/Product.php,php:S100,MAJOR,"Magic number"
app/code/Magento/Checkout/Controller/Index.php,php:S125,MINOR,"Remove commented code"
app/code/Magento/Catalog/Model/Product.php,php:S3776,MAJOR,"Cognitive complexity too high"
```
Top-K will count two occurrences for `Product.php` and one for `Index.php`.

---

For exact loading behavior, see `StaLLM_core.py` (`load_ground_truth`, `detect_language_from_zip`) and UI logs.
