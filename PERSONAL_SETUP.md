# Personal Ollama Configuration for Your Models

Based on your available Ollama models, here's a recommended configuration for your `.env` file:

## Your Available Models
- **gpt-oss:20b** (13 GB) - Large model for complex analysis
- **gemma3:1b** (815 MB) - Fast, lightweight model  
- **phi3:mini** (2.2 GB) - Balanced performance
- **deepseek-r1:14b** (9.0 GB) - Reasoning model
- **deepseek-coder-v2:latest** (8.9 GB) - **RECOMMENDED for code analysis**
- **llama2:latest** (3.8 GB) - Classic model
- **mistral:latest** (4.1 GB) - Alternative model

## Recommended .env Configuration

```bash
# === StaLLM Personal Configuration ===
STALLM_SLOTS=OL1,OL2,OL3,OL4,OL5,OL6,OL7
STALLM_DEFAULT_SLOT=OL5

# ----- Ollama (GPT-OSS 20B) - Large model for complex analysis -----
OL1_PROVIDER=ollama
OL1_LABEL=GPT-OSS 20B (Large)
OL1_HOST=http://localhost:11434
OL1_MODEL=gpt-oss:20b
OL1_PRICE_IN_PER_1K=0.0
OL1_PRICE_OUT_PER_1K=0.0

# ----- Ollama (Gemma3 1B) - Fast, lightweight model -----
OL2_PROVIDER=ollama
OL2_LABEL=Gemma3 1B (Fast)
OL2_HOST=http://localhost:11434
OL2_MODEL=gemma3:1b
OL2_PRICE_IN_PER_1K=0.0
OL2_PRICE_OUT_PER_1K=0.0

# ----- Ollama (Phi3 Mini) - Balanced performance -----
OL3_PROVIDER=ollama
OL3_LABEL=Phi3 Mini (Balanced)
OL3_HOST=http://localhost:11434
OL3_MODEL=phi3:mini
OL3_PRICE_IN_PER_1K=0.0
OL3_PRICE_OUT_PER_1K=0.0

# ----- Ollama (DeepSeek R1 14B) - Reasoning model -----
OL4_PROVIDER=ollama
OL4_LABEL=DeepSeek R1 14B (Reasoning)
OL4_HOST=http://localhost:11434
OL4_MODEL=deepseek-r1:14b
OL4_PRICE_IN_PER_1K=0.0
OL4_PRICE_OUT_PER_1K=0.0

# ----- Ollama (DeepSeek Coder V2) - Code specialist (RECOMMENDED) -----
OL5_PROVIDER=ollama
OL5_LABEL=DeepSeek Coder V2 (Code)
OL5_HOST=http://localhost:11434
OL5_MODEL=deepseek-coder-v2:latest
OL5_PRICE_IN_PER_1K=0.0
OL5_PRICE_OUT_PER_1K=0.0

# ----- Ollama (Llama2) - Classic model -----
OL6_PROVIDER=ollama
OL6_LABEL=Llama2 Latest (Classic)
OL6_HOST=http://localhost:11434
OL6_MODEL=llama2:latest
OL6_PRICE_IN_PER_1K=0.0
OL6_PRICE_OUT_PER_1K=0.0

# ----- Ollama (Mistral) - Alternative model -----
OL7_PROVIDER=ollama
OL7_LABEL=Mistral Latest (Alternative)
OL7_HOST=http://localhost:11434
OL7_MODEL=mistral:latest
OL7_PRICE_IN_PER_1K=0.0
OL7_PRICE_OUT_PER_1K=0.0

# ----- Ollama-specific configuration -----
STALLM_MODELS_OLLAMA=gpt-oss:20b,gemma3:1b,phi3:mini,deepseek-r1:14b,deepseek-coder-v2:latest,llama2:latest,mistral:latest
STALLM_OLLAMA_TIMEOUT=15

# ----- Global pricing fallback (for local models) -----
STALLM_PRICE_IN_PER_1K=0.0
STALLM_PRICE_OUT_PER_1K=0.0
```

## Model Recommendations for Static Analysis

### 🏆 **Best for Code Analysis: DeepSeek Coder V2**
- **Why**: Specifically trained for code understanding and analysis
- **Size**: 8.9 GB (good balance of capability and resource usage)
- **Use case**: Primary choice for static analysis experiments

### 🚀 **Fastest: Gemma3 1B**
- **Why**: Lightweight and fast for quick iterations
- **Size**: 815 MB (very fast loading)
- **Use case**: Rapid prototyping and testing

### 🧠 **Most Capable: GPT-OSS 20B**
- **Why**: Largest model with most reasoning capability
- **Size**: 13 GB (requires most resources)
- **Use case**: Complex analysis where accuracy is critical

### ⚖️ **Balanced: Phi3 Mini**
- **Why**: Good balance of speed and capability
- **Size**: 2.2 GB (reasonable resource usage)
- **Use case**: General-purpose analysis

## Usage Tips

1. **Start with DeepSeek Coder V2** - It's specifically designed for code analysis
2. **Use Gemma3 1B for quick tests** - Fastest to load and run
3. **Compare multiple models** - Use the "Compare LLM models" mode to see which works best for your specific codebases
4. **Monitor resource usage** - Larger models (GPT-OSS 20B, DeepSeek R1 14B) will use more RAM/VRAM

## Quick Start

1. Copy the configuration above to your `.env` file
2. Run `./.venv/bin/streamlit run StaLLM_app.py`
3. Select "DeepSeek Coder V2 (Code)" from the LLM dropdown
4. Upload your project ZIP and static analysis CSV
5. Run your first experiment!

## Model Comparison Strategy

To find the best model for your specific use case:

1. **Single Prompt Mode**: Test each model individually
2. **Compare LLM Models Mode**: Run the same prompt strategy across multiple models
3. **Focus on**: Precision, Recall, F1 scores, and response time
4. **Consider**: Resource usage vs. performance trade-offs

Your setup is excellent for comprehensive static analysis research with multiple model comparisons!
