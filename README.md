[![CI](https://github.com/rehanraza786/falcon/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/rehanraza786/falcon/actions/workflows/ci.yaml)

# FALCON: Factual Aware Logical COnsistency Network 

<p align="center">
  <img align="center" src="assets/falcon.png" alt="FALCON logo" height="300" width="600"/>
</p>

FALCON is a research-oriented, neuro-symbolic framework for reducing contradictions and hallucinations in Large Language Model (LLM) outputs.  
It operates at the *claim level*, using Natural Language Inference (NLI) and optimization-based reasoning to retain a maximally consistent subset of generated claims.

FALCON focuses on hallucination reduction and logical consistency, not preference or value alignment.

The system is model-agnostic, interpretable, and designed for rigorous evaluation on benchmark datasets.

## Key Ideas
- **Claim decomposition**: Breaks free-form text into atomic factual claims  
- **Contradiction detection**: Uses an NLI model to estimate pairwise contradictions  
- **Optimization-based filtering**: Selects a consistent subset of claims via MaxSAT-style reasoning implemented as a MILP  
- **Optional rewriting**: Rewrites filtered claims into fluent text using an LLM  
- **Reproducible evaluation**: Determni

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have **Python 3.9+** installed.
This project is optimized for:
- **Apple Silicon (M1/M2/M3)** via Metal Performance Shaders (MPS).
- **NVIDIA GPUs** via CUDA.
- **CPUs** (fallback).

### 2. Installation
Install the required dependencies. We use `sentencepiece` for the DeBERTa tokenizer and `accelerate` for hardware optimization.

```bash
  pip install -r requirements.txt
```
### Supported Datasets

TruthfulQA
- Hugging Face ID: truthful_qa
- Configuration: generation
- Split: validation

StrategyQA
- Hugging Face ID: wics/strategy-qa
- Split: test (only)

### Configuration

All runtime behavior is controlled through config.yaml.

Key sections:
- nli
- solver
- claims
- llm
- rewrite
- eval

### 3. Running the Baseline (No API Key Required)
You can run the full pipeline using the static datasets (TruthfulQA/StrategyQA) without any API keys. This uses a local NLI model to check for contradictions.

```
python run_experiments.py
```

This will:
#### 1. Load TruthfulQA and StrategyQA using script-free, secure dataset loading.
#### 2. Run both Hard Logic (MaxSAT-style reasoning via MILP) and Soft Logic (penalty-based relaxation using McCormick linearization).
#### 3. Save detailed JSON results to `outputs/`.
#### 4. Generate a `summary.csv` with the final metrics.

## 🧪 Experiments & Configuration
The project is controlled by `run_experiments.py`, which manages the configuration and process execution.

### Logic Modes
- **Hard Logic (`hard`)**: Enforces strict consistency. If Claim A conflicts with Claim B ($P_{conflict} > \tau$), one MUST be removed.
- **Soft Logic (`soft`)**: Uses a penalty function. Allows minor contradictions if the claim confidence is high enough.

### StrategyQA Notes
`wics/strategy-qa` exposes **only the `test` split**.
FALCON automatically loads this split, but configs should explicitly set:

```yaml
eval:
  dataset: strategyqa
  split: test
```

### Running

Single mode:
```
python main.py --mode single --text "Your input text"
```

Evaluation mode:
```
python main.py --mode eval --config config.yaml --logic hard --out outputs/results.json
```

### Using LLMs (Optional)
LLM usage is controlled via `config.yaml`. Environment variables are used **only for API keys**.

Example (`config.yaml`):
```yaml
llm:
  enabled: true
  provider: openai
  model: gpt-4o
```

### API Keys

Environment variables are used **only** for authentication.

OpenAI:
```
export OPENAI_API_KEY="sk-..."
```
Anthropic:
```
export ANTHROPIC_API_KEY="sk-ant-..."
```

LLM usage itself is enabled via `config.yaml`.

### For Local Hugging Face Models:

To use a local Hugging Face model, configure `config.yaml`:

```yaml
llm:
  enabled: true
  provider: hf
  model_id: <model-name>
```

Then run:
```
python run_experiments.py
```

## 📂 Project Structure
```
FALCON-Project/
├── main.py                             # Worker script (Processes text & runs evaluation)
├── run_experiments.py                  # Manager script (Orchestrates runs & configuration
├── config.yaml                         # Default configuration (NLI, Solver, & LLM settings)
├── requirements.txt                    # Project dependencies
├── falcon/
│   ├── llm.py                          # LLM Interface (OpenAI/Anthropic/HF/vLLM)
│   ├── models.py                       # NLI Judge (DeBERTa-v3) & Device handling
│   ├── pipeline.py                     # Core Logic: Extract -> Solve -> Rewrite
│   ├── rewriter.py                     # Reconstructs text from selected claims
│   ├── solver.py                       # PuLP Optimization (Hard/Soft logic)
│   └── adapters/                       # Dataset loaders (StrategyQA, TruthfulQA)
│       ├── anthropic_adapter.py
│       ├── hf_transformers_adapter.py
│       ├── openai_adapter.py
│       └── vllm_http_adapter.py
└── outputs/                            # Results JSONs and Summary CSV
```

## 🛠 Troubleshooting
### 1. `RuntimeError: Dataset scripts are no longer supported`
- **Status: Fixed.**
- **Detail**: FALCON uses **script-free dataset loading only**. No dataset relies on `trust_remote_code`, and `HF_DATASETS_TRUST_REMOTE_CODE` is **not required**. The project is compatible with `datasets >= 4.x`.

### 2. `ValueError: tokenizer wrapper / KeyError on load`
- **Status: Fixed.**
- **Detail**: This occurs if `sentencepiece` is missing. Ensure you ran `pip install -r requirements.txt` (which now includes it).

### 3. `RuntimeError: "auto" device not found`
- **Status: Fixed.**
- **Detail**: models.py now includes robust detection for:
  - **Mac**: `mps` (Metal)
  - **Nvidia**: `cuda`
  - **CPU**: `cpu` (Fallback)

### 4. `FileNotFoundError: outputs/temp_...yaml`
- **Status: Fixed.**
- **Detail**: The runner uses absolute paths. Ensure you run `python run_experiments.py` from the root project folder.

## 📊 Results Analysis
After running, check `outputs/summary.csv`. Evaluation produces JSON files with raw vs filtered outputs, metrics, and timing statistics.

Key metrics for analysis:
- `em_falcon`: Exact Match score (accuracy) against gold answers.
- `contra_after`: Number of contradictions remaining (Target: 0).
- `solve_s`: Average time taken by the constraint solver.
- `rewrite_rate`: How often the system intervened to fix a contradiction.

## License

Research and evaluation use only.