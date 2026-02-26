# FALCON: Factual-Aware Logical Consistency Optimization for LLM Outputs

[![CI](https://github.com/rehanraza786/falcon/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/rehanraza786/falcon/actions/workflows/ci.yaml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="assets/falcon.png" alt="FALCON logo" height="350" width="600"/>
</p>

FALCON is a post-generation filtering framework that improves the **logical consistency** of large language model (LLM) outputs by detecting and resolving internal contradictions between extracted claims. The system formulates contradiction-aware claim selection as an optimization problem and selects a maximally consistent subset of claims.

This project was developed as a **custom final project for Stanford CS224N (Winter 2026)**.

---

## Table of Contents

- [Motivation](#motivation)
- [✨ Key Features](#-key-features)
- [Method Overview](#method-overview)
- [Baselines](#baselines)
- [Datasets](#datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Quick Example](#quick-example)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Single Text Mode](#single-text-mode)
  - [Evaluation Mode](#evaluation-mode)
- [Running Experiments](#running-experiments)
- [🔬 Recent Improvements](#-recent-improvements)
- [Using FALCON as a Library](#using-falcon-as-a-library)
- [Result Interpretation](#result-interpretation)
- [Reproducibility & Determinism](#reproducibility--determinism)
- [❓ FAQ](#-faq)
- [Known Issues & Troubleshooting](#known-issues--troubleshooting)
- [Dependencies](#dependencies)
- [Compute & Performance](#compute--performance)
- [📂 Project Structure](#-project-structure)
- [Advanced Configuration](#advanced-configuration)
- [Limitations & Future Work](#limitations--future-work)
- [Ethical Considerations](#ethical-considerations)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

---

## Motivation

Modern LLMs frequently generate responses that contain **self-contradictory claims**, even when individual statements appear plausible in isolation. These inconsistencies reduce trustworthiness and downstream usability.

Rather than retraining or fine-tuning models, FALCON operates **post hoc**, treating consistency as a *global constraint satisfaction problem* over generated claims.

---

## ✨ Key Features

- **🎯 Post-hoc Filtering** — Works with any LLM without retraining
- **🧮 Optimal Selection** — MILP solver finds globally consistent claim subsets
- **⚡ Fast & Efficient** — <1s solve time per example on CPU
- **🔌 Multi-Provider** — OpenAI, Anthropic, HuggingFace, vLLM support
- **📊 Multiple Modes** — Hard constraints or soft penalties
- **🔍 NLI-Powered** — State-of-the-art contradiction detection via DeBERTa-v3
- **📈 Comprehensive Eval** — Greedy baseline + FALCON on StrategyQA & TruthfulQA
- **🛠️ Flexible Config** — YAML-based configuration for all parameters

---

## Method Overview

Given an LLM-generated response:

1. **Claim Extraction**
   - Decompose output into atomic claims using sentence boundaries and conjunction heuristics.
   - Cap extraction to **≤10 claims** for tractability.

2. **Pairwise Contradiction Scoring**
   - Use a pretrained NLI cross-encoder  
     (`cross-encoder/nli-deberta-v3-base`) to estimate contradiction probabilities.

3. **Claim Weighting**
   - Assign weights using either:
     - Uniform weighting, or
     - Optional LLM-based importance scoring.

4. **Consistency-Constrained Selection**
   - Solve a **Mixed Integer Linear Program (MILP)**:
     - **Hard mode**: disallow contradictory claim pairs above threshold τ.
     - **Soft mode**: allow contradictions with a penalty.

5. **Optional Rewrite**
   - Selected claims may be rewritten into a fluent response using an LLM
     (disabled during core evaluation).

### Mathematical Formulation

**Hard Mode (Binary Constraints):**
```
Maximize:     Σ(w_i * x_i)
Subject to:   x_i + x_j ≤ 1  ∀(i,j) where P_ij > τ
              x_i ∈ {0, 1}
```
Where:
- `x_i` = binary variable (1 if claim i is selected)
- `w_i` = weight/importance of claim i
- `P_ij` = contradiction probability between claims i and j
- `τ` = contradiction threshold (default 0.7)

**Soft Mode (Penalty-Based):**
```
Maximize:     Σ(w_i * x_i) - λ * Σ(P_ij * z_ij)
Subject to:   z_ij ≥ x_i + x_j - 1  (McCormick linearization)
              z_ij ≤ x_i, z_ij ≤ x_j
              x_i, z_ij ∈ {0, 1}
```
Where:
- `z_ij` = binary variable (1 if both claims i and j are selected)
- `λ` = penalty weight for contradictions (default 1.0)
   
---

## Baselines

- **B0 — Raw LLM Output**
- **B1 — Greedy Consistency Filter**
- **B2 — Solver without rewrite (ablation)**

FALCON’s MILP formulation provides a **globally optimal** alternative to greedy filtering.

---

## Datasets

- **StrategyQA** (test split)  
  Binary reasoning questions with explicit factual dependencies.  
  *Primary quantitative benchmark (Exact Match).*

- **TruthfulQA** (validation split, generation setting)  
  Open-ended questions designed to elicit hallucinations.  
  *Used for contradiction-rate reduction and qualitative analysis.*

Dataset access is handled via adapter classes that **standardize split usage and handle known dataset quirks** across HuggingFace versions.

---

## Evaluation Metrics

- **Exact Match (EM)** — StrategyQA
- **Contradiction Count**
  - Before filtering
  - After greedy filtering
  - After MILP optimization
- **Runtime**
  - End-to-end latency
  - Solver-only time
- **Qualitative Analysis** — TruthfulQA

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+** (Python 3.10–3.11 recommended)
- macOS (Apple Silicon supported via MPS), Linux, or Windows
- Optional: CUDA-enabled GPU for faster NLI inference

### Quick Example

```bash
# Install dependencies
pip install -r requirements.txt

# Run on sample text (no LLM needed)
python main.py \
  --mode single \
  --text "Paris is the capital of France. Paris is the capital of Germany." \
  --logic hard

# Output: Filtered claims with contradiction removed
```

### Installation
```bash
# Clone the repository
git clone https://github.com/rehanraza786/falcon.git
cd falcon

# Install dependencies
pip install -r requirements.txt
```

### Configuration

FALCON uses `config.yaml` for all settings. Key configuration options:

**LLM Provider Setup:**
```yaml
llm:
  enabled: true
  provider: openai  # openai | anthropic | hf | vllm_http
```

**Environment Variables (recommended for API keys):**
```bash
# OpenAI
export OPENAI_API_KEY=your_key_here

# Anthropic (Claude)
export ANTHROPIC_API_KEY=your_key_here
```

**Solver Configuration:**
```yaml
solver:
  mode: hard              # hard | soft
  tau: 0.7               # contradiction threshold
  lambda_penalty: 1.0    # soft mode penalty weight
```

⚠️ **Security Note**: Never commit API keys to version control. Use environment variables or a local `.env` file.

---

## Usage

### Single Text Mode

Process a single text input:
```bash
python main.py \
  --mode single \
  --text "Your text here. It might contain contradictory claims." \
  --config config.yaml \
  --logic hard \
  --out result.json
```

**Example:**
```bash
python main.py \
  --mode single \
  --text "The sky is blue. The sky is red." \
  --logic hard
```

### Evaluation Mode

Run evaluation on benchmarks:
```bash
python main.py \
  --mode eval \
  --config config.yaml \
  --logic hard \
  --out outputs/strategyqa_hard.json
```

**Command-line options:**
- `--mode`: `single` or `eval`
- `--text`: Input text (required for single mode)
- `--config`: Path to config file (default: `config.yaml`)
- `--logic`: Override solver mode (`hard` or `soft`)
- `--out`: Output JSON path
- `--seed`: Random seed for reproducibility
- `-v` / `-vv`: Increase verbosity (INFO / DEBUG)
- `--log-file`: Write logs to file

---

## Running Experiments

### Automated Evaluation

Run the full evaluation sweep across all datasets and modes:
```bash
python run_experiments.py
```

**Alternative: Shell script (faster)**
```bash
bash run_ablation.sh
```

**Environment Variables:**
```bash
export ENABLE_LLM=1              # Enable LLM generation (0 or 1)
export LLM_PROVIDER=openai       # openai | anthropic | hf | vllm_http
export OUTPUT_DIR=./outputs      # Output directory
```

Outputs are written to `./outputs/`:
```
strategyqa_hard_results.json
strategyqa_soft_results.json
truthfulqa_hard_results.json
truthfulqa_soft_results.json
summary.csv
```

---

## 🔬 Recent Improvements

FALCON now includes comprehensive improvements addressing evaluation accuracy, contradiction density, parameter sensitivity, and ethical considerations.

### 1. Fixed StrategyQA Evaluation ✅
- **Issue**: 0% EM scores due to verbose LLM outputs
- **Solution**: Smart yes/no extraction from explanatory text
- **Impact**: Accurate binary classification evaluation

### 2. Enhanced TruthfulQA Testing ✅
- **Issue**: Yes/no constraint prevented free-form answers
- **Solution**: Removed constraint + higher temperature (0.9)
- **Impact**: Increased contradiction density, exercises filtering

### 3. Systematic Ablation Studies ✅
- **Tau threshold**: 0.5 → 0.9 sensitivity analysis
- **Claim cap**: 5 → 25 claims scaling study
- **Solver modes**: Hard vs. soft robustness testing
- **Output**: `outputs/ablation/` with scaling reports

### 4. Qualitative Audits ✅
- **Information loss**: Assessment of removed content
- **Overconfidence**: Risk evaluation for filtered outputs
- **Logical validity**: Residual contradiction checks
- **Ethical concerns**: Bias and fairness analysis
- **Output**: `outputs/audit/` with detailed reports

### Running All Improvements

```bash
# Run comprehensive test suite + evaluations + studies
python run_all_improvements.py
```

**Individual components:**
```bash
python test_improvements.py           # Test suite
python run_ablation_study.py          # Ablation studies
python run_qualitative_audit.py       # Qualitative analysis
```
---

### Using FALCON as a Library

Import FALCON in your own Python code:

```python
from falcon_engine import run_single, run_benchmark

# Process a single text
result = run_single(
    cfg_path="config.yaml",
    text="The Earth is flat. The Earth is round."
)
print(result["output"])  # Filtered consistent claims
print(result["stats"])   # Contradiction statistics

# Run benchmark evaluation
results = run_benchmark(cfg_path="config.yaml")
print(results["aggregate"])  # Summary metrics
```

---

## Result Interpretation

### Expected Behavior

- **Zero scores** typically indicate generation was disabled (`llm.enabled=false`).
- Contradiction counts only appear when ≥2 claims are extracted.
- TruthfulQA results should be interpreted **qualitatively**.
- MILP is expected to reduce contradictions more than greedy filtering,
  sometimes at slight cost to EM.

### Performance Metrics Explained

**Exact Match (EM):**
- Proportion of outputs that exactly match gold answers
- Reported for: Raw LLM output, Greedy baseline, FALCON

**Contradiction Reduction:**
- Measures how many contradictory claim pairs remain after filtering
- Lower is better
- FALCON typically achieves 80-100% reduction vs. raw output

**Latency:**
- `avg_latency_s`: Total time per example (claim extraction + NLI + solver)
- `avg_solve_s`: MILP solver time only (typically <0.5s per example)

### Example Output Structure
```json
{
  "dataset": "strategyqa",
  "split": "test",
  "aggregate": {
    "em_raw": 0.42,
    "em_greedy": 0.38,
    "em_falcon": 0.40,
    "avg_contradictions_before": 2.3,
    "avg_contradictions_after_greedy": 0.8,
    "avg_contradictions_after_falcon": 0.1,
    "avg_latency_s": 1.24,
    "avg_solve_s": 0.12
  }
}
```

---

## Reproducibility & Determinism

- No model fine-tuning is performed.
- Results depend on:
  - LLM choice (if enabled)
  - Random seeds in generation
- For deterministic runs:
  - Disable LLM generation
  - Use uniform claim weights

---

## ❓ FAQ

**Q: Do I need an LLM API to use FALCON?**  
A: No! FALCON can work on pre-generated text. Set `llm.enabled: false` to disable generation.

**Q: How is FALCON different from other consistency methods?**  
A: FALCON uses MILP optimization for globally optimal solutions, unlike greedy heuristics. It also works post-hoc without model retraining.

**Q: What's the difference between hard and soft mode?**  
A: Hard mode strictly disallows contradictory pairs. Soft mode allows them with a penalty, providing more flexibility.

**Q: Can I use custom datasets?**  
A: Yes! Use the `JSONLAdapter` in `falcon/adapters/__init__.py` for custom JSONL files.

**Q: Why are my scores zero?**  
A: Check that `llm.enabled: true` and API keys are set. Zero often indicates disabled generation or missing credentials.

**Q: How long does evaluation take?**  
A: For 50 examples: ~1-2 minutes on GPU, ~5-10 minutes on CPU (dominated by NLI inference).

**Q: Can I change the contradiction threshold?**  
A: Yes! Adjust `solver.tau` in config.yaml (default: 0.7). Higher values are stricter.

**Q: Does FALCON modify my LLM?**  
A: No! FALCON only filters outputs post-generation. Your LLM remains unchanged.

---

## Dependencies

Core dependencies from `requirements.txt`:
- `torch` — PyTorch for NLI model
- `transformers` — HuggingFace models
- `datasets==3.6.0` — Dataset loading
- `pulp` — MILP optimization solver
- `pyyaml` — Configuration parsing
- `openai>=1.12.0` — OpenAI API
- `anthropic>=0.25.0` — Anthropic API
- `numpy<2` — Numerical operations
- `tqdm` — Progress bars

Optional:
- CUDA-enabled GPU for faster NLI inference
- Apple MPS for M1/M2 acceleration

---

## Compute & Performance

- ≤10 claims ⇒ ≤45 pairwise constraints
- MILP solves in **<1s per example** on CPU
- End-to-end runtime dominated by NLI scoring

---

## 📂 Project Structure

```
falcon/
├── main.py                    # CLI entry point
├── falcon_engine.py           # Library API
├── run_experiments.py         # Automated evaluation script
├── run_ablation.sh           # Shell script for ablation studies
├── config.yaml               # Main configuration file
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── README.md                 # This file
├── assets/
│   └── falcon.png           # Logo
├── falcon/
│   ├── __init__.py
│   ├── pipeline.py          # Core FALCON pipeline
│   ├── solver.py            # MILP optimization solver
│   ├── models.py            # NLI judge & weighting
│   ├── llm.py               # LLM protocol & scorer
│   ├── rewriter.py          # Claim rewriting logic
│   ├── utils.py             # Utilities (logging, seeding)
│   └── adapters/
│       ├── __init__.py      # Dataset adapters
│       ├── openai_adapter.py
│       ├── anthropic_adapter.py
│       ├── hf_transformers_adapter.py
│       └── vllm_http_adapter.py
└── outputs/
    ├── *_results.json       # Evaluation outputs
    └── summary.csv          # Aggregated results
```

---

## Advanced Configuration

### Claim Extraction Settings
```yaml
claims:
  split_on_conjunctions: true  # Split on "and", "but", etc.
  min_len_chars: 12            # Minimum claim length
  max_claims: 10               # Maximum claims per text
```

### NLI Model Configuration
```yaml
nli:
  model_name: cross-encoder/nli-deberta-v3-base
  device: auto                 # auto | cpu | cuda | mps
  batch_size: 8
  max_length: 192
```

### Rewriting (Optional)
```yaml
rewrite:
  enabled: false               # Enable claim rewriting
  style: concise               # concise | neutral | detailed
  preserve_facts: true         # Don't add new information
  max_output_tokens: 200
```

### Provider-Specific Examples

**OpenAI:**
```yaml
openai:
  model: gpt-4o-mini
  temperature: 0.2
  max_tokens: 256
  request_logprobs: true
```

**Anthropic (Claude):**
```yaml
anthropic:
  model: claude-3-haiku-20240307
  temperature: 0.2
  max_tokens: 256
```

**HuggingFace Transformers:**
```yaml
hf:
  model_id: gpt2
  device: auto
  dtype: auto
  temperature: 0.2
  max_new_tokens: 256
```

**vLLM HTTP Server:**
```yaml
vllm_http:
  base_url: http://localhost:8000/v1
  model: mistral-7b
  temperature: 0.2
  max_tokens: 256
```

---

## Limitations & Future Work

### Current Limitations

1. **Claim Extraction** — Heuristic-based splitting may miss complex claims
2. **Scalability** — Limited to ≤10 claims (45 pairs) for tractability
3. **NLI Accuracy** — Depends on cross-encoder quality; may miss nuanced contradictions
4. **Language Support** — Optimized for English (NLI model limitation)
5. **Context Loss** — Removing claims may lose important context/nuance

### Future Directions

- **Smarter Extraction** — LLM-based claim decomposition
- **Scalability** — Approximate solvers for larger claim sets
- **Multilingual** — Support for non-English languages
- **Uncertainty** — Incorporate confidence scores in claim selection
- **Interactive Mode** — Human-in-the-loop for ambiguous cases
- **Domain Adaptation** — Specialized NLI models for technical domains

---

## Ethical Considerations

FALCON removes content rather than generating new facts. While this may improve internal consistency, it may also remove nuance. The system should be used as a **decision-support tool**, not an authoritative filter.

---

## Citation

If you reference this project:
```bibtex
@misc{azam2026falcon,
  author = {Azam, Rehan},
  title = {FALCON: Factual-Aware Logical Consistency Optimization for LLM Outputs},
  year = {2026},
  publisher = {Stanford CS224N},
  howpublished = {\url{https://github.com/rehanraza786/falcon}}
}
```

---

## Acknowledgments

- **Stanford CS224N (Winter 2026)** — Course framework and guidance
- **HuggingFace** — Transformers library and datasets
- **Cross-Encoder Team** — NLI DeBERTa-v3 model
- **PuLP** — Open-source MILP solver

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## Contact

**Author:** Rehan Azam  
**Project:** Stanford CS224N Final Project (Winter 2026)  
**Repository:** [https://github.com/rehanraza786/falcon](https://github.com/rehanraza786/falcon)

For questions or issues, please open a GitHub issue.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.
