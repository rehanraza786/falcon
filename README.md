# FALCON: Factual-Aware Logical Consistency Optimization for LLM Outputs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

<p align="center">
  <img src="assets/falcon.png" alt="FALCON logo" height="350" width="600"/>
</p>

**FALCON** is a post-generation filtering framework that improves the **logical consistency** of large language model (LLM) outputs by detecting and resolving internal contradictions between extracted claims. The system formulates contradiction-aware claim selection as an optimization problem and selects a maximally consistent subset of claims using Mixed Integer Linear Programming (MILP).

This project was developed as a **custom final project for Stanford CS224N (Winter 2026)**.

---

## 🚀 TL;DR

**Problem:** LLMs generate self-contradictory text that reduces trustworthiness  
**Solution:** Post-hoc MILP-based filtering that removes contradictions optimally  
**Key Innovation:** Global optimization (not greedy) + works with any LLM (no retraining)

**Quick Start:**
```bash
pip install -r requirements.txt
python main.py --mode single --text "Paris is in France. Paris is in Germany." --logic hard
# Output: Removes contradictory claims, keeps consistent ones
```

**Results:** ~90% contradiction reduction on benchmarks, <1s solve time, 3600+ LOC implementation

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

## Project Overview

**FALCON** addresses a critical limitation of modern LLMs: the generation of self-contradictory text. Unlike approaches that require model retraining or fine-tuning, FALCON operates as a **post-processing filter** that:

1. **Extracts** atomic claims from generated text
2. **Scores** pairwise contradictions using a pretrained NLI model (DeBERTa-v3)
3. **Optimizes** claim selection via MILP to maximize consistency while preserving information
4. **Reconstructs** coherent output from the filtered claim set

**Key Innovations:**
- 🎯 **Global Optimization**: MILP guarantees finding the optimal consistent subset (vs. greedy heuristics)
- ⚡ **Efficient**: Solves in <1s per example on CPU for typical inputs
- 🔌 **Modular**: Works with any LLM provider (OpenAI, Anthropic, HuggingFace, vLLM)
- 📊 **Validated**: Comprehensive evaluation on StrategyQA and TruthfulQA benchmarks
- 🔬 **Extensible**: Supports both hard constraints and soft penalty modes

**Technical Stack:** PyTorch, HuggingFace Transformers, PuLP (MILP solver), Python 3.9+

---

## Motivation

Modern LLMs frequently generate responses that contain **self-contradictory claims**, even when individual statements appear plausible in isolation. These inconsistencies reduce trustworthiness and downstream usability.

Rather than retraining or fine-tuning models, FALCON operates **post hoc**, treating consistency as a *global constraint satisfaction problem* over generated claims.

---

## ✨ Key Features

- **🎯 Post-hoc Filtering** — Works with any LLM without retraining or fine-tuning
- **🧮 Optimal Selection** — MILP solver finds globally consistent claim subsets (~3600 LOC implementation)
- **⚡ Fast & Efficient** — <1s solve time per example on CPU (tested on M-series Macs and Linux)
- **🔌 Multi-Provider Support** — OpenAI (GPT-4o-mini), Anthropic (Claude), HuggingFace Transformers, vLLM HTTP
- **📊 Dual Modes** — Hard constraints (strict) or soft penalties (flexible trade-offs)
- **🔍 NLI-Powered** — State-of-the-art contradiction detection via `cross-encoder/nli-deberta-v3-base`
- **📈 Comprehensive Evaluation** — Greedy baseline + FALCON on StrategyQA & TruthfulQA with ablation studies
- **🛠️ Flexible Configuration** — YAML-based configuration for all parameters (solver, NLI, LLM, extraction)
- **🔬 Research-Ready** — Includes ablation study tools, qualitative audit scripts, and visualization utilities
- **📦 Library & CLI** — Use as a Python library or command-line tool

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

**Basic Usage (No LLM Required):**
```bash
# Install dependencies
pip install -r requirements.txt

# Run on sample text with contradictions
python main.py \
  --mode single \
  --text "Paris is the capital of France. Paris is the capital of Germany. The Eiffel Tower is in Paris." \
  --logic hard

# Output: Filters out "Paris is the capital of Germany" while keeping consistent claims
```

**Expected Output:**
```
Selected claims (2/3):
  1. Paris is the capital of France.
  2. The Eiffel Tower is in Paris.

Contradictions removed: 1
Solve time: 0.03s
```

**With LLM Generation (Requires API Key):**
```bash
# Set API key
export OPENAI_API_KEY=your_key_here

# Run on benchmark dataset
python main.py \
  --mode eval \
  --config config.yaml \
  --logic hard \
  --out outputs/results.json
```

### Installation

**Requirements:**
- Python 3.9+ (tested on 3.10, 3.11, 3.12, 3.13)
- 4GB+ RAM for NLI model
- Optional: CUDA-enabled GPU for faster inference

**Setup:**
```bash
# Clone the repository
git clone https://github.com/rehanraza786/falcon.git
cd falcon

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Verify Installation:**
```bash
# Test basic functionality
python main.py --mode single --text "Test claim." --logic hard
```

**First Run:** The NLI model (~1.5GB) will be automatically downloaded from HuggingFace on first use.

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

FALCON includes comprehensive improvements addressing evaluation accuracy, contradiction density, parameter sensitivity, and ethical considerations.

### 1. Fixed StrategyQA Evaluation ✅
- **Issue**: 0% EM scores due to verbose LLM outputs like "Yes, because..."
- **Solution**: Smart yes/no extraction using pattern matching and heuristics
- **Implementation**: `extract_yes_no()` in `falcon/pipeline.py`
- **Impact**: Accurate binary classification evaluation on StrategyQA

### 2. Enhanced TruthfulQA Testing ✅
- **Issue**: Yes/no constraint prevented free-form answers, reducing contradiction density
- **Solution**: Removed constraint + higher temperature (0.7 → 0.9) for diverse outputs
- **Implementation**: Updated `TruthfulQAAdapter.get_question()`
- **Impact**: Increased contradiction density, better exercises filtering capabilities

### 3. Systematic Ablation Studies ✅
- **Tau threshold**: 0.5 → 0.9 sensitivity analysis (9 data points)
- **Claim cap**: 5 → 25 claims scaling study (5 data points)
- **Solver modes**: Hard vs. soft robustness testing
- **Script**: `run_ablation_study.py` (289 lines)
- **Output**: `outputs/ablation/` with detailed scaling reports and plots

### 4. Qualitative Audits ✅
- **Information loss**: Assessment of removed content importance
- **Overconfidence**: Risk evaluation for filtered outputs
- **Logical validity**: Residual contradiction checks
- **Ethical concerns**: Bias and fairness analysis
- **Script**: `run_qualitative_audit.py` (300+ lines)
- **Output**: `outputs/audit/` with comprehensive audit reports

### 5. Additional Enhancements ✅
- **Self-reflection module**: Optional iterative refinement (`self_reflect.py`)
- **Visualization tools**: Chart generation for papers/presentations (`create_output_charts.py`)
- **Test suite**: Comprehensive unit tests (`test_improvements.py`, 218 lines)
- **Multiple configs**: Separate YAML files for different datasets

### Running All Improvements

**Comprehensive Test Suite:**
```bash
# Run all tests, evaluations, and studies
python run_all_improvements.py
```

**Individual Components:**
```bash
# Unit tests for improvements
python test_improvements.py

# Ablation studies (tau, claim cap, mode sensitivity)
python run_ablation_study.py

# Qualitative analysis and audits
python run_qualitative_audit.py

# Generate visualizations
python create_output_charts.py
```

**Shell Script (Parallel Processing):**
```bash
# Faster execution using shell parallelization
bash run_ablation.sh
```
---

## Performance Results

### Key Findings

**StrategyQA (Binary QA):**
- Contradiction reduction: ~90% reduction vs. raw output
- Exact Match: Comparable to baseline (filtering preserves accuracy)
- Solve time: <0.1s per example (MILP optimization)
- Total latency: ~0.5s per example (dominated by NLI)

**TruthfulQA (Open-ended QA):**
- Contradiction density: Higher with free-form prompting
- Filtering effectiveness: 80-100% contradiction removal
- Information preservation: Avg. 7/10 claims retained
- Qualitative improvement: More coherent outputs

**Ablation Studies:**
- **Tau sensitivity**: τ=0.7 offers best accuracy/coverage trade-off
- **Claim cap scaling**: Performance stable up to 20 claims
- **Hard vs. Soft**: Hard mode slightly faster; soft mode retains more information
- **Greedy vs. MILP**: MILP finds 15-20% better solutions on average

### Benchmark Metrics

| Metric | Raw LLM | Greedy | FALCON (Hard) | FALCON (Soft) |
|--------|---------|--------|---------------|---------------|
| **StrategyQA EM** | 0.42 | 0.38 | 0.40 | 0.41 |
| **Contradictions** | 2.3 | 0.8 | 0.1 | 0.3 |
| **Claims Retained** | 10.0 | 6.5 | 7.2 | 8.1 |
| **Solve Time (s)** | - | 0.05 | 0.08 | 0.12 |

*Note: Results based on 50-example samples. Actual numbers may vary depending on LLM choice and configuration.*

### Visualization

Run `python create_output_charts.py` after evaluation to generate:
- Performance comparison bar charts
- Contradiction reduction plots
- Claim selection graphs

---

## Using FALCON as a Library

Import FALCON in your own Python code for programmatic access:

**Basic Usage:**
```python
from falcon_engine import run_single, run_benchmark

# Process a single text (no LLM needed for pre-generated text)
result = run_single(
    cfg_path="config.yaml",
    text="The Earth is flat. The Earth is round. The Earth has one moon."
)

print(f"Input claims: {len(result['claims'])}")
print(f"Selected claims: {len(result['stats']['selected_indices'])}")
print(f"Contradictions removed: {result['stats']['contradictions_before'] - result['stats']['contradictions_after']}")
print(f"Output: {result['output']}")
```

**Advanced Usage:**
```python
import yaml
from falcon.models import load_nli_judge
from falcon.pipeline import run_falcon_on_text
from falcon.adapters.openai_adapter import OpenAIAdapter, OpenAIConfig

# Load custom configuration
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Initialize NLI judge
nli = load_nli_judge(
    model_name="cross-encoder/nli-deberta-v3-base",
    device="auto"  # auto-detects CUDA/MPS/CPU
)

# Optional: Initialize LLM for generation
llm = OpenAIAdapter(OpenAIConfig(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=256
))

# Run FALCON pipeline
filtered_text, stats, P, claims, weights = run_falcon_on_text(
    text="Your input text here",
    nli=nli,
    solver_cfg=cfg["solver"],
    claim_cfg=cfg["claims"],
    llm=llm,
    rewrite_cfg=cfg["rewrite"]
)

print(f"Solve time: {stats.solve_seconds:.3f}s")
print(f"Total time: {stats.total_seconds:.3f}s")
```

**Benchmark Evaluation:**
```python
from falcon_engine import run_benchmark

# Run on StrategyQA or TruthfulQA
results = run_benchmark(cfg_path="config.yaml")

print(f"Dataset: {results['dataset']}")
print(f"EM (Raw): {results['aggregate']['em_raw']:.2f}")
print(f"EM (FALCON): {results['aggregate']['em_falcon']:.2f}")
print(f"Avg contradictions removed: {results['aggregate']['avg_contradictions_before'] - results['aggregate']['avg_contradictions_after_falcon']:.2f}")
```

**Custom Dataset Integration:**
```python
from falcon.adapters import JSONLAdapter

# Use your own JSONL dataset
adapter = JSONLAdapter(file_path="my_dataset.jsonl")
dataset = adapter.load()

for example in dataset:
    question = adapter.get_question(example)
    gold = adapter.get_gold(example)
    # Process with FALCON...
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
A: No! FALCON can work on pre-generated text. Set `llm.enabled: false` in `config.yaml` to disable generation and use `--mode single` with `--text` to process existing text.

**Q: How is FALCON different from other consistency methods?**  
A: FALCON uses MILP optimization for **globally optimal** solutions, unlike greedy heuristics that may miss better combinations. It also works post-hoc without model retraining, making it compatible with any LLM.

**Q: What's the difference between hard and soft mode?**  
A: 
- **Hard mode**: Strictly disallows contradictory pairs above threshold τ (binary constraints: `x_i + x_j ≤ 1`)
- **Soft mode**: Allows contradictions with a penalty λ (continuous optimization with McCormick linearization)
- Hard mode is faster and more conservative; soft mode offers better flexibility and information preservation.

**Q: Can I use custom datasets?**  
A: Yes! Use the `JSONLAdapter` in `falcon/adapters/__init__.py` for custom JSONL files with `question` and `reference` fields. You can also extend the adapter protocol for other formats.

**Q: Why are my scores zero?**  
A: Common causes:
1. `llm.enabled: false` in config (generation disabled)
2. Missing API keys (check `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variables)
3. Max examples set too low (`eval.max_examples` in config)
4. Wrong dataset split (StrategyQA only supports `split: test`)

**Q: How long does evaluation take?**  
A: Performance varies by hardware:
- **50 examples** on GPU: ~1-2 minutes
- **50 examples** on M1 Mac: ~3-5 minutes  
- **50 examples** on CPU: ~5-10 minutes
- Time is dominated by NLI inference (batch processing helps)

**Q: Can I change the contradiction threshold?**  
A: Yes! Adjust `solver.tau` in `config.yaml` (default: 0.7, range: 0.0-1.0). Higher values are stricter:
- τ=0.5: More permissive (keeps more claims)
- τ=0.7: Balanced (default)
- τ=0.9: Very strict (aggressive filtering)

**Q: Does FALCON modify my LLM?**  
A: No! FALCON is a **post-processing filter** that only works on generated outputs. Your LLM remains completely unchanged.

**Q: What NLI models can I use?**  
A: Default is `cross-encoder/nli-deberta-v3-base` (best accuracy). You can configure any HuggingFace cross-encoder NLI model via `nli.model_name` in config.

**Q: Can I process non-English text?**  
A: The default NLI model is optimized for English. For other languages, you'll need a multilingual NLI model (e.g., `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`).

**Q: How do I visualize results?**  
A: Run `python create_output_charts.py` after evaluation to generate performance comparison charts and contradiction reduction plots.

**Q: What's the maximum number of claims FALCON can handle?**  
A: Default is 10 claims (45 pairs). The system has been tested up to 25 claims (300 pairs) in ablation studies. Beyond 32 claims, you may need approximate solvers.

---

## Known Issues & Troubleshooting

### Common Issues and Solutions

**Issue: `ModuleNotFoundError: No module named 'falcon'`**
- **Cause**: Python can't find the falcon package
- **Solution**: Make sure you're running from the project root directory, or install in development mode: `pip install -e .`

**Issue: `ValueError: Feature type 'List' not found`**
- **Cause**: HuggingFace datasets cache conflict between versions
- **Solution**: This is automatically handled by the adapters (force redownload), but you can manually clear cache: `rm -rf ~/.cache/huggingface/datasets/truthful_qa`

**Issue: NLI model download fails or is very slow**
- **Cause**: Network issues or firewall blocking HuggingFace CDN
- **Solution**: 
  - Try using a VPN or different network
  - Manually download model to `~/.cache/huggingface/hub/` and point to local path
  - Use `HF_ENDPOINT` environment variable for mirrors (e.g., China)

**Issue: CUDA out of memory**
- **Cause**: NLI model too large for GPU
- **Solution**: 
  - Reduce `nli.batch_size` in config (default: 8 → try 4 or 2)
  - Use CPU: `nli.device: cpu` in config
  - Use mixed precision if supported

**Issue: Solver takes too long (>10s)**
- **Cause**: Too many claims or complex constraints
- **Solution**:
  - Reduce `claims.max_claims` (default: 10 → try 5)
  - Use hard mode instead of soft mode
  - Check if CBC solver is installed correctly: `pulp-test`

**Issue: Zero EM scores on StrategyQA**
- **Cause**: Several possible reasons
- **Solution**: Check that:
  1. `llm.enabled: true` and API key is set
  2. Using `split: test` (not validation)
  3. `eval.max_examples` > 0
  4. Review extraction: run with `-vv` for debug logs

**Issue: API rate limit errors (OpenAI/Anthropic)**
- **Cause**: Too many requests
- **Solution**:
  - Reduce `eval.max_examples` for testing
  - Add delays between requests (modify adapter code)
  - Use HuggingFace local models instead: `llm.provider: hf`

**Issue: Windows: `FileNotFoundError` for solver**
- **Cause**: CBC solver not in PATH
- **Solution**:
  - Install CBC: Download from COIN-OR or use conda: `conda install -c conda-forge coincbc`
  - Or switch to GLPK solver in PuLP code

**Issue: MacOS: "killed: 9" error**
- **Cause**: Rosetta translation or memory pressure
- **Solution**:
  - Use native ARM Python for M1/M2 Macs
  - Reduce batch size
  - Close other applications

**Issue: Results not reproducible**
- **Cause**: LLM temperature > 0 or no seed set
- **Solution**:
  - Set `--seed 42` in command line
  - Lower temperature in config: `temperature: 0.0`
  - Disable LLM and use pre-generated text

### Debug Tips

**Enable verbose logging:**
```bash
python main.py --mode single --text "Test" -vv --log-file debug.log
```

**Check model loading:**
```python
from falcon.models import load_nli_judge
nli = load_nli_judge(device="cpu")
print(f"Model loaded: {nli.model_name}")
```

**Test claim extraction:**
```python
from falcon.pipeline import extract_claims
claims = extract_claims("Test sentence one. Test sentence two.")
print(f"Claims: {claims}")
```

**Verify solver installation:**
```bash
python -c "import pulp; print(pulp.listSolvers(onlyAvailable=True))"
```

### Performance Optimization

**For faster evaluation:**
1. Use GPU if available: `nli.device: cuda`
2. Increase batch size: `nli.batch_size: 16` (if memory allows)
3. Reduce max examples during testing: `eval.max_examples: 10`
4. Use hard mode: `solver.mode: hard`
5. Lower claim cap: `claims.max_claims: 5`

**For better accuracy:**
1. Increase claim cap: `claims.max_claims: 15`
2. Use soft mode: `solver.mode: soft`
3. Lower tau threshold: `solver.tau: 0.6`
4. Enable conjunction splitting: `claims.split_on_conjunctions: true`

---

## Dependencies

**Core dependencies** from `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | latest | PyTorch for NLI model inference |
| `transformers` | latest | HuggingFace models (DeBERTa-v3, optional local LLMs) |
| `datasets` | 3.6.0 | Dataset loading (StrategyQA, TruthfulQA) |
| `pulp` | latest | MILP optimization solver (CBC backend) |
| `pyyaml` | latest | Configuration file parsing |
| `openai` | ≥1.12.0 | OpenAI API client (GPT-4o-mini, etc.) |
| `anthropic` | ≥0.25.0 | Anthropic API client (Claude) |
| `numpy` | <2.0 | Numerical operations |
| `tqdm` | latest | Progress bars for batch processing |
| `matplotlib` | latest | Visualization utilities |
| `sentencepiece` | latest | Tokenization for some HF models |
| `accelerate` | latest | HuggingFace model loading optimization |
| `protobuf` | latest | Model serialization |
| `pyarrow` | 14-15 | Dataset backend |
| `requests` | latest | HTTP requests for vLLM adapter |

**Hardware acceleration:**
- ✅ CUDA-enabled GPU for faster NLI inference (5-10x speedup)
- ✅ Apple Silicon MPS for M1/M2/M3 Macs (2-3x speedup)
- ✅ CPU fallback supported (slower but functional)

**Installation size:** ~2-3 GB (including PyTorch and NLI model)

---

## Compute & Performance

**Computational Complexity:**
- Claim extraction: O(n) where n = text length
- NLI scoring: O(k²) where k = number of claims (max 10, so ≤45 pairs)
- MILP solving: Typically <1s for k≤10 claims using CBC solver

**Performance Benchmarks:**
| Setup | NLI Inference | MILP Solve | Total/Example |
|-------|---------------|------------|---------------|
| M1 Mac (MPS) | ~0.2s | <0.1s | ~0.5s |
| Linux CPU | ~0.5s | <0.1s | ~0.8s |
| CUDA GPU | ~0.05s | <0.1s | ~0.3s |

**Scalability:**
- ✅ Default: ≤10 claims → ≤45 pairwise constraints
- ✅ Tested up to 25 claims in ablation studies
- ⚠️ Beyond 32 claims may require approximate solvers (future work)

**Memory Requirements:**
- NLI model: ~1.5 GB GPU/CPU RAM
- Runtime overhead: ~500 MB per batch
- Recommended: 4 GB+ total RAM

**Optimization Notes:**
- Batch processing for NLI speeds up multi-example evaluation
- Claim cap (default: 10) balances accuracy vs. tractability
- Hard mode typically faster than soft mode (fewer constraints)

---

## 📂 Project Structure

**Total Lines of Code:** ~3,600 Python LOC

```
falcon/
├── main.py                       # CLI entry point (501 lines)
├── falcon_engine.py              # Library API for external use (70 lines)
├── run_experiments.py            # Automated evaluation script (269 lines)
├── run_ablation_study.py         # Parameter sensitivity analysis (289 lines)
├── run_qualitative_audit.py      # Qualitative evaluation tools (300+ lines)
├── run_all_improvements.py       # Comprehensive test runner (150+ lines)
├── test_improvements.py          # Test suite for improvements (218 lines)
├── create_output_charts.py       # Visualization utilities (93 lines)
├── run_ablation.sh               # Shell script for batch processing
├── config.yaml                   # Main configuration (StrategyQA defaults)
├── config_truthfulqa.yaml        # TruthfulQA-specific config
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── README.md                     # This file
├── assets/
│   └── falcon.png                # Project logo
├── falcon/                       # Core package (~1,400 LOC)
│   ├── __init__.py
│   ├── pipeline.py               # Core FALCON pipeline (500 lines)
│   ├── solver.py                 # MILP optimization solver (72 lines)
│   ├── models.py                 # NLI judge & weighting (75 lines)
│   ├── llm.py                    # LLM protocol & unified scorer (93 lines)
│   ├── rewriter.py               # Claim rewriting logic (100+ lines)
│   ├── self_reflect.py           # Self-reflection module
│   ├── metrics.py                # Evaluation metrics
│   ├── utils.py                  # Utilities (logging, seeding)
│   └── adapters/                 # Dataset & LLM adapters (~600 LOC)
│       ├── __init__.py           # Dataset adapters (TruthfulQA, StrategyQA, JSONL)
│       ├── openai_adapter.py     # OpenAI API integration
│       ├── anthropic_adapter.py  # Anthropic Claude integration
│       ├── hf_transformers_adapter.py  # HuggingFace local models
│       └── vllm_http_adapter.py  # vLLM HTTP server support
└── outputs/
    ├── *_results.json            # Evaluation outputs
    ├── summary.csv               # Aggregated results
    ├── ablation/                 # Ablation study results
    └── audit/                    # Qualitative audit reports
```

**Key Modules:**
- **`falcon/pipeline.py`**: Core workflow (claim extraction → NLI scoring → MILP solving)
- **`falcon/solver.py`**: MILP formulation using PuLP (hard & soft modes)
- **`falcon/models.py`**: NLI contradiction detection using DeBERTa-v3
- **`falcon/adapters/`**: Modular LLM and dataset interfaces
- **`run_*.py`**: Experiment automation and analysis scripts

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

1. **Claim Extraction Heuristics**
   - Relies on sentence splitting and conjunction patterns
   - May miss complex nested claims or implicit assumptions
   - Future: LLM-based semantic claim decomposition

2. **Scalability Constraints**
   - Default cap of 10 claims (45 pairs) for tractability
   - Quadratic growth in pairwise comparisons
   - Future: Approximate solvers (greedy+MILP hybrid), graph-based pruning

3. **NLI Model Limitations**
   - Cross-encoder requires all-pairs comparison (no caching)
   - May miss nuanced contradictions (sarcasm, implicit negation)
   - Domain-specific jargon may confuse general-purpose NLI
   - Future: Bi-encoder pre-filtering, domain-adapted NLI models

4. **Language Support**
   - Optimized for English (NLI model limitation)
   - Degraded performance on code-switching or multilingual text
   - Future: Multilingual NLI models (mBERT, XLM-R based)

5. **Context Loss Risk**
   - Removing claims may eliminate important nuance or caveats
   - No consideration of claim importance hierarchy
   - Future: Hierarchical claim structures, user-guided filtering

6. **Computational Cost**
   - O(k²) NLI inference can be expensive for large k
   - No incremental/streaming processing
   - Future: Lazy evaluation, claim clustering

### Planned Enhancements

**Short-term (next release):**
- [ ] Confidence intervals for contradiction probabilities
- [ ] Interactive CLI mode with claim preview
- [ ] Support for streaming/incremental processing
- [ ] Docker containerization for easy deployment

**Medium-term:**
- [ ] LLM-based claim extraction using GPT-4o/Claude
- [ ] Bi-encoder pre-filtering for faster pairwise scoring
- [ ] Domain-specific NLI fine-tuning toolkit
- [ ] Web UI for interactive contradiction exploration

**Long-term research directions:**
- [ ] Uncertainty-aware claim selection (Bayesian MILP)
- [ ] Multi-document consistency checking
- [ ] Temporal consistency (detecting time-based contradictions)
- [ ] Explainable contradiction detection (attention visualization)
- [ ] Integration with retrieval-augmented generation (RAG)

### Known Issues

- **Issue**: Extremely long sentences (>512 tokens) are truncated by NLI model
  - **Workaround**: Pre-split long sentences or use `claims.max_claims` wisely
- **Issue**: CBC solver may be slow on Windows
  - **Workaround**: Consider GLPK or Gurobi solver backends (PuLP supports both)
- **Issue**: HuggingFace datasets cache conflicts between versions
  - **Fix**: Implemented auto-retry with `FORCE_REDOWNLOAD` in adapters

---

## Ethical Considerations

FALCON is a **content filtering system** that removes information rather than generating new facts. While this approach improves internal consistency, it raises important ethical considerations:

### Information Loss and Bias

**Risk**: Filtering may disproportionately remove minority perspectives or nuanced viewpoints
- Claims expressing uncertainty ("might be", "could be") may be flagged as contradicting confident claims
- Complex multi-faceted answers may be oversimplified
- **Mitigation**: Qualitative audits (see `run_qualitative_audit.py`) assess information loss patterns

**Risk**: NLI models may reflect biases in training data
- Certain topics (politics, religion, cultural practices) may have skewed contradiction detection
- **Mitigation**: Use domain-specific NLI models when available; validate on diverse test sets

### Overconfidence and Misuse

**Risk**: Filtered output may appear more authoritative than warranted
- Removing contradictions doesn't guarantee factual correctness
- Users may over-trust "consistent" outputs
- **Warning**: FALCON should be used as a **decision-support tool**, not an authoritative filter

**Risk**: Adversarial use for deceptive content
- Could be used to make false claims appear internally consistent
- **Responsibility**: Users must ensure ethical application; consider provenance tracking

### Transparency and Accountability

**Best Practices**:
1. **Disclose filtering**: Inform downstream users that content has been processed
2. **Provide alternatives**: Show both filtered and unfiltered versions when possible
3. **Track provenance**: Log which claims were removed and why (see `stats` output)
4. **Human oversight**: Use for high-stakes applications only with expert review

### Research Ethics (CS224N Context)

This project was developed for **educational purposes** as part of Stanford CS224N. Deployment in production systems should consider:
- Regulatory compliance (e.g., GDPR, AI Act)
- Domain-specific requirements (medical, legal, financial)
- Accessibility and fairness across user populations

### Reporting Issues

If you identify ethical concerns or unexpected behaviors:
1. Open a GitHub issue with details
2. Tag with `ethical-concern` label
3. Provide reproducible examples
4. Suggest mitigation strategies if possible

**Acknowledgment**: We recognize that consistency ≠ correctness, and filtering is not a substitute for fact-checking or expert validation.

---

## Citation

If you use FALCON in your research or reference this project, please cite:

```bibtex
@misc{azam2026falcon,
  author = {Azam, Rehan},
  title = {FALCON: Factual-Aware Logical Consistency Optimization for LLM Outputs},
  year = {2026},
  publisher = {GitHub},
  journal = {Stanford CS224N Final Project},
  howpublished = {\url{https://github.com/rehanraza786/falcon}},
  note = {A post-generation filtering framework using MILP optimization for contradiction resolution in LLM outputs}
}
```

**Related Work:**

If you use the underlying NLI model, please also cite:
```bibtex
@inproceedings{he2021deberta,
  title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
  author={He, Pengcheng and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu},
  booktitle={ICLR},
  year={2021}
}
```

---

## Acknowledgments

This project was developed as part of **Stanford CS224N: Natural Language Processing with Deep Learning (Winter 2026)**.

**Special Thanks:**
- **Stanford CS224N Teaching Staff** — Course framework, guidance, and feedback
- **HuggingFace Team** — Transformers library, datasets, and model hub
- **Cross-Encoder Team** — NLI DeBERTa-v3 model ([cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base))
- **PuLP Developers** — Open-source MILP solver interface
- **COIN-OR CBC** — Fast, open-source linear programming solver
- **PyTorch Team** — Deep learning framework
- **OpenAI & Anthropic** — API access for evaluation

**Datasets:**
- **StrategyQA** — [Geva et al., 2021](https://allenai.org/data/strategyqa) (wics/strategy-qa on HF)
- **TruthfulQA** — [Lin et al., 2021](https://arxiv.org/abs/2109.07958) (truthful_qa on HF)

**Inspiration:**
This work builds on research in factual consistency, natural language inference, and constrained optimization for NLP. Key influences include:
- Factual consistency checking in summarization (Kryscinski et al., 2020)
- NLI-based contradiction detection (Williams et al., 2018)
- Optimization-based text generation (Hokamp & Liu, 2017)

**Tools & Infrastructure:**
- GitHub for version control and hosting
- Python ecosystem (NumPy, PyYAML, tqdm, matplotlib)
- Development environments: PyCharm, VS Code, Jupyter

---

## Contributing

Contributions are welcome! This project follows standard open-source practices.

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/falcon.git
   cd falcon
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-improvement
   ```

3. **Make your changes**
   - Follow existing code style (PEP 8)
   - Add docstrings to new functions
   - Update tests if applicable

4. **Test your changes**
   ```bash
   # Run test suite
   python test_improvements.py
   
   # Test on sample data
   python main.py --mode single --text "Test claim."
   ```

5. **Commit and push**
   ```bash
   git commit -am 'Add feature: description'
   git push origin feature/your-improvement
   ```

6. **Open a Pull Request**
   - Describe your changes clearly
   - Reference any related issues
   - Include before/after examples if applicable

### Areas for Contribution

**High Priority:**
- [ ] Additional dataset adapters (SQuAD, FEVER, etc.)
- [ ] Alternative MILP solvers (Gurobi, CPLEX)
- [ ] Performance optimizations (caching, parallelization)
- [ ] Documentation improvements

**Medium Priority:**
- [ ] Alternative NLI models comparison
- [ ] Web UI development
- [ ] Docker containerization
- [ ] CI/CD pipeline improvements

**Research Extensions:**
- [ ] Multi-document consistency
- [ ] Temporal contradiction detection
- [ ] Explainability features
- [ ] Multilingual support

### Code Style

- Use type hints where appropriate
- Maximum line length: 120 characters
- Docstrings: Google style format
- Variable naming: descriptive and consistent

### Testing

- Add unit tests for new features
- Ensure backward compatibility
- Test on both CPU and GPU if applicable

### Questions?

Open an issue with the `question` label or reach out via the contact info below.

---

## Quick Reference

### Command Cheat Sheet

```bash
# Basic filtering (no LLM)
python main.py --mode single --text "Your text here" --logic hard

# Evaluate on StrategyQA
python main.py --mode eval --config config.yaml --out results.json

# Evaluate on TruthfulQA
python main.py --mode eval --config config_truthfulqa.yaml --out results.json

# Run all experiments
python run_experiments.py

# Run ablation studies
python run_ablation_study.py

# Run qualitative audits
python run_qualitative_audit.py

# Run all improvements
python run_all_improvements.py

# Generate visualizations
python create_output_charts.py

# Enable debug logging
python main.py --mode eval -vv --log-file debug.log
```

### Configuration Quick Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `solver.mode` | `hard` | `hard` or `soft` |
| `solver.tau` | `0.7` | Contradiction threshold (0.0-1.0) |
| `solver.lambda_penalty` | `1.0` | Soft mode penalty weight |
| `claims.max_claims` | `10` | Maximum claims per text |
| `claims.split_on_conjunctions` | `true` | Split on "and", "but", etc. |
| `nli.device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `nli.batch_size` | `8` | NLI inference batch size |
| `llm.provider` | `hf` | `openai`, `anthropic`, `hf`, `vllm_http` |
| `eval.max_examples` | `50` | Number of examples to evaluate |

### File Structure Quick Reference

```
falcon/
├── main.py              # CLI entry point
├── falcon_engine.py     # Library API
├── config.yaml          # Main config
├── falcon/
│   ├── pipeline.py      # Core pipeline
│   ├── solver.py        # MILP solver
│   ├── models.py        # NLI model
│   └── adapters/        # LLM & dataset adapters
└── outputs/             # Results directory
```

### Python API Quick Reference

```python
# Single text processing
from falcon_engine import run_single
result = run_single("config.yaml", "Your text here")

# Benchmark evaluation
from falcon_engine import run_benchmark
results = run_benchmark("config.yaml")

# Direct pipeline access
from falcon.pipeline import run_falcon_on_text
from falcon.models import load_nli_judge

nli = load_nli_judge()
filtered, stats, P, claims, weights = run_falcon_on_text(
    text="Your text",
    nli=nli,
    solver_cfg={"mode": "hard", "tau": 0.7},
    claim_cfg={"max_claims": 10},
    llm=None,
    rewrite_cfg={"enabled": False}
)
```

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# HuggingFace (optional)
export HF_TOKEN=hf_...
export HF_ENDPOINT=https://huggingface.co  # or mirror

# Experiment config
export ENABLE_LLM=1
export LLM_PROVIDER=openai
export OUTPUT_DIR=./outputs
```

---

## Contact

**Author:** Rehan Azam  
**Affiliation:** Stanford University  
**Course:** CS224N - Natural Language Processing with Deep Learning (Winter 2026)  
**Project Type:** Custom Final Project  
**Repository:** [https://github.com/rehanraza786/falcon](https://github.com/rehanraza786/falcon)

### Get Help

- **Issues & Bugs:** [GitHub Issues](https://github.com/rehanraza786/falcon/issues)
- **Feature Requests:** [GitHub Discussions](https://github.com/rehanraza786/falcon/discussions)
- **Questions:** Open an issue with the `question` label

### Stay Updated

- ⭐ Star the repository to follow updates
- 👀 Watch for release notifications
- 🔔 Subscribe to issue notifications for specific features

For academic inquiries or collaboration opportunities, please open a GitHub discussion.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.
