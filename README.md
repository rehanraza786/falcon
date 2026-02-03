# FALCON: Factual-Aware Logical Consistency Optimization for LLM Outputs

[![CI](https://github.com/rehanraza786/falcon/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/rehanraza786/falcon/actions/workflows/ci.yaml)

<p align="center">
  <img src="assets/falcon.png" alt="FALCON logo" height="350" width="600"/>
</p>

FALCON is a post-generation filtering framework that improves the **logical consistency** of large language model (LLM) outputs by detecting and resolving internal contradictions between extracted claims. The system formulates contradiction-aware claim selection as an optimization problem and selects a maximally consistent subset of claims.

This project was developed as a **custom final project for Stanford CS224N (Winter 2026)**.

---

## Motivation

Modern LLMs frequently generate responses that contain **self-contradictory claims**, even when individual statements appear plausible in isolation. These inconsistencies reduce trustworthiness and downstream usability.

Rather than retraining or fine-tuning models, FALCON operates **post hoc**, treating consistency as a *global constraint satisfaction problem* over generated claims.

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
- Optional: CUDA-enabled GPU

### Installation
```bash
  pip install -r requirements.txt
```

Optional (for generation):
```bash
  export OPENAI_API_KEY=your_key_here
```

---

## Running Experiments

Run the full evaluation sweep:
```bash
  python run_experiments.py
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

## Result Interpretation

- **Zero scores** typically indicate generation was disabled (`llm.enabled=false`).
- Contradiction counts only appear when ≥2 claims are extracted.
- TruthfulQA results should be interpreted **qualitatively**.
- MILP is expected to reduce contradictions more than greedy filtering,
  sometimes at slight cost to EM.

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

## Known Issues & Troubleshooting

- **HuggingFace dataset schema errors**  
  → Clear cache:  
  ```bash
  rm -rf ~/.cache/huggingface/datasets
  ```

- **TruthfulQA has no `test` split**  
  → Always use `validation`.

- **Pandas warnings about numexpr/bottleneck**  
  → Safe to ignore or upgrade dependencies.

---

## Compute & Performance

- ≤10 claims ⇒ ≤45 pairwise constraints
- MILP solves in **<1s per example** on CPU
- End-to-end runtime dominated by NLI scoring

---

## 📂 Project Structure

```
FALCON-Project/
├── main.py
├── run_experiments.py
├── config.yaml
├── requirements.txt
├── falcon/
│   ├── pipeline.py
│   ├── solver.py
│   ├── models.py
│   ├── llm.py
│   ├── rewriter.py
│   └── adapters/
└── outputs/
```

---

## Ethical Considerations

FALCON removes content rather than generating new facts. While this may improve internal consistency, it may also remove nuance. The system should be used as a **decision-support tool**, not an authoritative filter.

---

## Citation

If you reference this project:
```
Rehan Azam. FALCON: Factual-Aware Logical Consistency Optimization for LLM Outputs.
Stanford CS224N Final Project, 2026.
```

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.
