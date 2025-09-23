

# HumanEvalComm V2 — Robust Benchmark Development Plan

## 🎯 Goal
Extend the HumanEvalComm benchmark (ambiguous/incomplete/inconsistent programming problems) into a reliable multi-dimensional benchmark that evaluates:
- Communication performance (original focus)
- Code correctness (beyond Pass@1, include fuzzing)
- Trustworthiness (readability, maintainability, security, efficiency)
- Evaluation reliability (multi-judge + calibration)

---

## 🛠️ Step 1: Dataset Preparation
- **Dataset Location:**
  - All HumanEvalComm data files are in the `Benchmark/` directory of the repository.
  - Main file: `Benchmark/HumanEvalComm.jsonl` (JSON Lines format, one problem per line).
  - Other related files: `HumanEvalComm_v2.jsonl`, `HumanEvalComm_v2.csv`, `HumanEvalComm_dry_run.jsonl`, etc.

- **Dataset Format:**
  - Each line in `HumanEvalComm.jsonl` is a JSON object with keys such as:
    - `name`: problem ID
    - `prompt`, `prompt1a`, `prompt1c`, `prompt1p`: original and modified prompts
    - `test_case`: list of test cases
    - `entry_point`, `solution`: reference solution and function name
    - Additional metadata and prompt variants

- **How to Use:**
  - Load the dataset using a Python class or function that reads the JSONL file line by line.
  - For each problem, extract the required fields (ID, prompts, tests, metadata).
  - The loader should support both original and modified problems, and allow filtering/subsetting.
  - Example usage: `Benchmark/HumanEvalComm.jsonl` is the default input for all benchmark scripts and notebooks.
- Use the existing HumanEvalComm dataset from the repo.
- Dataset contains:
  - Original HumanEval problems
  - Modified versions (ambiguous, incomplete, inconsistent, mixed).
- Confirm structure: each problem has prompt, tests, modified_prompt, and possibly clarifications.
- **Tasks:**
  - Clone repo and inspect dataset format (problems/ folder).
  - Validate number of tasks = 164 × (original + modified versions).
  - Prepare a dataset loader function (Python class) that returns task objects:
    ```python
    {
      "id": "problem_id",
      "original_prompt": "...",
      "modified_prompt": "...",
      "tests": [...],
      "metadata": {...}
    }
    ```

---

## 🛠️ Step 2: Model Inference
- Baseline: run existing evaluation with GPT-3.5, GPT-4, DeepSeek, CodeLlama (matching the paper).
- Each run should log:
  - Model clarifying questions
  - Final code
  - Test execution results
- **Tasks:**
  - Implement a model interface wrapper:
    - For OpenAI models → requires API key (OPENAI_API_KEY).
    - For HuggingFace models (DeepSeek, CodeLlama) → load via transformers.
  - Run inference with configurable model + seed + dataset subset.
  - Save raw JSON logs (per problem, per model). Example:
    ```python
    {
      "id": "problem_42",
      "model": "gpt-3.5-turbo",
      "clarifying_questions": ["..."],
      "final_code": "...",
      "test_results": {"passed": 7, "failed": 2},
      "runtime": 0.42,
      "memory_mb": 32
    }
    ```

---

## 🛠️ Step 3: Communication Metrics (reuse from paper)
- Communication Rate
  - Did the model ask any clarifying question before coding? (Y/N).
  - % across all tasks.
- Good Question Rate
  - Multi-judge evaluation of whether question is useful.
  - Each judge outputs JSON { "useful": 1/0, "confidence": 0-1 }.
  - Use consensus + calibration (see Step 5).

---

## 🛠️ Step 4: Code Correctness Metrics
- Pass@1
  - Standard HumanEval correctness measure.
- Test Pass Rate
  - Fraction of provided test cases passed.
- Fuzz/Property Testing
  - Extend unit tests with Hypothesis (auto-generate edge cases).
  - Metric = % of generated cases passed.
- Sandbox Execution
  - Run all code in Docker with CPU/memory/time limits.
  - Capture runtime + memory → feeds into efficiency metrics.

---

## 🛠️ Step 5: Reliability Evaluation
- Multi-judge scoring
  - Use ≥3 LLMs (GPT-4, Claude, DeepSeek) as judges for “good questions”.
  - Aggregate via weighted average.
- Human calibration
  - Select ~10–15% tasks (esp. disagreement cases).
  - Collect human labels (good/bad question, correct/incorrect code).
  - Train a calibration model (logistic regression) to adjust judge scores.
- Reliability metrics
  - Judge Consensus Confidence = agreement % among LLMs.
  - Calibration Gap = difference between consensus vs human truth.

---

## 🛠️ Step 6: Trustworthiness Metrics
- Readability
  - Pylint/flake8 style score.
  - Cyclomatic complexity (radon).
- Maintainability
  - Maintainability Index (from radon).
  - Docstring + comment density.
- Security
  - Bandit static analysis (counts of vulnerabilities → normalized score).
- Efficiency
  - Runtime (sec, normalized).
  - Peak memory (MB, normalized).

---

## 🛠️ Step 7: Aggregation & Composite Score
- Produce a multi-dimensional result table per model.
- Example scoring formula (configurable):
  ```python
  Final Score = 0.40 * Correctness
              + 0.20 * Communication
              + 0.15 * Readability
              + 0.10 * Security
              + 0.10 * Efficiency
              + 0.05 * Maintainability
  ```
- Also publish individual metric scores for transparency.

---

## 🛠️ Step 8: Leaderboard & Reporting
- Store results in results_v2/{model}/{run_id}/metrics.json.
- Build leaderboard script → outputs CSV + Markdown + plots.
- Radar charts for trust dimensions.
- Heatmaps for model × metric comparison.
- Publish a web UI (e.g., Streamlit or static HTML) for interactive browsing.

---

## 🛠️ Step 9: Documentation
- Write a V2_README.md with:
  - Dataset usage
  - Evaluation pipeline
  - Metrics definitions
  - Safety precautions (sandboxing)
  - How to reproduce results

---

## 🔹 Deliverables
- Code: evaluators/ folder with new metrics modules.
- Data: Raw JSON logs + calibration annotations.
- Outputs: Leaderboard tables + plots.
- Docs: Detailed README + config files for reproducibility.
