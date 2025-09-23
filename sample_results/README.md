# HumanEvalComm V2 Benchmark Suite

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Complete implementation of the HumanEvalComm V2 benchmark framework**

This directory contains the production-ready implementation of the HumanEvalComm V2 benchmark, featuring comprehensive code generation evaluation with advanced metrics and multi-dimensional analysis.

## 📁 Directory Structure

```bash
benchmark_v2/
├── v2_benchmark_completely_fixed.py    # Main benchmark script
├── README.md                           # This documentation
├── v2_complete_results_*.json          # Benchmark evaluation results
├── v2_fixed_results_*.json             # Processed results data
├── v2_leaderboard_*.csv               # Generated leaderboards
└── v2_fixed_leaderboard_*.csv         # Final leaderboard outputs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages: `openai`, `pandas`, `numpy`, `tqdm`

### Running the Benchmark

#### Option 1: Using the Wrapper Script (Recommended)

From the project root directory:

```bash
# Run with default settings
./run_v2_benchmark.sh

# Run with custom output directory
./run_v2_benchmark.sh --output-dir ./my_results

# Run with different models
./run_v2_benchmark.sh --models "gpt4:gpt-4:openai" --models "claude:claude-3-sonnet:anthropic"

# Run with custom dataset and more problems
./run_v2_benchmark.sh --dataset-path ./custom_dataset.jsonl --max-problems 10 --verbose
```

#### Option 2: Direct Python Execution

```bash
cd benchmark_v2
python v2_benchmark_completely_fixed.py --dataset-path ../Benchmark/HumanEvalComm.jsonl \
                                        --output-dir ../results \
                                        --models llama3-8b:meta-llama/Llama-3.1-8B-Instruct:cerebras \
                                        --models qwen-coder:Qwen/Qwen2.5-Coder-32B-Instruct:together \
                                        --max-problems 5
```

### Command Line Options

- `--dataset-path PATH`: Path to the dataset file (default: `Benchmark/HumanEvalComm.jsonl`)
- `--output-dir DIR`: Directory to save results (default: `./benchmark_results`)
- `--models SPEC`: Model specifications (required, can be used multiple times)
  - Format: `name:model_id[:provider][:max_tokens][:temperature]`
  - Examples:
    - `llama3-8b:meta-llama/Llama-3.1-8B-Instruct:cerebras`
    - `gpt4:gpt-4:openai:2048:0.2`
- `--max-problems N`: Maximum number of problems to evaluate (default: 3)
- `--request-delay SEC`: Delay between API requests in seconds (default: 6.0)
- `--verbose`: Enable verbose logging

The script will:

1. Load HumanEvalComm V2 problem set
2. Evaluate specified models  
3. Generate comprehensive results
4. Create leaderboard rankings

## 📊 Generated Outputs

### Results Files (`v2_*_results_*.json`)

- Raw evaluation data for each model-problem pair
- Execution results, test outcomes, and error logs
- Multi-LLM judge evaluations and consensus scores

### Leaderboard Files (`v2_*_leaderboard_*.csv`)

- Aggregated performance metrics per model
- V2 composite scores and component breakdowns
- Communication quality assessments
- Trustworthiness evaluations

## 🔧 Configuration

The benchmark script includes configurable parameters:

- **Models**: Specify which language models to evaluate
- **Problem Sets**: Choose evaluation datasets
- **Metrics**: Select evaluation dimensions
- **Output Paths**: Customize result file locations

## 📈 Evaluation Metrics

### V2 Composite Score

Weighted combination of:

- **Test Performance**: Unit test pass rates
- **Static Analysis**: Code quality metrics
- **Security Assessment**: Vulnerability detection
- **Communication Quality**: Documentation and clarity

### Additional Dimensions

- **Readability**: Code structure and naming
- **Efficiency**: Performance characteristics
- **Reliability**: Error handling and robustness

## 🏆 Leaderboard Rankings

Models are ranked based on the V2 composite score, with tie-breaking using:

1. Test pass rate
2. Communication quality
3. Security score
4. Efficiency metrics

## 📝 Notes

- This is the **final working version** with all known issues resolved
- Results are timestamped for reproducibility
- Large result files may require significant storage space
- GPU acceleration recommended for faster evaluation

## 🤝 Contributing

When modifying the benchmark:

1. Test changes on a small subset first
2. Validate results against known baselines
3. Update documentation for any new features
4. Ensure backward compatibility

---

## Part of the HumanEvalComm V2 Evaluation Framework

[📖 Main Documentation](../README.md) •
[🏠 Leaderboard](../flask_leaderboard/) •
[🐛 Report Issues](https://github.com/your-repo/humaneval-comm/issues)
