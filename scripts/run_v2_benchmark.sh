#!/bin/bash
# HumanEvalComm V2 Benchmark Runner
# Small wrapper script to run the configurable V2 benchmark

# Default configuration
DATASET_PATH="Benchmark/HumanEvalComm.jsonl"
OUTPUT_DIR="./benchmark_results"
MODELS=(
    "llama3-8b:meta-llama/Llama-3.1-8B-Instruct:cerebras"
    "qwen-coder:Qwen/Qwen2.5-Coder-32B-Instruct:together"
)
MAX_PROBLEMS=3
REQUEST_DELAY=6.0
VERBOSE=false

# Function to display usage
usage() {
    echo "HumanEvalComm V2 Benchmark Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset-path PATH    Path to dataset file (default: $DATASET_PATH)"
    echo "  --output-dir DIR       Directory to save results (default: $OUTPUT_DIR)"
    echo "  --models MODEL_SPEC    Model specifications (can be used multiple times)"
    echo "                         Format: name:model_id[:provider][:max_tokens][:temperature]"
    echo "                         Default models:"
    for model in "${MODELS[@]}"; do
        echo "                           $model"
    done
    echo "  --max-problems N       Maximum number of problems to evaluate (default: $MAX_PROBLEMS)"
    echo "  --request-delay SEC    Delay between API requests in seconds (default: $REQUEST_DELAY)"
    echo "  --verbose              Enable verbose logging"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --max-problems 10"
    echo "  $0 --output-dir ./my_results --models llama3-8b:meta-llama/Llama-3.1-8B-Instruct:cerebras"
    echo "  $0 --dataset-path ./custom_dataset.jsonl --verbose"
}

# Parse command line arguments
CUSTOM_MODELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --models)
            CUSTOM_MODELS+=("$2")
            shift 2
            ;;
        --max-problems)
            MAX_PROBLEMS="$2"
            shift 2
            ;;
        --request-delay)
            REQUEST_DELAY="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Use custom models if provided, otherwise use defaults
if [ ${#CUSTOM_MODELS[@]} -gt 0 ]; then
    MODELS=("${CUSTOM_MODELS[@]}")
fi

# Build the command
CMD="cd benchmark_v2 && python v2_benchmark_completely_fixed.py"
CMD="$CMD --dataset-path ../$DATASET_PATH"
CMD="$CMD --output-dir ../$OUTPUT_DIR"
CMD="$CMD --max-problems $MAX_PROBLEMS"
CMD="$CMD --request-delay $REQUEST_DELAY"

# Add models
for model in "${MODELS[@]}"; do
    CMD="$CMD --models $model"
done

# Add verbose flag if requested
if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

echo "🚀 Running HumanEvalComm V2 Benchmark"
echo "====================================="
echo "Dataset: $DATASET_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Models: ${#MODELS[@]}"
echo "Max Problems: $MAX_PROBLEMS"
echo "Request Delay: ${REQUEST_DELAY}s"
echo ""
echo "Command: $CMD"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the benchmark
eval "$CMD"