#!/bin/bash

# Usage:
# ./benchmarks_script.sh [GPU] [PORT]
# Example:
# ./benchmarks_script.sh 0 8000

# Get GPU and Port from command-line arguments
GPU=2
PORT=8002

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


# Array of concurrency values from 128 to 16, decrementing by 16
concurrency_values=(128 112 96 80 64 48 32 16)

# Loop through each concurrency value and execute the command
for concurrency in "${concurrency_values[@]}"; do
  bash benchmarks_script.sh \
    --max-input-len 8192 \
    --feature original \
    --prefill \
    --concurrency "$concurrency" \
    --gpu "$GPU" --port "$PORT"
done
