#!/bin/bash
port=8000
model_name=Meta-Llama-3.1-8B-Instruct
csv_path=/home/jovyan/vol-1/jh/sqzb/gaudi/benchmark_vllm_v062_1106/
max_input_len=1024

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) port="$2"; shift ;;
        --max-input-len) max_input_len="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

concurrency_values=(128 128 112 96 80 64 48 32 16 32 16 4 1)

for concurrency in "${concurrency_values[@]}"; do
    python benchmarks/benchmark_sqzb.py \
    --tokenizer /home/jovyan/vol-1/models/Meta-Llama-3.1-8B-Instruct \
    --num-requests 512 \
    --max-input-len $max_input_len \
    --max-output-len 1024 \
    --port $port \
    --concurrency $concurrency \
    --csv-path $csv_path
done