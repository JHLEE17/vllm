#! /bin/bash
# Default values
port=8000
max_input_len=1024
model=Meta-Llama-3.1-8B-Instruct
# model=Meta-Llama-3-8B-Instruct
csv_path=/home/jovyan/vol-1/jh/benchmark_vllm_v062_1008/
guided_json_template=/home/jovyan/vol-1/jh/vllm/benchmarks/guided_json_template.json
num_requests=1024
sleep_time=30
max_num_batched_tokens=1024

decode=false
prefill=false
dynamic_data=false
guided_json=false
enable_lora=false
prefix_caching=false
all=false

error() {
    echo "Error: Invalid option '$1'"
    echo "Usage: $0 [--port <port>] [--max-input-len <length>] [--guided-json] [--enable-lora] [--prefix-caching] [--all]"
    exit 1
}

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) port="$2"; shift ;;
        --max-input-len) max_input_len="$2"; shift ;;
        --num-requests) num_requests="$2"; shift ;;
        --max-num-batched-tokens) max_num_batched_tokens="$2"; shift ;;
        --decode) decode=true ;;
        --prefill) prefill=true ;;
        --dynamic-data) dynamic_data=true ;;
        --guided-json) guided_json=true ;;
        --enable-lora) enable_lora=true ;;
        --prefix-caching) prefix_caching=true ;;
        --all) 
            guided_json=true
            enable_lora=true
            prefix_caching=true
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

prefix_len=$((max_input_len / 4))
dataset_path=/home/jovyan/vol-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix_${prefix_len}_max_${max_input_len}_1024_sampled.parquet

# Construct base command
cmd_base="python benchmarks/benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --max-input-len $max_input_len"

mkdir -p $csv_path

# Function to run a specific benchmark
run_benchmark() {
    local num_requests=$1
    local max_output_len=$2

    cmd="$cmd_base --max-output-len $max_output_len --num-requests $num_requests --max-num-batched-tokens $max_num_batched_tokens"

    if [[ "$dynamic_data" == "true" ]]; then
        cmd+=" --dataset $dataset_path"
    fi

    if [[ "$enable_lora" == "true" ]]; then
        cmd+=" --lora-pattern lora-1,lora-2 --random-lora"
    fi

    if [[ "$prefix_caching" == "true" ]]; then
        cmd+=" --prefix-caching"
    fi

    if [[ "$guided_json" == "true" ]]; then
        cmd+=" --json-template $guided_json_template"
    fi

    if [[ "$num_requests" -ne 1024 ]]; then
        cmd+=" --warmup"
    fi

    echo "Running: $cmd"
    eval $cmd
    sleep 30
}

# Run benchmark (num_requests / max_out_len)


if [[ "$prefill" == "true" ]]; then
    # run_benchmark 10 10
    run_benchmark 1024 1
fi  

if [[ "$decode" == "true" ]]; then
    run_benchmark 10 10
    run_benchmark 1024 128
fi
