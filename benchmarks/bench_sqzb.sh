#! /bin/bash
model=Meta-Llama-3.1-8B-Instruct
# model=Meta-Llama-3-8B-Instruct
port=8001
max_input_len=1024
max_output_len=1024
csv_path=/home/jovyan/vol-1/jh/benchmark_sqzb_0925/
random_data=false
guided_json=false
enable_lora=false
prefix_caching=false
kv_fp8=false
enforce_eager=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) port="$2"; shift ;;
        --max-input-len) max_input_len="$2"; shift ;;
        --max-output-len) max_output_len="$2"; shift ;;
        --csv-path) csv_path="$2"; shift ;;
        --random-data) random_data=true ;;
        --json-template) guided_json=true ;;
        --enable-lora) enable_lora=true ;;
        --prefix-caching) prefix_caching=true ;;
        --kv-fp8) kv_fp8=true ;;
        --enforce-eager) enforce_eager=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

prefix_len=$((max_input_len / 4))
dataset_path=/home/jovyan/vol-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix\_$prefix_len\_max\_$max_input_len\_1024_sampled.parquet

python benchmark_sqzb.py \
    --port $port \
    --tokenizer /home/jovyan/vol-1/models/$model \
     $( [[ "$random_data" == "false" ]] && echo "--dataset $dataset_path" ) \
    --max-input-len $max_input_len \
    --csv-path $csv_path \
    --max-output-len $max_output_len \
    --num-requests 1024 \
    $( [[ "$guided_json" == "true" ]] && echo "--json-template" ) \
    $( [[ "$enable_lora" == "true" ]] && echo "--enable-lora" ) \
    $( [[ "$prefix_caching" == "true" ]] && echo "--prefix-caching" ) \
    $( [[ "$kv_fp8" == "true" ]] && echo "--kv-fp8" ) \
    $( [[ "$enforce_eager" == "true" ]] && echo "--enforce-eager" )