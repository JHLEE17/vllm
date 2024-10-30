#! /bin/bash
# Default values
port=8000
max_input_len=1024
model=Meta-Llama-3.1-8B-Instruct
csv_path=/home/jovyan/vol-1/jh/benchmark_vllm_v061_0926/
prefix_len=$((max_input_len / 4))
dataset_path=/home/jovyan/vol-1/datasets/dynamic_sonnet_llama3/dynamic_sonnet_llama_3_prefix\_$prefix_len\_max\_$max_input_len\_1024_sampled.parquet
guided_json_template=guided_json_template.json
sleep_time=30

guided_json=false
enable_lora=false
prefix_caching=false
all=false

error() {
    echo "Error: Invalid option '$1'"
    echo "Usage: $0 [--port <port>] [--max-input-len <length>] [--guided-json] [--enable-lora] [---prefix-caching] [--all]"
    exit 1
}

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) port="$2"; shift ;;
        --max-input-len) max_input_len="$2"; shift ;;
        --guided-json) guided_json=true ;;
        --enable-lora) enable_lora=true ;;
        --prefix-caching) prefix_caching=true ;;
        --all) 
            all=true
            guided_json=true
            enable_lora=true
            prefix_caching=true
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Construct base command
cmd_base_warmup="python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 10 --max-input-len $max_input_len"
cmd_base="python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 --max-input-len $max_input_len"

# Function to run a specific benchmark
run_benchmark() {
    local max_output_len=$1
    local dynamic_data=$2

    cmd_warmup="$cmd_base_warmup --max-output-len 1"
    cmd="$cmd_base --max-output-len $max_output_len"

    if [[ "$dynamic_data" == "true" ]]; then
        cmd+=" --dataset $dataset_path"
    fi

    if [[ "$enable_lora" == "true" ]]; then
        cmd_warmup+=" --lora-pattern lora-1, lora-2 --random-lora"
        cmd+=" --lora-pattern lora-1, lora-2 --random-lora"
    fi

    if [[ "$prefix_caching" == "true" ]]; then
        cmd_warmup+=" --prefix-caching"
        cmd+=" --lora-pattern lora-1, lora-2 --random-lora"
    fi

    if [[ "$guided_json" == "true" ]]; then
        echo "Warmup for guide-json: $cmd_warmup"
        eval $cmd_warmup
        sleep 20
        cmd+=" --json-template $guided_json_template"
    fi

    echo "Running: $cmd"
    eval $cmd
    sleep 20
}

# Run benchmark (dynamic data)
# run_benchmark 1024 true
# run_benchmark 1024 false
# run_benchmark 1 true
run_benchmark 1 false


# Ogirinal
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024 --random-data
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1 --random-data
#  sleep $sleep_time

# # Guided Json
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024 --json-template $guided_json_template
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024 --json-template $guided_json_template --random-data
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1 --json-template $guided_json_template
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1 --json-template $guided_json_template --random-data
#  sleep $sleep_time


# # # Multi-LoRA
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024 --lora-pattern lora-1, lora-2 --random-lora
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024 --lora-pattern lora-1, lora-2 --random-lora --random-data
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1 --lora-pattern lora-1, lora-2 --random-lora
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1 --lora-pattern lora-1, lora-2 --random-lora --random-data
#  sleep $sleep_time

# # # Prefix Caching
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024 --prefix-caching
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024 --prefix-caching --random-data
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1 --prefix-caching
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1 --prefix-caching --random-data
#  sleep $sleep_time

# # Multi-LoRA & Prefix-Caching & Guided-Json
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024 --json-template $guided_json_template --lora-pattern lora-1, lora-2 --random-lora --prefix-caching
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1024 --json-template $guided_json_template --lora-pattern lora-1, lora-2 --random-lora --prefix-caching --random-data
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1 --json-template $guided_json_template --lora-pattern lora-1, lora-2 --random-lora --prefix-caching
#  sleep $sleep_time
# python benchmark_sqzb.py --port $port --tokenizer /home/jovyan/vol-1/models/$model --csv-path $csv_path --num-requests 1024 \
# --max-input-len $max_input_len --max-output-len 1 --json-template $guided_json_template --lora-pattern lora-1, lora-2 --random-lora --prefix-caching --random-data
#  sleep $sleep_time