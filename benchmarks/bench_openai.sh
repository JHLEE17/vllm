#! /bin/bash
model=Meta-Llama-3.1-8B-Instruct
port=8000
max_input_len=1024
max_output_len=1024
csv_path=/home/jovyan/vol-1/jh/benchmark_vllm_v061_0926
random_data=true
guided_json=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) port="$2"; shift ;;
        --max-input-len) max_input_len="$2"; shift ;;
        --max-output-len) max_output_len="$2"; shift ;;
        --csv-path) csv_path="$2"; shift ;;
        --random-data) random_data="$2"; shift ;;
        --guided-json) guided_json=true ;; # 플래그 옵션이므로 값을 받지 않고 바로 true로 설정
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python openai_serving.py \
    --port $port \
    --tokenizer-path /home/jovyan/vol-1/models/$model \
    --dataset-path /home/jovyan/vol-1/datasets/gaudi/$model.$max_input_len\_sampled.pkl \
    --max-input-len $max_input_len \
    --csv-path $csv_path/$max_input_len\_random_data\_$random_data\_guided\_$guided_json.csv \
    --max-output-len $max_output_len \
    --llama3-prompt \
    --random-data $random_data \
    --num-query 1024 \
    $( [[ "$guided_json" == "true" ]] && echo "--guided-json" )