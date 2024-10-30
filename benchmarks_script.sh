#!/bin/bash

GPU=0
port=8000
decode=false
prefill=false
dynamic_data=false
concurrency=None
max_num_seqs=128
num_requests=512

features=()
max_input_lens=()

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift ;;
        --port) port="$2"; shift ;;
        --concurrency) concurrency="$2"; shift ;;
        --max-input-len) max_input_lens+=("$2"); shift ;;
        --feature) features+=("$2"); shift ;;
        --decode) decode=true ;;
        --prefill) prefill=true ;;
        --dynamic-data) dynamic_data=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Function to find an available port
find_available_port() {
    local port=$1
    while lsof -i:$port >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo $port
}

# In your script
port=$(find_available_port $port)


# Default features if none specified
if [ ${#features[@]} -eq 0 ]; then
    features=("original" "guided-json" "multi-lora" "prefix-caching" "all")
fi

# Default max_input_lens if none specified
if [ ${#max_input_lens[@]} -eq 0 ]; then
    max_input_lens=(1024)
fi

wait_for_server() {
    local port=$1
    timeout 60s bash -c "
        until curl -s localhost:$port/v1/models; do
            sleep 1
        done" && return 0 || {
            echo "Server failed to start or crashed."
            return 1
        }
}


# GPU 프로세스를 종료하는 함수
kill_gpu_processes() {
    # 특정 포트를 사용하는 프로세스 종료
    port_pids=$(lsof -t -i:$port)
    if [ -n "$port_pids" ]; then
        echo "포트 $port에서 실행 중인 프로세스 종료: $port_pids"
        kill -15 $port_pids
        sleep 5
        # 아직 실행 중인 경우 강제 종료
        port_pids=$(lsof -t -i:$port)
        if [ -n "$port_pids" ]; then
            echo "포트 $port에서 실행 중인 프로세스 강제 종료: $port_pids"
            kill -9 $port_pids
        fi
    else
        echo "포트 $port에서 실행 중인 프로세스가 없습니다."
    fi

    sleep 10

    # GPU를 사용하는 프로세스 종료
    gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i $GPU)
    if [ -n "$gpu_processes" ]; then
        echo "GPU $GPU에서 실행 중인 프로세스 종료: $gpu_processes"
        for pid in $gpu_processes; do
            kill -9 $pid
        done
    else
        echo "GPU $GPU에서 실행 중인 프로세스가 없습니다."
    fi

    sleep 10

    # GPU 메모리 사용량 출력
    gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    echo "GPU $GPU 메모리 사용량: $gpu_memory_usage MB"
}


run_benchmark() {
    local feature="$1"
    local max_input_len="$2"

    local max_model_len=$((max_input_len + 1024))

    kill_gpu_processes

    echo "Running feature: $feature with max_input_len: $max_input_len"

    # Prepare options for openai_server.sh
    server_options="--port $port --max-model-len $max_model_len --max-num-seqs $max_num_seqs"

    # Prepare options for client.sh
    client_options="--port $port --max-input-len $max_input_len --num-requests $num_requests"

    # Conditionally add concurrency option if it's a valid integer
    if [[ "$concurrency" =~ ^[0-9]+$ ]]; then
        client_options+=" --concurrency $concurrency"
    fi

    # Add prefill and decode options
    [[ "$prefill" == "true" ]] && client_options+=" --prefill"
    [[ "$decode" == "true" ]] && client_options+=" --decode"
    [[ "$dynamic_data" == "true" ]] && client_options+=" --dynamic-data"

    case "$feature" in
        "original")
            # No additional options
            ;;
        "guided-json")
            client_options+=" --guided-json"
            ;;
        "multi-lora")
            server_options+=" --enable-lora"
            client_options+=" --enable-lora"
            ;;
        "prefix-caching")
            server_options+=" --prefix-caching"
            client_options+=" --prefix-caching"
            ;;
        "all")
            server_options+=" --enable-lora --prefix-caching"
            client_options+=" --enable-lora --prefix-caching --guided-json"
            ;;
        *)
            echo "Unknown feature: $feature"
            exit 1
            ;;
    esac

    # Ensure the logs directory exists
    mkdir -p ../logs/

    bash openai_server.sh $server_options > ../logs/Server_${max_input_len}_${feature}_prefill_${prefill}_decode_${decode}_dynamic_${dynamic_data}.log 2>&1 &
    wait_for_server $port

    bash benchmarks/client.sh $client_options #> ../logs/Client_${max_input_len}_${feature}_prefill_${prefill}_decode_${decode}_dynamic_${dynamic_data}.log 2>&1

    kill_gpu_processes
}

for max_input_len in "${max_input_lens[@]}"; do
    for feature in "${features[@]}"; do
        run_benchmark "$feature" "$max_input_len"
        sleep 30
    done
done
