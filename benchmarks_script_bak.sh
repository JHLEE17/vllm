GPU=0
port=8000
max_input_len=1024
decode=false
prefill=false

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) GPU="$2"; shift ;;
        --port) port="$2"; shift ;;
        --max-input-len) max_input_len="$2"; shift ;;
        --decode) decode=true ;;
        --prefill) prefill=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

max_model_len=$((max_input_len + 1024))

wait_for_server() {
    local port=$1  # Accept the port as an argument
    # wait for vllm server to start
    # return 1 if vllm server crashes
    timeout 60s bash -c "
        until curl -s localhost:$port/v1/completions; do
            sleep 1
        done" && return 0 || return 1
}

kill_gpu_processes() {
    # Get the process IDs using the specific GPU
    gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i $GPU | tr -d '[:space:]')

    # Kill the processes on the specific GPU
    if [ -n "$gpu_processes" ]; then
        echo "Killing processes on GPU $GPU: $gpu_processes"
        for pid in $gpu_processes; do
            kill -9 $pid
        done
    else
        echo "No processes found on GPU $GPU"
    fi

    sleep 10

    # Print the GPU memory usage
    # so that we know if all GPU processes are killed.
    gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
    # The memory usage should be 0 MB.
    echo "GPU $GPU Memory Usage: $gpu_memory_usage MB"
}

# Original
bash openai_server.sh --port $port --max-model-len $max_model_len &
wait_for_server

bash benchmarks/client.sh --port $port --max-input-len $max_input_len \
$( [[ "$prefill" == "true" ]] && echo "--prefill" ) $( [[ "$decode" == "true" ]] && echo "--decode" )


#Guided-json
bash openai_server.sh --port $port --max-model-len $max_model_len &
wait_for_server

bash benchmarks/client.sh --port $port --max-input-len $max_input_len --guided-json --warmup\
$( [[ "$prefill" == "true" ]] && echo "--prefill" ) $( [[ "$decode" == "true" ]] && echo "--decode" )
kill_gpu_processes


# Multi-LoRA
bash openai_server.sh --port $port --max-model-len $max_model_len --enable-lora &
wait_for_server

bash benchmarks/client.sh --port $port --max-input-len $max_input_len --enable-lora \
$( [[ "$prefill" == "true" ]] && echo "--prefill" ) $( [[ "$decode" == "true" ]] && echo "--decode" )
kill_gpu_processes


# Prefix Caching
bash openai_server.sh --port $port --max-model-len $max_model_len --prefix-caching &
wait_for_server

bash benchmarks/client.sh --port $port --max-input-len $max_input_len --prefix-caching \
$( [[ "$prefill" == "true" ]] && echo "--prefill" ) $( [[ "$decode" == "true" ]] && echo "--decode" )
kill_gpu_processes


# ALL
bash openai_server.sh --port $port --max-model-len $max_model_len --enable-lora --prefix-caching &
wait_for_server

bash benchmarks/client.sh --port $port --max-input-len $max_input_len --enable-lora --prefix-caching --guided-json --warmup \
$( [[ "$prefill" == "true" ]] && echo "--prefill" ) $( [[ "$decode" == "true" ]] && echo "--decode" )
kill_gpu_processes