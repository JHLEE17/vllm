INPUT_LEN=128
OUTPUT_LEN=128
PREFIX_LEN=0

MODEL_DIR=/workspace/Meta-Llama-3-8B/

wait_for_server() {
    # wait for vllm server to start
    # return 1 if vllm server crashes
    timeout 1200 bash -c '
        until curl -s localhost:8000/v1/models; do
            sleep 1
        done' && return 0 || return 1
}

kill_gpu_processes() {
    # kill all processes on GPU.
    pkill pt_main_thread
    sleep 10

    # Print the GPU memory usage
    # so that we know if all GPU processes are killed.
    gpu_memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    # The memory usage should be 0 MB.
    echo "GPU 0 Memory Usage: $gpu_memory_usage MB"
}

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_DIR \
    --tokenizer $MODEL_DIR \
    --max-model-len 256 \
    &

wait_for_server

# run benchmark
python /workspace/vllm/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $MODEL_DIR \
    --num-prompts 1 \
    --dataset-name random --random-input-len 128 --random-output-len 128 # --random-prefix-len 0

kill_gpu_processes