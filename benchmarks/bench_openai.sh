#! /bin/bash
# model=Meta-Llama-3-8B-Instruct
model=Meta-Llama-3.1-70B-Instruct
port=9000
max_input_len=4096

python openai_serving.py \
    --port $port \
    --tokenizer-path /home/work/.models/$model \
    --max-input-len $max_input_len \
    --max-output-len 4096 \
    --llama3-prompt \
    --random-data true \
    --num-query 1024

# --dataset-path /home/sdp/works/datasets/$model.$max_input_len\_sampled.pkl \