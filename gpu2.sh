# bash benchmarks_script.sh \
# --max-input-len 1024 \
# --max-input-len 2048 \
# --max-input-len 4096 \
# --max-input-len 8192 \
# --feature prefix-caching \
# --gpu 2 --port 8002 \
# --prefill

# sleep 20

# bash benchmarks_script.sh \
# --max-input-len 1024 \
# --max-input-len 2048 \
# --max-input-len 4096 \
# --max-input-len 8192 \
# --feature prefix-caching \
# --gpu 2 --port 8002 \
# --decode

# sleep 20

# bash benchmarks_script.sh \
# --max-input-len 1024 \
# --max-input-len 2048 \
# --max-input-len 4096 \
# --max-input-len 8192 \
# --feature prefix-caching \
# --gpu 2 --port 8002 \
# --prefill --dynamic-data

# sleep 20

# bash benchmarks_script.sh \
# --max-input-len 1024 \
# --max-input-len 2048 \
# --max-input-len 4096 \
# --max-input-len 8192 \
# --feature prefix-caching \
# --gpu 2 --port 8002 \
# --decode --dynamic-data

# sleep 20



bash benchmarks_script.sh \
--max-input-len 4096 \
--max-input-len 8192 \
--feature multi-lora \
--gpu 0 --port 8000 \
--decode

bash benchmarks_script.sh \
--max-input-len 1024 \
--max-input-len 2048 \
--max-input-len 4096 \
--max-input-len 8192 \
--feature multi-lora \
--gpu 0 --port 8000 \
--prefill --dynamic-data

bash benchmarks_script.sh \
--max-input-len 1024 \
--max-input-len 2048 \
--max-input-len 4096 \
--max-input-len 8192 \
--feature multi-lora \
--gpu 0 --port 8000 \
--decode --dynamic-data