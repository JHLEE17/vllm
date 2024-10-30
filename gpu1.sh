# bash benchmarks_script.sh \
# --max-input-len 1024 \
# --max-input-len 2048 \
# --max-input-len 4096 \
# --max-input-len 8192 \
# --feature guided-json \
# --gpu 1 --port 8001 \
# --prefill

# sleep 20

# bash benchmarks_script.sh \
# --max-input-len 1024 \
# --max-input-len 2048 \
# --max-input-len 4096 \
# --max-input-len 8192 \
# --feature guided-json \
# --gpu 1 --port 8001 \
# --decode

# sleep 20

bash benchmarks_script.sh \
--max-input-len 1024 \
--max-input-len 2048 \
--max-input-len 4096 \
--max-input-len 8192 \
--feature all \
--gpu 1 --port 8001 \
--prefill --dynamic-data

sleep 20

bash benchmarks_script.sh \
--max-input-len 1024 \
--max-input-len 2048 \
--max-input-len 4096 \
--max-input-len 8192 \
--feature all \
--gpu 1 --port 8001 \
--decode --dynamic-data

sleep 20