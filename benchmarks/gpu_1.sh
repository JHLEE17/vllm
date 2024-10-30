port=8001
max_input_len=2048

# bash bench_sqzb.sh --port $port --max-input-len 1024 --max-output-len 1024 --random-data

# Ogirinal, Guided Json
bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --random-data
sleep 30
bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --random-data
sleep 30
bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --guided-json --random-data
sleep 30
bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --guided-json --random-data
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --guided-json
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --guided-json

# # Multi-LoRA
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --enable-lora --random-data
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --enable-lora --random-data
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --enable-lora
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --enable-lora


# # Prefix Caching
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --prefix-caching --random-data
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --prefix-caching --random-data
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --prefix-caching
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --prefix-caching

# # KV fp8
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --kv-fp8 --random-data
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --kv-fp8 --random-data
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --kv-fp8
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --kv-fp8


# Multi-LoRA & Prefix-Caching & Guided-Json
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --enable-lora --guided-json --prefix-caching --random-data
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --enable-lora --guided-json --prefix-caching --random-data
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1 --enable-lora --guided-json --prefix-caching
# sleep 30
# bash bench_sqzb.sh --port $port --max-input-len $max_input_len --max-output-len 1024 --enable-lora --guided-json --prefix-caching



# bash bench_sqzb.sh --port $port --max-input-len 1024 --max-output-len 1 --guided-json --random-data
# bash bench_sqzb.sh --port $port --max-input-len 1024 --max-output-len 1024 --guided-json --random-data


# bash bench_sqzb.sh --port $port --max-input-len 1024 --max-output-len 1
# bash bench_sqzb.sh --port $port --max-input-len 1024 --max-output-len 1024

# bash bench_sqzb.sh --port $port --max-input-len 1024 --max-output-len 1 --guided-json
# bash bench_sqzb.sh --port $port --max-input-len 1024 --max-output-len 1024 --guided-json