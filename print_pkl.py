import pickle

file_dir = '/home/jovyan/vol-1/jh/vllm/'

N = 20

file_path = file_dir + "Meta-Llama-3.1-8B-Instruct_qps_inf_total_101.40829830989242_in_786432_out_131072_random_1024_max_max_num_batched_tokens1024.pkl"
# file_path = file_dir + "Meta-Llama-3.1-8B-Instruct_qps_inf_total_6.967689549550414_in_7680_out_100_random_10_max_max_num_batched_tokens2048.pkl"
# file_path = file_dir + "Meta-Llama-3.1-8B-Instruct_qps_inf_total_7.654746642336249_in_7680_out_100_random_10_max_max_num_batched_tokens4096.pkl"
# file_path = file_dir + "Meta-Llama-3.1-8B-Instruct_qps_inf_total_6.443940052762628_in_7680_out_100_random_10_max_max_num_batched_tokens8192.pkl"

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# print(data)

print(data[:20])
print(data[-20:])
