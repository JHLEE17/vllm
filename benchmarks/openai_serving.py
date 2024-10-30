import aiohttp
import argparse
import asyncio
import csv
import json
import random
import time
from dataclasses import dataclass, fields, asdict
from typing import AsyncGenerator
import os

import numpy as np
import pandas as pd
import torch
import tqdm
import tqdm.asyncio
from transformers import AutoTokenizer
from datetime import datetime


@dataclass
class Result:
    num_input_tokens: int
    num_generated_tokens: int
    request_latency: float
    generation_latency: float
    TTFT: float
    TPOT: float
    text: str


# (input token len, num generated tokens, request latency, TTFT, TPOT)
RESULTS: list[Result] = []


def print_results():
    for filed in fields(Result):
        data = [asdict(result).get(filed.name) for result in RESULTS]

        max = np.max(data)
        min = np.min(data)
        mean = np.mean(data)
        median = np.median(data)
        percentile_90 = np.percentile(data, 90)
        percentile_99 = np.percentile(data, 99)

        print(f"{filed.name} :")
        print(f"\tmax: {max}")
        print(f"\tmin: {min}")
        print(f"\tmean: {mean}")
        print(f"\tmedian: {median}")
        print(f"\t90%: {percentile_90}")
        print(f"\t99%: {percentile_99}")


def create_request(
    dataset_path: str,
    random_data: bool,
    vocab_size: int,
    max_input_len: int,
    max_output_len: int,
    num_query: int,
    llama3_request: bool,
) -> list[tuple[list[int], int, int]]:
    if random_data:
        prompts_tok_ids = [
            torch.randint(0, vocab_size, (max_input_len,)).tolist()
            for _ in range(num_query)
        ]

    else:
        dataframe = pd.read_pickle(dataset_path)
        prompts_tok_ids = dataframe["tok_input"].to_list()

    requests = []
    llama3_prompt_len = 14
    llama3_pre = [128000, 128006, 9125, 128007, 271, 128009, 128006, 882, 128007, 271]
    llama3_post = [128009, 128006, 78191, 128007]
    for tok_ids in prompts_tok_ids:
        if llama3_request:
            if len(tok_ids) > max_input_len - llama3_prompt_len:
                new_tok_ids = (
                    llama3_pre
                    + tok_ids[: max_input_len - llama3_prompt_len]
                    + llama3_post
                )
            else:
                new_tok_ids = llama3_pre + tok_ids + llama3_post
            requests.append((new_tok_ids, len(new_tok_ids), max_output_len))
        else:
            requests.append((tok_ids, len(tok_ids), max_output_len))

    return requests


async def get_request(
    input_requests: list[tuple[list[int], int, int]],
    request_rate: float,
) -> AsyncGenerator[tuple[list[int], int, int], None]:
    input_requests = iter(input_requests)  # type: ignore
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        # interval = np.random.exponential(1.0 / request_rate)
        interval = 1.0 / request_rate
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    api_url: str,
    token_ids: list[int],
    input_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    ignore_eos: bool,
    stop_token_ids: list[int],
    guided_json: bool,
    writer: csv.DictWriter = None,
    csvfile = None,
    n: int = 1,
) -> None:
    request_start_time = time.perf_counter()
    headers = {"Content-Type": "application/json"}

    regex_template = "[Pp]ositive format |[Nn]egative format"
    json_template = {
        "$defs": {
            "SearchQuery": {
                "description": "Search query for the retrieval task.",
                "properties": {
                    "query_type": {
                        "description": "The type of query most effective for handling the retrieval task.",
                        "title": "Query Type",
                        "type": "string",
                    },
                    "query": {
                        "description": "A random user's search query.",
                        "title": "Query",
                        "type": "string",
                    },
                },
                "required": ["query_type", "query"],
                "title": "SearchQuery",
                "type": "object",
            }
        },
        "description": "A list of search queries anticipating a user looking for information from a given web page.",
        "properties": {
            "queries": {
                "description": "Brainstormed search queries for the given web page.",
                "items": {"$ref": "#/$defs/SearchQuery"},
                "title": "Queries",
                "type": "array",
            }
        },
        "required": ["queries"],
        "title": "Brainstorm",
        "type": "object",
    }

    # pload = {
    #     "model": "/models/Meta-Llama-3.1-8B-Instruct",
    #     "messages": [
    #         {"role": "user", "content": "Tell me about pablo picasso."}
    #     ],
    #     "n": n,
    #     "best_of": best_of,
    #     "use_beam_search": use_beam_search,
    #     "temperature": 0.0,
    #     "top_p": 1.0,
    #     "max_tokens": output_len,
    #     "ignore_eos": ignore_eos,
    #     "stop_token_ids": stop_token_ids,
    #     # "guided_regex": regex_template,
    #     "guided_json": json_template,
    # }

    pload = {
        "model": "/home/jovyan/vol-1/models/Meta-Llama-3.1-8B-Instruct",
        "prompt": token_ids,
        "n": n,
        "best_of": best_of,
        "use_beam_search": use_beam_search,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": output_len,
        "ignore_eos": ignore_eos,
        "stop_token_ids": stop_token_ids,
        # "guided_regex": regex_template,
    }
    if guided_json:
        pload["guided_json"] = json_template

    timeout = aiohttp.ClientTimeout(total=48 * 3600)
    first_token_arrival_time = None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                last_chunk = None
                async for chunk, _ in response.content.iter_chunks():
                    if last_chunk is None:
                        first_token_arrival_time = time.perf_counter()
                    last_chunk = chunk
            assert last_chunk is not None
            result = json.loads(last_chunk)
            # print(result, flush=True)
            num_generated_tokens = result["usage"]["completion_tokens"]
            break

    request_end_time = time.perf_counter()
    assert first_token_arrival_time

    request_latency = request_end_time - request_start_time
    generation_latency = request_end_time - first_token_arrival_time
    time_per_output_token = generation_latency / num_generated_tokens
    result = Result(
        num_input_tokens=input_len,
        num_generated_tokens=num_generated_tokens,
        request_latency=request_latency,
        generation_latency=generation_latency,
        TTFT=0,
        TPOT=time_per_output_token,
        # text=result["choices"][0]["message"]["content"]
        text=result["choices"][0]["text"],
    )
    if writer is not None and csvfile is not None:
        writer.writerow(asdict(result))
        csvfile.flush()
    RESULTS.append(result)


async def benchmark(
    api_url: str,
    input_requests: list[tuple[list[int], int, int]],
    best_of: int,
    use_beam_search: bool,
    ignore_eos: bool,
    request_rate: float,
    stop_token_ids: list[int],
    guided_json: bool,
    writer: csv.DictWriter = None,
    csvfile = None,
) -> None:
    tasks: list[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, input_len, output_len = request
        task = asyncio.create_task(
            send_request(
                api_url,
                prompt,
                input_len,
                output_len,
                best_of,
                use_beam_search,
                ignore_eos,
                stop_token_ids,
                guided_json,
                writer,
                csvfile,
            )
        )
        tasks.append(task)

    [await t for t in tqdm.asyncio.tqdm.as_completed(tasks)]

def get_unique_filepath(filepath: str) -> str:
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath

    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{ext}"
        counter += 1
    return new_filepath

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.tokenizer_path.strip("/").split("/")[-1]
    # if not os.path.exists(f"/home/sdp/works/taesu/results/{model_name}"):
    #     os.makedirs(f"/home/sdp/works/taesu/results/{model_name}")
    # if not os.path.exists(f"/home/sdp/works/taesu/results/{model_name}/{'random' if args.random_data else 'orca'}"):
    #     os.makedirs(f"/home/sdp/works/taesu/results/{model_name}/{'random' if args.random_data else 'orca'}")
    # with open(f"/home/sdp/works/taesu/results/{model_name}/{'random' if args.random_data else 'orca'}/input_{args.max_input_len}_output_{args.max_output_len}_qps_{args.request_rate}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv", mode="wt") as csvfile:
    csvfile = None
    writer = None
    # with open("/home/jovyan/vol-1/jh/results/doe.csv", mode="wt") as csvfile:
    if not args.disable_saving:
        csv_path = args.csv_path
        csvfile = open(csv_path, mode="wt")
        csv_path = get_unique_filepath(csv_path)
        field_names = [field.name for field in fields(Result)]
        writer = csv.DictWriter(csvfile, field_names)
        writer.writeheader()

    api_url = f"http://{args.host}:{args.port}/v1/completions"
    # api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    stop_token_ids = [tokenizer.eos_token_id]

    if eot_id := tokenizer.get_vocab().get("<|eot_id|>", None):
        stop_token_ids.append(eot_id)

    if args.random_data:
        stop_token_ids = []

    input_requests = create_request(
        args.dataset_path,
        args.random_data,
        tokenizer.vocab_size,
        args.max_input_len,
        args.max_output_len,
        args.num_query,
        args.llama3_prompt,
    )
    num_requests = len(input_requests)

    benchmark_start_time = time.perf_counter()

    asyncio.run(
        benchmark(
            api_url,
            input_requests,
            1,
            False,
            args.random_data,
            args.request_rate,
            stop_token_ids,
            args.guided_json,
            writer,
            csvfile,
        )
    )
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"# requests: {num_requests}")
    total_input_tokens = np.sum([result.num_input_tokens for result in RESULTS])
    print(f"Total input tokens: {total_input_tokens}")
    total_generated_tokens = np.sum(
        [result.num_generated_tokens for result in RESULTS]
    )
    print(f"Total generated tokens: {total_generated_tokens}")
    print(f"Total latency: {benchmark_time} s")

    # just use the last row to write summary
    if not args.disable_saving:
        writer.writerow(
            {
                "num_input_tokens": total_input_tokens,
                "num_generated_tokens": total_generated_tokens,
                "request_latency": benchmark_time,
            }
        )
        writer.writerow(
            {
                "TTFT": np.median([result.TTFT for result in RESULTS]),
                "TPOT": np.median([result.TPOT for result in RESULTS]),
            }
        )
        writer.writerow(
            {
                "TTFT": np.average([result.TTFT for result in RESULTS]),
                "TPOT": np.average([result.TPOT for result in RESULTS]),
            }
        )
        writer.writerow(
            {
                "TTFT": np.max([result.TTFT for result in RESULTS]),
                "TPOT": np.max([result.TPOT for result in RESULTS]),
            }
        )
    if csvfile:
        csvfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )

    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--random-data", type=bool, default=False)
    parser.add_argument("--max-input-len", type=int, choices=[1024, 2048, 4096, 8192])
    parser.add_argument("--max-output-len", type=int, default=1024)

    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mem-util", type=float, default=0.9)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--num-query", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--llama3-prompt", action="store_true", help="add llama3 instruct prompt"
    )
    parser.add_argument("--guided-json", action="store_true", help="enable guided json")
    parser.add_argument("--disable-saving", action="store_true", help="Disable saving results to file.")
    parser.add_argument("--csv-path", type=str, default="./openai_serving_results", help="Specify the path to save the CSV file.")
    args = parser.parse_args()
    main(args)