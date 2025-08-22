import random
import os
import json
import requests
import tqdm
import numpy as np

from json import JSONDecodeError
from typing import Optional, List
from transformers import PreTrainedTokenizerBase


SHAREGPT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
ASSISTANT_SUFFIX = "Assistant:"


class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int

def is_file_valid_json(path):
    if not os.path.isfile(path):
        return False

    try:
        with open(path) as f:
            json.load(f)
        return True
    except JSONDecodeError as e:
        print(
            f"{path} exists but json loading fails ({e=}), thus treat as invalid file"
        )
        return False

def remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text

def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if is_file_valid_json(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
) -> List[DatasetRow]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if not is_file_valid_json(dataset_path) and dataset_path == "":
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[DatasetRow] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            DatasetRow(prompt=prompt, prompt_len=prompt_len, output_len=output_len)
        )

    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset

def generate_workload(num_requests=100):
    random.seed(0)
    tasks = []
    # Define bursty intervals for high concurrency
    bursts = [
        (0.0, 1.0, int(0.2 * num_requests)),   # 20% of requests in [0, 1) sec
        (5.0, 1.0, int(0.2 * num_requests)),   # 20% in [5, 6) sec
        (10.0, 1.0, int(0.3 * num_requests)),  # 30% in [10, 11) sec
        (15.0, 1.0, num_requests - int(0.2*num_requests)*2 - int(0.3*num_requests))  
        # remaining in [15, 16) sec
    ]
    for start, duration, count in bursts:
        for _ in range(count):
            arrival = start + random.random() * duration
            prompt_length = random.randint(50, 300)
            output_length = random.randint(1, 50)
            tasks.append({
                'arrival_time': arrival,
                'prompt_length': prompt_length,
                'output_length': output_length
            })
    tasks.sort(key=lambda x: x['arrival_time'])
    return tasks

if __name__ == "__main__":
    sample_sharegpt_requests(
        dataset_path=SHAREGPT_URL,
        num_requests=100,
        tokenizer=tokenizer,
        fixed_output_len=50,
        context_len=1024,
        prompt_suffix="",
        apply_chat_template=False
    )