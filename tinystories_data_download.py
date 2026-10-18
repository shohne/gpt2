import argparse
import os
import glob
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import requests
import tiktoken
from tqdm import tqdm
from transformers import AutoTokenizer


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "dev/data/tinystories")

def download_file(url: str, fname: str, chunk_size = 1024):
    resp = requests.get(url, stream = True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc = fname,
        total = total,
        unit = "iB",
        unit_scale = True,
        unit_divisor = 1024
    ) as bar:
        for data in resp.iter_content(chunk_size = chunk_size):
            size = file.write(data)
            bar.update(size)

HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint32,
    }
}

def write_datafile(filename, toks):
    info = HEADERS_INFO['gpt-2']
    header = np.zeros(256, dtype = np.int32)
    header[0] = info['magic']
    header[1] = info['version']
    header[2] = len(toks)
    toks_np = np.array(toks, dtype = info['token_dtype'])
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the gpt-2 format")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())



def download():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "tinystories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename} ... ")
        download_file(data_url, data_filename)
    else:
        print(f"Found {data_filename} ... ")

    data_dir = os.path.join(DATA_CACHE_DIR, "tinystories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok = True)
        print(f"Unpacking {data_filename}  ... ")
        os.system(f"tar -xvf {data_filename} -C {data_dir}")
    else:
        print(f"Found {data_dir} ... ")

    shard_filename = sorted(glob.glob(os.path.join(data_dir, "*.json)")))
    print("Download done")
    print(f"Number of shards: {len(shard_filename)}")

def process_shard(shard_index, shard_filename):
    enc = tiktoken.get_encoding('gpt2')
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>']
    with open(shard_filename, 'r') as f:
        data = json.load(f)
    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    for example in data:
        text = example["story"]
        text = text.strip()
        tokens = encode(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    return all_tokens

def tokenize():
    data_dir = os.path.join(DATA_CACHE_DIR, "tinystories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    for split_name, split_shards in [("val", val_shards), ("train", train_shards)]:
        print(f"Tokenizing {split_name} ... ")
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_shard, shard_index, shard_filename) for shard_index, shard_filename in enumerate(split_shards)]
            for future in as_completed(futures):
                all_tokens.extend(future.result())
        split_filename = os.path.join(DATA_CACHE_DIR, f"tinystories_{split_name}.bin")
        write_datafile(split_filename, all_tokens)

if __name__ == "__main__":
    download()
    tokenize()




