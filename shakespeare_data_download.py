
import os
import tiktoken
from transformers import AutoTokenizer
import requests
from tqdm import tqdm
import numpy as np

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "dev/data/tinyshakespeare/tinyshakespeare")


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

def write_evalfile(filename, datas):
    header = np.zeros(256, dtype = np.int32)
    header[0] = 20240522
    header[1] = 1
    header[2] = len(datas)
    header[3] = 0
    longest_example_bytes = 0
    full_stream = []
    for idx, data in enumerate(datas):
        stream = []
        stream.append(2**16-1)
        stream.append(0)
        stream.append(idx)
        stream.append(data['label'])
        ending_tokens = data['ending_tokens']
        stream.append(len(ending_tokens))
        ctx_tokens = data['ctx_tokens']
        stream.append(len(ctx_tokens))
        stream.append(ctx_tokens)
        for end_token in ending_tokens:
            stream.append(len(end_tokens))
            stream.append(end_tokens)
        nbytes = len(stream) * 2
        stream[1] = nbytes
        longest_example_bytes = max(longest_example_bytes, nbytes)
        full_stream.extend(stream)
    stream_np = np.array(full_stream, dtype = np.uint16)
    header[3] = longest_example_bytes
    print(f"writing {len(datas):,} examples to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(stream_np.tobytes())

def download():
    os.makedirs(DATA_CACHE_DIR, exist_ok = True)
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"File {data_filename} already exists")

def tokenize():
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>']
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    text = open(data_filename, 'r').read()
    sections = text.split("\n\n")
    tokens = []
    for i, s in enumerate(sections):
        tokens.append(eot)
        spad = s + "\n\n" if i != len(sections) - 1 else s
        tokens.extend(encode(spad))
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    val_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_tokens)
    write_datafile(train_filename, train_tokens)
    
if __name__ == "__main__":
    download()
    tokenize()
    