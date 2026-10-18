

import os
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class NewGELU(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

FLASH = 0

class CasualSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        if FLASH:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList([ Block(config) for _ in range(config.n_layer) ]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1
        self.transformer.wte.weight = self.lm_head.weight
        self.init_rng = torch.Generator()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02 / math.sqrt(2 * self.config.n_layer)
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets = None, return_logits = True):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0,t, dtype=torch.long, device = device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s"  % model_type)
        config_args = {
            'gpt2':         dict(n_layer = 12, n_head = 12, n_embd = 768),
            'gpt2-medium':  dict(n_layer = 24, n_head = 16, n_embd = 1024),
            'gpt2-large':   dict(n_layer = 36, n_head = 20, n_embd = 1280),
            'gpt2-xl':      dict(n_layer = 48, n_head = 25, n_embd = 1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias') ]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameters tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameters tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")

        if zero_stage == 1:
            print0(f"using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(
                **optim_groups[0], 
                optimizer_class=torch.optim.AdamW,
                lr = learning_rate,
                betas = betas,
                fused = use_fused
            )
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = betas, fused = use_fused)
        
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat( (idx, idx_next), dim = 1 )
        return idx


def _peek_data_shard(filename):
    with open(filename, 'rb') as f:
        header = np.frombuffer(f.read(256 * 4), dtype = np.int32)
    
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file")
        exit(1)

    ntok = header[2]
    return ntok

def _load_data_shard(filename):
    with open(filename, 'rb') as f:
        header = np.frombuffer(f.read(256 * 4), dtype = np.int32)
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype = np.uint16)
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {self.ntok_total:,} across {len(self.files)} files.")
        self.current_shard = None
        self.reset()

    def reset(self):
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype = torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T*self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    t = t.view(torch.int16)
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file, dtype):
    write_fun = write_fp32 if dtype == 'float32' else write_bf16
    write_fun(model_tensors['transformer.wte.weight'], file)
    write_fun(model_tensors['transformer.wpe.weight'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.ln_1.weight'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.ln_1.bias'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.attn.c_attn.weight'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.attn.c_attn.bias'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.attn.c_proj.weight'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.attn.c_proj.bias'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.ln_2.weight'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.ln_2.bias'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.mlp.c_fc.weight'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.mlp.c_fc.bias'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.mlp.c_proj.weight'], file)
    for i in range(L):
        write_fun(model_tensors[f'transformer.h.{i}.mlp.c_proj.bias'], file)
    write_fun(model_tensors['transformer.ln_f.weight'], file)
    write_fun(model_tensors['transformer.ln_f.bias'], file)

@torch.no_grad()
def pad_vocab(tensor, multiple = 128, value = 0):
    V, C = tensor.shape
    Vp = (( V + multiple - 1) // multiple) * multiple
    pad_rows = Vp - V
    padded = tensor if pad_rows == 0 else F.pad(tensor, (0,0,0,pad_rows), value = value)
    return padded

def write_model(model, filename, dtype):
    version = {
        'float32': 3,
        'bfloat16': 5
    }[dtype]
    header = torch.zeros(256, dtype = torch.int32)
    header[0] = 20240326
    header[1] = version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    params = {name: param.cpu() for name, param in model.named_parameters()}
    wte = params["transformer.wte.weight"]
    wte_padded = pad_vocab(wte)
    params["transformer.wte.weight"] = wte_padded
    print(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    header[7] = wte_padded.size(0)
    with open(filename, 'wb') as file:
        file.write(header.numpy().tobytes())
        write_tensors(params, model.config.n_layer, file, dtype)
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    header = torch.zeros(256, dtype = torch.int32)
    header[0] = 20240327
    header[1] = 2
    header[2] = x.size(0)
    header[3] = x.size(1)
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    wte_grad = grads['transformer.wte.weight']
    wte_grad_padded = pad_vocab(wte_grad, value = 0)
    grads['transformer.wte.weight'] = wte_grad_padded
    print(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
    with open(filename, 'wb') as file:
        file.write(header.numpy().tobytes())
        file.write(x.cpu().numpy().astype("int32").tobytes())
        file.write(y.cpu().numpy().astype("int32").tobytes())
        write_fp32(logits.cpu(), file)
        write_fp32(loss.cpu(), file)
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

def write_tokenizer(enc, filename):
    n = enc.max_token_value + 1
    header = torch.zeros(256, dtype = torch.int32)
    header[0] = 20240328
    header[1] = 2
    header[2] = n
    header[3] = enc.eot_token
    with open(filename, 'wb') as file:
        file.write(header.numpy().tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            file.write(struct.pack("<B", length))
            file.write(b)
    print(f"wrote {filename}")

def print0(*args, **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    import time
    import argparse
    import tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_bin", type=str, default="dev/data/tinystories/tinystories_train.bin", help="input .bin to train on")
#    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_train.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="dev/data/tinystories/tinystories_val.bin", help="input .bin to eval validation loss on")
#    parser.add_argument("--input_val_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=3000, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=5, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=10, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=int, default=1, help="write tensors to disk")
    args = parser.parse_args()


    B, T = args.batch_size, args.sequence_length
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        init_process_group(backend = 'nccl')
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = 0
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        if args.device:
            device = args.device
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = 'mps'
    print(f"using devide: {device}")
    device_type = "cuda" if device == "cuda" else "cpu"
    tokens_per_fwdbwd = B * T * ddp_world_size
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type = device_type, dtype = ptdtype) if device_type == "cuda" else nullcontext()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    if args.tensorcores:
        torch.set_float32_manual_precision('high')
    FLASH = args.flash
    
    enc = tiktoken.get_encoding('gpt2')
    if master_process and args.write_tensors:
        write_tokenizer(enc, 'gpt2_tokenizer.bin')

    if args.model[0] == "d":
        model_config = {
            'd12': GPTConfig(block_size = 1024, vocab_size = 50257, n_layer = 12, n_head = 12, n_embd = 768),
            'd24': GPTConfig(block_size = 1024, vocab_size = 50257, n_layer = 24, n_head = 16, n_embd = 1024),
            'd36': GPTConfig(block_size = 1024, vocab_size = 50257, n_layer = 36, n_head = 20, n_embd = 1280),
            'd48': GPTConfig(block_size = 1024, vocab_size = 50257, n_layer = 48, n_head = 25, n_embd = 1600)
        }[args.model]
        model = GPT(model_config)
    else:
        model = GPT.from_pretrained(args.model)
    model.train()
    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descente_tuning = True 
        print0("compiling model ...")
        model = torch.compile(model)

    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None

    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    if master_process and args.write_tensors and (not args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        model_to_size = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M" }
        model_to_size.update({f"d{d}": f"d{d}" for d in [12,24,36,48]})
        model_size_str = model_to_size[args.model]
        write_model(model, f"gpt2_{model_size_str}.bin", dtype = "float32")
        write_model(model, f"gpt2_{model_size_str}_bf16.bin", dtype = "bfloat16")
        write_state(model, x, y, logits, loss, f"gpt2_{model_size_str}_debug_state.bin")
        train_loader.reset()

    if ddp:
        model = DDP(model, device_ids = [ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay = args.weight_decay,
        learning_rate = args.learning_rate,
        betas = (0.9, 0.95),
        device_type = device,
        zero_stage = zero_stage
    )

    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        if it > args.num_iterations:
            return min_lr
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate + min_lr)

    logfile = None

    if args.output_dir:
        os.makedirs(args.output_dir, exists_ok = True)
        logfile = os.path.join(args.output_dir, "main.log")
        with open(logfile, "w"):
            pass
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)
        if (args.val_loss_every > 0 and (step % args.val_loss_every == 0 or last_step)):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits = False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))
        if (args.sample_every > 0 and (step % args.sample_every == 0 or last_step )) and master_process:
            model.eval()
            start_ids = [enc.eot_token]
            xg = (torch.tensor(start_ids, dtype = torch.long, device = device)[None, ...])
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature = temperature, top_k = top_k)
            print0("----------------")
            print0(enc.decode(yg[0].tolist()))
            print0("----------------")

        if last_step:
            break

        model.train()
        optimizer.zero_grad(set_to_none = True)
        if args.overfit_single_batch:
            train_loader.reset()
        lossf = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with ctx:
                _, loss = model(x, y, return_logits = False)
                loss = loss / grad_accum_steps
                lossf += loss.detach()
            if not args.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op = dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1 - t0)
        print0(f"Step {step+1:4d} / {args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1 - t0)* 1000:.2f} ms | {tokens_per_second:.0f} tok/s)")

        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1 - t0)

    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings) * 1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if ddp:
        destroy_process_group()
           

            






                









