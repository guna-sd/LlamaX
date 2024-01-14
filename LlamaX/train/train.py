import math
import os
import json
import time
import numpy as np
from contextlib import nullcontext
from datetime import datetime
from functools import partial
import torch
from .LlamaX import Transformer, ModelArgs


device ='cuda' if torch.cuda.is_available() else 'cpu'
out_dir :str ='./models/'
init_from : str = 'new' # or 'resume'
always_save_checkpoint : bool = True 
dtype : str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
compile : bool = False
eval_model : bool = False
log_interval : int = 1
running_mfu = -1.0

# Default hyperparameters for the LlamaX Mini models ---> Sparse attention implementation...
dim = 512
n_layers = 8
n_heads = 8
n_kv_heads = 8
vocab_size = 32000
max_seq_len: int = 1024
dropout: float = 0.0
block_size = 1024
bias : bool = False

# Train Parameter for the LlamaX Mini models
eval_interval = 2000 
eval_iters = 200
max_seq_len = 256
max_iters = 10000
max_lr = 5e-4
max_batch_size=48
gradient_accumulation_steps = 40
batch_size : int = 12
block_size = 1024
best_val_loss = 1e9

model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    max_seq_len=max_seq_len,
    dropout=dropout,
    bias=bias,
)

learning_rate = 6e-4 # max lr_rate
max_iters = 10000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 
decay_lr = True 
warmup_iters = 2000
lr_decay_iters = 600000 
min_lr = 6e-5 

train_data = np.memmap('prepare/train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('prepare/val.bin', dtype=np.uint16, mode='r')

def get_batch(split : str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def get_lr(iter):
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    if iter > lr_decay_iters:
        return min_lr
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def write_params(params: dict, path: str):
    with open(os.path.join(path,'params.json'), 'w') as file:
        json.dump(params, file)

def read_params(path: str) -> dict or None:
    try:
        with open(path, 'r') as file:
            params = json.load(file)
        return params
    except Exception as e:
        print(f"Error reading parameters from {path}: {e}")
        return None

def get_batch_size():
    pass 
  # not yet implemented ---> will be later...


class NanoLlamaTrainer:
    def __init__(self):
        self.config, self.model = self.init_model()
        self.optimizer = self.model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))        
 
    def init_model(self, init_from : str) -> Transformer:
        if init_from == 'new':
            print("Initializing New Model...")
            write_params(model_args, out_dir)
            config = ModelArgs(**model_args)
            model = Transformer(config)

        elif init_from == 'resume':
            print(f"Resuming training from {out_dir}...")
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            checkpoint = torch.load(ckpt_path, map_location=device)
            ckpt_model_args = checkpoint['model_args']
            for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
                model_args[k] = ckpt_model_args[k]
            config = ModelArgs(**model_args)
            model = Transformer(config)
            ckpt = checkpoint['model']
            prefix = "_orig_mod."
            ckpt = {key[len(prefix):]: value for key, value in ckpt.items() if key.startswith(prefix)}
            model.load_state_dict(ckpt)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            del ckpt
        return config, model.to(device)
    
    def save_ckpt(self, step:int = None) -> None:
        if step != None:
            ckpt_name = f'ckpt{step}.pt'
            ckpt = {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': step,
                    'best_val_loss': best_val_loss,
                    'config': self.config,
            }
            torch.save(obj=ckpt,f = os.path.join(out_dir,ckpt_name))
            print(f'{ckpt_name} model saved at {out_dir}...')

        else:
            ckpt_name = f'ckpt.pt'
            torch.save(obj=self.model.state_dict(),f = os.path.join(out_dir,ckpt_name))
            print(f'{ckpt_name} model saved at {out_dir}...')
    
    def compile(self):
        if compile:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(model)
        
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train(self):
        X, Y = get_batch('train')
        t0 = time.time()
        estimate_iter : int = 0
        iter_num : int = 0
        while True:
            learning_rate = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['learning_rate'] = learning_rate
            if iter_num % eval_iters == 0:
                loss = self.estimate_loss()
                print(f"step {iter_num}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")
                if loss['val'] < best_val_loss or always_save_checkpoint:
                    best_val_loss = loss['val']
                    if iter_num > 0:
                        self.save_ckpt(step=iter_num)
            if iter_num == 0 and eval_model:
                break

            for step in range(gradient_accumulation_steps):
                with ctx:
                    logits, loss = self.model(X,Y)
                    loss = loss / gradient_accumulation_steps
                X,Y = get_batch('train')
                self.scaler.scale(loss).backward()
            if grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0:
                lossf = loss.item() * gradient_accumulation_steps
                if estimate_iter >=5:
                    mfu = self.model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            estimate_iter += 1
            if iter_num > max_iters:
                print("training completed...")
                break
