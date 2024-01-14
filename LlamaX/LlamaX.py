''' Re-Implementation of Llama2 with SparseAttention'''
import torch
from torch import nn , Tensor
import torch.nn.functional as F
from dataclasses import dataclass
import math
from typing import Optional, List

@dataclass
class ModelArgs:
    # default hyperparameters for the LlamaX Mini models [42M]
    def __init__(self, **kwargs) -> None:
        self.dim: int = kwargs.get('dim', 512)
        self.n_embd : int = kwargs.get('dim', 512)
        self.n_layers: int = kwargs.get('n_layer', 8)
        self.n_heads: int = kwargs.get('n_heads', 8)
        self.n_kv_heads: int = kwargs.get('n_kv_heads', 8)
        self.vocab_size: int = kwargs.get('vocab_size', 32000)
        self.max_seq_len: int = kwargs.get('max_seq_len', 1024)
        self.dropout: float = kwargs.get('dropout', 0.0)
        self.topk: int = kwargs.get('topk', 24)
        self.use_block: bool = kwargs.get('use_block', False)
        self.block_size : int = kwargs.get('block_size', 1024)
        self.bias : bool = kwargs.get('bias',True)
        self.hidden_dim : int = kwargs.get('hidden_dim',1376)

class RMSNorm(torch.nn.Module):

    def __init__(self,dim: int,eps: float = 1e-6):
        """RMS Normaliation module

        Arguments:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class FeedForward(nn.Module):
    def __init__(self, args : ModelArgs):
        super().__init__()
        if args.hidden_dim is None:
            hidden_dim = 4 * args.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    
#TODO: Optimizing the Algoritm based on Original Papers from OPENAI 'https://arxiv.org/abs/1904.10509'
class SparseAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.topk = args.topk
        self.use_block_sparsity = args.use_block

        # creates triangular mask
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + self.mask[:, :, :seq_len, :seq_len]

        if self.use_block_sparsity:
            # applies block sparsity
            block_size = 128
            block_scores = []
            for i in range(0, seq_len, block_size):
                for j in range(0, seq_len, block_size):
                    block_score = scores[:, :, i:i + block_size, j:j + block_size]
                    block_scores.append(block_score)
            scores = torch.cat(block_scores, dim=-1)
        else:
            # applies top-k sparsity
            topk_scores, topk_indices = torch.topk(scores, min(self.topk, scores.size(-1)), dim=-1, largest=True)
            scores = torch.zeros_like(scores).scatter(-1, topk_indices, topk_scores)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, layer_id : int, args : ModelArgs):
        super().__init__()
        self.head_dim = args.dim // args.n_heads
        self.attention = SparseAttention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x,):
        h = x + self.attention.forward(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))

class Transformer(nn.Module):
    def __init__(self, args : ModelArgs,) -> None:
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = torch.nn.ModuleList()
        self.last_loss: Optional[torch.Tensor] = None

        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))
        self.norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens : Tensor, targets : Optional[Tensor]=None) -> Tensor:
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        if targets != None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),ignore_index=-1)
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None
        return logits
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
