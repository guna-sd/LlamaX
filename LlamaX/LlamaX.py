''' Re-Implementation of Llama2 '''

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


@dataclass
class ModelArgs:
# Deafault parameters for Llama 42M
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    vocab_size: int = 32000
    hidden_dim: int = 1376
    multiple_of: int = 32
    norm_eps: float = 1e-05
    max_seq_len: int = 1024
    dropout: float = 0.0

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
