"""Llama shaped decoder, written from scratch. Every weight starts random.

RMSNorm, rotary positions, grouped query attention, SwiGLU, tied embeddings.
Llama shaped on purpose: llama.cpp runs this architecture on a phone with no
extra work once exported to GGUF.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 16384
    dim: int = 640
    n_layers: int = 10
    n_heads: int = 10
    n_kv_heads: int = 5
    ffn: int = 1728
    seq: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self):
        return self.dim // self.n_heads


# hard cap from the user: at most 80 MB on the phone. Q8_0 is about 1.06 bytes
# per parameter, so 56m is 59 MB and 72m is 77 MB. Nothing bigger is defined.
SIZES = {
    "56m": Config(dim=640, n_layers=10, n_heads=10, n_kv_heads=5, ffn=1728),
    "72m": Config(dim=768, n_layers=10, n_heads=12, n_kv_heads=4, ffn=1920),
}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).type_as(x)


def rope_cache(seq, head_dim, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq, device=device).float()
    freqs = torch.outer(t, inv)                       # seq, head_dim/2
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: B, H, T, D   rotate pairs (x[..., :D/2], x[..., D/2:]) the way llama does
    T = x.size(2)
    c, s = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.kv, self.d = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.d).transpose(1, 2)
        k = self.wk(x).view(B, T, self.kv, self.d).transpose(1, 2)
        v = self.wv(x).view(B, T, self.kv, self.d).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.kv != self.h:
            rep = self.h // self.kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).reshape(B, T, self.h * self.d))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn, bias=False)   # gate
        self.w3 = nn.Linear(cfg.dim, cfg.ffn, bias=False)   # up
        self.w2 = nn.Linear(cfg.ffn, cfg.dim, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.ffn_norm(x))


class Llama(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        cos, sin = rope_cache(cfg.seq, cfg.head_dim, cfg.rope_theta, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():            # scale residual projections
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        x = self.norm(x)
        logits = F.linear(x, self.tok_emb.weight)         # tied output head
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=64, temperature=0.8, top_k=40):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.seq:])
            logits = logits[:, -1].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(logits.softmax(-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
