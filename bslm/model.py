"""A transformer encoder written from scratch, plus the two task heads.

No pretrained weights, no huggingface. Every parameter in this model starts
from random init and is learned on our own synthetic corpus.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)   # B,h,T,dk
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # B,h,T,T
        att = att.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, pad_mask):
        x = x + self.attn(self.n1(x), pad_mask)
        x = x + self.ff(self.n2(x))
        return x


class BSLM(nn.Module):
    """Joint intent classifier and BIO slot tagger."""

    def __init__(self, vocab_size, n_intents, n_slot_tags, d_model=256, n_layers=6,
                 n_heads=4, d_ff=768, max_len=64, dropout=0.1):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, n_intents=n_intents,
                        n_slot_tags=n_slot_tags, d_model=d_model, n_layers=n_layers,
                        n_heads=n_heads, d_ff=d_ff, max_len=max_len, dropout=dropout)
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.case = nn.Embedding(3, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_intents))
        self.slot_head = nn.Linear(d_model, n_slot_tags)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, cases):
        B, T = ids.shape
        pad_mask = ids.ne(0)
        pos = torch.arange(T, device=ids.device)
        x = self.tok(ids) + self.pos(pos)[None] + self.case(cases)
        x = self.drop(x)
        for b in self.blocks:
            x = b(x, pad_mask)
        x = self.norm(x)
        # sentence vector: cls token plus masked mean, they complement each other
        cls = x[:, 0]
        m = pad_mask.unsqueeze(-1).float()
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return self.intent_head(cls + mean), self.slot_head(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
