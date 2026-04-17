"""
model/gpt.py — GPTModel, straight from Notebook 3.
No changes — swap in your own architecture here if you want.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var  = x.var(-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / (var + 1e-5).sqrt() + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.d_out     = d_out
        self.W_query   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key     = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj  = nn.Linear(d_out, d_out)
        self.dropout   = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        b, T, _ = x.shape
        q = self.W_query(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5
        scores.masked_fill_(self.mask[:T, :T], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(b, T, self.d_out)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att  = MultiHeadAttention(
            config["emb_dim"], config["emb_dim"],
            config["context_length"], config["drop_rate"],
            config["n_heads"], config["qkv_bias"],
        )
        self.ff    = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop  = nn.Dropout(config["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb    = nn.Embedding(config["vocab_size"],     config["emb_dim"])
        self.pos_emb    = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop       = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head   = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        x    = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))
