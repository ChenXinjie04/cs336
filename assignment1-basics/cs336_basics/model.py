import torch.nn as nn
import torch
from einops import einsum, rearrange
import math
from jaxtyping import Bool, Float
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weights, 0, std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "out_features in_features, ... in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embeddings, 0, 1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        l2_norm = einsum(x, x, "... d_model, ... d_model -> ...")
        sigma = l2_norm / self.d_model
        denom = (sigma + self.eps) ** 0.5
        denom = rearrange(denom, "... -> ... 1")
        result = (x / denom) * self.weights
        return result.to(in_type)


class SWiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.w1 = Linear(d_model, self.d_ff, device, dtype)
        self.w2 = Linear(self.d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, self.d_ff, device, dtype)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_in = self.w1.forward(x)
        gate_out = gate_in * self.sigmoid(gate_in)
        return self.w2.forward(gate_out * self.w3.forward(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        pair_idx = torch.arange(0, d_k // 2, device=device)
        exponent = pair_idx * -2 / d_k
        inv_freq = torch.exp(exponent * math.log(theta))
        m = torch.arange(0, max_seq_len, device=device)
        angles = torch.outer(m, inv_freq)
        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)
        self.register_buffer(
            "cos_table",
            cos_table,
            persistent=False,
        )
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = rearrange(x, "... (d two) -> ... d two", two=2)
        x_even = x[..., 0]
        x_odd = x[..., 1]
        cos = self.cos_table[token_positions].to(in_type)
        sin = self.sin_table[token_positions].to(in_type)
        x_even_out = x_even * cos - x_odd * sin
        x_odd_out = x_even * sin + x_odd * cos
        x = torch.stack([x_even_out, x_odd_out], dim=-1)
        return rearrange(x, "... pair pos -> ... (pair pos)")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_elem = torch.max(x, dim=dim, keepdim=True)[0]
    x = x - max_elem
    x = torch.exp(x)
    x_sum = torch.sum(x, dim=dim, keepdim=True)
    return x / x_sum


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        qk = qk.masked_fill(~mask, float("-inf"))
    qk = softmax(qk / (d_k**0.5), dim=-1)
    return einsum(qk, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, RoPE: RotaryPositionalEmbedding | None = None, device=None, dtype=None
    ):
        super().__init__()
        self.q_proj_weight = Linear(d_model, d_model, device, dtype)
        self.k_proj_weight = Linear(d_model, d_model, device, dtype)
        self.v_proj_weight = Linear(d_model, d_model, device, dtype)
        self.o_proj_weight = Linear(d_model, d_model, device, dtype)
        self.RoPE = RoPE
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.d_model = d_model

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        seq_len = x.shape[1]
        q = self.q_proj_weight.forward(x)
        k = self.k_proj_weight.forward(x)
        v = self.v_proj_weight.forward(x)
        q = rearrange(q, "batch seq (num_heads d_q) -> batch num_heads seq d_q", num_heads=self.num_heads)
        k = rearrange(k, "batch seq (num_heads d_k) -> batch num_heads seq d_k", num_heads=self.num_heads)
        v = rearrange(v, "batch seq (num_heads d_v) -> batch num_heads seq d_v", num_heads=self.num_heads)
        if self.RoPE is not None and token_positions is not None:
            q = self.RoPE.forward(q, token_positions)
            k = self.RoPE.forward(k, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool))
        v = scaled_dot_product_attention(q, k, v, mask)
        v = rearrange(v, "... heads seq d_v -> ... seq (heads d_v)")
        return self.o_proj_weight.forward(v)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        RoPE: RotaryPositionalEmbedding | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.multihead_self_attention = MultiheadSelfAttention(d_model, num_heads, RoPE, device, dtype)
        self.position_wise_ff = SWiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        token_positions = torch.arange(0, x.shape[1], dtype=torch.int)
        x1 = self.norm1.forward(x)
        x1 = self.multihead_self_attention.forward(x1, token_positions)
        x = x + x1
        x2 = self.norm2(x)
        x2 = self.position_wise_ff.forward(x2)
        x = x + x2
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        d_k = d_model // num_heads
        self.RoPE = RotaryPositionalEmbedding(theta, d_k, context_length, device, dtype)
        self.embedding = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, self.RoPE, device, dtype) for _ in range(num_layers)]
        )
        self.ln = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        x = self.embedding.forward(in_indices)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.ln.forward(x)
        x = self.linear(x)
        return x
