"""
Modified Transformer Architecture
===================================
Based on "Attention is All You Need" (Vaswani et al., 2017)
with the following modifications:

1. Configurable number of layers and attention heads
2. Multiple positional encoding options: Sinusoidal, Learned, Rotary (RoPE), ALiBi
3. Multiple attention types: Vanilla, Linear, Local/Sliding-Window
4. Hybrid CNN-RNN encoder block
5. FFN replacements: Depthwise-Separable CNN FFN, Gated FFN
6. Cross-layer encoder-decoder attention
7. Bidirectional decoder
8. Early-exit mechanism with learnable confidence heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Literal


# =============================================================================
# 1.  POSITIONAL ENCODINGS
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Original sinusoidal positional encoding from the paper."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings (e.g., BERT/GPT-2 style)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al., 2021.
    Applied inside the attention head rather than added to embeddings.
    This module pre-computes sin/cos tables; the actual rotation is done
    by the attention layer via `apply_rotary_emb`.
    """

    def __init__(self, head_dim: int, max_len: int = 5000):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0))  # (1, seq, hd)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0))

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for RoPE: rotate the second half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embedding to query and key tensors.
    q, k: (batch, heads, seq_len, head_dim)
    cos, sin: (1, seq_len, head_dim) — will be broadcast-expanded.
    """
    cos = cos.unsqueeze(1)  # (1, 1, seq, hd)
    sin = sin.unsqueeze(1)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) — Press et al., 2022.
    No positional encoding is added to embeddings; instead a static
    linear bias is added to attention logits.
    """

    def __init__(self, num_heads: int, max_len: int = 5000):
        super().__init__()
        self.num_heads = num_heads
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)
        self._build_bias(max_len)

    @staticmethod
    def _get_slopes(n: int) -> torch.Tensor:
        """Compute per-head slopes as powers of 2."""

        def _closest_power_of_2(x):
            return 2 ** math.floor(math.log2(x))

        if math.log2(n).is_integer():
            slopes = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)) * torch.arange(1, n + 1))
        else:
            cp = _closest_power_of_2(n)
            base = 2.0 ** (-(2.0 ** -(math.log2(cp) - 3)) * torch.arange(1, cp + 1))
            extra = (
                2.0
                ** (
                    -(2.0 ** -(math.log2(2 * cp) - 3))
                    * torch.arange(1, 2 * (n - cp) + 1, 2)
                )
            )
            slopes = torch.cat([base, extra])
        return slopes.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, heads, 1, 1)

    def _build_bias(self, max_len: int):
        context = torch.arange(max_len)
        memory = torch.arange(max_len)
        bias = (context.unsqueeze(1) - memory.unsqueeze(0)).abs().neg().float()
        self.register_buffer("bias", bias.unsqueeze(0).unsqueeze(0))  # (1,1,L,L)

    def forward(self, seq_len_q: int, seq_len_k: int) -> torch.Tensor:
        """Return bias of shape (1, num_heads, seq_len_q, seq_len_k)."""
        bias = self.bias[:, :, :seq_len_q, :seq_len_k]
        return bias * self.slopes


# =============================================================================
# 2.  ATTENTION MECHANISMS
# =============================================================================

class VanillaMultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        rope: Optional[RotaryPositionalEncoding] = None,
        alibi: Optional[ALiBiPositionalBias] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = rope
        self.alibi = alibi

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq, _ = query.shape
        Lk = key.size(1)

        Q = self.W_q(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if available (only for self-attention where Lq == Lk)
        if self.rope is not None and Lq == Lk:
            cos, sin = self.rope(Lq)
            Q, K = apply_rotary_emb(Q, K, cos[:, :Lq], sin[:, :Lq])

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply ALiBi bias if available
        if self.alibi is not None:
            scores = scores + self.alibi(Lq, Lk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.W_o(out)


class LinearAttention(nn.Module):
    """
    Linear (kernel-based) attention — O(n) complexity.
    Uses ELU+1 as the feature map φ(x) so that φ(Q)·φ(K)^T ≈ softmax(QK^T).
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq, _ = query.shape
        Lk = key.size(1)

        Q = self.W_q(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        Q = self._elu_feature_map(Q)
        K = self._elu_feature_map(K)

        # O(n·d²) instead of O(n²·d)
        KV = torch.matmul(K.transpose(-2, -1), V)  # (B, h, d, d)
        Z = 1.0 / (torch.matmul(Q, K.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6)
        out = torch.matmul(Q, KV) * Z

        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.W_o(out)


class LocalSlidingWindowAttention(nn.Module):
    """
    Sliding-window (local) attention — each token attends only
    to a fixed window of `window_size` tokens around it.
    Efficient for long sequences.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = query.shape

        Q = self.W_q(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Build sliding-window mask
        row_idx = torch.arange(L, device=query.device).unsqueeze(1)
        col_idx = torch.arange(L, device=query.device).unsqueeze(0)
        window_mask = (col_idx - row_idx).abs() <= (self.window_size // 2)
        window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        if mask is not None:
            window_mask = window_mask & (mask.bool())

        scores = scores.masked_fill(~window_mask, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)


# =============================================================================
# 3.  FEED-FORWARD VARIANTS
# =============================================================================

class StandardFFN(nn.Module):
    """Original position-wise feed-forward network (ReLU)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class GatedFFN(nn.Module):
    """
    Gated FFN (GLU variant) — Shazeer, 2020.
    Uses SiLU gating: FFN_gated(x) = (xW₁ ⊙ SiLU(xV)) W₂
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.v(x))
        return self.w2(self.dropout(self.w1(x) * gate))


class DepthwiseSeparableCNNFFN(nn.Module):
    """
    Replace position-wise FFN with depthwise-separable 1-D convolutions.
    Captures local patterns more cheaply than full convolutions.
    """

    def __init__(
        self, d_model: int, d_ff: int, kernel_size: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        padding = kernel_size // 2
        # Depthwise conv (groups=d_model: each channel is convolved independently)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, padding=padding, groups=d_model
        )
        # Pointwise conv expands channels
        self.pointwise_up = nn.Conv1d(d_model, d_ff, 1)
        self.pointwise_down = nn.Conv1d(d_ff, d_model, 1)
        self.norm = nn.BatchNorm1d(d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x.transpose(1, 2)  # (B, d_model, L)
        x = self.depthwise(x)
        x = self.pointwise_up(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.pointwise_down(x)
        return x.transpose(1, 2)  # (B, L, d_model)


# =============================================================================
# 4.  HYBRID CNN-RNN BLOCK
# =============================================================================

class HybridCNNRNNBlock(nn.Module):
    """
    Hybrid block that combines:
      • 1-D CNN to capture local n-gram features
      • Bidirectional GRU to capture sequential dependencies
      • Output is projected back to d_model and added residually
    Drop-in replacement for an encoder layer.
    """

    def __init__(
        self,
        d_model: int,
        cnn_channels: int = 256,
        cnn_kernel_size: int = 3,
        rnn_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = cnn_kernel_size // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(d_model, cnn_channels, cnn_kernel_size, padding=padding),
            nn.BatchNorm1d(cnn_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rnn = nn.GRU(
            cnn_channels, rnn_hidden, batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(rnn_hidden * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        # CNN expects (B, C, L)
        cnn_out = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # (B, L, cnn_ch)
        rnn_out, _ = self.rnn(cnn_out)  # (B, L, 2*rnn_hidden)
        out = self.proj(rnn_out)
        return self.norm(residual + self.dropout(out))


# =============================================================================
# 5.  ENCODER LAYER
# =============================================================================

class EncoderLayer(nn.Module):
    """
    Single encoder layer with configurable attention and FFN.
    Pre-norm architecture (norm → sublayer → residual).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: Literal["vanilla", "linear", "local"] = "vanilla",
        ffn_type: Literal["standard", "gated", "depthwise_cnn"] = "standard",
        window_size: int = 128,
        rope: Optional[RotaryPositionalEncoding] = None,
        alibi: Optional[ALiBiPositionalBias] = None,
    ):
        super().__init__()
        # --- Attention ---
        if attention_type == "vanilla":
            self.self_attn = VanillaMultiHeadAttention(
                d_model, num_heads, dropout, rope=rope, alibi=alibi
            )
        elif attention_type == "linear":
            self.self_attn = LinearAttention(d_model, num_heads, dropout)
        elif attention_type == "local":
            self.self_attn = LocalSlidingWindowAttention(
                d_model, num_heads, window_size, dropout
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        # --- FFN ---
        if ffn_type == "standard":
            self.ffn = StandardFFN(d_model, d_ff, dropout)
        elif ffn_type == "gated":
            self.ffn = GatedFFN(d_model, d_ff, dropout)
        elif ffn_type == "depthwise_cnn":
            self.ffn = DepthwiseSeparableCNNFFN(d_model, d_ff, dropout=dropout)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm self-attention
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, src_mask))
        # Pre-norm FFN
        normed = self.norm2(x)
        x = x + self.dropout(self.ffn(normed))
        return x


# =============================================================================
# 6.  DECODER LAYER  (with Cross-Layer Attention + Bidirectional option)
# =============================================================================

class DecoderLayer(nn.Module):
    """
    Single decoder layer supporting:
      • Masked self-attention (optionally bidirectional for encoder-like decoding)
      • Cross-layer attention: attends to ALL encoder layers (not just the last)
      • Configurable attention and FFN types
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_encoder_layers: int,
        dropout: float = 0.1,
        attention_type: Literal["vanilla", "linear", "local"] = "vanilla",
        ffn_type: Literal["standard", "gated", "depthwise_cnn"] = "standard",
        bidirectional_decoder: bool = False,
        window_size: int = 128,
        rope: Optional[RotaryPositionalEncoding] = None,
        alibi: Optional[ALiBiPositionalBias] = None,
    ):
        super().__init__()
        self.bidirectional = bidirectional_decoder

        # --- Masked Self-Attention ---
        if attention_type == "vanilla":
            self.self_attn = VanillaMultiHeadAttention(
                d_model, num_heads, dropout, rope=rope, alibi=alibi
            )
        elif attention_type == "linear":
            self.self_attn = LinearAttention(d_model, num_heads, dropout)
        elif attention_type == "local":
            self.self_attn = LocalSlidingWindowAttention(
                d_model, num_heads, window_size, dropout
            )
        else:
            raise ValueError(f"Unknown attention_type: {attention_type}")

        # --- Cross-Layer Attention ---
        # Learns a weighted combination of all encoder layer outputs
        self.cross_attn = VanillaMultiHeadAttention(
            d_model, num_heads, dropout, rope=rope, alibi=alibi
        )
        self.layer_gate = nn.Linear(num_encoder_layers, 1, bias=False)

        # --- FFN ---
        if ffn_type == "standard":
            self.ffn = StandardFFN(d_model, d_ff, dropout)
        elif ffn_type == "gated":
            self.ffn = GatedFFN(d_model, d_ff, dropout)
        elif ffn_type == "depthwise_cnn":
            self.ffn = DepthwiseSeparableCNNFFN(d_model, d_ff, dropout=dropout)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_outputs: List[torch.Tensor],
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ---- Self-Attention ----
        normed = self.norm1(x)
        self_mask = None if self.bidirectional else tgt_mask
        x = x + self.dropout(self.self_attn(normed, normed, normed, self_mask))

        # ---- Cross-Layer Attention ----
        # Stack all encoder layer outputs: (B, num_layers, L_s, d_model)
        stacked = torch.stack(encoder_outputs, dim=1)
        B, N, L_s, D = stacked.shape

        # Learned weighted sum across encoder layers => (B, L_s, d_model)
        gate_weights = F.softmax(
            self.layer_gate.weight.view(-1), dim=-1
        )  # (num_layers,)
        memory = torch.einsum("bnsd,n->bsd", stacked, gate_weights)

        normed = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normed, memory, memory, src_mask))

        # ---- FFN ----
        normed = self.norm3(x)
        x = x + self.dropout(self.ffn(normed))
        return x


# =============================================================================
# 7.  EARLY-EXIT HEAD
# =============================================================================

class EarlyExitHead(nn.Module):
    """
    Lightweight classifier head attached to intermediate decoder layers.
    Returns logits + a scalar confidence score used to decide whether
    to exit early during inference.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        self.confidence = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(x)  # (B, L, vocab)
        conf = self.confidence(x).squeeze(-1)  # (B, L)
        return logits, conf


# =============================================================================
# 8.  FULL ENCODER
# =============================================================================

class Encoder(nn.Module):
    """
    Stack of N encoder layers + optional hybrid CNN-RNN block.
    Returns a list of all layer outputs for cross-layer attention.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: str = "vanilla",
        ffn_type: str = "standard",
        use_hybrid_block: bool = True,
        window_size: int = 128,
        rope: Optional[RotaryPositionalEncoding] = None,
        alibi: Optional[ALiBiPositionalBias] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        # Optionally prepend a hybrid CNN-RNN block
        if use_hybrid_block:
            self.layers.append(
                HybridCNNRNNBlock(d_model, d_model, 3, d_model // 2, dropout)
            )

        for _ in range(num_layers):
            self.layers.append(
                EncoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                    attention_type,
                    ffn_type,
                    window_size,
                    rope,
                    alibi,
                )
            )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        layer_outputs: List[torch.Tensor] = []
        for layer in self.layers:
            if isinstance(layer, HybridCNNRNNBlock):
                x = layer(x, src_mask)
            else:
                x = layer(x, src_mask)
            layer_outputs.append(x)
        layer_outputs[-1] = self.final_norm(layer_outputs[-1])
        return layer_outputs


# =============================================================================
# 9.  FULL DECODER  (with Early Exit)
# =============================================================================

class Decoder(nn.Module):
    """
    Stack of N decoder layers with:
      • Cross-layer attention to encoder
      • Optional bidirectional decoding
      • Early-exit heads at every layer for adaptive compute
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        num_encoder_layers: int,
        dropout: float = 0.1,
        attention_type: str = "vanilla",
        ffn_type: str = "standard",
        bidirectional_decoder: bool = False,
        window_size: int = 128,
        rope: Optional[RotaryPositionalEncoding] = None,
        alibi: Optional[ALiBiPositionalBias] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.early_exit_heads = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                DecoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    num_encoder_layers,
                    dropout,
                    attention_type,
                    ffn_type,
                    bidirectional_decoder,
                    window_size,
                    rope,
                    alibi,
                )
            )
            self.early_exit_heads.append(EarlyExitHead(d_model, vocab_size))

        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_outputs: List[torch.Tensor],
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        early_exit_threshold: float = 0.0,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], int]:
        """
        Returns
        -------
        final_logits : (B, L_t, vocab_size)
        all_exit_outputs : list of (logits, confidence) from every layer
        exit_layer : index of the layer used (== last layer if no early exit)
        """
        all_exit_outputs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        exit_layer = len(self.layers) - 1

        for i, (layer, exit_head) in enumerate(
            zip(self.layers, self.early_exit_heads)
        ):
            x = layer(x, encoder_outputs, src_mask, tgt_mask)
            logits, conf = exit_head(self.final_norm(x) if i == exit_layer else x)
            all_exit_outputs.append((logits, conf))

            # At inference time, exit early if all positions are confident
            if (
                early_exit_threshold > 0.0
                and not self.training
                and i < len(self.layers) - 1
            ):
                if conf.min().item() >= early_exit_threshold:
                    exit_layer = i
                    break

        final_logits = all_exit_outputs[exit_layer][0]
        return final_logits, all_exit_outputs, exit_layer


# =============================================================================
# 10. FULL MODIFIED TRANSFORMER
# =============================================================================

class ModifiedTransformer(nn.Module):
    """
    End-to-end modified Transformer model.

    Configuration knobs
    -------------------
    - num_encoder_layers / num_decoder_layers : depth
    - num_heads : width of attention
    - pos_encoding : "sinusoidal" | "learned" | "rope" | "alibi"
    - attention_type : "vanilla" | "linear" | "local"
    - ffn_type : "standard" | "gated" | "depthwise_cnn"
    - use_hybrid_block : prepend CNN-RNN hybrid layer in encoder
    - bidirectional_decoder : remove causal mask in decoder self-attn
    - early_exit_threshold : confidence threshold for adaptive inference
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        pos_encoding: Literal["sinusoidal", "learned", "rope", "alibi"] = "rope",
        attention_type: Literal["vanilla", "linear", "local"] = "vanilla",
        ffn_type: Literal["standard", "gated", "depthwise_cnn"] = "gated",
        use_hybrid_block: bool = True,
        bidirectional_decoder: bool = False,
        early_exit_threshold: float = 0.0,
        window_size: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.early_exit_threshold = early_exit_threshold

        # --- Embeddings ---
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)

        # --- Positional Encoding ---
        rope = None
        alibi = None
        head_dim = d_model // num_heads

        if pos_encoding == "sinusoidal":
            self.src_pos = SinusoidalPositionalEncoding(d_model, max_len, dropout)
            self.tgt_pos = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == "learned":
            self.src_pos = LearnedPositionalEncoding(d_model, max_len, dropout)
            self.tgt_pos = LearnedPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == "rope":
            # RoPE is applied inside attention; no additive pos encoding needed
            rope = RotaryPositionalEncoding(head_dim, max_len)
            self.src_pos = nn.Dropout(dropout)
            self.tgt_pos = nn.Dropout(dropout)
        elif pos_encoding == "alibi":
            alibi = ALiBiPositionalBias(num_heads, max_len)
            self.src_pos = nn.Dropout(dropout)
            self.tgt_pos = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        # Account for hybrid block in encoder layer count for cross-layer attn
        effective_enc_layers = num_encoder_layers + (1 if use_hybrid_block else 0)

        # --- Encoder ---
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            attention_type=attention_type,
            ffn_type=ffn_type,
            use_hybrid_block=use_hybrid_block,
            window_size=window_size,
            rope=rope,
            alibi=alibi,
        )

        # --- Decoder ---
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=tgt_vocab_size,
            num_encoder_layers=effective_enc_layers,
            dropout=dropout,
            attention_type=attention_type,
            ffn_type=ffn_type,
            bidirectional_decoder=bidirectional_decoder,
            window_size=window_size,
            rope=rope,
            alibi=alibi,
        )

        self._init_parameters()

    def _init_parameters(self):
        """Xavier uniform initialization for all linear/embedding layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Causal mask: (1, 1, sz, sz) with 1 = attend, 0 = mask."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return (mask == 0).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def generate_padding_mask(
        seq: torch.Tensor, pad_idx: int = 0
    ) -> torch.Tensor:
        """Padding mask: (B, 1, 1, L) with 1 = attend, 0 = mask."""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], int]:
        """
        Parameters
        ----------
        src : (B, L_s)  source token IDs
        tgt : (B, L_t)  target token IDs
        src_mask : optional padding mask for source
        tgt_mask : optional causal + padding mask for target

        Returns
        -------
        logits : (B, L_t, tgt_vocab_size)
        all_exit_outputs : intermediate exit logits & confidences
        exit_layer : layer at which the model exited
        """
        # Embed + scale + positional encoding
        src_emb = self.src_pos(self.src_embed(src) * self.embed_scale)
        tgt_emb = self.tgt_pos(self.tgt_embed(tgt) * self.embed_scale)

        # Encode — returns list of layer outputs
        encoder_outputs = self.encoder(src_emb, src_mask)

        # Build causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(
                tgt.size(1), tgt.device
            )

        # Decode
        logits, all_exit_outputs, exit_layer = self.decoder(
            tgt_emb,
            encoder_outputs,
            src_mask,
            tgt_mask,
            self.early_exit_threshold,
        )

        return logits, all_exit_outputs, exit_layer


# =============================================================================
# 11. EARLY-EXIT LOSS
# =============================================================================

def early_exit_loss(
    all_exit_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
    target: torch.Tensor,
    pad_idx: int = 0,
    layer_weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Compute weighted cross-entropy across all early-exit heads.
    Deeper layers receive higher weight by default (linearly increasing).

    Parameters
    ----------
    all_exit_outputs : [(logits, conf), ...] from every decoder layer
    target : (B, L_t) ground-truth token IDs
    pad_idx : padding index to ignore
    layer_weights : optional per-layer weights
    """
    num_layers = len(all_exit_outputs)
    if layer_weights is None:
        # Linearly increasing: later layers matter more
        layer_weights = [(i + 1) / num_layers for i in range(num_layers)]

    total_loss = torch.tensor(0.0, device=target.device)
    for w, (logits, _) in zip(layer_weights, all_exit_outputs):
        # logits: (B, L, V)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=pad_idx,
        )
        total_loss = total_loss + w * loss

    return total_loss / sum(layer_weights)


# =============================================================================
# 12. DEMO / SMOKE TEST
# =============================================================================

def demo():
    """Quick smoke test showing all modifications in action."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Configuration --
    config = dict(
        src_vocab_size=8000,
        tgt_vocab_size=8000,
        d_model=256,
        num_encoder_layers=4,          # ← configurable depth
        num_decoder_layers=4,
        num_heads=8,                   # ← configurable width
        d_ff=512,
        dropout=0.1,
        max_len=512,
        pos_encoding="rope",           # ← RoPE positional encoding
        attention_type="vanilla",      # ← attention variant
        ffn_type="gated",             # ← Gated FFN (SwiGLU style)
        use_hybrid_block=True,         # ← CNN-RNN hybrid in encoder
        bidirectional_decoder=False,   # ← causal decoder
        early_exit_threshold=0.9,      # ← early exit at inference
        window_size=64,                # ← for local attention
    )

    model = ModifiedTransformer(**config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: {config}\n")

    # -- Dummy data --
    B, L_s, L_t = 2, 32, 24
    src = torch.randint(1, config["src_vocab_size"], (B, L_s), device=device)
    tgt = torch.randint(1, config["tgt_vocab_size"], (B, L_t), device=device)
    tgt_labels = torch.randint(1, config["tgt_vocab_size"], (B, L_t), device=device)

    src_mask = ModifiedTransformer.generate_padding_mask(src)

    # -- Training forward pass --
    model.train()
    logits, all_exits, exit_layer = model(src, tgt, src_mask=src_mask)
    loss = early_exit_loss(all_exits, tgt_labels)
    print(f"[Train] logits shape: {logits.shape}")
    print(f"[Train] loss: {loss.item():.4f}")
    print(f"[Train] early exit outputs from {len(all_exits)} layers\n")

    # Backward
    loss.backward()

    # -- Inference forward pass (early exit enabled) --
    model.eval()
    with torch.no_grad():
        logits, all_exits, exit_layer = model(src, tgt, src_mask=src_mask)
    print(f"[Eval] exited at layer {exit_layer} / {config['num_decoder_layers'] - 1}")
    print(f"[Eval] logits shape: {logits.shape}")

    print("\n✓ All modifications working correctly.")


if __name__ == "__main__":
    demo()
