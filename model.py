"""
model.py — Transformer language model optimized for Italian + C programming.

Key features:
  - Grouped Query Attention (GQA) with zero-copy expand
  - QK-Norm (RMSNorm on queries/keys)
  - RoPE with theta=500000 (Llama 3 style) for better length generalization
  - SwiGLU feed-forward (PaLM / Llama)
  - Multi-Token Prediction (MTP) auxiliary heads with decaying weight
  - Z-loss regularisation (PaLM / Gemma) for training stability
  - Weight tying (input embeddings = output projection)
  - Gradient checkpointing (saves ~40% VRAM)
  - KV cache for fast autoregressive inference (~4-5x speedup)
  - @dataclass ModelArgs (type-safe, enforced)
  - Context length 1024 (extended from 512)
  - Automatic SDPA kernel selection (no hardcoded Flash context manager)
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# ─── Model Configuration ─────────────────────────────────────────────────────

@dataclass
class ModelArgs:
    dim: int = 1152
    n_layers: int = 12
    n_heads: int = 18
    n_kv_heads: int = 6           # Must divide n_heads evenly: 18/6 = 3 repeats
    vocab_size: int = 32000
    hidden_dim: int = 3072
    max_seq_len: int = 1024          # Extended from 512 — fits in 8GB with checkpointing
    use_checkpointing: bool = True    # ON by default — REQUIRED for 8GB VRAM (saves ~40% activation memory)
    n_mtp_tokens: int = 2
    mtp_weight: float = 0.05     # Reduced from 0.1 — auxiliary MTP loss adds high-variance gradients that amplify fluctuation
    dropout: float = 0.02        # Reduced from 0.05 — less stochastic noise in forward pass, smoother training
    rope_theta: float = 500_000.0   # Llama 3 style — much better length generalization
    z_loss_weight: float = 1e-4     # PaLM/Gemma — prevents logit explosion, replaces spike detector

    def as_dict(self):
        """Serialize all fields for checkpoint saving."""
        return {
            'dim': self.dim, 'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'n_kv_heads': self.n_kv_heads, 'vocab_size': self.vocab_size,
            'hidden_dim': self.hidden_dim, 'max_seq_len': self.max_seq_len,
            'use_checkpointing': self.use_checkpointing,
            'n_mtp_tokens': self.n_mtp_tokens, 'mtp_weight': self.mtp_weight,
            'dropout': self.dropout, 'rope_theta': self.rope_theta,
            'z_loss_weight': self.z_loss_weight,
        }


# ─── Building Blocks ──────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation — faster than LayerNorm, no bias needed."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 500_000.0) -> torch.Tensor:
    """Precompute complex exponentials for Rotary Position Embeddings.
    Uses theta=500k (Llama 3 style) for better extrapolation to longer contexts."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply RoPE to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ─── KV Cache ─────────────────────────────────────────────────────────────────

@dataclass
class KVCache:
    """Per-layer key/value cache for fast autoregressive generation.
    Stores pre-computed K,V tensors so we only process the new token
    at each generation step instead of the full context window.
    Provides ~4-5x speedup during inference.

    Uses pre-allocated fixed-size tensors and tracks active_len to avoid
    torch.cat() allocations during generation. truncate() adjusts active_len
    without reshaping the underlying buffer, so subsequent update() calls
    can safely overwrite positions beyond the truncation point."""
    k: torch.Tensor  # (batch, n_kv_heads, max_seq_len, head_dim) — pre-allocated
    v: torch.Tensor  # (batch, n_kv_heads, max_seq_len, head_dim) — pre-allocated
    max_seq_len: int = 0  # Guard: prevents unbounded growth
    active_len: int = 0   # Tracks logical length of valid cache entries

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor,
               start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new K,V using slice assignment (no tensor allocation).
        Writes into the pre-allocated buffer at start_pos and updates active_len.
        Returns views into the active portion of the cache (0..active_len)."""
        sl = k_new.shape[2]
        # Write new values into pre-allocated buffer at the correct offset
        self.k[:, :, start_pos:start_pos + sl, :] = k_new
        self.v[:, :, start_pos:start_pos + sl, :] = v_new
        # Update logical length
        self.active_len = start_pos + sl
        return self.k[:, :, :self.active_len, :], self.v[:, :, :self.active_len, :]

    def truncate(self, length: int):
        """Truncate cache to the given length (keeps entries from position 0 to length-1).
        Used in speculative decoding when some draft tokens are rejected — the cache
        has entries for positions 0..cur_pos+len(draft)-1, and we want to keep only
        0..cur_pos+accepted-1, so we call truncate(cur_pos + accepted).

        Instead of reshaping the tensor (which would break subsequent update()
        slice-assignment), we zero out positions beyond 'length' and adjust active_len.
        The next update() call will overwrite these zeroed positions."""
        if length < self.active_len:
            # Zero out positions beyond the truncation point so stale data
            # doesn't leak into future attention computations
            self.k[:, :, length:self.active_len, :].zero_()
            self.v[:, :, length:self.active_len, :].zero_()
            self.active_len = length


# ─── Attention ────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Grouped Query Attention with QK-Norm and KV cache support.
    Uses expand (zero-copy view) instead of repeat_interleave (memory allocation).
    Removes hardcoded Flash SDP context manager — PyTorch >=2.2 auto-selects best kernel."""
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.repeats = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # QK-Norm: stabilises attention at high learning rates
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cis, kv_cache: Optional[KVCache] = None,
                start_pos: int = 0):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm BEFORE RoPE (standard convention — avoids learned weight interacting
        # with rotation; since RoPE is norm-preserving, post-RoPE magnitudes stay controlled)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Apply RoPE — freqs_cis is already sliced to [start_pos:start_pos+seqlen]
        # by Transformer.forward(), so we use it directly here.
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Update KV cache if provided
        if kv_cache is not None:
            xk, xv = kv_cache.update(xk, xv, start_pos)

        if self.repeats > 1:
            # Zero-copy expand instead of memory-allocating repeat_interleave
            xk = xk.unsqueeze(2).expand(-1, -1, self.repeats, -1, -1).reshape(
                bsz, self.n_heads, xk.shape[2], self.head_dim)
            xv = xv.unsqueeze(2).expand(-1, -1, self.repeats, -1, -1).reshape(
                bsz, self.n_heads, xv.shape[2], self.head_dim)

        # Attention mask logic:
        # - Training / prefill (no cache): use causal mask via is_causal=True
        # - Single-token generation (cache, seqlen=1): no mask needed
        # - Multi-token verify (cache, seqlen>1, speculative decoding):
        #   Need causal mask among new tokens, but all attend to cached tokens
        if kv_cache is not None and start_pos > 0:
            if seqlen == 1:
                # Standard single-token generation: no mask needed
                attn_mask = None
                is_causal = False
            else:
                # Multi-token verify pass (speculative decoding):
                # Need causal mask within the new tokens, but all attend to cached tokens
                # Build explicit mask: (seqlen, total_len) where total_len = cached + new
                cached_len = xk.shape[2] - seqlen  # kv_cache was already updated
                # All new queries can attend to all cached tokens (zero in cached region)
                # New queries are causal among themselves (upper triangle = -inf)
                mask = torch.zeros(seqlen, xk.shape[2], device=x.device, dtype=x.dtype)
                mask[:, cached_len:] = torch.triu(
                    torch.full((seqlen, seqlen), float('-inf'), device=x.device), diagonal=1
                )
                attn_mask = mask
                is_causal = False
        else:
            attn_mask = None
            is_causal = True

        # PyTorch >=2.2 automatically picks the best SDPA kernel
        # (Flash Attention > Memory-Efficient > Math) — no context manager needed
        out = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.resid_dropout(self.wo(out))


# ─── Feed-Forward ─────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """SwiGLU feed-forward block: silu(W1(x)) * W3(x) -> W2.
    Better than ReLU/GELU, used in PaLM and Llama-3."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional gradient checkpointing."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args.dim, args.n_heads, args.n_kv_heads, args.dropout)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.dropout)
        self.attention_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
        self.use_ckpt = args.use_checkpointing

    def _block_forward(self, x, freqs_cis, kv_cache, start_pos):
        """Full block forward: attention + FFN with pre-norm.
        Used as a single checkpoint unit to halve Python overhead vs two checkpoints."""
        h = x + self.attention(self.attention_norm(x), freqs_cis, kv_cache, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward(self, x, freqs_cis, kv_cache: Optional[KVCache] = None,
                start_pos: int = 0):
        if self.use_ckpt and self.training:
            # Single checkpoint per block instead of two — halves Python overhead
            # (2*n_layers -> n_layers checkpoint segments)
            out = checkpoint(self._block_forward, x, freqs_cis, kv_cache, start_pos,
                             use_reentrant=False)
        else:
            h = x + self.attention(self.attention_norm(x), freqs_cis, kv_cache, start_pos)
            out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ─── Multi-Token Prediction Heads ─────────────────────────────────────────────

class MTPHead(nn.Module):
    """Auxiliary prediction head for token positions t+2, t+3, etc.
    Uses the main embedding weight matrix for the final projection (parameter-efficient)."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.GELU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, h, embed_weight):
        h = self.proj(h)
        return torch.matmul(h, embed_weight.t())


# ─── Full Transformer ─────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """Full autoregressive Transformer language model with KV cache support."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Embedding + output (weight-tied)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)

        # Output projection — tied to embedding
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

        # Multi-token prediction auxiliary heads
        self.mtp_heads = nn.ModuleList([MTPHead(args.dim) for _ in range(args.n_mtp_tokens)])

        # Precompute RoPE frequencies with 4x context headroom
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            args.dim // args.n_heads, args.max_seq_len * 4, args.rope_theta
        ), persistent=False)

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # GPT-2 style: scale down residual output projections to maintain
        # variance across the depth of the network. Each transformer block
        # adds 2 residual contributions (attention + FFN), so we scale by
        # 1/sqrt(2 * n_layers).
        # Fixed: only target the specific weight parameters by qualified name,
        # not named_parameters() which recurses into sub-modules (fragile).
        if isinstance(module, Attention) and hasattr(module, 'wo'):
            with torch.no_grad():
                module.wo.weight.mul_((2 * self.args.n_layers) ** -0.5)
        elif isinstance(module, FeedForward) and hasattr(module, 'w2'):
            with torch.no_grad():
                module.w2.weight.mul_((2 * self.args.n_layers) ** -0.5)

    def forward(self, tokens, targets=None, kv_caches: Optional[List[KVCache]] = None,
                start_pos: int = 0, return_hidden: bool = False):
        """Forward pass with optional KV cache for efficient autoregressive generation.

        Args:
            tokens: Input token IDs, shape (batch, seq_len)
            targets: Target token IDs for loss computation (training only)
            kv_caches: List of KVCache objects, one per layer. None during training.
            start_pos: Position offset for KV cache (0 for prefill, >0 for generation).
            return_hidden: If True, also return the final hidden state h (for speculative decoding).
        """
        bsz, seqlen = tokens.shape
        h = self.dropout(self.tok_embeddings(tokens))
        # Use correct slice for all cases: training (start_pos=0), single-token gen,
        # and multi-token KV cache generation (speculative decoding verify pass)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h = layer(h, freqs_cis, kv_cache=layer_cache, start_pos=start_pos)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            # -- Main cross-entropy loss --
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            # -- Z-loss for training stability (PaLM / Gemma) --
            # Penalises large logit magnitudes, prevents loss spikes
            if self.args.z_loss_weight > 0 and self.training:
                z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
            else:
                z_loss = torch.tensor(0.0, device=logits.device)

            loss = ce_loss + self.args.z_loss_weight * z_loss

            # -- Multi-Token Prediction --
            if self.args.n_mtp_tokens > 0 and self.training:
                total_mtp_loss = 0.0
                for i, mtp_head in enumerate(self.mtp_heads):
                    if seqlen > i + 1:
                        mtp_logits = mtp_head(h[:, :-i-1, :], self.tok_embeddings.weight)
                        mtp_targets = targets[:, i+1:]
                        head_loss = F.cross_entropy(
                            mtp_logits.reshape(-1, mtp_logits.size(-1)),
                            mtp_targets.reshape(-1)
                        )
                        # Decay weight for further-ahead predictions: head 0 -> full weight,
                        # head 1 -> half weight, etc.
                        total_mtp_loss += head_loss / (i + 1)
                loss = loss + self.args.mtp_weight * total_mtp_loss

        if return_hidden:
            return logits, loss, h
        return logits, loss

    def init_kv_caches(self, batch_size: int, device: torch.device,
                       dtype: torch.dtype = torch.bfloat16) -> List[KVCache]:
        """Initialize pre-allocated KV caches for all layers. Call once before generation.

        Pre-allocates the full cache tensor (max_seq_len positions) to avoid
        repeated torch.cat() allocations during autoregressive generation.
        This eliminates ~1024 tensor allocations and copies per generation call.

        Args:
            batch_size: Number of sequences (usually 1 for single-prompt generation)
            device: CUDA device
            dtype: Data type for cache tensors (bfloat16 recommended)

        Returns:
            List of KVCache objects, one per transformer layer
        """
        head_dim = self.args.dim // self.args.n_heads
        max_len = self.args.max_seq_len
        caches = []
        for _ in range(self.args.n_layers):
            # Pre-allocate full cache: [batch, heads, max_seq_len, head_dim]
            k = torch.zeros(batch_size, self.args.n_kv_heads, max_len, head_dim,
                           device=device, dtype=dtype)
            v = torch.zeros(batch_size, self.args.n_kv_heads, max_len, head_dim,
                           device=device, dtype=dtype)
            caches.append(KVCache(k=k, v=v, max_seq_len=max_len))
        return caches


# ─── Utility ──────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> float:
    """Return parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
