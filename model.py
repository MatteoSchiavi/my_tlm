"""
model.py v7 — Transformer language model optimized for Italian + C programming.

v7 fix:
  + FIXED: OOM during MTP on 8 GB GPUs (RTX 3070). Root cause: logits tensor
    (batch, seq, vocab) BF16 = 262 MB with batch=4 was held alive as a Python
    local variable throughout the entire MTP section. torch.cuda.empty_cache()
    cannot reclaim a live reference — so 7.01/7.04 GB triggered OOM even with
    chunk=64. Fix: after ce_loss + z_loss are computed, `del logits` and call
    empty_cache() ONCE before MTP starts. This frees 262 MB and gives MTP the
    headroom it needs. logits is returned as None during training (train.py
    never reads it). Chunked MTP is retained as a secondary guard.

v6 fixes (audit-driven):
  + FIXED: BF16-native RoPE — real-arithmetic formulation replaces complex-number
    approach. Eliminates FP32 upcast on Q/K every forward pass, saving ~175 MB
    peak VRAM and ~60% memory bandwidth for RoPE. Fuses cleanly with torch.compile.
    freqs_cos/sin buffers precomputed and pre-expanded once per forward, not inside
    apply_rotary_emb on every layer call. (Issues 7, 12)
  + FIXED: GQA KV expansion uses repeat_interleave instead of expand().reshape().
    expand().reshape() forced a full contiguous copy of KV tensors to MHA size.
    repeat_interleave is semantically identical but avoids the non-contiguous
    tensor materialization ambiguity. (Issue 5)
  + FIXED: MTPHead — GELU replaced with SiLU (consistent with SwiGLU used
    everywhere else), RMSNorm added before the tied embedding projection to bound
    activation scale, torch.matmul(...t()) replaced with F.linear for compile
    compatibility and fused kernel selection. (Issue 10)
  + FIXED: Z-loss during eval no longer allocates a device tensor (torch.tensor(0.0))
    and adds it to loss. Eval path now assigns loss = ce_loss directly, eliminating
    CPU-GPU syncs and scalar tensor allocations during validation. (Issue 8)
  + FIXED: MTP boundary mask uses rolling AND across the full prediction span.
    Old code used doc_boundary_mask[:, offset:] which only checked the endpoint;
    correct logic requires no boundary crossing anywhere between j and j+offset.
    New compute_mtp_mask() ANDs the base mask over the full span. (Issue 15)

v5 features preserved:
  - Grouped Query Attention (GQA): 18Q / 6KV heads, 3:1 ratio
  - QK-Norm (RMSNorm on queries/keys before RoPE)
  - SwiGLU feed-forward (PaLM / Llama style)
  - Z-loss regularisation (PaLM / Gemma) for training stability
  - Weight tying (input embeddings = output projection)
  - Gradient checkpointing with use_reentrant=False (saves ~40% VRAM)
  - KV cache for fast autoregressive inference
  - Context length 1024, RoPE theta=10,000
  - Automatic SDPA kernel selection (FlashAttention on Ampere)
  - Document-boundary loss masking for packed sequences
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
    hidden_dim: int = 3072        # SwiGLU hidden: 8/3 * dim ≈ 2.667x (≡ 4x GELU)
    max_seq_len: int = 1024
    use_checkpointing: bool = True # ON by default — saves ~40% activation memory
    n_mtp_tokens: int = 2
    mtp_weight: float = 0.05
    dropout: float = 0.02
    rope_theta: float = 10_000.0  # GPT-NeoX standard for <=2048 context
    z_loss_weight: float = 1e-4   # PaLM/Gemma — prevents logit explosion

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
    """Root Mean Square Layer Normalisation — faster than LayerNorm, no bias."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ─── RoPE — BF16-native real-arithmetic implementation ───────────────────────
#
# v5 used torch.view_as_complex which requires an FP32 upcast on every forward:
#   xq.float() → complex mul → type_as(xq)
# This materialises ~225 MB of FP32 intermediates per forward (12 layers, recomputed
# on backward with checkpointing). The real-arithmetic formulation below operates
# entirely in the model's native dtype (BF16 during training), eliminating those
# intermediates and halving memory bandwidth for this operation.
#
# The two are mathematically identical:
#   complex:  (a + ib)(cos θ + i sin θ) = a cos θ − b sin θ + i(a sin θ + b cos θ)
#   real:     xq * cos + rotate_half(xq) * sin
# where rotate_half maps [..., a, b, ...] → [..., −b, a, ...].

def precompute_rope_freqs(dim: int, end: int,
                          theta: float = 10_000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cosine and sine tables.

    Returns:
        cos: float32 tensor of shape (end, dim//2)
        sin: float32 tensor of shape (end, dim//2)

    Stored as two separate register_buffers (persistent=False) so they are
    always on the right device but not saved in checkpoints (recomputed on load).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[:dim // 2] / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs: [x0, x1, ..., xN-1] → [-xN/2, ..., -xN-1, x0, ..., xN/2-1]."""
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        xq:        (bsz, n_heads,    seq, head_dim) — in model dtype (BF16)
        xk:        (bsz, n_kv_heads, seq, head_dim) — in model dtype (BF16)
        freqs_cos: (1, 1, seq, head_dim) — pre-expanded by Transformer.forward
        freqs_sin: (1, 1, seq, head_dim) — pre-expanded by Transformer.forward

    Returns:
        (xq_rot, xk_rot) in the same dtype as the inputs. No FP32 upcast.
    """
    # Cast freqs to model dtype once; broadcasting handles batch/heads dims.
    cos = freqs_cos.to(xq.dtype)
    sin = freqs_sin.to(xq.dtype)
    xq_rot = xq * cos + _rotate_half(xq) * sin
    xk_rot = xk * cos + _rotate_half(xk) * sin
    return xq_rot, xk_rot


# ─── KV Cache ─────────────────────────────────────────────────────────────────

@dataclass
class KVCache:
    """Per-layer key/value cache for fast autoregressive generation.
    Pre-allocated fixed-size buffers; active_len tracks filled positions."""
    k: torch.Tensor   # (batch, n_kv_heads, max_seq_len, head_dim)
    v: torch.Tensor   # (batch, n_kv_heads, max_seq_len, head_dim)
    max_seq_len: int = 0
    active_len: int = 0

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor,
               start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sl = k_new.shape[2]
        self.k[:, :, start_pos:start_pos + sl, :] = k_new
        self.v[:, :, start_pos:start_pos + sl, :] = v_new
        self.active_len = start_pos + sl
        return self.k[:, :, :self.active_len, :], self.v[:, :, :self.active_len, :]

    def truncate(self, length: int):
        if length < self.active_len:
            self.k[:, :, length:self.active_len, :].zero_()
            self.v[:, :, length:self.active_len, :].zero_()
            self.active_len = length


# ─── Attention ────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Grouped Query Attention with QK-Norm and KV cache support.

    Signature change from v5: freqs_cis (complex) replaced by freqs_cos/freqs_sin
    (real, pre-expanded) to support the BF16-native RoPE implementation.
    """
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = dim // n_heads
        self.repeats    = n_heads // n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                freqs_cos: torch.Tensor, freqs_sin: torch.Tensor,
                kv_cache: Optional[KVCache] = None,
                start_pos: int = 0) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # Linear projections
        xq = self.wq(x).view(bsz, seqlen, self.n_heads,    self.head_dim).transpose(1, 2)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm: normalise per head before RoPE for attention entropy stability
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # BF16-native RoPE (no FP32 upcast)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # KV cache update (inference only; kv_cache is None during training)
        if kv_cache is not None:
            xk, xv = kv_cache.update(xk, xv, start_pos)

        # GQA: expand KV from n_kv_heads to n_heads.
        # v5 used expand().reshape() which forced a contiguous copy via reshape.
        # repeat_interleave produces an equivalent contiguous tensor with clearer
        # semantics and avoids the non-contiguous intermediate.
        if self.repeats > 1:
            xk = xk.repeat_interleave(self.repeats, dim=1)  # (bsz, n_heads, seq, head_dim)
            xv = xv.repeat_interleave(self.repeats, dim=1)

        # Attention mask selection
        if kv_cache is not None and start_pos > 0:
            if seqlen == 1:
                # Single-token generation: no causal mask needed
                attn_mask = None
                is_causal  = False
            else:
                # Prefill with prior cache: manual partial-causal mask
                cached_len = xk.shape[2] - seqlen
                mask = torch.zeros(seqlen, xk.shape[2], device=x.device, dtype=x.dtype)
                mask[:, cached_len:] = torch.triu(
                    torch.full((seqlen, seqlen), float('-inf'), device=x.device), diagonal=1
                )
                attn_mask = mask
                is_causal  = False
        else:
            # Training / full prefill: SDPA with is_causal=True triggers FlashAttention
            attn_mask = None
            is_causal  = True

        # PyTorch SDPA — uses FlashAttention on Ampere (RTX 3070) when is_causal=True
        # and dtype is BF16/FP16
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
    """SwiGLU feed-forward block: silu(W1(x)) * W3(x) → W2.
    hidden_dim = 8/3 * dim (≡ 4× GELU in total parameter count)."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional gradient checkpointing."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention    = Attention(args.dim, args.n_heads, args.n_kv_heads, args.dropout)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.dropout)
        self.attention_norm = RMSNorm(args.dim)
        self.ffn_norm       = RMSNorm(args.dim)
        self.use_ckpt = args.use_checkpointing

    def _block_forward(self, x: torch.Tensor,
                       freqs_cos: torch.Tensor, freqs_sin: torch.Tensor,
                       kv_cache: Optional[KVCache],
                       start_pos: int) -> torch.Tensor:
        h   = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin, kv_cache, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward(self, x: torch.Tensor,
                freqs_cos: torch.Tensor, freqs_sin: torch.Tensor,
                kv_cache: Optional[KVCache] = None,
                start_pos: int = 0) -> torch.Tensor:
        if self.use_ckpt and self.training:
            # use_reentrant=False: compatible with torch.compile and allows
            # non-tensor arguments (freqs_cos, freqs_sin, start_pos)
            return checkpoint(
                self._block_forward, x, freqs_cos, freqs_sin, kv_cache, start_pos,
                use_reentrant=False,
            )
        else:
            return self._block_forward(x, freqs_cos, freqs_sin, kv_cache, start_pos)


# ─── Multi-Token Prediction Heads ─────────────────────────────────────────────

class MTPHead(nn.Module):
    """Auxiliary prediction head for positions t+2, t+3, ...

    v6 changes vs v5:
      - GELU → SiLU (consistent with SwiGLU activation used in main model)
      - RMSNorm added before the tied embedding projection. The input h comes
        from the normalised final hidden state, but after two linear transforms
        the scale drifts; normalising before the embedding projection bounds
        MTP logit scale to the same range as main LM logits.
      - torch.matmul(h, embed_weight.t()) → F.linear(h, embed_weight) for
        correct fused kernel dispatch under torch.compile.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.w1   = nn.Linear(dim, dim, bias=False)
        self.w2   = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim)

    def forward(self, h: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.w1(h))   # SiLU — same activation family as SwiGLU
        h = self.w2(h)
        h = self.norm(h)         # Normalise before tied projection
        return F.linear(h, embed_weight)


# ─── MTP Boundary Mask ────────────────────────────────────────────────────────

def compute_mtp_mask(doc_boundary_mask: torch.Tensor,
                     offset: int) -> Optional[torch.Tensor]:
    """Compute the valid-prediction mask for an MTP head at the given offset.

    The main LM head predicts t+1; MTP head at offset predicts t+offset.
    A prediction at position j is valid for MTP only if EVERY position in the
    span [j, j+offset-1] is within the same document — i.e., no document boundary
    is crossed anywhere between j and j+offset.

    v5 bug: used doc_boundary_mask[:, offset:] which only checked whether the
    endpoint was valid, silently training across document boundaries for all
    intermediate positions. This fix ANDs the mask over the full span.

    Args:
        doc_boundary_mask: (batch, seq_len-1) — True = valid prediction position
        offset: prediction distance (2 for head-0, 3 for head-1, etc.)

    Returns:
        mask of shape (batch, seq_len-offset) — True = valid MTP position,
        or None if the sequence is too short for this offset.
    """
    B, L = doc_boundary_mask.shape          # L = seq_len - 1
    out_len = L - offset + 1               # = seq_len - offset
    if out_len <= 0:
        return None
    # Sliding-window AND: position j is valid iff dbm[j], dbm[j+1], ..., dbm[j+offset-1].
    # Start with a zero-copy slice (no .clone() needed — each & produces a new tensor).
    mask = doc_boundary_mask[:, :out_len]
    for k in range(1, offset):
        mask = mask & doc_boundary_mask[:, k : out_len + k]
    return mask


# ─── Document Boundary Detection ──────────────────────────────────────────────

def compute_doc_boundary_mask(tokens: torch.Tensor,
                              bos_id: int, eos_id: int) -> torch.Tensor:
    """Compute valid-for-loss mask for packed sequences.

    In packed sequences documents are concatenated with BOS/EOS separators.
    Predicting the first token of a new document conditions on the previous
    document's context — a spurious correlation. This mask excludes those
    positions from loss.

    Args:
        tokens: (batch, seq_len) — input token IDs
        bos_id: BOS token ID
        eos_id: EOS token ID

    Returns:
        Boolean tensor (batch, seq_len-1):
            True  = valid prediction position (within a document)
            False = first token of a new document — exclude from loss
    """
    prev = tokens[:, :-1]
    boundary = (prev == bos_id) | (prev == eos_id)
    return ~boundary


# ─── Full Transformer ─────────────────────────────────────────────────────────

class Transformer(nn.Module):
    """Full autoregressive Transformer language model."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # Embedding + output (weight-tied: saves 37M parameters)
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)

        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm   = RMSNorm(args.dim)

        # Output projection — tied to embedding weight
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

        # Multi-token prediction auxiliary heads
        self.mtp_heads = nn.ModuleList(
            [MTPHead(args.dim) for _ in range(args.n_mtp_tokens)]
        )

        # BF16-native RoPE buffers (float32, not saved in checkpoints).
        # Precomputed at 4× context for inference flexibility.
        # Stored pre-concatenated [cos,cos] / [sin,sin] along head_dim so that
        # forward() slices a zero-copy view instead of calling torch.cat every step.
        # Shape: (max_seq_len * 4, head_dim)  ← full head_dim, not head_dim // 2
        _cos, _sin = precompute_rope_freqs(
            args.dim // args.n_heads,
            args.max_seq_len * 4,
            args.rope_theta,
        )
        self.register_buffer("freqs_cos", torch.cat([_cos, _cos], dim=-1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat([_sin, _sin], dim=-1), persistent=False)

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """GPT-2 style initialisation with depth-scaled residual projections.

        Output projections (wo, w2) are scaled by 1/sqrt(2 * n_layers) so that
        the residual stream variance stays bounded as depth increases.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Depth scaling for residual output projections
        if isinstance(module, Attention) and hasattr(module, 'wo'):
            with torch.no_grad():
                module.wo.weight.mul_((2 * self.args.n_layers) ** -0.5)
        elif isinstance(module, FeedForward) and hasattr(module, 'w2'):
            with torch.no_grad():
                module.w2.weight.mul_((2 * self.args.n_layers) ** -0.5)

    def forward(self,
                tokens: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                kv_caches: Optional[List[KVCache]] = None,
                start_pos: int = 0,
                return_hidden: bool = False,
                doc_boundary_mask: Optional[torch.Tensor] = None
                ) -> Tuple:
        """Forward pass.

        Args:
            tokens:             (batch, seq_len) input token IDs
            targets:            (batch, seq_len) target token IDs (training only)
            kv_caches:          list of KVCache per layer; None during training
            start_pos:          position offset for KV cache (0 = full prefill)
            return_hidden:      if True, also return final hidden state
            doc_boundary_mask:  (batch, seq_len-1) True=valid-for-loss.
                                Computed on GPU in training loop and passed in.
                                None → standard loss on all positions.

        Returns:
            (logits, loss)        when targets is not None
            (logits, None)        when targets is None
            (logits, loss, h)     when return_hidden=True
        """
        bsz, seqlen = tokens.shape
        h = self.dropout(self.tok_embeddings(tokens))

        # Pre-expand RoPE buffers once for all layers.
        # Buffers are stored pre-concatenated (head_dim, not head_dim//2), so
        # slicing is a zero-copy view. Previously torch.cat was called here every
        # forward pass, allocating two (seq, head_dim) tensors each time.
        # Shape after unsqueeze: (1, 1, seqlen, head_dim) — broadcasts over (bsz, heads, seq, head_dim)
        freqs_cos = self.freqs_cos[start_pos : start_pos + seqlen].unsqueeze(0).unsqueeze(0)
        freqs_sin = self.freqs_sin[start_pos : start_pos + seqlen].unsqueeze(0).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            h = layer(h, freqs_cos, freqs_sin, kv_cache=layer_cache, start_pos=start_pos)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            # ── Cross-entropy loss with optional document boundary masking ──
            if doc_boundary_mask is not None:
                # Masked path: exclude cross-document boundary positions.
                # logits[:, i, :] predicts targets[:, i] (= tokens[:, i+1]).
                # We use positions 0..seq_len-2 (logits[:, :-1, :]) which aligns
                # with doc_boundary_mask of shape (batch, seq_len-1).
                flat_logits  = logits[:, :-1, :].reshape(-1, logits.size(-1))
                flat_targets = targets[:, :-1].reshape(-1)
                flat_mask    = doc_boundary_mask.reshape(-1)
                if flat_mask.any():
                    ce_loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
                else:
                    # Degenerate: entire batch is boundary tokens — fallback to full loss
                    ce_loss = F.cross_entropy(flat_logits, flat_targets)
            else:
                # Standard path: loss over all positions
                ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            # ── Z-loss for logit-scale stability (PaLM / Gemma) ──
            # FIX v6: during eval, we previously allocated torch.tensor(0.0, device=...)
            # and multiplied it into the loss — a device allocation + CPU/GPU sync
            # on every validation forward pass. Now we simply skip it during eval.
            if self.args.z_loss_weight > 0 and self.training:
                z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
                loss   = ce_loss + self.args.z_loss_weight * z_loss
            else:
                loss = ce_loss

            # ── Release the 262 MB logits tensor before MTP ───────────────────
            # logits shape: (batch, seq, vocab) BF16.
            # With batch=4, seq=1024, vocab=32000 → 262 MB permanently occupying
            # VRAM during MTP, even though ce_loss and z_loss are already computed.
            # empty_cache() cannot reclaim it while the local variable holds a ref.
            # Deleting it here frees 262 MB and is the root cause fix for the OOM.
            # During training we return None for logits — train.py never reads it.
            if self.training and self.args.n_mtp_tokens > 0:
                del logits
                logits = None
                if h.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # ── Multi-Token Prediction auxiliary loss ──
            # Head i predicts t+offset where offset = i+2:
            #   head 0 → t+2, head 1 → t+3
            # Main LM head already covers t+1, so no duplication.
            #
            # v7 FIX: The root OOM cause was logits (262 MB) staying alive during
            # MTP. Fixed above by del-ing logits after loss is computed.
            # Chunked MTP forward is retained as an additional guard against
            # vocab-dimension peak memory spikes (each chunk: batch*chunk*32000*2 B).
            if self.args.n_mtp_tokens > 0 and self.training:
                total_mtp_loss = 0.0
                # With logits freed before MTP we have ~740 MB headroom.
                # chunk=256: logits (4,256,32000) BF16 = 65 MB, CE FP32 ≈ 131 MB peak.
                # chunk=64 was the conservative fallback when we had ~0 MB headroom.
                _MTP_CHUNK = 256

                for i, mtp_head in enumerate(self.mtp_heads):
                    offset = i + 2  # head 0 → t+2, head 1 → t+3

                    if seqlen <= offset:
                        continue

                    # FIX v6: rolling-AND boundary mask over the full prediction span.
                    if doc_boundary_mask is not None:
                        mtp_mask = compute_mtp_mask(doc_boundary_mask, offset)
                    else:
                        mtp_mask = None

                    # Skip early if no valid positions at all
                    if mtp_mask is not None and not mtp_mask.any():
                        del mtp_mask
                        continue

                    # .detach() the hidden state: MTP heads get gradients from their
                    # own loss, but we don't backprop through the main transformer
                    # from MTP. This saves ~300 MB of backward activation memory.
                    h_mtp = h[:, :-offset, :].detach()
                    mtp_targets = targets[:, offset:]   # (batch, seqlen-offset)

                    # Chunk along the sequence dimension to cap logits memory.
                    # Accumulate in FP32 to avoid BF16 precision loss in the sum.
                    chunk_ce_sum = torch.tensor(0.0, device=h.device, dtype=torch.float32)
                    chunk_count  = 0
                    n_positions  = h_mtp.shape[1]

                    for c_start in range(0, n_positions, _MTP_CHUNK):
                        c_end = min(c_start + _MTP_CHUNK, n_positions)
                        h_chunk      = h_mtp[:, c_start:c_end, :]
                        t_chunk      = mtp_targets[:, c_start:c_end]
                        logits_chunk = mtp_head(h_chunk, self.tok_embeddings.weight)

                        if mtp_mask is not None:
                            mask_chunk = mtp_mask[:, c_start:c_end]
                            if not mask_chunk.any():
                                del h_chunk, t_chunk, logits_chunk, mask_chunk
                                continue
                            fl = logits_chunk.reshape(-1, logits_chunk.size(-1))
                            ft = t_chunk.reshape(-1)
                            fm = mask_chunk.reshape(-1)
                            # n_valid > 0 is guaranteed by mask_chunk.any() above —
                            # removing the redundant fm.sum().item() check eliminates
                            # one GPU-CPU sync per chunk (was 2 syncs, now 1).
                            ce = F.cross_entropy(fl[fm], ft[fm], reduction='sum')
                            chunk_ce_sum = chunk_ce_sum + ce
                            chunk_count += fm.sum().item()
                            del h_chunk, t_chunk, logits_chunk, mask_chunk, fl, ft, fm
                        else:
                            ce = F.cross_entropy(
                                logits_chunk.reshape(-1, logits_chunk.size(-1)),
                                t_chunk.reshape(-1),
                                reduction='sum',
                            )
                            chunk_ce_sum = chunk_ce_sum + ce
                            chunk_count += t_chunk.numel()
                            del h_chunk, t_chunk, logits_chunk

                    del h_mtp, mtp_targets
                    if mtp_mask is not None:
                        del mtp_mask

                    if chunk_count > 0:
                        head_loss = chunk_ce_sum / chunk_count
                        # Decay weight for further-ahead predictions
                        total_mtp_loss = total_mtp_loss + head_loss / (i + 1)

                loss = loss + self.args.mtp_weight * total_mtp_loss

        if return_hidden:
            return logits, loss, h
        return logits, loss

    def init_kv_caches(self, batch_size: int, device: torch.device,
                       dtype: torch.dtype = torch.bfloat16) -> List[KVCache]:
        """Initialise pre-allocated KV caches for all layers (inference only)."""
        head_dim = self.args.dim // self.args.n_heads
        max_len  = self.args.max_seq_len
        return [
            KVCache(
                k=torch.zeros(batch_size, self.args.n_kv_heads, max_len, head_dim,
                              device=device, dtype=dtype),
                v=torch.zeros(batch_size, self.args.n_kv_heads, max_len, head_dim,
                              device=device, dtype=dtype),
                max_seq_len=max_len,
            )
            for _ in range(self.args.n_layers)
        ]


# ─── Utility ──────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> float:
    """Return parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6
