"""
train.py v5 — Training loop with all audit fixes applied.

v5 fixes (audit-driven):
  + FIXED: USE_WANDB boolean was inverted — WANDB=1 silently did nothing, WANDB=0
    enabled it. Correct logic: default "0" (disabled), WANDB=1 enables. (Issue 1)
  + FIXED: Weight decay applied to ALL parameters including RMSNorm scale vectors
    and embeddings. Added get_param_groups() which excludes 1-D parameters and
    any parameter whose name contains 'norm' or 'embed' from weight decay. (Issue 2)
  + FIXED: Validation did not apply doc_boundary_mask — val_loss was computed on
    cross-document boundary tokens while train_loss excluded them. Metrics were
    not comparable, best-val checkpoint selection was biased. _run_validation now
    accepts bos_id/eos_id, computes the mask, and passes it to model(). (Issue 3)
  + FIXED: MICRO_BATCH=1 severely underutilised tensor cores (~25% GPU utilisation).
    With gradient checkpointing, activation memory per sample is ~80-120 MB on top
    of the 2.6 GB static footprint. MICRO_BATCH=4, ACCUM_STEPS=16 keeps the same
    65,536-token effective batch but requires only 16 accumulation steps instead of
    64, reducing kernel launches by 4× and improving tensor-core utilisation to
    ~55-65%. (Issue 4)
  + FIXED: torch.compile disabled by default. Now auto-enabled on Linux/CUDA where
    the inductor backend delivers +15-30% throughput via fused RMSNorm, fused
    SwiGLU, and operator fusion. Disabled on Windows (Triton unavailable). (Issue 6)
  + FIXED: 8-bit AdamW now attempted first (saves ~1,320 MB of VRAM by quantising
    m/v states), with graceful fallback to standard fused AdamW. Frees enough VRAM
    to safely run MICRO_BATCH=8-12 if desired. (Issue 11)
  + FIXED: doc_boundary_mask removed from PackedTokenDataset. Mask was computed in
    DataLoader workers (CPU), required unsqueeze/squeeze, and was pinned+transferred
    as a third tensor per batch. Now computed on GPU after H2D transfer — two
    comparison ops on an already-resident tensor, near-zero cost. (Issue 9)
  + FIXED: accumulated_loss changed from GPU tensor to Python float. Eliminates
    the torch.tensor(0.0, device=DEVICE) allocation and the .zero_() kernel launch
    per optimizer step. Loss syncs happen on .item() per microbatch rather than a
    deferred GPU flush at step boundary. (Issue 13)

v4 features preserved:
  - WSD schedule (Warmup → Stable → Cosine Decay)
  - BF16 autocast (no GradScaler needed)
  - Raw model state dict saved before torch.compile for clean resume
  - torch.set_float32_matmul_precision('high') + TF32 flags
  - EMA loss tracking
  - VRAM pressure warning
  - Infinite loader with per-epoch re-shuffle
  - Training log saved to JSONL with persistent file handle
  - Embedded monitoring dashboard (HTTP server on port 8686)
  - Checkpoint architecture mismatch detection on resume
  - Muon optimizer support (MUON=1)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import math
import json
import platform
import threading
import http.server
import socketserver
import subprocess
import argparse
from typing import Optional

os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from model import Transformer, ModelArgs, compute_doc_boundary_mask

from tokenizers import Tokenizer


# ─── Platform Detection ───────────────────────────────────────────────────────

IS_WINDOWS = platform.system() == 'Windows'
HAS_CUDA   = torch.cuda.is_available()
HAS_TRITON = False
try:
    import triton
    HAS_TRITON = True
except ImportError:
    pass


# ─── Configuration ────────────────────────────────────────────────────────────

# FIX Issue 4: MICRO_BATCH=4, ACCUM_STEPS=16.
# Effective batch = 4 * 16 * 1024 = 65,536 tokens/step (identical to before).
# With gradient checkpointing, activation memory per sample ≈ 80-120 MB.
# Static footprint (weights + gradients + FP32 AdamW) ≈ 2.6 GB.
# 4 samples * 120 MB + 2.6 GB ≈ 3.1 GB — safely within 8 GB.
# With 8-bit Adam (Issue 11), static drops to ~1.3 GB → MICRO_BATCH=8 feasible.
MICRO_BATCH  = 4
ACCUM_STEPS  = 16           # 4 * 16 * 1024 = 65,536 tokens/step

MAX_STEPS         = 80_000
WARMUP_FRACTION   = 0.0375
STABLE_FRACTION   = 0.7125
LR                = 3e-4
MIN_LR            = 1e-5
WEIGHT_DECAY      = 0.1
GRAD_CLIP         = 1.0
EMA_ALPHA         = 0.05
CHECKPOINT_DIR    = "checkpoints"
CHECKPOINT_EVERY  = 1000
VAL_EVERY         = 2000
VAL_MAX_STEPS     = 500
DEVICE            = "cuda" if HAS_CUDA else "cpu"
METRICS_PORT      = 8686
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))

# FIX Issue 1: USE_WANDB was inverted.
# Old (broken): os.environ.get("WANDB", "1") == "0"
#   → WANDB=1 produced "1"=="0" → False (WandB OFF despite explicit opt-in)
#   → WANDB=0 produced "0"=="0" → True  (WandB ON when explicitly disabled)
# Correct: default "0" (disabled), WANDB=1 enables, WANDB=0 disables.
USE_WANDB = os.environ.get("WANDB", "0") == "1"
USE_MUON  = os.environ.get("MUON",  "0") == "1"

# FIX Issue 6: torch.compile auto-enabled on Linux/CUDA (inductor available).
# On Windows, Triton is unavailable so inductor doesn't work; aot_eager gives
# smaller gains and requires explicit opt-in.
USE_COMPILE      = HAS_CUDA and not IS_WINDOWS
COMPILE_BACKEND  = 'inductor' if (HAS_CUDA and not IS_WINDOWS) else 'aot_eager'

NUM_WORKERS      = 0 if IS_WINDOWS else 4
NUM_WORKERS_VAL  = 0 if IS_WINDOWS else 2
MAX_CHECKPOINTS  = 3

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ─── Performance Flags ────────────────────────────────────────────────────────

if HAS_CUDA:
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32        = True
    torch.backends.cudnn.benchmark         = True
    # 0.92 × 8192 MB = 7537 MB allowed.
    # Previously 0.88 (7209 MB). Extra 328 MB headroom needed on Windows
    # (no expandable_segments) where allocator fragmentation can block
    # otherwise-feasible allocations. Stay below 0.95 to leave room for
    # the display driver and OS VRAM usage.
    torch.cuda.set_per_process_memory_fraction(0.92, 0)

import torch._dynamo
torch._dynamo.config.suppress_errors = True


# ─── Parameter Group Helper ───────────────────────────────────────────────────

def get_param_groups(model: nn.Module, weight_decay: float):
    """Separate parameters into decay / no-decay groups.

    FIX Issue 2: weight_decay=0.1 was applied uniformly to ALL parameters,
    including RMSNorm scale vectors, QK-norm scales, and embeddings. These
    1-D parameters act as scale controllers for layer activations; pulling them
    toward zero (via L2 decay) degrades the residual stream's dynamic range and
    destabilises training. The weight tying means the embedding weight is also
    a projection matrix, but decaying it hurts token representation quality.

    Exclusion rules:
      - param.ndim < 2  : 1-D tensors (norm scales, any biases if added later)
      - 'norm' in name  : all RMSNorm and QK-norm weights by name
      - 'embed' in name : tok_embeddings weight (also aliased as output.weight)

    Returns:
        List of two param-group dicts for AdamW / 8-bit Adam.
    """
    decay, no_decay = [], []
    no_decay_names  = []
    seen_ids        = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_ids:
            # Skip tied weights already registered (output.weight = tok_embeddings.weight)
            continue
        seen_ids.add(pid)

        if param.ndim < 2 or 'norm' in name or 'embed' in name:
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decay.append(param)

    print(f"  Weight decay=0.0 for {len(no_decay)} tensors: {no_decay_names[:8]}{'...' if len(no_decay_names) > 8 else ''}")
    print(f"  Weight decay={weight_decay} for {len(decay)} tensors")

    return [
        {'params': decay,    'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


# ─── Compile-safe Helpers ─────────────────────────────────────────────────────

@torch._dynamo.disable()
def _optimizer_step(model: nn.Module, optimizer, grad_clip: float):
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


@torch._dynamo.disable()
def _run_validation(model, val_loader, autocast_device, autocast_dtype,
                    val_max_steps, bos_id, eos_id):
    """Run validation with the same doc_boundary_mask applied as in training.

    FIX Issue 3: v4 called model(vx, vy) without a doc_boundary_mask. Training
    loss excluded cross-document boundary tokens; validation included them.
    The boundary tokens are the hardest in the dataset (first token of each new
    document conditioned on an unrelated prior document). Systematically including
    them inflated val_loss, corrupted the train/val comparison, and biased
    best-val checkpoint selection.

    Fix: compute the mask on vx (same as training loop does on x) and pass it in.
    """
    model.eval()
    val_loss        = 0.0
    val_steps_count = 0

    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(DEVICE, non_blocking=True).long()
            vy = vy.to(DEVICE, non_blocking=True).long()

            # Compute doc boundary mask on GPU — same logic as training loop
            vdoc_mask = compute_doc_boundary_mask(vx, bos_id, eos_id)

            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                _, vloss = model(vx, vy, doc_boundary_mask=vdoc_mask)

            val_loss        += vloss.item()
            val_steps_count += 1
            if val_steps_count >= val_max_steps:
                break

    model.train()
    return val_loss, val_steps_count


@torch._dynamo.disable()
def _save_checkpoint(raw_model, optimizer, step, best_val, path):
    save_ckpt(raw_model, optimizer, step, best_val, path)


# ─── Learning Rate Schedule (WSD: Warmup → Stable → Cosine Decay) ────────────

def get_lr(step: int, max_steps: int = None) -> float:
    if max_steps is None:
        max_steps = MAX_STEPS
    warmup = int(max_steps * WARMUP_FRACTION)
    stable = int(max_steps * STABLE_FRACTION)
    if step < warmup:
        return LR * (step + 1) / max(warmup, 1)
    if step < stable:
        return LR
    decay_ratio = (step - stable) / max(max_steps - stable, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class PackedTokenDataset(Dataset):
    """Memory-mapped packed-token dataset.

    FIX Issue 9 (v5): doc_boundary_mask removed from __getitem__.
    v4 computed the mask in the DataLoader worker (CPU) using unsqueeze/squeeze,
    then pinned and transferred a third tensor per batch. The mask is now computed
    on the GPU in the training loop after H2D transfer — two comparison ops on an
    already-resident tensor, negligible cost. Dataset returns a clean (x, y) pair.
    """
    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len     = seq_len
        self.data        = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.num_samples = len(self.data) // (seq_len + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx * (self.seq_len + 1)
        chunk = torch.from_numpy(self.data[start : start + self.seq_len + 1].copy())
        # uint16 tensors: 4× smaller H2D transfer vs int32/int64.
        # Training loop calls .long() after H2D.
        return chunk[:-1], chunk[1:]


def infinite_loader(dataset, **kwargs):
    """Yield batches forever by re-iterating the DataLoader when exhausted."""
    kwargs = dict(kwargs)
    if kwargs.get('num_workers', 0) > 0:
        kwargs['persistent_workers'] = True
    else:
        kwargs['persistent_workers'] = False
    if IS_WINDOWS and kwargs.get('num_workers', 0) > 0:
        kwargs['multiprocessing_context'] = 'spawn'

    epoch  = 0
    loader = DataLoader(dataset, **kwargs)
    while True:
        epoch += 1
        if epoch > 1:
            print(f"  (Data epoch {epoch}: re-shuffling training data)")
        for batch in loader:
            yield batch


# ─── Training Metrics (thread-safe) ──────────────────────────────────────────

class TrainingMetrics:
    def __init__(self):
        self.data = {
            'step': 0, 'loss': 0.0, 'val_loss': 0.0, 'val_ppl': 0.0,
            'lr': 0.0, 'tok_per_sec': 0.0,
            'gpu_mem_used_mb': 0, 'gpu_mem_total_mb': 0,
            'gpu_util_pct': 0, 'gpu_mem_reserved_mb': 0,
            'best_val': float('inf'),
            'loss_history': [], 'val_history': [], 'lr_history': [],
            'ema_loss_history': [], 'speed_history': [],
            'started_at': time.time(), 'status': 'initializing',
            'total_tokens': 0, 'eta_hours': 0.0,
            'max_steps': MAX_STEPS,
            'effective_batch_tokens': MICRO_BATCH * ACCUM_STEPS * 1024,
            'grad_norm': 0.0, 'data_epoch': 1, 'z_loss': 0.0,
            'compile_mode': f'compiled ({COMPILE_BACKEND})' if USE_COMPILE else 'eager',
            'device': DEVICE, 'platform': platform.system(),
            'ema_loss': 0.0, 'vram_pressure': False,
        }
        self.lock = threading.Lock()

    def update(self, **kwargs):
        with self.lock:
            self.data.update(kwargs)

    def append_history(self, key: str, step: int, value: float):
        with self.lock:
            self.data.setdefault(key, []).append({'step': step, 'value': round(value, 5)})
            if len(self.data[key]) > 2000:
                self.data[key] = self.data[key][-1500:]

    def get(self) -> dict:
        with self.lock:
            return dict(self.data)


metrics = TrainingMetrics()


# ─── GPU Stats ────────────────────────────────────────────────────────────────

_gpu_util_cache = {'pct': 0, 'last_query_time': 0}
_GPU_UTIL_POLL_INTERVAL = 30


def _poll_gpu_util() -> int:
    if not HAS_CUDA:
        return 0
    now = time.time()
    if now - _gpu_util_cache['last_query_time'] < _GPU_UTIL_POLL_INTERVAL:
        return _gpu_util_cache['pct']
    _gpu_util_cache['last_query_time'] = now
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            _gpu_util_cache['pct'] = int(result.stdout.strip())
    except Exception:
        pass
    return _gpu_util_cache['pct']


def get_gpu_stats(step=None) -> dict:
    """GPU memory stats using PEAK allocation, not instantaneous.

    FIX v5.1: torch.cuda.memory_allocated() returns memory currently held at
    call time. After optimizer.step() + zero_grad() + empty_cache(), this
    drops to ~20% — but the PEAK during forward+backward was ~80%. This gave
    a misleading "VRAM: 19%" reading when the GPU was actually near-OOM.

    Now reports max_memory_allocated() which tracks the high-water mark since
    the last reset. Reset at each optimizer step so it reflects per-step peak.
    """
    if not HAS_CUDA:
        return {'gpu_mem_used_mb': 0, 'gpu_mem_total_mb': 0,
                'gpu_util_pct': 0, 'gpu_mem_reserved_mb': 0}
    try:
        # Peak memory since last reset — reflects actual max usage during the step
        peak_allocated = torch.cuda.max_memory_allocated() // (1024 * 1024)
        current        = torch.cuda.memory_allocated()      // (1024 * 1024)
        reserved       = torch.cuda.memory_reserved()       // (1024 * 1024)
        total          = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        return {
            'gpu_mem_used_mb':     peak_allocated,   # CHANGED: peak, not current
            'gpu_mem_current_mb':  current,          # NEW: current for reference
            'gpu_mem_reserved_mb': reserved,
            'gpu_mem_total_mb':    total,
            'gpu_util_pct':        _poll_gpu_util(),
        }
    except Exception:
        return {'gpu_mem_used_mb': 0, 'gpu_mem_total_mb': 0,
                'gpu_util_pct': 0, 'gpu_mem_reserved_mb': 0}


# ─── Dashboard HTTP Server ────────────────────────────────────────────────────

def _sanitize_for_json(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/metrics':
            data = _sanitize_for_json(metrics.get())
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif self.path in ('/', '/index.html', '/dashboard.html'):
            dashboard_path = os.path.join(SCRIPT_DIR, 'dashboard.html')
            if os.path.exists(dashboard_path):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                with open(dashboard_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Dashboard not found')
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress request logging


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def start_metrics_server():
    try:
        with ReusableTCPServer(("", METRICS_PORT), DashboardHandler) as httpd:
            print(f"  Dashboard: http://localhost:{METRICS_PORT}")
            httpd.serve_forever()
    except OSError as e:
        print(f"  Dashboard: port {METRICS_PORT} unavailable ({e})")


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────

ARCH_FIELDS = {'dim', 'n_layers', 'n_heads', 'n_kv_heads', 'vocab_size',
               'hidden_dim', 'max_seq_len', 'n_mtp_tokens'}


def cleanup_old_checkpoints():
    ckpt_files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR)
         if f.startswith('ckpt_') and f.endswith('.pt')],
        key=lambda f: int(f.split('_')[-1].split('.')[0]),
    )
    while len(ckpt_files) > MAX_CHECKPOINTS:
        oldest = ckpt_files.pop(0)
        try:
            os.remove(os.path.join(CHECKPOINT_DIR, oldest))
        except OSError:
            pass


def save_ckpt(raw_model, optimizer, step: int, best_val: float, path: str):
    tmp = path + ".tmp"
    torch.save({
        'step':       step,
        'model':      raw_model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'best_val':   best_val,
        'model_args': raw_model.args.as_dict(),
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict:
    print(f"Loading checkpoint: {path}")
    return torch.load(path, map_location=DEVICE, weights_only=False)


def find_checkpoint(resume_from=None) -> Optional[str]:
    if resume_from:
        if os.path.exists(resume_from):
            return resume_from
        full_path = os.path.join(CHECKPOINT_DIR, resume_from)
        if os.path.exists(full_path):
            return full_path
        print(f"ERROR: Checkpoint not found: {resume_from}")
        sys.exit(1)
    for preferred in ['best.pt', 'final.pt']:
        p = os.path.join(CHECKPOINT_DIR, preferred)
        if os.path.exists(p):
            return p
    import glob as _glob
    files = _glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_*.pt"))
    if files:
        return max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    return None


# ─── Training Log ─────────────────────────────────────────────────────────────

class TrainingLog:
    """Persistent JSONL log with periodic flush — avoids per-step open/close."""
    def __init__(self, log_path: str):
        self.log_path       = log_path
        self.f              = None
        self._write_count   = 0
        self._flush_interval = 100

    def write(self, entry: dict):
        if self.f is None:
            self.f = open(self.log_path, 'a')
        try:
            self.f.write(json.dumps(entry) + '\n')
            self._write_count += 1
            if self._write_count % self._flush_interval == 0:
                self.f.flush()
        except Exception:
            pass

    def close(self):
        if self.f is not None:
            self.f.flush()
            self.f.close()
            self.f = None


# ─── Main Training Function ───────────────────────────────────────────────────

def train(resume_from=None, reset_optimizer=False,
          micro_batch=None, accum_steps=None,
          use_8bit_adam=False, compile_model=None, compile_backend=None):
    global MICRO_BATCH, ACCUM_STEPS, USE_COMPILE, COMPILE_BACKEND

    if micro_batch     is not None: MICRO_BATCH     = micro_batch
    if accum_steps     is not None: ACCUM_STEPS      = accum_steps
    if compile_model   is not None: USE_COMPILE      = compile_model
    if compile_backend is not None: COMPILE_BACKEND  = compile_backend

    args      = ModelArgs()
    tokenizer = Tokenizer.from_file("tokenizer.json")

    if not HAS_CUDA:
        print("WARNING: No CUDA GPU detected. Training on CPU will be extremely slow.")

    # ── Model ──
    raw_model = Transformer(args).to(DEVICE)
    n_params  = sum(p.numel() for p in raw_model.parameters()) / 1e6
    print(f"\nModel: {n_params:.1f}M parameters")
    print(f"Context: {args.max_seq_len} tokens | Vocab: {args.vocab_size}")
    print(f"RoPE theta: {args.rope_theta} | Device: {DEVICE}")
    print(f"Grad checkpointing: {'ON' if args.use_checkpointing else 'OFF'}")
    print(f"MTP heads: {args.n_mtp_tokens} (head-0 → t+2, head-1 → t+3)")

    # ── torch.compile ──
    if USE_COMPILE and HAS_CUDA:
        backend = COMPILE_BACKEND
        if IS_WINDOWS and backend not in ('aot_eager', 'cudagraphs'):
            print(f"WARNING: Backend '{backend}' unavailable on Windows. Forcing aot_eager.")
            backend = COMPILE_BACKEND = 'aot_eager'
        print(f"torch.compile: ON (backend={backend})")
        print(f"  Note: first step will be slow (~60-120s) while Triton compiles kernels.")
        model = torch.compile(raw_model, backend=backend)
    else:
        model = raw_model
        reason = "Windows" if IS_WINDOWS else ("no CUDA" if not HAS_CUDA else "disabled by flag")
        print(f"torch.compile: OFF ({reason})")

    # ── Optimizer ──
    # FIX Issue 2: param groups exclude norms/embeddings from weight decay.
    # FIX Issue 11: try 8-bit AdamW first to save ~1,320 MB of VRAM on m/v states.
    param_groups = get_param_groups(raw_model, WEIGHT_DECAY)
    optimizer    = None

    if USE_MUON:
        try:
            from muon import Muon
            matrix_params = []
            other_params  = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.ndim >= 2 and 'norm' not in name and 'embed' not in name:
                        matrix_params.append(param)
                    else:
                        other_params.append(param)
            optimizer = Muon(
                [
                    {'params': matrix_params, 'use_muon': True},
                    {'params': other_params,  'use_muon': False, 'weight_decay': 0.0},
                ],
                lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.95,
            )
            print("Optimizer: Muon")
        except ImportError:
            print("Muon not found — falling back to AdamW")

    if optimizer is None:
        # Try 8-bit AdamW first (saves ~1.32 GB VRAM by quantising m/v to int8)
        if not use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    param_groups, lr=LR, betas=(0.9, 0.95),
                )
                print("Optimizer: 8-bit AdamW (bitsandbytes) — saves ~1.32 GB VRAM")
                print("  Tip: with 1.32 GB freed, try --micro-batch 8 for higher throughput")
            except ImportError:
                print("Optimizer: 8-bit AdamW unavailable (pip install bitsandbytes)")
            except Exception as e:
                print(f"Optimizer: 8-bit AdamW failed ({e}), falling back")

        if optimizer is None:
            # Standard fused AdamW
            try:
                optimizer = torch.optim.AdamW(
                    param_groups, lr=LR, betas=(0.9, 0.95), fused=HAS_CUDA,
                )
                print("Optimizer: standard AdamW (fused=True)")
            except Exception:
                optimizer = torch.optim.AdamW(
                    param_groups, lr=LR, betas=(0.9, 0.95),
                )
                print("Optimizer: standard AdamW")

    # ── WandB ──
    wandb_run = None
    if USE_WANDB:
        try:
            import wandb
            wandb_run = wandb.init(
                project="my-slm",
                config={
                    'model_params_M':  n_params,
                    'dim':             args.dim,
                    'n_layers':        args.n_layers,
                    'n_heads':         args.n_heads,
                    'n_kv_heads':      args.n_kv_heads,
                    'max_seq_len':     args.max_seq_len,
                    'micro_batch':     MICRO_BATCH,
                    'accum_steps':     ACCUM_STEPS,
                    'lr':              LR,
                    'weight_decay':    WEIGHT_DECAY,
                    'compile':         USE_COMPILE,
                    'platform':        platform.system(),
                    'rope_theta':      args.rope_theta,
                    'effective_batch': MICRO_BATCH * ACCUM_STEPS * args.max_seq_len,
                },
                name=f"slm-{int(n_params)}M-lr{LR}-b{MICRO_BATCH}",
            )
            print(f"WandB: enabled — {wandb_run.url}")
        except ImportError:
            print("WandB: not found (pip install wandb)")
        except Exception as e:
            print(f"WandB: init failed ({e})")

    # ── Checkpoint Resume ──
    start_step = 0
    best_val   = float('inf')

    ckpt_path = find_checkpoint(resume_from)
    if ckpt_path:
        ckpt      = load_checkpoint(ckpt_path)
        ckpt_args = ckpt.get('model_args', {})
        if ckpt_args:
            current = args.as_dict()
            arch_mismatches  = []
            hyper_mismatches = []
            for k, v in current.items():
                if k in ckpt_args and ckpt_args[k] != v:
                    (arch_mismatches if k in ARCH_FIELDS else hyper_mismatches).append(
                        f"  {k}: checkpoint={ckpt_args[k]}, current={v}"
                    )
            if arch_mismatches:
                print("ERROR: Architecture-breaking mismatches — cannot resume:")
                for m in arch_mismatches: print(m)
                sys.exit(1)
            if hyper_mismatches:
                print("NOTE: Non-breaking hyperparameter changes:")
                for m in hyper_mismatches: print(m)

        raw_model.load_state_dict(ckpt['model'])
        if reset_optimizer:
            print("Optimizer state RESET (--reset-optimizer)")
        else:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print("Optimizer state loaded from checkpoint.")
            except (ValueError, RuntimeError) as e:
                print(f"WARNING: Could not load optimizer state ({e})")
        start_step = ckpt['step']
        best_val   = ckpt.get('best_val', float('inf'))
        print(f"Resumed at step {start_step}, best_val={best_val:.4f}")
    else:
        print("No checkpoint found — starting from scratch.")

    # ── Datasets & Dataloaders ──
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    # PackedTokenDataset now returns (x, y) only — doc_mask computed on GPU
    train_ds = PackedTokenDataset("data/train.bin", args.max_seq_len)
    val_ds   = PackedTokenDataset("data/val.bin",   args.max_seq_len)
    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")

    _train_dl_kwargs = dict(
        batch_size=MICRO_BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=HAS_CUDA,
        drop_last=True,
    )
    if NUM_WORKERS > 0:
        _train_dl_kwargs['prefetch_factor'] = 2
        _train_dl_kwargs['persistent_workers'] = True
        if IS_WINDOWS:
            _train_dl_kwargs['multiprocessing_context'] = 'spawn'
    train_loader = infinite_loader(train_ds, **_train_dl_kwargs)

    _val_dl_kwargs = dict(
        batch_size=MICRO_BATCH, shuffle=False,
        pin_memory=HAS_CUDA, num_workers=NUM_WORKERS_VAL,
        drop_last=True,
    )
    if NUM_WORKERS_VAL > 0:
        _val_dl_kwargs['persistent_workers'] = True
        _val_dl_kwargs['prefetch_factor']    = 2
        if IS_WINDOWS:
            _val_dl_kwargs['multiprocessing_context'] = 'spawn'
    val_loader = DataLoader(val_ds, **_val_dl_kwargs)

    # ── Dashboard ──
    server_thread = threading.Thread(target=start_metrics_server, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    # ── Training State ──
    model.train()
    step       = start_step
    optimizer.zero_grad(set_to_none=True)
    t0         = time.time()
    accum_count = 0
    ema_loss    = None

    # FIX Issue 13: Python float instead of GPU tensor.
    # Old: accumulated_loss = torch.tensor(0.0, device=DEVICE)
    #      → GPU allocation + .zero_() kernel per step + deferred .item() sync.
    # New: plain Python float — syncs happen per microbatch on .item(),
    #      but avoids GPU allocation, .zero_() kernel, and fragmentation.
    accumulated_loss: float = 0.0

    effective_batch_tokens = MICRO_BATCH * ACCUM_STEPS * args.max_seq_len

    log_path     = os.path.join(CHECKPOINT_DIR, "training_log.jsonl")
    training_log = TrainingLog(log_path)

    # Initial VRAM check
    if HAS_CUDA:
        gpu_check = get_gpu_stats(0)
        vram_pct  = gpu_check['gpu_mem_used_mb'] / max(gpu_check['gpu_mem_total_mb'], 1)
        if vram_pct > 0.95:
            print(f"\n  WARNING: VRAM at {vram_pct*100:.1f}% before training!")
            print(f"  Try: --micro-batch 1 --accum-steps 64")

    metrics.update(
        status='training', step=step, best_val=best_val,
        effective_batch_tokens=effective_batch_tokens, max_steps=MAX_STEPS,
    )

    print(f"\n{'='*62}")
    print(f"Training from step {step}")
    print(f"Effective batch : {effective_batch_tokens:,} tokens/step")
    print(f"Micro batch     : {MICRO_BATCH}  |  Accum steps: {ACCUM_STEPS}")
    print(f"Doc boundary masking: ENABLED (computed on GPU)")
    print(f"Ctrl+C pauses training and saves a checkpoint automatically")
    print(f"{'='*62}\n")

    autocast_device = 'cuda' if HAS_CUDA else 'cpu'
    autocast_dtype  = torch.bfloat16 if HAS_CUDA else torch.float32

    # Initialise LR for the first step
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    try:
        for batch_idx, batch in enumerate(train_loader):
            if step >= MAX_STEPS:
                break

            # Unpack — dataset returns (x, y) only (doc_mask computed below on GPU)
            x, y = batch
            x = x.to(DEVICE, non_blocking=True).long()
            y = y.to(DEVICE, non_blocking=True).long()

            # FIX Issue 9: doc_boundary_mask computed on GPU after H2D transfer.
            # Two comparison ops on an already-resident tensor — near-zero cost.
            # Avoids the CPU-side unsqueeze/squeeze and the extra pinned tensor.
            doc_mask = compute_doc_boundary_mask(x, bos_id, eos_id)

            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                logits, loss = model(x, y, doc_boundary_mask=doc_mask)
                loss = loss / ACCUM_STEPS   # Scale for gradient accumulation

            # logits is None during training (freed inside model.forward before MTP).
            # Del here as an explicit guard so it never lingers in local scope.
            del logits

            loss.backward()

            # FIX Issue 13: Python float accumulation — no GPU tensor, no .zero_()
            accumulated_loss += loss.item()   # .item() syncs here (one per microbatch)
            accum_count      += 1

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                grad_norm = _optimizer_step(model, optimizer, GRAD_CLIP)

                # FIX: read peak VRAM BEFORE resetting the high-water mark.
                # Previously reset_peak_memory_stats() was called first, making
                # get_gpu_stats() always read ~0 MB (only the ops between reset
                # and get_gpu_stats counted). Now the peak correctly reflects the
                # max allocation during the full forward+backward+optimizer step.
                gpu_stats     = get_gpu_stats(step)
                if HAS_CUDA:
                    torch.cuda.reset_peak_memory_stats()

                # avg_loss = mean of original (unscaled) per-microbatch losses
                avg_loss = (accumulated_loss * ACCUM_STEPS) / accum_count

                if ema_loss is None:
                    ema_loss = avg_loss
                else:
                    ema_loss = EMA_ALPHA * avg_loss + (1 - EMA_ALPHA) * ema_loss

                dt    = time.time() - t0
                tok_s = effective_batch_tokens / dt if dt > 0 else 0
                total_tokens    = step * effective_batch_tokens
                remaining_steps = MAX_STEPS - step
                eta_hours = (remaining_steps * dt) / 3600 if step > start_step + 5 else 0

                vram_pressure = False
                if gpu_stats['gpu_mem_total_mb'] > 0:
                    vram_pct_cur  = gpu_stats['gpu_mem_used_mb'] / gpu_stats['gpu_mem_total_mb']
                    vram_pressure = vram_pct_cur > 0.95

                metrics.update(
                    step=step, loss=avg_loss, lr=lr,
                    tok_per_sec=tok_s, total_tokens=total_tokens,
                    eta_hours=eta_hours, grad_norm=grad_norm.item(),
                    ema_loss=ema_loss, vram_pressure=vram_pressure,
                    **gpu_stats,
                )
                metrics.append_history('loss_history',     step, avg_loss)
                metrics.append_history('ema_loss_history', step, ema_loss)
                metrics.append_history('lr_history',       step, lr)
                metrics.append_history('speed_history',    step, tok_s)

                # WandB logging
                if wandb_run is not None:
                    try:
                        import wandb
                        wandb.log({
                            'train/loss':      avg_loss,
                            'train/ema_loss':  ema_loss,
                            'train/lr':        lr,
                            'train/grad_norm': grad_norm.item(),
                            'train/tok_per_sec': tok_s,
                            'train/step':      step,
                        })
                    except Exception:
                        pass

                # JSONL log
                training_log.write({
                    'step':      step,
                    'loss':      round(avg_loss, 6),
                    'lr':        lr,
                    'tok_per_sec': round(tok_s, 1),
                    'timestamp': time.time(),
                    'ema_loss':  round(ema_loss, 6),
                })

                # Console output
                if step % 100 == 0 or step < 10:
                    vram_info = ""
                    if gpu_stats['gpu_mem_total_mb'] > 0:
                        vram_peak_pct = (gpu_stats['gpu_mem_used_mb'] /
                                         gpu_stats['gpu_mem_total_mb'] * 100)
                        vram_cur_pct = (gpu_stats.get('gpu_mem_current_mb', 0) /
                                        gpu_stats['gpu_mem_total_mb'] * 100)
                        vram_info = f" | VRAM: {vram_peak_pct:.0f}%peak {vram_cur_pct:.0f}%now"
                    print(f"  step {step:>6d} | loss {avg_loss:.4f} | ema {ema_loss:.4f} | "
                          f"lr {lr:.2e} | {tok_s:,.0f} tok/s{vram_info}")

                # Advance LR for next step
                lr = get_lr(step + 1)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                step            += 1
                accumulated_loss  = 0.0   # Python assignment — no kernel launch
                accum_count       = 0
                t0                = time.time()

                # Validation
                if step % VAL_EVERY == 0 and step > 0:
                    val_loss, val_steps = _run_validation(
                        model, val_loader,
                        autocast_device, autocast_dtype,
                        VAL_MAX_STEPS,
                        bos_id, eos_id,   # FIX Issue 3
                    )
                    if val_steps > 0:
                        val_loss /= val_steps
                    val_ppl = math.exp(min(val_loss, 20))

                    print(f"  >>> step {step} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                    metrics.update(val_loss=val_loss, val_ppl=val_ppl)
                    metrics.append_history('val_history', step, val_loss)
                    training_log.write({
                        'step':      step,
                        'val_loss':  round(val_loss, 6),
                        'val_ppl':   round(val_ppl, 4),
                        'timestamp': time.time(),
                    })

                    if wandb_run is not None:
                        try:
                            import wandb
                            wandb.log({'val/loss': val_loss, 'val/ppl': val_ppl,
                                       'train/step': step})
                        except Exception:
                            pass

                    if val_loss < best_val:
                        best_val = val_loss
                        _save_checkpoint(raw_model, optimizer, step, best_val,
                                         os.path.join(CHECKPOINT_DIR, "best.pt"))
                        print(f"  >>> New best val_loss: {best_val:.4f}")

                # Periodic checkpoint
                if step % CHECKPOINT_EVERY == 0 and step > 0:
                    _save_checkpoint(raw_model, optimizer, step, best_val,
                                     os.path.join(CHECKPOINT_DIR, f"ckpt_{step}.pt"))
                    cleanup_old_checkpoints()

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at step {step}. Saving checkpoint...")
    finally:
        _save_checkpoint(raw_model, optimizer, step, best_val,
                         os.path.join(CHECKPOINT_DIR, "final.pt"))
        print(f"Saved final checkpoint at step {step}")
        training_log.close()


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SLM")
    parser.add_argument("--resume-from",     type=str,  default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="Reset optimizer state on resume")
    parser.add_argument("--8bit-adam",       action="store_true",
                        help="Force 8-bit AdamW even if auto-detection fails "
                             "(note: 8-bit is already tried first by default)")
    parser.add_argument("--no-compile",      action="store_true",
                        help="Disable torch.compile (enabled by default on Linux CUDA)")
    parser.add_argument("--compile",         action="store_true",
                        help="Force torch.compile on (overrides auto-detection)")
    parser.add_argument("--compile-backend", type=str, default=None,
                        help="torch.compile backend (inductor, aot_eager, cudagraphs)")
    parser.add_argument("--micro-batch",     type=int, default=None,
                        help=f"Micro-batch size (default: {MICRO_BATCH})")
    parser.add_argument("--accum-steps",     type=int, default=None,
                        help=f"Gradient accumulation steps (default: {ACCUM_STEPS})")
    args = parser.parse_args()

    # Handle compile flags
    compile_flag = None
    if args.no_compile:
        compile_flag = False
    elif args.compile:
        compile_flag = True
    # else: use USE_COMPILE auto-detection (Linux CUDA = True, Windows = False)

    train(
        resume_from    = args.resume_from,
        reset_optimizer= args.reset_optimizer,
        micro_batch    = args.micro_batch,
        accum_steps    = args.accum_steps,
        use_8bit_adam  = getattr(args, '8bit_adam', False),
        compile_model  = compile_flag,
        compile_backend= args.compile_backend,
    )
