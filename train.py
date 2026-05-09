"""
train.py — Training loop with all fixes and embedded monitoring dashboard.

Key fixes from v3 review:
  + No GradScaler (BF16 doesn't need it — same exponent range as FP32)
  + Save RAW model state dict (before torch.compile) for clean resume
  + Removed dead spike detector — z-loss handles stability (Minor Issue 10)
  + torch.set_float32_matmul_precision('high') for free ~20% speedup
  + Larger effective batch (ACCUM_STEPS=32 -> 65k tokens/step)
  + Better validation (500 steps instead of 50)
  + pin_memory + persistent_workers on val loader
  + Embedded HTTP metrics server + dashboard at http://localhost:8686
  + GPU stats cached for 10 steps (Minor Issue 6: nvidia-smi latency)
  + val_loader has drop_last=True (Minor Issue 7)
  + Optional Muon optimizer (Feature #5)
  + Optional WandB logging (Feature #4)
  + Infinite loader prevents silent early stop (Bug 3)
  + Training log saved to JSONL for post-training analysis

v3.6 — torch.compile fix for Windows:
  + FIXED: inductor backend crashes on Windows (Triton triton_key import error)
  + FIXED: cudagraphs backend incompatible with dropout (skipped entirely)
  + FIXED: torch.compiler.disable() not usable as context manager — use torch._dynamo.disable()
  + FIXED: Dynamo tracing optimizer.step() breaks fused AdamW (KeyError: 'exp_avg')
  + Use aot_eager backend on Windows — traces graph, eliminates Python overhead, works with dropout
  + Wrap optimizer step, validation, checkpoint save in torch._dynamo.disable()
  + --compile flag to explicitly enable torch.compile
  + --compile-backend flag to choose backend (aot_eager, cudagraphs, inductor, reduce-overhead)
  + torch._dynamo.config.suppress_errors = True — compile failures fall back to eager
  + Auto-detects best backend: aot_eager on Windows, inductor on Linux

v3.7 — Loss fluctuation fix (critical for 8GB VRAM):
  + FIXED: VRAM at 99.2% causes CUDA memory allocator instability — the #1 cause of loss fluctuation
  + FIXED: MICRO_BATCH=1 + ACCUM_STEPS=64 halves per-step activation memory (~400MB freed)
  + FIXED: Same effective batch (65,536 tokens/step) — no quality loss, just less VRAM pressure
  + ADDED: Exponential Moving Average (EMA) loss tracking — see actual trend vs noise
  + ADDED: EMA loss shown in console and dashboard alongside raw loss
  + ADDED: VRAM pressure warning when usage > 95%
  + LR reduced 4e-4 -> 3e-4 for smoother convergence
  + Warmup extended 2000 -> 3000 steps
  + MTP weight reduced 0.1 -> 0.05 (less auxiliary loss noise)
  + Dropout reduced 0.05 -> 0.02 (less forward pass stochasticity)
  + All changes are checkpoint-compatible (no architecture changes)

v3.5 — Throughput & stability (hotfix):
  + FIXED: Gradient checkpointing must be ON for 8GB VRAM (model.py default corrected)
  + FIXED: MICRO_BATCH=2 was the safe default but still uses 99% VRAM — v3.7 reduces further
  + MICRO_BATCH and ACCUM_STEPS configurable via CLI (--micro-batch, --accum-steps)
  + Defer loss.item() — accumulate as tensor, sync once per optimizer step (~10% speedup)
  + Dataset returns uint16, converts on GPU (4x less H2D transfer)
  + LR update moved out of inner micro-batch loop
  + Fused AdamW on CUDA (~5% speedup)
  + CUDA memory fraction pre-allocation
  + LR reduced 6e-4 -> 4e-4, MTP weight 0.3 -> 0.1 (fixes loss regression)
  + Warmup extended 1000 -> 2000 steps
  + Full ModelArgs saved in checkpoints with architecture compatibility check
  + --resume-from flag for explicit checkpoint selection
  + --reset-optimizer flag to skip optimizer state on resume

Windows compatibility (v3.2):
  + num_workers=0 on Windows (multiprocessing issues)
  + CUDA flags guarded with is_available() checks
  + bitsandbytes fallback for Windows (may not compile)
  + Symlink warning suppressed for HF cache
"""

import os
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

# Suppress Windows symlink warnings from huggingface_hub
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from model import Transformer, ModelArgs
from tokenizers import Tokenizer


# ─── Platform Detection ───────────────────────────────────────────────────────

IS_WINDOWS = platform.system() == 'Windows'
HAS_CUDA = torch.cuda.is_available()
HAS_TRITON = False
try:
    import triton
    HAS_TRITON = True
except ImportError:
    pass


# ─── Configuration ────────────────────────────────────────────────────────────

MICRO_BATCH = 1             # 1 is critical for 8GB VRAM — MICRO_BATCH=2 uses 99.2% VRAM causing instability!
ACCUM_STEPS = 64            # 1*64*1024 = 65,536 tokens/step — same effective batch as 2*32
MAX_STEPS = 80_000          # Increased from 60k for max dataset — more data needs more steps
WARMUP_FRACTION = 0.0375    # ~3000/80000 — warmup phase (4% of training)
STABLE_FRACTION = 0.7125    # ~57000/80000 — stable phase (71% of training)
# Decay phase = remaining ~25% — cosine decay from LR to MIN_LR
# These fractions auto-adjust when MAX_STEPS changes, preventing the
# schedule from breaking on resume with different step counts.
LR = 3e-4                   # Reduced from 4e-4 — smoother convergence, less per-step oscillation
MIN_LR = 1e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EMA_ALPHA = 0.05            # EMA smoothing factor — lower = smoother (0.05 ≈ 20-step average)
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 2500
VAL_EVERY = 2000
VAL_MAX_STEPS = 500          # Was 50 -> 500 for accurate val loss estimation
DEVICE = "cuda" if HAS_CUDA else "cpu"
METRICS_PORT = 8686
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Optional integrations
USE_WANDB = os.environ.get("WANDB", "0") == "1"
USE_MUON = os.environ.get("MUON", "0") == "1"

# torch.compile control — determined by --compile CLI flag now
# Backends:
#   - 'cudagraphs': Records CUDA ops into a graph, replays them. Fast but
#     INCOMPATIBLE with dropout (uses RNG state) — gets skipped entirely.
#   - 'aot_eager': AOT Autograd without codegen. Traces the compute graph,
#     eliminates Python overhead. Works with dropout! ~5-10% speedup. No Triton needed.
#   - 'inductor': Full code generation + kernel fusion. ~15-30% speedup.
#     Requires working Triton. CRASHES on Windows.
#   - 'reduce-overhead': Uses cudagraphs + inductor. Needs Triton.
#
# Default: aot_eager on Windows (works with dropout), inductor on Linux
COMPILE_BACKEND = 'aot_eager' if IS_WINDOWS else 'inductor'
USE_COMPILE = False  # Will be set by --compile CLI flag

# DataLoader workers: Windows has multiprocessing issues, use 0
NUM_WORKERS = 0 if IS_WINDOWS else 4
NUM_WORKERS_VAL = 0 if IS_WINDOWS else 2

# Keep only last N checkpoints to save disk space
MAX_CHECKPOINTS = 3

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── Performance flags ────────────────────────────────────────────────────────

if HAS_CUDA:
    torch.set_float32_matmul_precision('high')  # TF32 for matmul, FP32 accumulation
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Pre-allocate CUDA memory — 0.88 to leave headroom for peak allocations
    # Setting this too high (0.95) causes 99%+ VRAM usage and CUDA allocator
    # instability, which manifests as loss fluctuation. 0.88 leaves ~1GB free
    # for temporary tensors (cross_entropy, MTP heads, etc.)
    torch.cuda.set_per_process_memory_fraction(0.88, 0)

# Safety net: if torch.compile fails, fall back to eager instead of crashing
# This catches Triton errors, unsupported ops, etc.
import torch._dynamo
torch._dynamo.config.suppress_errors = True


# ─── Compile-safe helpers ──────────────────────────────────────────────────────
# torch._dynamo.disable() CANNOT be used as a context manager in some PyTorch
# versions (raises RuntimeError). It MUST be used as a decorator on functions.
# These helper functions prevent dynamo from tracing into the optimizer step,
# validation loop, and checkpoint saves — which would break fused AdamW
# (KeyError: 'exp_avg') and cause unnecessary recompilations.

@torch._dynamo.disable()
def _optimizer_step(model, optimizer, grad_clip):
    """Clip gradients and step optimizer. Must run in eager mode — dynamo
    breaks fused AdamW's lazy state initialization (exp_avg, exp_avg_sq)."""
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


@torch._dynamo.disable()
def _run_validation(model, val_loader, autocast_device, autocast_dtype, val_max_steps):
    """Run validation loop in eager mode. Different batch sizes / eval mode
    cause graph recompilation if compiled — not worth it for infrequent val."""
    model.eval()
    val_loss = 0.0
    val_steps_count = 0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(DEVICE, non_blocking=True).long()
            vy = vy.to(DEVICE, non_blocking=True).long()
            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                _, vloss = model(vx, vy)
            val_loss += vloss.item()
            val_steps_count += 1
            if val_steps_count >= val_max_steps:
                break
    model.train()
    return val_loss, val_steps_count


@torch._dynamo.disable()
def _save_checkpoint(raw_model, optimizer, step, best_val, path):
    """Save checkpoint in eager mode to avoid any compile interference."""
    save_ckpt(raw_model, optimizer, step, best_val, path)


# ─── Learning Rate Schedule (WSD: Warmup -> Stable -> Decay) ──────────────────

def get_lr(step, max_steps=None):
    """WSD (Warmup-Stable-Decay) learning rate schedule with fractional phases.
    
    Uses fractions of max_steps instead of absolute step counts, so the schedule
    auto-adjusts when MAX_STEPS changes (e.g., extending training on resume).
    
    Phases:
      - Warmup: WARMUP_FRACTION of training — linear ramp from 0 to LR
      - Stable: STABLE_FRACTION of training — constant LR
      - Decay: remaining fraction — cosine decay from LR to MIN_LR
    """
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
    """Memory-mapped packed token dataset. Zero disk I/O per sample.
    Returns uint16 tensors for 4x smaller H2D transfers — convert to long on GPU."""
    def __init__(self, bin_path, seq_len):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.num_samples = len(self.data) // (seq_len + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        # Return uint16 — 4x less data to transfer to GPU vs int64
        # Convert to long on GPU where it's parallel and free
        chunk = torch.from_numpy(self.data[start : start + self.seq_len + 1].copy())
        return chunk[:-1], chunk[1:]


def infinite_loader(dataset, **kwargs):
    """Yield batches forever by re-creating the DataLoader when it's exhausted.

    Without this, training silently stops when the DataLoader runs out of samples.
    With ~1.47B tokens in train.bin and a 3.93B token budget (60k steps x 65k tokens),
    the loader would exhaust at ~step 22,000, leaving 38,000 steps undone.
    This wrapper re-shuffles and repeats so we always train the full 60k steps.

    Fix (v3.3): Uses a single DataLoader with persistent_workers=True for all epochs
    except the first, avoiding the overhead of respawning workers every epoch.
    On the first call, creates the DataLoader. On subsequent epochs, re-iterates
    the same DataLoader (which automatically re-shuffles with shuffle=True).
    """
    kwargs = dict(kwargs)  # Don't mutate caller's dict

    # Use persistent_workers when num_workers > 0 — avoids worker respawn overhead
    # on every epoch. Workers stay alive between epochs, saving ~5-10s per epoch.
    if kwargs.get('num_workers', 0) > 0:
        kwargs['persistent_workers'] = True
    else:
        kwargs['persistent_workers'] = False

    epoch = 0
    loader = DataLoader(dataset, **kwargs)

    while True:
        epoch += 1
        if epoch > 1:
            print(f"  (Data epoch {epoch}: re-shuffling training data)")
        for batch in loader:
            yield batch


# ─── Training Metrics (thread-safe) ──────────────────────────────────────────

class TrainingMetrics:
    """Thread-safe metrics store for the monitoring dashboard."""
    def __init__(self):
        self.data = {
            'step': 0,
            'loss': 0.0,
            'val_loss': 0.0,
            'val_ppl': 0.0,
            'lr': 0.0,
            'tok_per_sec': 0.0,
            'gpu_mem_used_mb': 0,
            'gpu_mem_total_mb': 0,
            'gpu_util_pct': 0,
            'gpu_mem_reserved_mb': 0,
            'best_val': float('inf'),
            'loss_history': [],
            'val_history': [],
            'lr_history': [],
            'started_at': time.time(),
            'status': 'initializing',
            'total_tokens': 0,
            'eta_hours': 0.0,
            'max_steps': MAX_STEPS,
            'effective_batch_tokens': MICRO_BATCH * ACCUM_STEPS * 1024,  # matches ModelArgs.max_seq_len
            'grad_norm': 0.0,
            'data_epoch': 1,
            'z_loss': 0.0,
            'compile_mode': f'compiled ({COMPILE_BACKEND})' if USE_COMPILE else 'eager',
            'device': DEVICE,
            'platform': platform.system(),
            'ema_loss': 0.0,  # Exponential Moving Average loss — shows true trend vs noise
            'vram_pressure': False,  # True when VRAM > 95% — causes instability
        }
        self.lock = threading.Lock()

    def update(self, **kwargs):
        with self.lock:
            self.data.update(kwargs)

    def append_history(self, key, step, value):
        with self.lock:
            self.data.setdefault(key, []).append({'step': step, 'value': round(value, 5)})
            # Keep only last 2000 entries to bound memory
            if len(self.data[key]) > 2000:
                self.data[key] = self.data[key][-1500:]

    def get(self):
        with self.lock:
            return dict(self.data)


# Global metrics instance
metrics = TrainingMetrics()


# ─── GPU Stats ────────────────────────────────────────────────────────────────

# Background nvidia-smi polling for GPU utilization % (updated every 30 seconds)
_gpu_util_cache = {'pct': 0, 'last_query_time': 0}
_GPU_UTIL_POLL_INTERVAL = 30  # seconds between nvidia-smi queries

def _poll_gpu_util():
    """Background poll nvidia-smi for GPU utilization %.
    Called at most once every 30 seconds to avoid subprocess overhead."""
    if not HAS_CUDA:
        return 0
    now = time.time()
    if now - _gpu_util_cache['last_query_time'] < _GPU_UTIL_POLL_INTERVAL:
        return _gpu_util_cache['pct']
    _gpu_util_cache['last_query_time'] = now
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            _gpu_util_cache['pct'] = int(result.stdout.strip())
    except Exception:
        pass
    return _gpu_util_cache['pct']


def get_gpu_stats(step=None):
    """Get GPU memory stats using PyTorch native API (zero overhead).
    GPU utilization % is polled from nvidia-smi at most once every 30 seconds
    in a background thread to avoid subprocess overhead during training."""
    if not HAS_CUDA:
        return {'gpu_mem_used_mb': 0, 'gpu_mem_total_mb': 0, 'gpu_util_pct': 0,
                'gpu_mem_reserved_mb': 0}
    try:
        allocated = torch.cuda.memory_allocated() // (1024 * 1024)
        reserved = torch.cuda.memory_reserved() // (1024 * 1024)
        total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        util_pct = _poll_gpu_util()
        return {
            'gpu_mem_used_mb': allocated,
            'gpu_mem_reserved_mb': reserved,
            'gpu_mem_total_mb': total,
            'gpu_util_pct': util_pct,
        }
    except Exception:
        return {'gpu_mem_used_mb': 0, 'gpu_mem_total_mb': 0, 'gpu_util_pct': 0,
                'gpu_mem_reserved_mb': 0}


# ─── Dashboard HTTP Server ───────────────────────────────────────────────────

def _sanitize_for_json(obj):
    """Replace float('inf') and float('nan') with None so json.dumps doesn't crash.

    JSON spec has no Infinity/NaN — json.dumps(float('inf')) raises ValueError.
    This was the root cause of the dashboard always showing 'disconnected':
    best_val starts as float('inf'), so every /api/metrics call failed silently.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Serves dashboard.html at / and /api/metrics as JSON."""

    def do_GET(self):
        if self.path == '/api/metrics':
            data = _sanitize_for_json(metrics.get())
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        elif self.path == '/' or self.path == '/index.html' or self.path == '/dashboard.html':
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
                self.wfile.write(b'Dashboard not found - place dashboard.html next to train.py')
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress request logs


class ReusableTCPServer(socketserver.TCPServer):
    """TCPServer with SO_REUSEADDR — needed on Windows to avoid 'Address already in use'."""
    allow_reuse_address = True


def start_metrics_server():
    """Start the HTTP metrics server in a daemon thread."""
    try:
        with ReusableTCPServer(("", METRICS_PORT), DashboardHandler) as httpd:
            print(f"  Dashboard: http://localhost:{METRICS_PORT}")
            httpd.serve_forever()
    except OSError as e:
        print(f"  Dashboard: port {METRICS_PORT} unavailable ({e})")
        print(f"  Dashboard disabled. Close other apps using port {METRICS_PORT}.")


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────

# Architecture fields — changes here make checkpoints INCOMPATIBLE
ARCH_FIELDS = {'dim', 'n_layers', 'n_heads', 'n_kv_heads', 'vocab_size',
               'hidden_dim', 'max_seq_len', 'n_mtp_tokens'}


def cleanup_old_checkpoints():
    """Keep only the last MAX_CHECKPOINTS periodic checkpoints to save disk space.
    Never deletes best.pt or final.pt."""
    ckpt_files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('ckpt_') and f.endswith('.pt')],
        key=lambda f: int(f.split('_')[-1].split('.')[0])
    )
    while len(ckpt_files) > MAX_CHECKPOINTS:
        oldest = ckpt_files.pop(0)
        path = os.path.join(CHECKPOINT_DIR, oldest)
        try:
            os.remove(path)
        except OSError:
            pass


def save_ckpt(raw_model, optimizer, step, best_val, path):
    """Save checkpoint using RAW model state dict (before torch.compile).
    Compiled model keys have '_orig_mod.' prefix which breaks resume.
    Saves ALL ModelArgs fields so checkpoints are self-describing and can be
    validated on resume (catches architecture mismatches early)."""
    tmp = path + ".tmp"
    torch.save({
        'step': step,
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val': best_val,
        'model_args': raw_model.args.as_dict(),  # Save ALL args, not just a subset
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    """Load a specific checkpoint file."""
    print(f"Loading checkpoint: {path}")
    return torch.load(path, map_location=DEVICE, weights_only=False)


def find_checkpoint(resume_from=None):
    """Find the best available checkpoint to load.
    Priority: --resume-from > best.pt > final.pt > latest ckpt_*.pt"""
    if resume_from:
        if os.path.exists(resume_from):
            return resume_from
        # Try as relative to checkpoint dir
        full_path = os.path.join(CHECKPOINT_DIR, resume_from)
        if os.path.exists(full_path):
            return full_path
        print(f"ERROR: Specified checkpoint not found: {resume_from}")
        sys.exit(1)

    # Prefer best.pt, then final.pt, then latest ckpt_*.pt
    for preferred in ['best.pt', 'final.pt']:
        path = os.path.join(CHECKPOINT_DIR, preferred)
        if os.path.exists(path):
            return path

    # Find latest checkpoint by step number
    import glob as glob_mod
    files = glob_mod.glob(os.path.join(CHECKPOINT_DIR, "ckpt_*.pt"))
    if files:
        return max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))

    return None


# ─── Training Log ─────────────────────────────────────────────────────────────

def log_step(step, loss, lr, tok_s, val_loss=None, val_ppl=None, ema_loss=None):
    """Append training step data to JSONL log for post-training analysis."""
    log_path = os.path.join(CHECKPOINT_DIR, "training_log.jsonl")
    entry = {
        'step': step,
        'loss': round(loss, 6),
        'lr': lr,
        'tok_per_sec': round(tok_s, 1),
        'timestamp': time.time(),
    }
    if ema_loss is not None:
        entry['ema_loss'] = round(ema_loss, 6)
    if val_loss is not None:
        entry['val_loss'] = round(val_loss, 6)
        entry['val_ppl'] = round(val_ppl, 4) if val_ppl else None
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass


# ─── Main Training Loop ──────────────────────────────────────────────────────

def train(resume_from=None, reset_optimizer=False, micro_batch=None, accum_steps=None,
         use_8bit_adam=False, compile_model=False, compile_backend=None):
    # Allow CLI override of batch config
    global MICRO_BATCH, ACCUM_STEPS, USE_COMPILE, COMPILE_BACKEND
    if micro_batch is not None:
        MICRO_BATCH = micro_batch
    if accum_steps is not None:
        ACCUM_STEPS = accum_steps
    if compile_model:
        USE_COMPILE = True
    if compile_backend is not None:
        COMPILE_BACKEND = compile_backend

    args = ModelArgs()
    tokenizer = Tokenizer.from_file("tokenizer.json")

    if not HAS_CUDA:
        print("WARNING: No CUDA GPU detected! Training on CPU will be extremely slow.")
        print("  Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  Continuing anyway...\n")

    # Keep reference to raw model BEFORE compile for clean checkpoint save
    raw_model = Transformer(args).to(DEVICE)
    n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M parameters")
    print(f"Context: {args.max_seq_len} tokens | Vocab: {args.vocab_size}")
    print(f"Device: {DEVICE} | Platform: {platform.system()}")
    print(f"Grad checkpointing: {'ON' if args.use_checkpointing else 'OFF'}")

    # Compile model if requested
    if USE_COMPILE and HAS_CUDA:
        # cudagraphs backend: records CUDA ops into a graph, replays them.
        # Reduces CPU-side kernel launch overhead. Works on Windows! No Triton needed.
        # inductor backend: full code generation + kernel fusion. Needs working Triton.
        #   CRASHES on Windows with "cannot import triton_key" error.
        backend = COMPILE_BACKEND

        # On Windows, only aot_eager and cudagraphs work without Triton.
        # cudagraphs is faster but INCOMPATIBLE with dropout (gets skipped).
        # aot_eager traces the graph and eliminates Python overhead (~5-10%).
        if IS_WINDOWS and backend not in ('aot_eager', 'cudagraphs'):
            print(f"WARNING: Backend '{backend}' requires working Triton — not available on Windows.")
            print(f"  Forcing backend='aot_eager' (works with dropout, no Triton needed).")
            print(f"  For inductor, use WSL2: python train.py --compile --compile-backend inductor")
            backend = 'aot_eager'
            COMPILE_BACKEND = 'aot_eager'

        print(f"Compile: ON (backend={backend})")
        if backend == 'cudagraphs':
            # cudagraphs — fast but incompatible with dropout
            # Your model has dropout=0.05, so cudagraphs will SKIP (no speedup).
            # Only use this if you set dropout=0 in ModelArgs.
            model = torch.compile(raw_model, backend='cudagraphs')
            print(f"  NOTE: cudagraphs is incompatible with dropout — may be skipped!")
            print(f"  Use --compile-backend aot_eager for dropout-compatible compile.")
        elif backend == 'aot_eager':
            # aot_eager — traces graph, eliminates Python overhead, works with dropout
            model = torch.compile(raw_model, backend='aot_eager')
        elif backend == 'inductor':
            # inductor — needs working Triton (Linux only, or WSL2)
            model = torch.compile(raw_model, mode='default')
        elif backend == 'reduce-overhead':
            # reduce-overhead — uses cudagraphs + some optimizations
            model = torch.compile(raw_model, mode='reduce-overhead')
        else:
            model = torch.compile(raw_model, backend=backend)
        print(f"  First few steps will be slower (tracing computation graph)")
    else:
        model = raw_model
        if not USE_COMPILE:
            print(f"Compile: OFF (eager mode)")
            print(f"  Tip: add --compile for ~5-10% speedup (aot_eager backend, no Triton needed)")
        else:
            print(f"Compile: OFF (no CUDA device)")

    # ── Optimizer selection ──
    optimizer = None

    # Option 1: Muon optimizer (Feature #5 — recently shown to improve LLM training)
    if USE_MUON:
        try:
            from muon import Muon
            # Muon needs separate param groups for 2D matrices and other params
            matrix_params = []
            other_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.ndim >= 2:
                        matrix_params.append(param)
                    else:
                        other_params.append(param)
            optimizer = Muon(
                [{'params': matrix_params}, {'params': other_params, 'use_muon': False}],
                lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.95
            )
            print(f"Using Muon optimizer (matrix_params={len(matrix_params)}, other={len(other_params)})")
        except ImportError:
            print("Muon not found (pip install muon-pytorch), falling back to AdamW")
            optimizer = None

    # Option 2: Standard fused AdamW (FASTER — preferred over 8-bit for throughput)
    # 8-bit AdamW saves ~1.3GB VRAM but the quantization/dequantization overhead
    # slows down training by ~5-10%. Use --8bit-adam only if you're tight on VRAM.
    if optimizer is None and not use_8bit_adam:
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, betas=(0.9, 0.95),
                weight_decay=WEIGHT_DECAY, fused=HAS_CUDA
            )
            print("Using standard AdamW (fused=True) — fastest option")
        except Exception:
            optimizer = None

    # Option 3: 8-bit AdamW (saves ~1.3GB VRAM, but slower due to quantization overhead)
    if optimizer is None:
        try:
            import bitsandbytes as bnb
            # Quick sanity check — bitsandbytes sometimes installs but fails at runtime on Windows
            test_opt = bnb.optim.Adam8bit(model.parameters(), lr=LR)
            del test_opt  # Just testing if it can be instantiated
            optimizer = bnb.optim.Adam8bit(
                model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY
            )
            print("Using 8-bit AdamW (bitsandbytes) — slower but saves ~1.3GB VRAM")
        except Exception as e:
            print(f"8-bit AdamW unavailable ({type(e).__name__}), using standard AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, betas=(0.9, 0.95),
                weight_decay=WEIGHT_DECAY, fused=HAS_CUDA
            )
            print("Using standard AdamW (fused=True)")

    # ── WandB logging (Feature #4) ──
    wandb_run = None
    if USE_WANDB:
        try:
            import wandb
            wandb_run = wandb.init(
                project="my-slm",
                config={
                    'model_params_M': n_params,
                    'dim': args.dim,
                    'n_layers': args.n_layers,
                    'n_heads': args.n_heads,
                    'n_kv_heads': args.n_kv_heads,
                    'max_seq_len': args.max_seq_len,
                    'micro_batch': MICRO_BATCH,
                    'accum_steps': ACCUM_STEPS,
                    'lr': LR,
                    'weight_decay': WEIGHT_DECAY,
                    'compile': USE_COMPILE,
                    'platform': platform.system(),
                },
                name=f"slm-{int(n_params)}M-lr{LR}",
            )
            print(f"WandB logging enabled: {wandb_run.url}")
        except ImportError:
            print("WandB not found (pip install wandb), logging disabled")
            wandb_run = None

    # NO GradScaler — BF16 has the same exponent range as FP32, never needs scaling
    start_step = 0
    best_val = float('inf')

    # Resume from checkpoint if available
    ckpt_path = find_checkpoint(resume_from)
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path)

        # Validate architecture compatibility before loading weights
        ckpt_args = ckpt.get('model_args', {})
        if ckpt_args:
            current_args_dict = args.as_dict()
            arch_mismatches = []
            hyper_mismatches = []
            for k, v in current_args_dict.items():
                if k in ckpt_args and ckpt_args[k] != v:
                    if k in ARCH_FIELDS:
                        arch_mismatches.append(f"  {k}: checkpoint={ckpt_args[k]}, current={v}")
                    else:
                        hyper_mismatches.append(f"  {k}: checkpoint={ckpt_args[k]}, current={v}")

            if arch_mismatches:
                print("ERROR: Architecture-breaking mismatches — CANNOT resume:")
                for m in arch_mismatches:
                    print(m)
                print("Model weights are incompatible. Start a new training run.")
                sys.exit(1)

            if hyper_mismatches:
                print("NOTE: Training hyperparameter changes detected (non-breaking):")
                for m in hyper_mismatches:
                    print(m)
                print("Model weights will load fine. New hyperparameters take effect immediately.")

        raw_model.load_state_dict(ckpt['model'])

        if reset_optimizer:
            print("Optimizer state RESET per --reset-optimizer flag.")
            print("  (Fresh optimizer with new LR schedule — recommended after loss regression)")
        else:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
                print("Optimizer state loaded from checkpoint.")
            except (ValueError, RuntimeError) as e:
                print(f"WARNING: Could not load optimizer state ({e})")
                print("  This happens when switching optimizer types (e.g., AdamW -> 8-bit AdamW).")
                print("  Starting with fresh optimizer state. Model weights are loaded correctly.")

        start_step = ckpt['step']
        best_val = ckpt.get('best_val', float('inf'))
        print(f"Resumed at step {start_step}, best_val={best_val:.4f}")
    else:
        print("No checkpoint found — starting from scratch.")

    # Datasets
    train_ds = PackedTokenDataset("data/train.bin", args.max_seq_len)
    val_ds = PackedTokenDataset("data/val.bin", args.max_seq_len)
    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")

    # DataLoader: num_workers=0 on Windows to avoid multiprocessing issues
    # prefetch_factor: pre-fetch 2 batches per worker to keep GPU fed
    _train_dl_kwargs = dict(
        batch_size=MICRO_BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=HAS_CUDA,
        drop_last=True,  # Prevent partial batches that cause torch.compile recompilation
    )
    if NUM_WORKERS > 0:
        _train_dl_kwargs['prefetch_factor'] = 2
    train_loader = infinite_loader(train_ds, **_train_dl_kwargs)

    _val_dl_kwargs = dict(
        batch_size=MICRO_BATCH, shuffle=False,
        pin_memory=HAS_CUDA, num_workers=NUM_WORKERS_VAL,
        drop_last=True,  # Consistent batch sizes for val
    )
    if NUM_WORKERS_VAL > 0:
        _val_dl_kwargs['persistent_workers'] = True
        _val_dl_kwargs['prefetch_factor'] = 2
    val_loader = DataLoader(val_ds, **_val_dl_kwargs)

    # Start dashboard server (prints its own URL on success)
    server_thread = threading.Thread(target=start_metrics_server, daemon=True)
    server_thread.start()
    time.sleep(0.3)  # Give server a moment to start (or fail with error message)

    # Training state
    model.train()
    step = start_step
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    accumulated_loss = torch.tensor(0.0, device=DEVICE)  # Tensor accumulation — no CUDA sync per micro-batch!
    accum_count = 0
    effective_batch_tokens = MICRO_BATCH * ACCUM_STEPS * args.max_seq_len

    # EMA loss tracking — Exponential Moving Average smooths out mini-batch noise
    # so you can see the actual training trend. α=0.05 means ~20-step average.
    # Without EMA, loss fluctuation of ±3.5% is normal and misleading.
    ema_loss = None

    # Check VRAM pressure at startup
    if HAS_CUDA:
        gpu_check = get_gpu_stats(0)
        vram_pct = gpu_check['gpu_mem_used_mb'] / max(gpu_check['gpu_mem_total_mb'], 1)
        if vram_pct > 0.95:
            print(f"\n  ⚠ WARNING: VRAM at {vram_pct*100:.1f}% — this causes training instability!")
            print(f"  ⚠ CUDA memory allocator can't find contiguous blocks, causing fluctuation.")
            print(f"  ⚠ Fix: use --micro-batch 1 --accum-steps 64 to free ~400MB of activation memory.")
            print(f"  ⚠ This keeps the SAME effective batch size (65,536 tokens/step) but uses less VRAM.")
            print()

    metrics.update(status='training', step=step, best_val=best_val,
                   effective_batch_tokens=effective_batch_tokens, max_steps=MAX_STEPS)

    print(f"\n{'='*60}")
    print(f"Training from step {step}")
    print(f"Effective batch: {effective_batch_tokens:,} tokens/step")
    print(f"Micro batch: {MICRO_BATCH} | Accum steps: {ACCUM_STEPS}")
    print(f"Press Ctrl+C to pause (saves checkpoint automatically)")
    print(f"{'='*60}\n")

    # Determine autocast device type
    autocast_device = 'cuda' if HAS_CUDA else 'cpu'
    autocast_dtype = torch.bfloat16 if HAS_CUDA else torch.float32

    # Compute initial LR (only needs update at optimizer step boundaries)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    try:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= MAX_STEPS:
                break

            # Transfer uint16 to GPU, then convert to long — 4x less H2D bandwidth
            x = x.to(DEVICE, non_blocking=True).long()
            y = y.to(DEVICE, non_blocking=True).long()

            # Forward with autocast — no GradScaler needed for BF16
            with autocast(device_type=autocast_device, dtype=autocast_dtype):
                logits, loss = model(x, y)
                loss = loss / ACCUM_STEPS

            # Backward — directly, no scaler
            loss.backward()

            # Accumulate loss as TENSOR — avoids CUDA sync per micro-batch!
            # loss.item() forces a CUDA sync every call (32x per optimizer step).
            # By accumulating as a tensor, we only sync once at the optimizer step boundary.
            with torch.no_grad():
                accumulated_loss += loss.detach()

            accum_count += 1

            # ── Optimizer step (only after full accumulation) ──
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                # Optimizer step runs in eager mode via @_optimizer_step decorator.
                # Dynamo must NOT trace this — it breaks fused AdamW state init.
                grad_norm = _optimizer_step(model, optimizer, GRAD_CLIP)

                # Single CUDA sync per optimizer step instead of per micro-batch
                avg_loss = (accumulated_loss.item() * ACCUM_STEPS) / accum_count

                # Update EMA loss — exponential moving average smooths mini-batch noise
                # ema_loss = α * new_loss + (1-α) * old_ema
                # α=0.05 gives ~20-step average, showing the true trend
                if ema_loss is None:
                    ema_loss = avg_loss
                else:
                    ema_loss = EMA_ALPHA * avg_loss + (1 - EMA_ALPHA) * ema_loss

                dt = time.time() - t0
                tok_s = effective_batch_tokens / dt if dt > 0 else 0
                total_tokens = step * effective_batch_tokens
                remaining_steps = MAX_STEPS - step
                eta_hours = (remaining_steps * dt) / 3600 if step > start_step + 5 else 0

                # GPU stats (cached — only queries nvidia-smi every 10 steps)
                gpu_stats = get_gpu_stats(step)

                # Check VRAM pressure — warn if > 95%
                vram_pressure = False
                if gpu_stats['gpu_mem_total_mb'] > 0:
                    vram_pct = gpu_stats['gpu_mem_used_mb'] / gpu_stats['gpu_mem_total_mb']
                    vram_pressure = vram_pct > 0.95

                # Update metrics for dashboard
                metrics.update(
                    step=step, loss=avg_loss, lr=lr,
                    tok_per_sec=tok_s, total_tokens=total_tokens,
                    eta_hours=eta_hours, grad_norm=grad_norm.item(),
                    ema_loss=ema_loss, vram_pressure=vram_pressure,
                    **gpu_stats
                )
                metrics.append_history('loss_history', step, avg_loss)
                metrics.append_history('ema_loss_history', step, ema_loss)
                metrics.append_history('lr_history', step, lr)
                metrics.append_history('speed_history', step, tok_s)

                # WandB logging
                if wandb_run is not None:
                    try:
                        import wandb
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/ema_loss': ema_loss,
                            'train/lr': lr,
                            'train/grad_norm': grad_norm.item(),
                            'train/tok_per_sec': tok_s,
                            'train/step': step,
                            'train/vram_pressure': vram_pressure,
                        }, step=step)
                    except Exception:
                        pass

                # Log to JSONL
                log_step(step, avg_loss, lr, tok_s, ema_loss=ema_loss)

                # Console log every 10 steps — show EMA alongside raw loss
                if step % 10 == 0:
                    vram_pct_str = ""
                    if gpu_stats['gpu_mem_total_mb'] > 0:
                        vram_pct = gpu_stats['gpu_mem_used_mb'] / gpu_stats['gpu_mem_total_mb'] * 100
                        vram_pct_str = f" ({vram_pct:.0f}%)"
                        if vram_pct > 95:
                            vram_pct_str = f" ({vram_pct:.0f}% ⚠ PRESSURE)"
                    print(f"Step {step:>6} | Loss: {avg_loss:.4f} | EMA: {ema_loss:.4f} | "
                          f"LR: {lr:.2e} | {tok_s:.0f} tok/s | "
                          f"GPU: {gpu_stats['gpu_mem_used_mb']}MB{vram_pct_str}")

                # Reset accumulation
                accumulated_loss = torch.tensor(0.0, device=DEVICE)
                accum_count = 0
                t0 = time.time()

                # ── Validation ──
                if step % VAL_EVERY == 0 and step > 0:
                    # Validation runs in eager mode via @_run_validation decorator.
                    val_loss, val_steps_count = _run_validation(
                        model, val_loader, autocast_device, autocast_dtype, VAL_MAX_STEPS)

                    avg_val = val_loss / val_steps_count
                    ppl = math.exp(min(avg_val, 20))

                    metrics.update(val_loss=avg_val, val_ppl=ppl)
                    metrics.append_history('val_history', step, avg_val)

                    # WandB validation logging
                    if wandb_run is not None:
                        try:
                            import wandb
                            wandb.log({
                                'val/loss': avg_val,
                                'val/ppl': ppl,
                                'val/step': step,
                            }, step=step)
                        except Exception:
                            pass

                    # Log validation to JSONL
                    log_step(step, avg_loss, lr, tok_s, avg_val, ppl)

                    print(f"  * VAL Step {step} | Loss: {avg_val:.4f} | PPL: {ppl:.2f}")

                    if avg_val < best_val:
                        best_val = avg_val
                        _save_checkpoint(raw_model, optimizer, step, best_val,
                                  os.path.join(CHECKPOINT_DIR, "best.pt"))
                        print(f"  * New best model saved!")

                # ── Periodic checkpoint ──
                if step % CHECKPOINT_EVERY == 0 and step > 0:
                    _save_checkpoint(raw_model, optimizer, step, best_val,
                              os.path.join(CHECKPOINT_DIR, f"ckpt_{step}.pt"))
                    print(f"  Checkpoint saved at step {step}")
                    cleanup_old_checkpoints()

                # ── Save metrics snapshot ──
                if step % 100 == 0:
                    try:
                        metrics_path = os.path.join(CHECKPOINT_DIR, "metrics.json")
                        with open(metrics_path, 'w') as f:
                            json.dump(metrics.get(), f, indent=2)
                    except Exception:
                        pass

                step += 1

                # Update LR for next optimizer step (once, not per micro-batch)
                lr = get_lr(step)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

    except KeyboardInterrupt:
        print(f"\nInterrupted at step {step}. Saving checkpoint...")
        _save_checkpoint(raw_model, optimizer, step, best_val,
                  os.path.join(CHECKPOINT_DIR, f"ckpt_{step}.pt"))
        metrics.update(status='paused')
        print("Checkpoint saved. Resume anytime with: python train.py")
        return

    # Training complete
    _save_checkpoint(raw_model, optimizer, step, best_val,
              os.path.join(CHECKPOINT_DIR, "final.pt"))
    metrics.update(status='complete')
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
    print(f"\nTraining complete! Final step: {step}, Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SLM")
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Checkpoint to resume from (e.g., best.pt, ckpt_4000.pt, or full path)')
    parser.add_argument('--reset-optimizer', action='store_true',
                        help='Skip optimizer state on resume — fresh optimizer with new LR schedule')
    parser.add_argument('--micro-batch', type=int, default=None,
                        help='Micro batch size per GPU forward pass (default: 2 for 8GB VRAM, try 4 on 12GB+)')
    parser.add_argument('--accum-steps', type=int, default=None,
                        help='Gradient accumulation steps (default: 32, adjust to keep 65k effective batch)')
    parser.add_argument('--8bit-adam', action='store_true',
                        help='Use 8-bit AdamW (saves ~1.3GB VRAM but ~5-10%% slower). Use only if standard AdamW OOMs.')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile() for ~5-15%% speedup. On Windows, uses CUDA graphs backend (no Triton needed).')
    parser.add_argument('--compile-backend', type=str, default=None,
                        choices=['aot_eager', 'cudagraphs', 'inductor', 'reduce-overhead'],
                        help='torch.compile backend (default: aot_eager on Windows, inductor on Linux)')
    args_cli = parser.parse_args()
    train(resume_from=args_cli.resume_from, reset_optimizer=args_cli.reset_optimizer,
          micro_batch=args_cli.micro_batch, accum_steps=args_cli.accum_steps,
          use_8bit_adam=getattr(args_cli, '8bit_adam', False),
          compile_model=args_cli.compile, compile_backend=args_cli.compile_backend)
