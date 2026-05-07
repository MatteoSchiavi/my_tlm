# My_SLM — v3.2 Build Guide

python train.py --resume-from best.pt --reset-optimizer --compile

wls: python train.py --resume-from best.pt --reset-optimizer --compile
A 212M parameter Small Language Model built from scratch for 8 GB VRAM (RTX 3070),
optimised for **Italian language + C programming**, with a real-time training dashboard.
Full **Windows compatibility** — no Triton required.

---

## What's New in v3.2

### Windows Compatibility (CRITICAL FIX)

| Issue                                                     | Cause                                                            | Fix                                                                              |
| --------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `RuntimeError: Cannot find a working triton installation` | `torch.compile()` requires Triton, which doesn't work on Windows | Auto-detect: compile on Linux, eager mode on Windows                             |
| 0 code files downloaded                                   | `codeparrot/github-code` uses deprecated loading script          | Replaced with `bigcode/starcoderdata` + `codeparrot/github-code-clean` fallbacks |
| OSCAR download failed                                     | `oscar-corpus/OSCAR-2301` is gated (requires auth)               | Primary: `allenai/c4` (mc4, open access). OSCAR-2301 as fallback                 |
| The Stack download failed                                 | `bigcode/the-stack` is gated                                     | Replaced with `bigcode/the-stack-smol` (open access)                             |
| `num_workers` crash on Windows                            | Windows multiprocessing issues with DataLoader                   | Auto-detect: `num_workers=0` on Windows, 4 on Linux                              |
| bitsandbytes import crash                                 | May fail on Windows even when installed                          | Try/except with sanity check, fallback to standard AdamW                         |
| HF cache symlink warnings                                 | Windows needs Developer Mode for symlinks                        | `HF_HUB_DISABLE_SYMLINKS_WARNING=1` set automatically                            |

### Key Changes

- `train.py`: torch.compile() auto-disabled on Windows (10-20% slower but stable)
- `train.py`: `COMPILE=1` env var to force compile, `NO_COMPILE=1` to disable on Linux
- `train.py`: CUDA flags guarded with `torch.cuda.is_available()` checks
- `download.py`: 3-tier fallback for code data (starcoderdata -> the-stack-smol -> github-code-clean)
- `download.py`: Sequential mode on Windows by default (parallel causes crashes)
- `download.py`: Better error messages with actionable fixes when sources fail

---

## What's New in v3

### Features Implemented

| #   | Feature                                 | Status      | Impact                                                          |
| --- | --------------------------------------- | ----------- | --------------------------------------------------------------- |
| 1   | **KV cache in inference**               | Implemented | ~4-5x faster generation (40 → 150-200 tok/s)                    |
| 2   | **Speculative decoding with MTP heads** | Implemented | 1.5-2.5x additional speedup over KV cache                       |
| 3   | **Muon optimizer**                      | Implemented | Drop-in replacement for AdamW (enable with `MUON=1`)            |
| 4   | **WandB logging**                       | Implemented | Experiment tracking alongside dashboard (enable with `WANDB=1`) |
| 5   | **5-tier data download**                | Implemented | smoke / quick / standard / full / max tiers                     |
| 6   | **New data sources**                    | Implemented | Italian Gutenberg, English Gutenberg, C++ code, StackOverflow   |
| 7   | **langdetect integration**              | Implemented | Accurate Italian filtering (fixes Bug 5)                        |

### Bugs Fixed

| #   | Bug                          | File                    | Fix                                                  |
| --- | ---------------------------- | ----------------------- | ---------------------------------------------------- |
| 1   | n_kv_heads=4 crash           | model.py                | Changed to n_kv_heads=6 (18/6=3 clean repeats)       |
| 2   | Double BOS/EOS               | preprocess.py           | Removed TemplateProcessing, manual insertion only    |
| 3   | Training stops early         | train.py                | Added infinite_loader()                              |
| 4   | Dead KVCache class           | model.py + inference.py | Full KV cache implementation                         |
| 5   | Weak Italian detection       | download.py + filter.py | langdetect with improved fallback                    |
| 6   | nvidia-smi every step        | train.py                | Cached GPU stats (every 10 steps)                    |
| 7   | val_loader missing drop_last | train.py                | Added drop_last=True                                 |
| 8   | Missing requirements         | requirements.txt        | Added langdetect, psutil, tqdm                       |
| 9   | Fragile \_init_weights       | model.py                | Direct module type check instead of named_parameters |
| 10  | Dead spike detector          | train.py                | Removed check_spike (z-loss handles stability)       |
| 11  | Insufficient data tiers      | download.py             | Added smoke + max tiers                              |

---

## Download Tiers — Full Explanation

### Why Do the Numbers Seem "Low"?

The previous tier system was designed for the original training settings
(seq_len=512, ACCUM_STEPS=8, effective batch of 8,192 tokens). The v2 settings
(seq_len=1024, ACCUM_STEPS=32) consume **8x more tokens per optimizer step**.

The fundamental constraint is this equation:

```
Training budget = MAX_STEPS x MICRO_BATCH x ACCUM_STEPS x seq_len
               = 60,000 x 2 x 32 x 1024
               = 3.93 billion tokens
```

Your dataset needs to hold AT LEAST this many tokens to avoid repeating documents
(repetition causes overfitting and kills generalisation). Given ~600 tokens per
document on average:

```
Minimum documents needed = 3.93B / 600 = ~6.5 million documents
```

The old `full` tier had 2.46M documents (~1.47B tokens) — enough for v1 settings,
but less than half of what v2 needs.

### Chinchilla Context

The "Chinchilla optimal" rule says a model of **N parameters** should see
**~20N tokens** during training for compute-optimal performance:

```
205M params x 20 = 4.1 billion tokens  <- minimum for this model
```

Training on significantly less leaves capability on the table. Training on
significantly more (2-3x) still helps (more data = better generalisation),
just at diminishing returns per hour of compute.

---

### Tier 0 — Smoke Test

**Purpose:** Verify the pipeline works end-to-end. Not for real training.

```bash
python download.py --tier smoke
```

| Source            | Documents   | Tokens (~) |
| ----------------- | ----------- | ---------- |
| Italian OSCAR     | 3,000       | 1.8M       |
| Italian Wikipedia | 1,500       | 0.9M       |
| Italian FineWeb   | 1,500       | 0.9M       |
| English Web       | 6,000       | 3.6M       |
| English Wikipedia | 3,000       | 1.8M       |
| English Edu       | 3,000       | 1.8M       |
| C Code            | 5,000       | 5.0M       |
| Other Code        | 5,000       | 5.0M       |
| **Total**         | **~28,000** | **~21M**   |

| Metric                  | Value                                   |
| ----------------------- | --------------------------------------- |
| Download time           | ~3-5 minutes                            |
| Preprocessing           | ~1 minute                               |
| Min training steps      | 500 (to see loss move)                  |
| Wall time for 500 steps | ~20 minutes                             |
| Good for                | "Does my GPU work? Does loss decrease?" |

---

### Tier 1 — Quick Iteration

**Purpose:** First real model. Enough data to see actual learning curves.

```bash
python download.py --tier quick
```

| Source            | Documents    | Tokens (~) |
| ----------------- | ------------ | ---------- |
| Italian OSCAR     | 30,000       | 18M        |
| Italian Wikipedia | 15,000       | 9M         |
| Italian FineWeb   | 15,000       | 9M         |
| English Web       | 60,000       | 36M        |
| English Wikipedia | 30,000       | 18M        |
| English Edu       | 30,000       | 18M        |
| C Code            | 40,000       | 40M        |
| Other Code        | 30,000       | 30M        |
| **Total**         | **~250,000** | **~178M**  |

| Metric                | Value                                        |
| --------------------- | -------------------------------------------- |
| Download time         | ~20-40 minutes                               |
| Preprocessing         | ~5 minutes                                   |
| Recommended MAX_STEPS | 5,000                                        |
| Wall time (RTX 3070)  | ~42 hours (~5 days at 8h/day)                |
| Good for              | Hyperparameter testing, architecture changes |

---

### Tier 2 — Standard

**Purpose:** A genuinely usable model. Solid Italian and C capability.

```bash
python download.py --tier standard
```

| Source            | Documents      | Tokens (~) |
| ----------------- | -------------- | ---------- |
| Italian OSCAR     | 200,000        | 120M       |
| Italian Wikipedia | 60,000         | 36M        |
| Italian FineWeb   | 150,000        | 90M        |
| English Web       | 400,000        | 240M       |
| English Wikipedia | 120,000        | 72M        |
| English Edu       | 250,000        | 150M       |
| C Code            | 200,000        | 200M       |
| Other Code        | 120,000        | 120M       |
| **Total**         | **~1,500,000** | **~1.03B** |

| Metric                | Value                           |
| --------------------- | ------------------------------- |
| Download time         | ~2-4 hours                      |
| Recommended MAX_STEPS | 20,000                          |
| Wall time (RTX 3070)  | ~166 hours (~21 days at 8h/day) |
| Good for              | First production-quality model  |

---

### Tier 3 — Full

**Purpose:** Best model achievable in a reasonable training run on one 3070.
Near Chinchilla-optimal for 205M params.

```bash
python download.py --tier full
```

| Source            | Documents      | Tokens (~) |
| ----------------- | -------------- | ---------- |
| Italian OSCAR     | 500,000        | 300M       |
| Italian Wikipedia | 60,000         | 36M        |
| Italian FineWeb   | 250,000        | 150M       |
| English Web       | 800,000        | 480M       |
| English Wikipedia | 300,000        | 180M       |
| English Edu       | 600,000        | 360M       |
| C Code            | 400,000        | 400M       |
| Other Code        | 200,000        | 200M       |
| **Total**         | **~3,110,000** | **~2.1B**  |

| Metric                         | Value                                 |
| ------------------------------ | ------------------------------------- |
| Download time                  | ~6-10 hours                           |
| Disk (raw + filtered + binary) | ~35-50 GB                             |
| Recommended MAX_STEPS          | 40,000                                |
| Wall time (RTX 3070)           | ~332 hours (~41 days at 8h/day)       |
| Expected val loss at 40k steps | ~2.1-2.3                              |
| Good for                       | Serious use, close to peak capability |

---

### Tier MAX — Maximum Intelligence

**Purpose:** Absolute best this architecture can achieve. Exceeds Chinchilla-optimal
(~2x the minimum token budget). Expect excellent Italian fluency, solid C completion,
and reasonable general English capability.

```bash
python download.py --tier max
```

> **Note:** Requires ~80-100 GB of free disk space during download + processing.
> Some sources (The Stack) require a HuggingFace account and licence agreement.

| Source                       | Documents      | Tokens (~) | Notes                         |
| ---------------------------- | -------------- | ---------- | ----------------------------- |
| Italian OSCAR                | 1,000,000      | 600M       | ~1/8 of the full corpus       |
| Italian Wikipedia            | 60,000         | 36M        | Full Italian Wikipedia        |
| Italian FineWeb-2            | 600,000        | 360M       | Best Italian web quality      |
| Italian Books (Gutenberg IT) | 20,000         | 40M        | Long-form Italian prose       |
| English Web (OpenWebText)    | 2,000,000      | 1,200M     | ~1/4 of the full corpus       |
| English Wikipedia            | 600,000        | 360M       | Top articles by length        |
| English Edu (FineWeb-Edu)    | 1,200,000      | 720M       | High educational score        |
| Project Gutenberg (EN)       | 50,000         | 200M       | Public domain books           |
| C Code (The Stack)           | 800,000        | 800M       | License-filtered C files      |
| C++ Code (The Stack)         | 300,000        | 300M       | Complements C understanding   |
| Python/JS/Rust/Shell         | 400,000        | 400M       | General programming structure |
| StackOverflow Q&A            | 200,000        | 160M       | C/Italian Q&A pairs           |
| **Total**                    | **~7,230,000** | **~5.18B** |                               |

| Metric                         | Value                                          |
| ------------------------------ | ---------------------------------------------- |
| Download time                  | ~16-28 hours                                   |
| Disk (raw + filtered + binary) | ~70-90 GB                                      |
| Recommended MAX_STEPS          | 80,000                                         |
| Training budget at 80k steps   | 5.24B tokens (~1.01 passes through data)       |
| Wall time (RTX 3070)           | ~665 hours (~83 days at 8h/day)                |
| Expected val loss              | ~1.85-2.05                                     |
| Expected perplexity            | ~6-8                                           |
| Good for                       | Maximum quality — leave it running for a month |

**Practical note:** At 8 hours/day, the max tier takes ~83 calendar days. A more
realistic approach is to run Tier 3 (40k steps, ~41 days at 8h) first and evaluate.
If the model is good, continue training from the checkpoint — the WSD schedule's
stable phase can simply be extended.

---

## Hardware Requirements

| Component | Minimum              | Recommended                             |
| --------- | -------------------- | --------------------------------------- |
| GPU       | NVIDIA 8 GB VRAM     | RTX 3070, 3080, 4060 Ti                 |
| CUDA      | 11.8                 | 12.1+                                   |
| RAM       | 16 GB                | 32 GB (for preprocessing tier 3+)       |
| Storage   | 40 GB                | 100 GB (tier max)                       |
| OS        | Windows 10/11 native | Ubuntu 22.04 LTS (torch.compile faster) |

**Tested on:**

- RTX 3070 Laptop (8 GB VRAM), 16 GB DDR4, WSL2 Ubuntu 22.04
- Windows 11 native (eager mode, ~10-20% slower than compiled Linux)

### Windows vs Linux

| Feature                  | Linux                  | Windows                                      |
| ------------------------ | ---------------------- | -------------------------------------------- |
| torch.compile()          | Yes (Triton available) | No (auto-disabled)                           |
| Training speed           | ~2,000-2,400 tok/s     | ~1,600-2,000 tok/s (eager mode)              |
| DataLoader workers       | 4 (parallel)           | 0 (sequential)                               |
| bitsandbytes 8-bit AdamW | Works                  | May not compile (fallback to standard AdamW) |
| Parallel downloads       | Yes                    | Sequential only                              |
| HF cache symlinks        | Yes                    | Needs Developer Mode (warning suppressed)    |

> **Tip:** If you have WSL2, training under Linux is ~20% faster due to torch.compile().
> The Windows native path works perfectly fine, just a bit slower.

---

## Project Structure

```
My_SLM/
├── download.py          # Tiered data download (smoke/quick/standard/full/max)
├── filter.py            # Quality filtering, deduplication, Italian-aware
├── mix.py               # Mix 45% Italian / 25% Code (C-primary) / 25% English / 5% C++
├── preprocess.py        # BPE tokenizer (500k sample) + binary packing
├── model.py             # 205M Transformer (GQA, RoPE, SwiGLU, MTP, z-loss, KV cache)
├── train.py             # Training loop + live dashboard + WandB + Muon support
├── inference.py         # Generation with KV cache + speculative decoding
├── dashboard.html       # Real-time training monitor (served by train.py)
├── requirements.txt     # All dependencies
├── tokenizer.json       # Trained BPE tokenizer (32k vocab)
├── training_log.jsonl   # Append-only training log (step, loss, lr, tok/s)
├── data/
│   ├── train.bin        # Packed training tokens (uint16)
│   └── val.bin          # Packed validation tokens (uint16)
├── data_raw/            # Downloaded JSONL files
│   ├── italian/
│   ├── english/
│   └── code/
├── data_filtered/       # After quality filtering
├── data_mixed/          # After mixing and shuffling
└── checkpoints/
    ├── ckpt_2500.pt     # Periodic checkpoint (only last 3 kept)
    ├── best.pt          # Lowest validation loss
    └── final.pt         # Training complete
```

---

## Step-by-Step Setup

### 1. Create environment

```bash
mkdir My_SLM && cd My_SLM
python3 -m venv venv && source venv/bin/activate   # Linux/WSL
python -m venv venv && venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
# CRITICAL: PyTorch with CUDA support (the default pip install gives CPU-only!)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA works:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
# Should print: CUDA: True, Device: NVIDIA GeForce RTX 3070 ...

# Core
pip install datasets huggingface-hub tokenizers numpy
pip install langdetect psutil tqdm

# Optional: 8-bit optimizer (saves ~50% memory, may not work on Windows)
pip install bitsandbytes

# Optional: experiment tracking
pip install wandb

# Optional: Muon optimizer
pip install muon-pytorch

# Optional: Triton (Linux only — enables torch.compile for ~20% speedup)
pip install triton
```

> **Windows users:** If `pip install bitsandbytes` fails, don't worry — the training
> script will automatically fall back to standard AdamW. The model will still train,
> just using slightly more VRAM for optimizer states.

### 3. HuggingFace login (optional but recommended)

Most datasets are open access, but some larger/higher-quality ones require authentication:

- `oscar-corpus/OSCAR-2301` (Italian) — gated, needs auth
- `bigcode/the-stack` (code) — gated, needs licence agreement

```bash
huggingface-cli login
# Paste your token from huggingface.co/settings/tokens

# Then visit these URLs and click "Access repository":
# https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
# https://huggingface.co/datasets/bigcode/the-stack
```

**Without HF login**, you'll still get data from open-access sources:

- `allenai/c4` (mc4 Italian) instead of OSCAR-2301
- `bigcode/starcoderdata` instead of The Stack
- `bigcode/the-stack-smol` as additional fallback
- `codeparrot/github-code-clean` as last resort for code

You'll get ~70-80% of the target data without login, which is still enough for training.

### 4. Run the pipeline

```bash
# Choose your tier: smoke / quick / standard / full / max
python download.py --tier quick

python filter.py
python mix.py
python preprocess.py

# Start training — dashboard at http://localhost:8686
python train.py

# With optional integrations:
WANDB=1 python train.py    # Enable WandB logging
MUON=1 python train.py     # Use Muon optimizer
WANDB=1 MUON=1 python train.py  # Both

# torch.compile control:
COMPILE=1 python train.py  # Force compile (crashes without Triton!)
NO_COMPILE=1 python train.py  # Force eager mode (for debugging on Linux)
```

### 5. Monitor and generate

Open `http://localhost:8686` in any browser while `train.py` is running.

```bash
# After training (or anytime after a checkpoint):

# Fast generation with KV cache (default, ~150-200 tok/s)
python inference.py

# Even faster with speculative decoding (uses MTP heads as draft)
python inference.py --speculative

# Single prompt
python inference.py --prompt "Il compilatore C mostra un errore:" --max-new 200

# Without KV cache (for debugging)
python inference.py --no-kv-cache

# Fine-tune generation style
python inference.py \
  --temperature 0.7 \
  --top-p 0.90 \
  --top-k 40 \
  --repetition-penalty 1.15
```

### Checkpoint Selection

The script auto-selects: `best.pt` -> `final.pt` -> latest `ckpt_*.pt`.
Force a specific checkpoint:

```bash
python inference.py --checkpoint checkpoints/ckpt_30000.pt
```

---

## Architecture

### Model Specifications (v3)

| Parameter           | Value   | Notes                     |
| ------------------- | ------- | ------------------------- |
| Total parameters    | ~205M   |                           |
| Embedding dimension | 1,152   |                           |
| Layers              | 12      |                           |
| Attention heads     | 18      |                           |
| KV heads (GQA)      | **6**   | 3 Q-heads share 1 KV-head |
| Feed-forward dim    | 3,072   | SwiGLU hidden             |
| Context length      | 1,024   |                           |
| Vocabulary size     | 32,000  | BPE, Italian+C weighted   |
| RoPE theta          | 500,000 | Llama 3 style             |
| Dropout             | 0.05    |                           |

### Architecture Features

**Grouped Query Attention (GQA)**
18 query heads share 6 key/value heads. Reduces KV memory by 3x vs standard MHA.
At inference with a KV cache, this allows much longer generation before memory pressure.

**QK-Norm (RMSNorm on Q and K)**
Prevents attention softmax saturation. Allows LR of 6e-4 without instability, even at
1024 context length where attention logits can grow large.

**RoPE with theta=500,000**
Llama-3's RoPE configuration. Compared to the original theta=10,000:
rotations are much slower across positions, which means the model loses relative
position signal more gradually — critical for coherent text beyond 512 tokens.

**SwiGLU Feed-Forward**
`FFN(x) = W2(SiLU(W1(x)) * W3(x))`. Two matmuls to gate the output. Used in every
major model since PaLM. Roughly 10-15% better than ReLU/GELU at the same parameter count.

**Z-Loss (PaLM/Gemma)**
`L_z = w x E[(log Sum exp(logits))^2]` with `w=1e-4`. Penalises logit magnitude growth.
Without it, logits can grow unboundedly under BF16, causing loss spikes that require
the fragile spike-detector rollback. With it, training is naturally stable.

**Multi-Token Prediction (MTP)**
Two auxiliary heads predict token positions t+2 and t+3 (in addition to the main
head at t+1). Loss contributions decay: head 0 at full `mtp_weight`, head 1 at half.
This forces intermediate hidden states to encode richer information and improves
representation quality. Also enables **speculative decoding** at inference (see below).

**KV Cache**
Each attention layer caches its computed key and value tensors. During generation,
only the new token is processed instead of the full context window. This reduces
generation from O(n^2) to O(n) per token, providing ~4-5x speedup.

**Speculative Decoding**
Uses the MTP heads as "draft models" to predict multiple future tokens at once.
The main model then verifies the draft tokens in a single forward pass. Accepted
tokens are kept; rejected tokens trigger resampling. Provides 1.5-2.5x additional
speedup on top of the KV cache. Enable with `--speculative` flag.

**Gradient Checkpointing**
Discards activation tensors after the forward pass, recomputes them during backward.
Trades ~20% more compute for ~40% less VRAM. Essential for fitting the full model
at 1024 context in 8 GB.

**Weight Tying**
Input embedding matrix = output projection matrix. Saves ~150M parameters (no separate
output head), and forces the model to use a unified semantic space for reading and writing.

**Residual Scaling Init (GPT-2 style)**
Output projections (`wo`, `w2`) initialised with `std = 0.02 / sqrt(2 x n_layers)`.
Without this, early training has residual activations that grow exponentially with depth.

---

## Training Configuration

### Parameters

| Hyperparameter         | v3 Value           | v1 Value    | Notes                  |
| ---------------------- | ------------------ | ----------- | ---------------------- |
| Micro-batch            | 2                  | 2           | Limited by VRAM        |
| Gradient accumulation  | 32                 | 8           | 4x smoother gradients  |
| Effective batch tokens | 65,536             | 8,192       | 8x larger              |
| Learning rate          | 6e-4               | 8e-4        | Lower for larger batch |
| Sequence length        | 1,024              | 512         | 2x longer context      |
| Max steps              | 60,000             | 60,000      |                        |
| Warmup steps           | 1,000              | 1,000       |                        |
| Stable steps           | 48,000             | 48,000      |                        |
| Validation batches     | 500                | 50          | 10x less noisy         |
| Optimizer              | 8-bit AdamW / Muon | 8-bit AdamW |                        |
| Checkpoint rotation    | Last 3 only        | All         | Saves disk space       |

### Optional Integrations

| Integration    | Enable                           | Notes                                   |
| -------------- | -------------------------------- | --------------------------------------- |
| WandB logging  | `WANDB=1 python train.py`        | Tracks loss, LR, grad norm, val metrics |
| Muon optimizer | `MUON=1 python train.py`         | Needs `pip install muon-pytorch`        |
| Both           | `WANDB=1 MUON=1 python train.py` |                                         |

### VRAM Budget (8 GB RTX 3070)

| Component                            | Size        |
| ------------------------------------ | ----------- |
| Model weights (BF16)                 | 410 MB      |
| Gradients (BF16)                     | 410 MB      |
| 8-bit Adam states                    | 260 MB      |
| Activations (checkpointed, 1024 ctx) | ~2.8 GB     |
| CUDA kernels + overhead              | ~1.4 GB     |
| **Total**                            | **~5.3 GB** |
| **Headroom**                         | **~2.7 GB** |

If you hit OOM: set `MICRO_BATCH=1` and `ACCUM_STEPS=64` to maintain the same
effective batch size at half the per-step VRAM usage.

### RTX 3070 Timing Reference

All times assume: `MICRO_BATCH=2, ACCUM_STEPS=32, seq_len=1024, BF16, torch.compile`

Throughput: **~2,000-2,400 tok/s** (effective, including gradient accumulation overhead)

| Steps  | Tokens processed | Wall time  | 8h/day calendar |
| ------ | ---------------- | ---------- | --------------- |
| 1,000  | 65.5M            | ~7.5 hours | ~1 day          |
| 5,000  | 328M             | ~38 hours  | ~5 days         |
| 10,000 | 655M             | ~76 hours  | ~10 days        |
| 20,000 | 1.31B            | ~152 hours | ~19 days        |
| 40,000 | 2.62B            | ~304 hours | ~38 days        |
| 60,000 | 3.93B            | ~456 hours | ~57 days        |
| 80,000 | 5.24B            | ~608 hours | ~76 days        |

**To train faster without sacrificing quality:**

1. **Reduce `ACCUM_STEPS` to 16** — halves wall time, slightly noisier gradients
   (still much better than v1's 8). Effective batch = 32k tokens.
2. **Use `seq_len=512` for steps 1-20k, then extend to 1024** — context extension
   curriculum saves ~40% wall time on the early training where short context is fine.
3. **Keep only the last 3 checkpoints** — checkpoint saves with torch.save take
   30-60 seconds each; with CHECKPOINT_EVERY=2500 that's 24 checkpoints x ~45s = 18 min total.

### Loss Targets (approximate, Italian+C mix)

| Step   | Tokens trained | Expected val loss | What the model can do                    |
| ------ | -------------- | ----------------- | ---------------------------------------- |
| 500    | 33M            | ~5.5-6.0          | Nothing meaningful                       |
| 2,000  | 131M           | ~4.0-4.5          | Recognisable words                       |
| 5,000  | 328M           | ~3.0-3.5          | Short Italian phrases, C keywords        |
| 15,000 | 983M           | ~2.5-2.8          | Italian sentences, C function stubs      |
| 30,000 | 1.97B          | ~2.2-2.5          | Italian paragraphs, working C snippets   |
| 60,000 | 3.93B          | ~2.0-2.2          | Solid Italian prose, useful C completion |
| 80,000 | 5.24B          | ~1.85-2.0         | Near-peak for this size                  |

---

## Inference

### Generation Modes

```bash
# Standard generation with KV cache (default, ~150-200 tok/s)
python inference.py

# Speculative decoding with MTP heads (~200-400 tok/s)
python inference.py --speculative

# Without KV cache (sliding window, ~40-60 tok/s, for debugging)
python inference.py --no-kv-cache

# Single prompt
python inference.py --prompt "La funzione principale in C e:" --max-new 256

# Fine-tune generation style
python inference.py \
  --temperature 0.7 \       # Lower = more deterministic
  --top-p 0.90 \            # Nucleus sampling threshold
  --top-k 40 \              # Top-k cutoff
  --repetition-penalty 1.15 # Penalise recently seen tokens
```

### How Speculative Decoding Works

1. The **MTP draft heads** predict 2 future tokens cheaply
2. The **main model** verifies them in one forward pass
3. Accepted tokens are kept — rejected tokens trigger resampling
4. On average, 1.5-2 tokens are accepted per step, giving 1.5-2x speedup
5. The output distribution is **mathematically identical** to standard autoregressive generation

### Inference Speed Comparison

| Mode                      | Speed (RTX 3070) | Notes                    |
| ------------------------- | ---------------- | ------------------------ |
| Sliding window (no cache) | ~40-60 tok/s     | Correct but slow         |
| KV cache                  | ~150-200 tok/s   | 4-5x faster              |
| KV cache + speculative    | ~200-400 tok/s   | Additional 1.5-2x on top |

---

## Dashboard

Open `http://localhost:8686` in any browser after starting `python train.py`.

The dashboard shows:

- Live training loss + smoothed curve
- Validation loss + perplexity (updated every 2000 steps)
- Learning rate schedule visualisation
- Token throughput (tok/s)
- GPU VRAM usage with colour-coded warning (yellow >75%, red >90%)
- GPU utilisation %
- Gradient norm
- Progress bar with step count, total tokens, and ETA
- Data epoch counter
- Best validation loss achieved

The dashboard polls the training process every 2 seconds via `/api/metrics` — no
browser refresh needed. The server runs in a daemon thread inside `train.py`; it
dies when training ends or is interrupted.

---

## Troubleshooting

### RuntimeError: Cannot find a working triton installation

This happens on Windows because Triton doesn't support Windows. The v3.2 training script
auto-detects this and falls back to eager mode. If you see this error, you're running an
older version of train.py — update to the latest version.

### RuntimeError: Torch not compiled with CUDA enabled

You have the CPU-only version of PyTorch installed. Fix:

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

### 0 code files downloaded

This was caused by `codeparrot/github-code` using a deprecated loading script. The v3.2
download.py uses `bigcode/starcoderdata` (primary), `bigcode/the-stack-smol` (fallback),
and `codeparrot/github-code-clean` (last resort). If all three fail:

1. Check your internet connection
2. Run: `huggingface-cli login`
3. Try: `python download.py --sources c_code --sequential`

### RuntimeError: shape mismatch in attention

`n_kv_heads=4` is incompatible with `n_heads=18`. Change `n_kv_heads=6` in `ModelArgs`.

### Training stops early without reaching MAX_STEPS

Dataset exhaustion — the infinite_loader should handle this. Verify you're using
the latest train.py with `infinite_loader()`.

### CUDA OOM

1. `MICRO_BATCH=1`, `ACCUM_STEPS=64` (same effective batch, half peak VRAM)
2. `max_seq_len=512` in `ModelArgs` (halves activation memory)
3. Close all other GPU applications (browsers, games)
4. Windows: disable Hardware-Accelerated GPU Scheduling

### Loss stuck at ~4.0+ after 5,000 steps

1. Verify tokenizer was trained correctly: `tokenizer.json` should be > 5 MB
2. Check for double BOS/EOS bug — see Bug #2 above
3. Run `python filter.py` again; very low quality data prevents learning
4. LR too high or too low — try 4e-4 or 8e-4

### Dashboard shows 0 tok/s

`nvidia-smi` may not be on PATH (WSL2 sometimes needs `export PATH=$PATH:/usr/lib/wsl/lib`).
GPU stats will fall back to PyTorch's memory API (no utilisation %, but VRAM still shows).

### Can I resume training after changing ModelArgs?

No. Checkpoints store the raw weight tensors; changing architecture invalidates them.
The only safe changes after a checkpoint: `dropout`, `max_seq_len` (if you extend it),
`n_mtp_tokens` (only if you zero-initialise the new heads).

### How do I retrain from scratch?

Delete the generated data files and restart the pipeline:

```bash
# Delete generated data (keep your code files!)
rm -rf data_raw/ data_filtered/ data_mixed/ data/ tokenizer.json checkpoints/

# Windows PowerShell:
Remove-Item -Recurse -Force data_raw, data_filtered, data_mixed, data
Remove-Item -Force tokenizer.json
Remove-Item -Recurse -Force checkpoints

# Then re-run the full pipeline:
python download.py --tier quick
python filter.py
python mix.py
python preprocess.py
python train.py
```

**What to keep:** All `.py` files, `dashboard.html`, `README.md`, `requirements.txt`
**What to delete:** `data_raw/`, `data_filtered/`, `data_mixed/`, `data/`, `tokenizer.json`, `checkpoints/`

> **Tip:** You don't need to delete `data_raw/` if you just want to re-mix or re-tokenize.
> Deleting from `data_filtered/` onward is enough if the download was successful.

### langdetect not found

Install it: `pip install langdetect`. Without it, download.py falls back to an improved
Italian word-list filter that works but is less precise.

---

## Performance Benchmarks (expected at 60k steps, Tier 3+)

| Metric                        | Value             |
| ----------------------------- | ----------------- |
| Training throughput           | 2,000-2,400 tok/s |
| Inference speed (KV cache)    | ~150-200 tok/s    |
| Inference speed (speculative) | ~200-400 tok/s    |
| Inference speed (no cache)    | ~40-60 tok/s      |
| Validation loss               | ~2.0-2.2          |
| Val perplexity                | ~7-9              |
| HellaSwag (EN)                | ~35-40%           |
| MMLU 5-shot (EN)              | ~28-33%           |
| C HumanEval pass@1            | ~8-14%            |

---

## Possible Future Improvements

In rough priority order:

1. **Context length extension** — fine-tune at 2048-4096 tokens after initial 1024 training (cheap with YaRN or ABF)
2. **Instruction tuning** — after base training, fine-tune on Italian instruction datasets (Dolly-IT, Alpaca-IT)
3. **Continuous batching** — for faster throughput when serving multiple prompts
4. **Flash Attention 2** — custom CUDA kernel for even faster attention (if not auto-selected)
5. **Quantisation (GPTQ/AWQ)** — 4-bit inference for even faster generation with minimal quality loss
6. **DPO/RLHF** — alignment training for safer, more helpful outputs
7. **Mixture of Experts** — sparse MoE layers for more capacity at the same inference cost

---

## License

Educational and research use. Training data from public datasets.
Respect source dataset licences before distributing trained weights:

- OpenWebText: MIT
- FineWeb-Edu / FineWeb-2: ODC-By
- Wikipedia: CC BY-SA 4.0
- The Stack: The Stack licence (requires agreement on HuggingFace)
- OSCAR: CC0 / CC BY 4.0 (varies by language subset)
- Project Gutenberg: Public domain (individual works may vary)

---

## Acknowledgements

Architecture inspired by Llama-3, Qwen-2, and TinyLlama.
Flash Attention by Dao et al. 8-bit AdamW by bitsandbytes.
Speculative decoding inspired by Leviathan et al. (DeepMind).
Muon optimizer by Jordan et al.
Datasets: HuggingFace, OSCAR corpus, Wikimedia, BigCode, Project Gutenberg.
