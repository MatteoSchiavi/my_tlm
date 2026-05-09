# My_SLM — v3.5 Build Guide

A 212M parameter Small Language Model built from scratch for 8 GB VRAM (RTX 3070),
optimised for **Italian language + C programming**, with a real-time training dashboard.
Full **Windows compatibility** — no Triton required.

---

## What's New in v3.5

### VRAM Stability & Loss Fluctuation Fix (CRITICAL)

Training at 99.2% VRAM was causing OOM crashes on the backward pass and ±3.5% loss
fluctuation. The root cause was a triple interaction: aggressive learning rate, high
dropout + MTP gradient noise, and zero VRAM headroom for peak allocations during the
backward pass. This release addresses all three:

| Issue                        | Cause                                                       | Fix                                                                      |
| ---------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| OOM on backward pass         | `MICRO_BATCH=2` + `aot_eager` compile allocated extra 250MB | `MICRO_BATCH=1`, `ACCUM_STEPS=64` (same effective batch, half peak VRAM) |
| Loss fluctuation (±3.5%)     | LR=4e-4 too aggressive at 99% VRAM                          | LR reduced to 3e-4                                                       |
| Loss spikes from MTP         | `mtp_weight=0.1` over-amplified gradient noise              | Reduced to 0.05                                                          |
| Gradient noise amplification | dropout=0.05 + MTP noise + VRAM pressure                    | Reduced to 0.02                                                          |
| CUDA OOM on compile          | `aot_eager` mode allocated extra buffers                    | Compile disabled by default on all platforms                             |
| No smooth loss metric        | Raw mini-batch loss is too noisy to track                   | EMA loss tracking (α=0.05) added                                         |
| Silent VRAM crisis           | No warning when GPU is about to OOM                         | VRAM pressure warning (>95% threshold)                                   |

### Data Mix Rebalancing

The C code tokenizer compression was poor (2.3 chars/token vs 6.5 for Italian),
meaning the model saw relatively fewer C characters per training step. The data mix
has been rebalanced to compensate:

| Category   | Old Ratio | New Ratio | Rationale                                                  |
| ---------- | --------- | --------- | ---------------------------------------------------------- |
| Italian    | 45%       | 35%       | Still primary language, but not over-represented           |
| C code     | 20%       | 35%       | 3x oversampling compensates for poor tokenizer compression |
| C++ code   | 5%        | 10%       | Shares syntax with C, improves structural understanding    |
| Other code | 5%        | 5%        | Python, JS, Bash, Rust — general programming patterns      |
| English    | 25%       | 15%       | Reduced to make room for more code                         |

**New effective mix: ~35% Italian, ~50% code (70% C), ~15% English**

### Other Changes

- `MAX_STEPS` extended from 60k to 80k (larger datasets need more training)
- `WARMUP_STEPS` extended from 2000 to 3000 (prevents early divergence with new LR)
- `STABLE_STEPS` adjusted: 3000 + 62,000 + 15,000 decay = 80,000 total
- `CUDA memory_fraction` reduced from 0.95 to 0.88 (12% headroom for peak allocations)
- EMA loss logged to WandB and JSONL for post-training analysis
- `TARGET_TOTAL` in mix.py increased from 7M to 10M documents
- Added `.gitignore` and `requirements.txt` (previously missing)
- Dashboard footer updated to v3.5

---

## What's New in v3.2

### Windows Compatibility (CRITICAL FIX)

| Issue                                                     | Cause                                                            | Fix                                                                                   |
| --------------------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `RuntimeError: Cannot find a working triton installation` | `torch.compile()` requires Triton, which doesn't work on Windows | Auto-detect: compile on Linux, eager mode on Windows                                  |
| 0 code files downloaded                                   | `codeparrot/github-code` uses deprecated loading script          | Replaced with `Fsoft-AIC/the-vault` (open) + `bigcode/starcoderdata` (gated fallback) |
| OSCAR download failed                                     | `oscar-corpus/OSCAR-2301` is gated (requires auth)               | Primary: `allenai/c4` (mc4, open access). OSCAR-2301 as fallback                      |
| The Stack download failed                                 | `bigcode/the-stack` is gated                                     | Replaced with `bigcode/the-stack-smol` (open access)                                  |
| `num_workers` crash on Windows                            | Windows multiprocessing issues with DataLoader                   | Auto-detect: `num_workers=0` on Windows, 4 on Linux                                   |
| bitsandbytes import crash                                 | May fail on Windows even when installed                          | Try/except with sanity check, fallback to standard AdamW                              |
| HF cache symlink warnings                                 | Windows needs Developer Mode for symlinks                        | `HF_HUB_DISABLE_SYMLINKS_WARNING=1` set automatically                                 |

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
| 6   | **New data sources**                    | Implemented | Italian FineWeb-2, C++ code, improved fallback chain            |
| 7   | **langdetect integration**              | Implemented | Accurate Italian filtering (fixes Bug 5)                        |
| 8   | **EMA loss tracking**                   | Implemented | Smoothed loss curve (α=0.05), logged to WandB + JSONL           |
| 9   | **VRAM pressure warning**               | Implemented | Prints actionable advice when VRAM >95%                         |

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
| 8   | Missing requirements         | requirements.txt        | Added langdetect, psutil, wandb                      |
| 9   | Fragile \_init_weights       | model.py                | Direct module type check instead of named_parameters |
| 10  | Dead spike detector          | train.py                | Removed check_spike (z-loss handles stability)       |
| 11  | Insufficient data tiers      | download.py             | Added smoke + max tiers                              |
| 12  | OOM on backward pass         | train.py                | MICRO_BATCH=1, CUDA memory_fraction=0.88             |
| 13  | Loss fluctuation ±3.5%       | model.py + train.py     | LR 3e-4, dropout 0.02, mtp_weight 0.05               |
| 14  | Poor C tokenizer compression | mix.py                  | C code 3x oversampled (35% of mix)                   |

---

## Download Tiers — Full Explanation

### Why Do the Numbers Seem "Low"?

The fundamental constraint is the training budget equation:

```
Training budget = MAX_STEPS x MICRO_BATCH x ACCUM_STEPS x seq_len
               = 80,000 x 1 x 64 x 1024
               = 5.24 billion tokens
```

Your dataset needs to hold AT LEAST this many tokens to avoid repeating documents
(repetition causes overfitting and kills generalisation). Given ~600 tokens per
document on average:

```
Minimum documents needed = 5.24B / 600 = ~8.7 million documents
```

### Chinchilla Context

The "Chinchilla optimal" rule says a model of **N parameters** should see
**~20N tokens** during training for compute-optimal performance:

```
212M params x 20 = 4.24 billion tokens  <- minimum for this model
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

| Metric             | Value                                   |
| ------------------ | --------------------------------------- |
| Download time      | ~3-5 minutes                            |
| Min training steps | 500 (to see loss move)                  |
| Good for           | "Does my GPU work? Does loss decrease?" |

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
| Recommended MAX_STEPS | 5,000                                        |
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

| Metric                | Value                          |
| --------------------- | ------------------------------ |
| Download time         | ~2-4 hours                     |
| Recommended MAX_STEPS | 20,000                         |
| Good for              | First production-quality model |

---

### Tier 3 — Full

**Purpose:** Best model achievable in a reasonable training run on one 3070.
Near Chinchilla-optimal for 212M params.

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
| Good for                       | Serious use, close to peak capability |

---

### Tier MAX — Maximum Intelligence

**Purpose:** Absolute best this architecture can achieve. Exceeds Chinchilla-optimal
(~1.2x the minimum token budget). Expect excellent Italian fluency, solid C completion,
and reasonable general English capability.

```bash
python download.py --tier max
```

> **Note:** Requires ~80-100 GB of free disk space during download + processing.
> Some sources (BigCode) require a HuggingFace account and licence agreement.

| Source                    | Documents      | Tokens (~) | Notes                    |
| ------------------------- | -------------- | ---------- | ------------------------ |
| Italian OSCAR             | 1,000,000      | 600M       | ~1/8 of the full corpus  |
| Italian Wikipedia         | 60,000         | 36M        | Full Italian Wikipedia   |
| Italian FineWeb-2         | 600,000        | 360M       | Best Italian web quality |
| English Web (OpenWebText) | 2,000,000      | 1,200M     | ~1/4 of the full corpus  |
| English Wikipedia         | 600,000        | 360M       | Top articles by length   |
| English Edu (FineWeb-Edu) | 1,200,000      | 720M       | High educational score   |
| C Code (The Vault)        | 800,000        | 800M       | License-filtered C files |
| Other Code                | 400,000        | 400M       | Python, JS, Rust, Shell  |
| **Total**                 | **~6,660,000** | **~4.48B** |                          |

| Metric                         | Value                                          |
| ------------------------------ | ---------------------------------------------- |
| Download time                  | ~16-28 hours                                   |
| Disk (raw + filtered + binary) | ~70-90 GB                                      |
| Recommended MAX_STEPS          | 80,000                                         |
| Training budget at 80k steps   | 5.24B tokens (~1.17 passes through data)       |
| Wall time (RTX 3070, 8h/day)   | ~83 days                                       |
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

- RTX 3070 Laptop (8 GB dedicated + 7 GB shared VRAM), 16 GB DDR4, Ryzen 7 5000 series, SSD
- Windows 11 native (eager mode, ~10-20% slower than compiled Linux)
- WSL2 Ubuntu 22.04

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
├── mix.py               # Mix 35% Italian / 35% C / 10% C++ / 5% other code / 15% English
├── preprocess.py        # BPE tokenizer (500k sample) + binary packing
├── model.py             # 212M Transformer (GQA, RoPE, SwiGLU, MTP, z-loss, KV cache)
├── train.py             # Training loop + live dashboard + WandB + Muon support
├── inference.py         # Generation with KV cache + speculative decoding
├── dashboard.html       # Real-time training monitor (served by train.py)
├── requirements.txt     # All dependencies
├── .gitignore           # Ignores data, checkpoints, and generated files
├── tokenizer.json       # Trained BPE tokenizer (32k vocab) — generated by preprocess.py
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

# Core + quality filtering
pip install datasets huggingface-hub tokenizers numpy
pip install langdetect

# Optional: experiment tracking
pip install wandb psutil

# Optional: 8-bit optimizer (saves ~1.3GB VRAM, may not work on Windows)
pip install bitsandbytes

# Optional: Muon optimizer (alternative to AdamW)
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
- `bigcode/starcoderdata` (code) — gated, needs licence agreement

```bash
huggingface-cli login
# Paste your token from huggingface.co/settings/tokens

# Then visit these URLs and click "Access repository":
# https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
# https://huggingface.co/datasets/bigcode/starcoderdata
```

**Without HF login**, you'll still get data from open-access sources:

- `allenai/c4` (mc4 Italian) instead of OSCAR-2301
- `Fsoft-AIC/the-vault` (truly open, ~1.9M C files) instead of StarCoder
- `code_search_net` for Python/JS

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

# Override batch config (keep 65k effective batch):
python train.py --micro-batch 1 --accum-steps 64    # Minimum VRAM (default)
python train.py --micro-batch 2 --accum-steps 32    # Faster but needs VRAM headroom

# Resume from checkpoint:
python train.py                                    # Auto-finds best.pt
python train.py --resume-from ckpt_4000.pt         # Specific checkpoint
python train.py --reset-optimizer                  # Fresh optimizer (recommended after data change)
python train.py --8bit-adam                        # Saves ~1.3GB VRAM but ~5-10% slower
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

### Model Specifications (v3.5)

| Parameter           | Value    | Notes                     |
| ------------------- | -------- | ------------------------- |
| Total parameters    | ~212M    | With weight tying         |
| Embedding dimension | 1,152    |                           |
| Layers              | 12       |                           |
| Attention heads     | 18       |                           |
| KV heads (GQA)      | **6**    | 3 Q-heads share 1 KV-head |
| Feed-forward dim    | 3,072    | SwiGLU hidden             |
| Context length      | 1,024    |                           |
| Vocabulary size     | 32,000   | BPE, Italian+C weighted   |
| RoPE theta          | 500,000  | Llama 3 style             |
| Dropout             | **0.02** | Reduced from 0.05 in v3.5 |
| MTP weight          | **0.05** | Reduced from 0.1 in v3.5  |
| Z-loss weight       | 1e-4     | PaLM/Gemma stability      |

### Architecture Features

**Grouped Query Attention (GQA)**
18 query heads share 6 key/value heads. Reduces KV memory by 3x vs standard MHA.
At inference with a KV cache, this allows much longer generation before memory pressure.

**QK-Norm (RMSNorm on Q and K)**
Prevents attention softmax saturation. Allows LR of 3e-4 without instability, even at
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
Weight reduced to 0.05 in v3.5 — higher values over-amplified gradient noise,
causing loss fluctuation at high VRAM utilisation.

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
at 1024 context in 8 GB. Enabled by default.

**Weight Tying**
Input embedding matrix = output projection matrix. Saves ~150M parameters (no separate
output head), and forces the model to use a unified semantic space for reading and writing.

**Residual Scaling Init (GPT-2 style)**
Output projections (`wo`, `w2`) initialised with `std = 0.02 / sqrt(2 x n_layers)`.
Without this, early training has residual activations that grow exponentially with depth.

**EMA Loss Tracking (v3.5)**
Exponential moving average of training loss with α=0.05. Provides a smooth, noise-free
loss curve that makes it easy to spot real trends vs mini-batch noise. Logged to WandB
(`train/ema_loss`) and JSONL (`ema_loss` field) alongside raw loss.

---

## Training Configuration

### Parameters

| Hyperparameter         | v3.5 Value  | v3.0 Value  | Notes                                                      |
| ---------------------- | ----------- | ----------- | ---------------------------------------------------------- |
| Micro-batch            | **1**       | 2           | Reduced — 99% VRAM was causing OOM + fluctuation           |
| Gradient accumulation  | **64**      | 32          | Compensates for smaller micro-batch                        |
| Effective batch tokens | 65,536      | 65,536      | Same total, half peak VRAM per step                        |
| Learning rate          | **3e-4**    | 6e-4        | Reduced — high LR + 99% VRAM = fluctuation                 |
| Sequence length        | 1,024       | 1,024       |                                                            |
| Max steps              | **80,000**  | 60,000      | Extended — larger datasets need more training              |
| Warmup steps           | **3,000**   | 2,000       | Extended — prevents early divergence                       |
| Stable steps           | **62,000**  | 44,000      | 3000 + 62000 + 15000 cosine decay = 80k                    |
| Dropout                | **0.02**    | 0.05        | Reduced — high dropout + MTP noise = instability           |
| MTP weight             | **0.05**    | 0.1         | Reduced — was over-amplifying gradient noise               |
| CUDA memory fraction   | **0.88**    | 0.95        | 12% headroom for backward pass peak allocations            |
| Validation batches     | 500         | 50          | 10x less noisy val loss                                    |
| Optimizer              | Fused AdamW | 8-bit AdamW | Standard AdamW is faster; use --8bit-adam if tight on VRAM |
| Checkpoint rotation    | Last 3 only | All         | Saves disk space                                           |
| EMA loss tracking      | **α=0.05**  | None        | Smooth loss curve, logged to WandB + JSONL                 |
| VRAM warning           | **>95%**    | None        | Prints actionable advice once                              |

### Optional Integrations

| Integration       | Enable                           | Notes                                             |
| ----------------- | -------------------------------- | ------------------------------------------------- |
| WandB logging     | `WANDB=1 python train.py`        | Tracks loss, EMA loss, LR, grad norm, val metrics |
| Muon optimizer    | `MUON=1 python train.py`         | Needs `pip install muon-pytorch`                  |
| 8-bit AdamW       | `python train.py --8bit-adam`    | Saves ~1.3GB VRAM but ~5-10% slower               |
| Both WandB + Muon | `WANDB=1 MUON=1 python train.py` |                                                   |

### VRAM Budget (8 GB RTX 3070)

With `MICRO_BATCH=1`, `ACCUM_STEPS=64`, `CUDA memory_fraction=0.88`:

| Component                            | Size        |
| ------------------------------------ | ----------- |
| Model weights (BF16)                 | 410 MB      |
| Gradients (BF16)                     | 410 MB      |
| AdamW states (FP32)                  | ~520 MB     |
| Activations (checkpointed, 1024 ctx) | ~2.8 GB     |
| CUDA kernels + overhead              | ~1.4 GB     |
| **Total**                            | **~5.5 GB** |
| **Headroom**                         | **~2.5 GB** |

With `MICRO_BATCH=2`, `ACCUM_STEPS=32` (faster but tighter):

| Component                                  | Size        |
| ------------------------------------------ | ----------- |
| Model weights (BF16)                       | 410 MB      |
| Gradients (BF16)                           | 410 MB      |
| AdamW states (FP32)                        | ~520 MB     |
| Activations (checkpointed, 2x micro-batch) | ~3.8 GB     |
| CUDA kernels + overhead                    | ~1.4 GB     |
| **Total**                                  | **~6.3 GB** |
| **Headroom**                               | **~1.7 GB** |

> **v3.5 recommendation:** Use `MICRO_BATCH=1, ACCUM_STEPS=64` by default. The 2.5 GB
> headroom prevents OOM during backward pass peak allocations (MTP heads, gradient
> buffers). You can try `MICRO_BATCH=2` if you have headroom, but watch for the VRAM
> pressure warning.

### RTX 3070 Timing Reference

All times assume: `MICRO_BATCH=1, ACCUM_STEPS=64, seq_len=1024, BF16, eager mode`

Throughput: **~1,600-2,000 tok/s** (eager mode on Windows), **~2,000-2,400 tok/s** (compiled Linux)

| Steps  | Tokens processed | Wall time  | 8h/day calendar |
| ------ | ---------------- | ---------- | --------------- |
| 1,000  | 65.5M            | ~7.5 hours | ~1 day          |
| 5,000  | 328M             | ~38 hours  | ~5 days         |
| 10,000 | 655M             | ~76 hours  | ~10 days        |
| 20,000 | 1.31B            | ~152 hours | ~19 days        |
| 40,000 | 2.62B            | ~304 hours | ~38 days        |
| 60,000 | 3.93B            | ~456 hours | ~57 days        |
| 80,000 | 5.24B            | ~608 hours | ~76 days        |

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

## Resuming Training After Data Changes

When you download new data and want to continue from a checkpoint:

```bash
# After re-running filter.py + mix.py + preprocess.py with new data:
python train.py --reset-optimizer
```

The `--reset-optimizer` flag loads model weights from the best checkpoint but starts
with a fresh optimizer state and LR schedule. This is recommended when:

- You've changed the dataset significantly (e.g., upgraded from `quick` to `max` tier)
- You've adjusted hyperparameters (LR, dropout, etc.)
- Training had loss regression issues

Without `--reset-optimizer`, the old optimizer momentum/variance estimates are loaded,
which may be mismatched with the new data distribution.

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

This happens on Windows because Triton doesn't support Windows. The v3.5 training script
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

The v3.5 download.py uses `Fsoft-AIC/the-vault` (truly open, no auth needed) as the
primary C code source, with `bigcode/starcoderdata` and `bigcode/the-stack-smol` as
fallbacks. If all fail:

1. Check your internet connection
2. Run: `huggingface-cli login`
3. Try: `python download.py --sources c_code --sequential`

### RuntimeError: shape mismatch in attention

`n_kv_heads=4` is incompatible with `n_heads=18`. Change `n_kv_heads=6` in `ModelArgs`.

### Training stops early without reaching MAX_STEPS

Dataset exhaustion — the infinite_loader should handle this. Verify you're using
the latest train.py with `infinite_loader()`.

### CUDA OOM

1. Use defaults: `MICRO_BATCH=1`, `ACCUM_STEPS=64` (same effective batch, minimum peak VRAM)
2. `max_seq_len=512` in `ModelArgs` (halves activation memory)
3. `python train.py --8bit-adam` (saves ~1.3GB optimizer VRAM)
4. Close all other GPU applications (browsers, games)
5. Windows: disable Hardware-Accelerated GPU Scheduling

### Loss fluctuation (±3%+)

This was the primary issue addressed in v3.5. If you still see large fluctuations:

1. Verify `MICRO_BATCH=1` and `ACCUM_STEPS=64` (not the old `MICRO_BATCH=2`)
2. Verify `LR=3e-4` (not the old `4e-4` or `6e-4`)
3. Verify `dropout=0.02` and `mtp_weight=0.05` in model.py
4. Check VRAM utilisation — if >95%, the GPU is struggling with memory pressure
5. The EMA loss (printed alongside raw loss) should be smooth — if raw loss fluctuates
   but EMA is stable, that's normal mini-batch noise

### Loss stuck at ~4.0+ after 5,000 steps

1. Verify tokenizer was trained correctly: `tokenizer.json` should be > 5 MB
2. Check for double BOS/EOS bug — see Bug #2 in the bugs table
3. Run `python filter.py` again; very low quality data prevents learning
4. Check dataset size — if you're on smoke/quick tier, there's not enough data
5. LR too high or too low — try 3e-4 (the default)

### Dashboard shows 0 tok/s

`nvidia-smi` may not be on PATH (WSL2 sometimes needs `export PATH=$PATH:/usr/lib/wsl/lib`).
GPU stats will fall back to PyTorch's memory API (no utilisation %, but VRAM still shows).

### Can I resume training after changing ModelArgs?

Architecture-breaking changes (dim, n_layers, n_heads, n_kv_heads, vocab_size,
hidden_dim, max_seq_len, n_mtp_tokens) invalidate checkpoints. The training script
will detect mismatches and refuse to load.

Safe changes after a checkpoint: `dropout`, `mtp_weight`, `z_loss_weight` — these
are training hyperparameters, not architecture. Use `--reset-optimizer` when changing
these to get a fresh LR schedule.

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

**What to keep:** All `.py` files, `dashboard.html`, `README.md`, `requirements.txt`, `.gitignore`
**What to delete:** `data_raw/`, `data_filtered/`, `data_mixed/`, `data/`, `tokenizer.json`, `checkpoints/`

> **Tip:** You don't need to delete `data_raw/` if you just want to re-mix or re-tokenize.
> Deleting from `data_filtered/` onward is enough if the download was successful.

### langdetect not found

Install it: `pip install langdetect`. Without it, download.py falls back to an improved
Italian word-list filter that works but is less precise.

---

## Performance Benchmarks (expected at 80k steps, Tier max)

| Metric                        | Value             |
| ----------------------------- | ----------------- |
| Training throughput (Windows) | 1,600-2,000 tok/s |
| Training throughput (Linux)   | 2,000-2,400 tok/s |
| Inference speed (KV cache)    | ~150-200 tok/s    |
| Inference speed (speculative) | ~200-400 tok/s    |
| Inference speed (no cache)    | ~40-60 tok/s      |
| Validation loss               | ~1.85-2.0         |
| Val perplexity                | ~6-8              |

---

## Possible Future Improvements

In rough priority order:

1. **Context length extension** — fine-tune at 2048-4096 tokens after initial 1024 training (cheap with YaRN or ABF)
2. **Instruction tuning** — after base training, fine-tune on Italian instruction datasets (Dolly-IT, Alpaca-IT)
3. **Tokenizer retraining** — train tokenizer with more C code to improve 2.3 chars/token compression
4. **Continuous batching** — for faster throughput when serving multiple prompts
5. **Flash Attention 2** — custom CUDA kernel for even faster attention (if not auto-selected)
6. **Quantisation (GPTQ/AWQ)** — 4-bit inference for even faster generation with minimal quality loss
7. **DPO/RLHF** — alignment training for safer, more helpful outputs
8. **Mixture of Experts** — sparse MoE layers for more capacity at the same inference cost

---

## License

Educational and research use. Training data from public datasets.
Respect source dataset licences before distributing trained weights:

- OpenWebText: MIT
- FineWeb-Edu / FineWeb-2: ODC-By
- Wikipedia: CC BY-SA 4.0
- The Vault (Fsoft-AIC): MIT
- OSCAR / mc4: CC0 / CC BY 4.0 (varies by language subset)
- CodeSearchNet: MIT

---

## Acknowledgements

Architecture inspired by Llama-3, Qwen-2, and TinyLlama.
Flash Attention by Dao et al. 8-bit AdamW by bitsandbytes.
Speculative decoding inspired by Leviathan et al. (DeepMind).
Muon optimizer by Jordan et al.
Datasets: HuggingFace, OSCAR corpus, Wikimedia, Fsoft-AIC, BigCode.
