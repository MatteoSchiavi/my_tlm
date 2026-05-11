# SLM — Italian + C Small Language Model

A from-scratch 212M-parameter transformer language model trained on Italian text and C/C++ programming code, with an end-to-end pipeline from data acquisition to inference.

## Overview

This project trains a small but capable language model (~212M parameters) optimized for Italian language understanding and C programming. It implements a modern Llama-style transformer architecture from scratch in PyTorch, with a full data pipeline that downloads, filters, mixes, tokenizes, and packs training data from HuggingFace datasets. The model trains on a single consumer GPU (RTX 3070, 8 GB VRAM) using BF16 mixed precision, gradient checkpointing, and 8-bit optimizers.

**Key result:** The model is designed to be Chinchilla-optimal for 220M parameters, targeting ~5.24B training tokens with a data mix of 35% Italian, 30% C code, 10% C++ code, 5% other code, and 20% English.

## Architecture

| Parameter | Value |
|---|---|
| Dimensions | 1152 |
| Layers | 12 |
| Attention heads (Q) | 18 |
| KV heads (GQA) | 6 (3:1 ratio) |
| Hidden dim (SwiGLU) | 3072 |
| Vocab size | 32,000 (BPE) |
| Context length | 1024 tokens |
| Parameters | ~212M |

### Features

- **Grouped Query Attention (GQA):** 18 query heads / 6 KV heads for efficient inference
- **SwiGLU feed-forward:** PaLM/Llama-style activation with 8/3 × dim hidden size
- **QK-Norm:** RMSNorm on queries and keys before RoPE for attention stability
- **BF16-native RoPE:** Real-arithmetic rotary position embeddings (no FP32 upcast)
- **Weight tying:** Input embeddings = output projection (saves ~37M parameters)
- **Z-loss regularisation:** Prevents logit explosion (PaLM/Gemma technique)
- **Multi-Token Prediction (MTP):** 2 auxiliary heads predicting t+2 and t+3
- **Gradient checkpointing:** Saves ~40% activation memory
- **KV cache:** Fast autoregressive inference (~4-5x speedup)
- **Speculative decoding:** Uses MTP heads as draft models for 1.5-2.5x additional speedup
- **Document-boundary loss masking:** Excludes cross-document boundary tokens from loss

## Project Structure

```
.
├── download.py          # Data download from HuggingFace (tiered: smoke → max)
├── filter.py            # Quality filtering, dedup, MinHash LSH fuzzy dedup
├── mix.py               # Token-budget-aware data mixing and shuffling
├── preprocess.py        # BPE tokenizer training + sequence packing
├── model.py             # Transformer model (GQA, RoPE, SwiGLU, MTP, KV cache)
├── train.py             # Training loop with WSD schedule, dashboard, WandB
├── inference.py         # Text generation (KV cache, speculative decoding)
├── dashboard.html       # Real-time training monitoring dashboard
├── requirements.txt     # Python dependencies
└── README.md
```

### Data Pipeline Directories

```
data_raw/                # Raw downloaded data (JSONL per source)
├── italian/oscar/       # Italian web crawl (FineWeb-2, Wikipedia, OSCAR)
├── italian/wiki/        # Italian Wikipedia
├── italian/fineweb/     # FineWeb-2 Italian subset
├── english/web/         # OpenWebText
├── english/wiki/        # English Wikipedia
├── english/edu/         # FineWeb-Edu
├── code/c/              # C source code (StarCoder, The-Stack, GitHub)
├── code/cpp/            # C++ source code
└── code/other/          # Python, JS, Rust code
data_filtered/           # Quality-filtered + deduplicated data
data_mixed/              # Token-budget-mixed train/val split
data/                    # Packed binary token data (train.bin, val.bin)
checkpoints/             # Model checkpoints (best.pt, final.pt, ckpt_*.pt)
```

## Pipeline

### Step 1: Download Data

```bash
python download.py --tier standard
```

Five download tiers are available:

| Tier | Docs | Time | Use case |
|---|---|---|---|
| `smoke` | ~30K | ~5 min | Pipeline testing |
| `quick` | ~300K | ~30 min | First training run |
| `standard` | ~1.5M | ~4 hrs | Production quality |
| `full` | ~3.3M | ~10 hrs | Near Chinchilla-optimal |
| `max` | ~12M | ~40 hrs | Maximum intelligence (filtering-aware) |

The download script sources data from multiple HuggingFace datasets with automatic fallbacks:

- **Italian:** FineWeb-2 → Italian Wikipedia → OSCAR-2301 → mc4 (last resort)
- **C code:** StarCoderData → The-Stack-dedup → GitHub-Code → The-Vault → The-Stack-smol
- **C++ code:** The-Stack-dedup (cpp) → StarCoderData + language filter → The-Vault
- **Other code:** GitHub-Code (Python, JS, Rust, file-level) → CodeSearchNet
- **English:** OpenWebText → English Wikipedia → FineWeb-Edu

**Important:** For C/C++ code from gated datasets (StarCoderData, The-Stack-dedup), you need HuggingFace authentication:

```bash
pip install huggingface_hub
huggingface-cli login
# Accept terms at: https://huggingface.co/datasets/bigcode/starcoderdata
# Optionally: https://huggingface.co/datasets/bigcode/the-stack-dedup
```

Additional options:

```bash
python download.py --tier standard                    # Download all sources
python download.py --sources c_code --tier max        # Download only C code
python download.py --check-auth                       # Verify HF authentication
```

### Step 2: Filter Data

```bash
python filter.py
```

The filtering pipeline applies multiple quality checks in parallel across all CPU cores:

1. **Quality filters:**
   - Minimum length (50 chars for code, 200 for text)
   - Sufficient vocabulary (scaled threshold: `min(50, max(10, n//3))` unique words)
   - No excessive long words (≤50 chars for text, ≤80 for code)
   - Repetition detection (trigram repetition threshold)
   - Reasonable Latin/script ratio (≥65%)
   - Sentence length sanity check
   - Not list-heavy (<70% list markers)
   - Code quality: not mostly comments (C: 90% threshold, other: 80%)

2. **Deduplication:**
   - Local dedup within chunks (MD5 hash)
   - Global dedup across files (persistent SQLite3 store, ~30-50 MB on disk)
   - Fuzzy dedup via MinHash LSH (word 3-grams, parallelized) or fast O(n) suffix-hash for large files

3. **Benchmark contamination screening:** 13-gram overlap detection against known benchmarks

Options:

```bash
python filter.py -j 8                  # Use 8 workers
python filter.py --skip-fuzzy          # Skip MinHash fuzzy dedup (faster)
python filter.py --chunk-size 25000    # Lines per processing chunk
```

### Step 3: Mix Data

```bash
python mix.py
```

Mixes filtered data into the target token ratios using a two-phase approach:

**Token budget allocation:**

| Category | Token ratio | Chars/token |
|---|---|---|
| Italian | 35% | 4.7 |
| C code | 30% | 2.9 |
| C++ code | 10% | 3.2 |
| Other code | 5% | 3.3 |
| English | 20% | 4.3 |

**Process:**
1. **Stats scan:** Parallel scan of all filtered files (reservoir sampling for unbiased estimates)
2. **Budget-aware sampling:** Hash-based deterministic streaming — O(1) memory per file
3. **Stream-shuffle:** Distribute to 16 shards via deterministic MD5 hashing, shuffle in memory, systematic val sampling (every 100th doc → val set)

Output: `data_mixed/train.jsonl` and `data_mixed/val.jsonl`

Options:

```bash
python mix.py -j 8              # Use 8 workers
python mix.py --allow-unknown   # Allow uncategorized files (excluded from training)
```

### Step 4: Preprocess (Tokenizer + Packing)

```bash
python preprocess.py
```

1. **Train BPE tokenizer** (32K vocab, ByteLevel, `add_prefix_space=True`) on training data only (not val — prevents vocabulary contamination)
2. **Encode + pack** all documents into fixed-length sequences (1024 tokens) with BOS/EOS tokens, saved as memory-mapped binary files

The tokenizer sample is balanced at ~50% code (1.5x oversample) to match the ~40% code training ratio.

Output: `tokenizer.json`, `data/train.bin`, `data/val.bin`

Options:

```bash
python preprocess.py -j 4    # Use 4 encoding workers
```

### Step 5: Train

```bash
python train.py
```

**Training configuration:**

| Setting | Value |
|---|---|
| Effective batch | 65,536 tokens/step (4 × 16 × 1024) |
| Max steps | 80,000 |
| Learning rate | 3e-4 |
| Schedule | WSD (3.75% warmup → 71.25% stable → 25% cosine decay) |
| Min LR | 1e-5 |
| Weight decay | 0.1 (excludes norms + embeddings) |
| Grad clip | 1.0 |
| Precision | BF16 autocast |
| Optimizer | 8-bit AdamW (fallback: fused AdamW) |
| torch.compile | Auto-enabled on Linux/CUDA (Inductor) |

**Features during training:**
- Real-time monitoring dashboard at `http://localhost:8686`
- Automatic checkpoint saving (best val, periodic, final, and on Ctrl+C)
- WandB logging (`WANDB=1 python train.py`)
- VRAM pressure warnings
- Architecture mismatch detection on checkpoint resume
- Infinite data loader with per-epoch re-shuffling
- Document-boundary loss masking (computed on GPU for zero overhead)

Options:

```bash
python train.py                              # Default training
python train.py --micro-batch 8 --accum-steps 8  # Larger micro-batch (needs more VRAM)
python train.py --resume best.pt             # Resume from specific checkpoint
python train.py --reset-optimizer            # Reset optimizer state on resume
python train.py --no-8bit-adam               # Force standard AdamW
python train.py --no-compile                 # Disable torch.compile
WANDB=1 python train.py                      # Enable WandB logging
MUON=1 python train.py                       # Use Muon optimizer
```

### Step 6: Inference

```bash
python inference.py --prompt "Write a C function that sorts an array"
```

**Generation features:**
- KV cache for ~4-5x faster generation (~150-200 tok/s on RTX 3070)
- Speculative decoding using MTP heads (1.5-2.5x additional speedup)
- Top-p (nucleus) sampling, top-k sampling
- Temperature control
- Repetition penalty
- EOS detection
- Interactive multi-turn chat mode

Options:

```bash
python inference.py --prompt "Scrivi una funzione C"     # Single prompt
python inference.py                                       # Interactive mode
python inference.py --speculative --draft-steps 2         # Speculative decoding
python inference.py --no-kv-cache                        # Disable KV cache (debugging)
python inference.py --temperature 0.6 --top-p 0.9        # Sampling parameters
python inference.py --max-new 512                         # Generate up to 512 tokens
python inference.py --checkpoint checkpoints/best.pt      # Use specific checkpoint
```

## Training Dashboard

The embedded web dashboard provides real-time monitoring at `http://localhost:8686` while training runs. It displays:

- Training progress bar with ETA
- Live metrics: training loss, validation loss, perplexity, learning rate, gradient norm
- Interactive charts: training loss, validation loss, LR schedule, throughput
- GPU resource monitoring: VRAM usage, GPU utilization

The dashboard auto-connects when `python train.py` is running and gracefully handles disconnections.

## Data Sources

| Source | Type | Access | Content |
|---|---|---|---|
| [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Italian web | Open | High-quality Italian web text with language labels |
| [Italian Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | Italian encyclopedic | Open | Italian Wikipedia dump |
| [OSCAR-2301](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) | Italian web | Gated | Clean Italian web crawl |
| [mc4](https://huggingface.co/datasets/allenai/c4) | Italian web | Open | Common Crawl Italian (low quality, last resort) |
| [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) | English web | Open | Reddit-linked English web text |
| [English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | English encyclopedic | Open | English Wikipedia dump |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | English educational | Open | Educational quality English web data |
| [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata) | Code | Gated | Per-language code from The Stack |
| [The-Stack-dedup](https://huggingface.co/datasets/bigcode/the-stack-dedup) | Code | Gated | Deduplicated permissively-licensed code |
| [GitHub-Code](https://huggingface.co/datasets/codeparrot/github-code) | Code | Open | Multi-language GitHub code (file-level) |
| [The-Vault](https://huggingface.co/datasets/Fsoft-AIC/the-vault) | Code | Open | Function-level code with language labels |

## Requirements

- Python 3.9+
- CUDA-capable GPU (8 GB+ VRAM recommended; tested on RTX 3070)
- Linux recommended (torch.compile with Triton backend); Windows supported with limitations

See [requirements.txt](requirements.txt) for Python package dependencies.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace (required for C/C++ code)
huggingface-cli login

# Download data (start with 'quick' tier for testing)
python download.py --tier quick

# Filter, mix, preprocess, train
python filter.py
python mix.py
python preprocess.py
python train.py

# Generate text
python inference.py --prompt "#include <stdio.h>" --speculative
```

## Design Decisions & Audit Trail

This project has undergone multiple internal audits. Key fixes include:

- **RoPE BF16-native:** Replaced complex-number RoPE with real-arithmetic formulation, eliminating FP32 upcast on every forward pass (~175 MB peak VRAM savings)
- **MTP OOM fix:** Root cause was 262 MB logits tensor held alive during MTP section; freed before MTP starts
- **Weight decay exclusion:** Norms and embeddings excluded from L2 decay (prevents residual stream degradation)
- **Doc-boundary masking:** Applied consistently in both training and validation (was missing from val)
- **Micro-batch tuning:** MICRO_BATCH=4 with ACCUM_STEPS=16 for tensor core utilisation (~55-65% vs ~25% with MICRO_BATCH=1)
- **8-bit AdamW:** Saves ~1.32 GB VRAM by quantising optimizer m/v states
- **Filter vocabulary threshold:** Scaled `min(50, max(10, n//3))` so short Italian docs aren't systematically rejected
- **Data mixing:** Hash-based deterministic sampling replaces biased reservoir sampling; per-category chars/token calibration from empirical measurement
- **Tokenizer integrity:** Trained only on train split (not val); `add_prefix_space=True` reduces Italian token fragmentation

## License

This project is provided as-is for research and educational purposes. The training data is sourced from datasets with various licenses — please check each dataset's terms of use on HuggingFace before redistribution.
