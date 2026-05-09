"""
preprocess.py v2 — Tokenizer training and data packing with all fixes.
Multi-core parallelized for maximum throughput on large datasets.

Key fixes from review:
  ✅ Dynamic eos_id from tokenizer (not hardcoded to 3)
  ✅ Add <bos> token at start of each document
  ✅ Remove byte_fallback=True (was double-encoding with ByteLevel pre-tokenizer)
  ✅ Train tokenizer on 500k docs instead of 50k (better vocab coverage)
  ✅ Maintain source ratios in tokenizer training sample
  ✅ Support for Italian-heavy vocabulary
  ✅ CODE OVERSAMPLING: 1.5x for BPE merges (#include, ->, &&, etc.)
  ✅ Function-level splitting for C/C++ code
  ✅ Train tokenizer ONLY on train split (not val) to prevent data leakage

v2 — MULTI-CORE PARALLEL:
  ✅ Parallel tokenizer.encode() via ProcessPoolExecutor (biggest CPU win)
  ✅ Batches of documents encoded in parallel, written sequentially
  ✅ --workers flag to control parallelism (default: all CPUs)
  ✅ Live throughput reporting (docs/s, tokens/s)
"""

import os
import re
import json
import random
import argparse
import time
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from concurrent.futures import ProcessPoolExecutor


# ─── Code Detection for Oversampling ──────────────────────────────────────────

CODE_PATTERNS = ['c_code', 'cpp_code', 'other_code', '/code/', 'multi_code']

def _is_code_path(path):
    """Check if a file path contains code data."""
    path_lower = path.lower()
    return any(p in path_lower for p in CODE_PATTERNS)


def _is_code_jsonl(jsonl_path):
    """Check if a JSONL file contains code data based on path patterns."""
    path_lower = jsonl_path.lower()
    return any(p in path_lower for p in CODE_PATTERNS)


# ─── Tokenizer Training ──────────────────────────────────────────────────────

def train_tokenizer(jsonl_paths, vocab_size=32000, save_path="tokenizer.json",
                    sample_size=500_000):
    """Train a BPE tokenizer on a representative sample of the data."""
    sample_path = "tokenizer_sample.txt"
    print(f"Sampling {sample_size:,} documents for tokenizer training...")

    # Count total available docs
    total_available = 0
    code_available = 0
    text_available = 0
    for path in jsonl_paths:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        is_code = _is_code_path(path)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                total_available += 1
                if is_code:
                    code_available += 1
                else:
                    text_available += 1

    code_pct = code_available / max(total_available, 1) * 100
    print(f"  Total docs available: {total_available:,} (code: {code_available:,}, text: {text_available:,})")
    print(f"  Code ratio: {code_pct:.1f}%")

    CODE_OVERSAMPLE = 1.5
    code_weight = code_pct / 100 * CODE_OVERSAMPLE
    text_weight = text_available / max(total_available, 1)
    total_weight = code_weight + text_weight
    code_sample_pct = code_weight / total_weight * 100

    print(f"  Base sampling ratio: {code_pct:.1f}% code (1.5x oversampled)")
    print(f"  Effective tokenizer sample: ~{code_sample_pct:.1f}% code")

    rng = random.Random(42)
    written = 0
    with open(sample_path, 'w', encoding='utf-8') as out:
        for path in jsonl_paths:
            if not os.path.exists(path):
                continue
            is_code = _is_code_path(path)
            base_rate = sample_size / max(total_available, 1)
            if is_code:
                sample_rate = min(base_rate * CODE_OVERSAMPLE, 1.0)
            else:
                sample_rate = base_rate
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if rng.random() >= sample_rate:
                        continue
                    try:
                        text = json.loads(line)['text']
                        if text.strip():
                            out.write(text + '\n')
                            written += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

    print(f"  Sampled {written:,} documents to {sample_path}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    print(f"Training tokenizer (vocab_size={vocab_size})...")
    tokenizer.train([sample_path], trainer)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.save(save_path)
    os.remove(sample_path)

    vocab = tokenizer.get_vocab()
    print(f"Tokenizer saved: {save_path}")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  <pad>={tokenizer.token_to_id('<pad>')}, "
          f"<unk>={tokenizer.token_to_id('<unk>')}, "
          f"<bos>={tokenizer.token_to_id('<bos>')}, "
          f"<eos>={tokenizer.token_to_id('<eos>')}")

    test_c = '#include <stdio.h>'
    enc = tokenizer.encode(test_c)
    print(f"  C code test: '{test_c}' -> {len(enc.tokens)} tokens: {enc.tokens}")

    return tokenizer


# ─── Function-Level Splitting for C/C++ Code ──────────────────────────────────

def split_c_at_functions(text, max_chars_estimate=3600):
    """Split a C/C++ file at top-level function boundaries."""
    if len(text) <= max_chars_estimate:
        return [text]

    func_start = re.compile(
        r'^\w[\w\s\*]+\s+\w+\s*\([^;]*\)\s*\{', re.MULTILINE
    )
    positions = [m.start() for m in func_start.finditer(text)]
    if not positions:
        return [text]

    chunks = []
    positions = [0] + positions + [len(text)]
    current_chunk_start = 0
    current_chunk = ""

    for i in range(1, len(positions) - 1):
        candidate = text[current_chunk_start:positions[i + 1]]
        if len(candidate) > max_chars_estimate and current_chunk:
            chunks.append(current_chunk)
            current_chunk_start = positions[i]
            current_chunk = text[current_chunk_start:positions[i + 1]]
        else:
            current_chunk = candidate

    if current_chunk and len(current_chunk.strip()) > 50:
        chunks.append(current_chunk)

    if not chunks:
        return [text]

    return chunks


# ─── Parallel Tokenization Workers ────────────────────────────────────────────

def _encode_batch(batch_texts, tokenizer_path, is_code_file):
    """Encode a batch of documents using the tokenizer.
    Called in worker processes — each worker loads its own tokenizer instance.

    Returns list of (token_ids_list, doc_char_count) tuples.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    results = []
    for text in batch_texts:
        char_count = len(text)
        text_chunks = [text]
        if is_code_file:
            text_chunks = split_c_at_functions(text)
        doc_token_seqs = []
        for chunk in text_chunks:
            if not chunk.strip():
                continue
            ids = tokenizer.encode(chunk).ids
            doc_token_seqs.append(ids)
        results.append((doc_token_seqs, char_count))
    return results


def pack_jsonl(jsonl_path, output_path, tokenizer_path, seq_len=1024, workers=4):
    """Pack JSONL documents into a flat binary file for efficient training.
    Multi-core: batches of documents are tokenized in parallel.

    Args:
        tokenizer_path: Path to tokenizer.json (workers load their own instance)
        workers: Number of parallel encoding workers
    """
    bos_id = Tokenizer.from_file(tokenizer_path).token_to_id("<bos>")
    eos_id = Tokenizer.from_file(tokenizer_path).token_to_id("<eos>")

    if eos_id is None:
        raise ValueError("Tokenizer missing <eos> token!")
    if bos_id is None:
        raise ValueError("Tokenizer missing <bos> token!")

    vocab_size = Tokenizer.from_file(tokenizer_path).get_vocab_size()
    if vocab_size > 65535:
        raise ValueError(
            f"Vocab size {vocab_size} exceeds uint16 max (65535). "
            f"Reduce vocab_size or change binary format to int32."
        )

    is_code_file = _is_code_jsonl(jsonl_path)
    tag = " [CODE — function-level splitting]" if is_code_file else ""
    print(f"Packing {jsonl_path} -> {output_path} (seq_len={seq_len}, {workers} workers){tag}")

    # ── Read all documents into memory for batching ──
    # This is necessary for parallel encoding. For max-tier datasets,
    # the mixed JSONL is ~5-10GB which fits in RAM on most systems.
    print(f"  Reading documents...")
    t_read = time.time()
    all_docs = []
    with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                text = json.loads(line)['text']
                if text.strip():
                    all_docs.append(text)
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"  Read {len(all_docs):,} documents in {time.time()-t_read:.1f}s")

    # ── Encode in parallel batches ──
    BATCH_SIZE = 500  # docs per batch — balances overhead vs parallelism
    total_tokens = 0
    doc_count = 0
    total_chars = 0

    CHUNK_SIZE = 2_000_000  # Flush buffer every 2M tokens (~4MB)
    buf = np.empty(CHUNK_SIZE, dtype=np.uint16)
    buf_pos = 0

    t_encode = time.time()
    with open(output_path, 'wb') as fout:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # Submit batches
            futures = []
            for i in range(0, len(all_docs), BATCH_SIZE):
                batch = all_docs[i:i + BATCH_SIZE]
                futures.append(pool.submit(
                    _encode_batch, batch, tokenizer_path, is_code_file
                ))

            # Collect results in order and write to disk
            for batch_idx, future in enumerate(futures):
                batch_results = future.result()
                for doc_token_seqs, char_count in batch_results:
                    for ids in doc_token_seqs:
                        token_seq = [bos_id] + ids + [eos_id]
                        seq_len_actual = len(token_seq)

                        if buf_pos + seq_len_actual > CHUNK_SIZE:
                            fout.write(buf[:buf_pos].tobytes())
                            total_tokens += buf_pos
                            buf_pos = 0

                        buf[buf_pos:buf_pos + seq_len_actual] = token_seq
                        buf_pos += seq_len_actual

                    doc_count += 1
                    total_chars += char_count

                if (batch_idx + 1) % 20 == 0:
                    elapsed = time.time() - t_encode
                    docs_done = min((batch_idx + 1) * BATCH_SIZE, len(all_docs))
                    rate = docs_done / max(elapsed, 0.001)
                    print(f"  {docs_done:,}/{len(all_docs):,} docs "
                          f"({rate:,.0f} docs/s, "
                          f"{(total_tokens+buf_pos)/1e6:.1f}M tokens)")

        # Flush remaining tokens
        if buf_pos > 0:
            fout.write(buf[:buf_pos].tobytes())
            total_tokens += buf_pos

    # Trim to exact multiple of (seq_len + 1)
    total_tokens_final = total_tokens
    remainder = total_tokens_final % (seq_len + 1)
    if remainder:
        with open(output_path, 'r+b') as f:
            f.truncate((total_tokens_final - remainder) * 2)
        total_tokens_final -= remainder

    elapsed = time.time() - t_encode
    print(f"  Saved {total_tokens_final:,} tokens ({total_tokens_final//(seq_len+1):,} sequences)")
    print(f"  Docs: {doc_count:,} | Avg chars/doc: {total_chars/max(doc_count,1):.0f}")
    print(f"  Encoding time: {elapsed:.1f}s ({doc_count/max(elapsed,0.001):,.0f} docs/s, "
          f"{total_tokens_final/max(elapsed,0.001)/1e6:.2f}M tok/s)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for training")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel encoding workers (default: CPU count)")
    args = parser.parse_args()

    workers = args.workers or os.cpu_count() or 4
    os.makedirs("data", exist_ok=True)

    train_path = "data_mixed/train.jsonl"
    val_path = "data_mixed/val.jsonl"

    if not os.path.exists(train_path):
        print("ERROR: data_mixed/train.jsonl not found. Run download.py → filter.py → mix.py first.")
        exit(1)

    # Train tokenizer if it doesn't exist
    if not os.path.exists("tokenizer.json"):
        print("\n=== Training Tokenizer ===")
        train_tokenizer([train_path])
    else:
        print("Tokenizer already exists. Delete tokenizer.json to retrain.")

    # Pack data (parallel encoding)
    print(f"\n=== Packing Training Data ({workers} workers) ===")
    pack_jsonl(train_path, "data/train.bin", "tokenizer.json", workers=workers)

    print(f"\n=== Packing Validation Data ({workers} workers) ===")
    pack_jsonl(val_path, "data/val.bin", "tokenizer.json", workers=workers)

    print(f"\nPreprocessing complete! Run: python train.py")
