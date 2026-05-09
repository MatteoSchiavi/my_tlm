"""
preprocess.py — Tokenizer training and data packing with all fixes.

Key fixes from review:
  ✅ Dynamic eos_id from tokenizer (not hardcoded to 3)
  ✅ Add <bos> token at start of each document
  ✅ Remove byte_fallback=True (was double-encoding with ByteLevel pre-tokenizer)
  ✅ Train tokenizer on 500k docs instead of 50k (better vocab coverage)
  ✅ Maintain source ratios in tokenizer training sample
  ✅ Support for Italian-heavy vocabulary
  ✅ CODE OVERSAMPLING: Code docs are 1.5x more likely to be sampled for
    tokenizer training, so the BPE merges learn code patterns like
    #include, ->, &&, ||, printf, struct, etc.
  ✅ Function-level splitting for C/C++ code to avoid mid-function truncation
  ✅ Train tokenizer ONLY on train split (not val) to prevent data leakage
"""

import os
import re
import json
import random
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
# NOTE: Do NOT add TemplateProcessing post-processor here.
# pack_jsonl() manually inserts <bos>/<eos> between documents.
# A post-processor would cause DOUBLE BOS/EOS: BOS BOS ... tokens ... EOS EOS.


# ─── Code Detection for Oversampling ──────────────────────────────────────────

CODE_PATTERNS = ['c_code', 'cpp_code', 'other_code', '/code/', 'multi_code']

def _is_code_path(path):
    """Check if a file path contains code data."""
    path_lower = path.lower()
    return any(p in path_lower for p in CODE_PATTERNS)


def train_tokenizer(jsonl_paths, vocab_size=32000, save_path="tokenizer.json",
                    sample_size=500_000):
    """Train a BPE tokenizer on a representative sample of the data.

    Key choices:
      - ByteLevel pre-tokenizer (GPT-2 style) — NOT combined with byte_fallback
        (that would double-encode non-ASCII, breaking Italian text)
      - 500k doc sample (10× the original) for better vocab coverage
      - CODE OVERSAMPLING: code docs are sampled at 1.5x the rate of text docs,
        so the BPE merges learn code patterns like #include, ->, &&, printf, etc.
        This gives ~50% code in the tokenizer sample, matching the actual
        training token ratio (~40% code) more closely than 3x oversampling.
      - Streams from disk to avoid loading everything into RAM
      - Should be trained ONLY on train split to prevent val data leakage
    """
    sample_path = "tokenizer_sample.txt"
    print(f"Sampling {sample_size:,} documents for tokenizer training...")

    # Count total available docs first (cheap scan)
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

    # Calculate sampling probabilities with code oversampling
    # Code docs get 1.5x higher probability of being selected
    CODE_OVERSAMPLE = 1.5

    # Effective sample: if code is 30% of data, after 1.5x oversampling it becomes
    # 30% * 1.5 / (30% * 1.5 + 70% * 1) = ~39% of the tokenizer training sample
    # This more closely matches the actual training data ratio (~40% code)
    code_weight = code_pct / 100 * CODE_OVERSAMPLE
    text_weight = text_available / max(total_available, 1)
    total_weight = code_weight + text_weight
    code_sample_pct = code_weight / total_weight * 100

    print(f"  Base sampling ratio: {code_pct:.1f}% code (C code: 1.5x oversampled)")
    print(f"  Effective tokenizer sample: ~{code_sample_pct:.1f}% code")

    # Two-pass sampling: first pass selects which lines to keep
    rng = random.Random(42)

    # We need to give code docs a higher probability of selection
    # Simple approach: adjust sampling ratio per file based on type
    written = 0
    with open(sample_path, 'w', encoding='utf-8') as out:
        for path in jsonl_paths:
            if not os.path.exists(path):
                continue
            is_code = _is_code_path(path)

            # Calculate sampling rate for this file
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

    # Train BPE tokenizer — NO byte_fallback (conflicts with ByteLevel pre-tokenizer)
    # ByteLevel alone handles all Unicode via byte representation (GPT-2 approach)
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Train
    print(f"Training tokenizer (vocab_size={vocab_size})...")
    tokenizer.train([sample_path], trainer)

    # Do NOT add a TemplateProcessing post-processor here.
    # pack_jsonl() manually inserts <bos>/<eos> between documents.
    # If we added a post-processor, every encode() call would also wrap with
    # BOS/EOS, resulting in BOS BOS ... tokens ... EOS EOS in the training data.

    # CRITICAL: Add ByteLevel decoder so tokenizer.decode() converts Ġ back to spaces.
    tokenizer.decoder = ByteLevelDecoder()

    # Save
    tokenizer.save(save_path)
    os.remove(sample_path)

    # Print stats + code token quality check
    vocab = tokenizer.get_vocab()
    print(f"Tokenizer saved: {save_path}")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  <pad>={tokenizer.token_to_id('<pad>')}, "
          f"<unk>={tokenizer.token_to_id('<unk>')}, "
          f"<bos>={tokenizer.token_to_id('<bos>')}, "
          f"<eos>={tokenizer.token_to_id('<eos>')}")

    # Quick check: does the tokenizer efficiently encode C code?
    test_c = '#include <stdio.h>'
    enc = tokenizer.encode(test_c)
    print(f"  C code test: '{test_c}' -> {len(enc.tokens)} tokens: {enc.tokens}")
    if len(enc.tokens) <= 3:
        print(f"    ✅ Good C compression!")
    elif len(enc.tokens) <= 6:
        print(f"    ⚠️  Moderate C compression (expected without code data)")
    else:
        print(f"    ❌ Poor C compression — tokenizer needs more code data")

    return tokenizer


# ─── Function-Level Splitting for C/C++ Code ──────────────────────────────────

def _is_code_jsonl(jsonl_path):
    """Check if a JSONL file contains code data based on path patterns."""
    path_lower = jsonl_path.lower()
    code_patterns = ['c_code', 'cpp_code', 'other_code', '/code/', 'multi_code']
    return any(p in path_lower for p in code_patterns)


def split_c_at_functions(text, max_chars_estimate=3600):
    """Split a C/C++ file at top-level function boundaries.

    This prevents long C files from being arbitrarily truncated mid-function
    when packing into fixed-length training sequences. Each chunk will contain
    one or more complete functions, ensuring the model sees self-contained
    code during training.

    max_chars_estimate: approximate char limit (~900 tokens * 4 chars/token)
    """
    if len(text) <= max_chars_estimate:
        return [text]

    # Find lines that look like function definitions: type name(...) {
    func_start = re.compile(
        r'^\w[\w\s\*]+\s+\w+\s*\([^;]*\)\s*\{', re.MULTILINE
    )
    positions = [m.start() for m in func_start.finditer(text)]
    if not positions:
        # No function boundaries found — return as-is
        return [text]

    # Build chunks between function boundaries
    chunks = []
    positions = [0] + positions + [len(text)]
    current_chunk_start = 0
    current_chunk = ""

    for i in range(1, len(positions) - 1):
        candidate = text[current_chunk_start:positions[i + 1]]
        if len(candidate) > max_chars_estimate and current_chunk:
            # Current chunk would be too long — flush what we have
            chunks.append(current_chunk)
            current_chunk_start = positions[i]
            current_chunk = text[current_chunk_start:positions[i + 1]]
        else:
            current_chunk = candidate

    if current_chunk and len(current_chunk.strip()) > 50:
        chunks.append(current_chunk)

    # If no chunks were produced (edge case), return original
    if not chunks:
        return [text]

    return chunks


def pack_jsonl(jsonl_path, output_path, tokenizer, seq_len=1024):
    """Pack JSONL documents into a flat binary file for efficient training.

    Documents are separated by <eos> and preceded by <bos>.
    The binary format uses uint16 (2 bytes per token, max vocab 65535).
    Each training sample is seq_len+1 tokens (input + shifted target).

    For code files, documents are split at function boundaries so that
    the model sees complete function definitions rather than arbitrarily
    truncated mid-function code.
    """
    eos_id = tokenizer.token_to_id("<eos>")
    bos_id = tokenizer.token_to_id("<bos>")

    if eos_id is None:
        raise ValueError("Tokenizer missing <eos> token!")
    if bos_id is None:
        raise ValueError("Tokenizer missing <bos> token!")

    # Validate vocab fits in uint16 (max 65535)
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65535:
        raise ValueError(
            f"Vocab size {vocab_size} exceeds uint16 max (65535). "
            f"Reduce vocab_size in ModelArgs or change binary format to int32."
        )

    print(f"Packing {jsonl_path} -> {output_path} (seq_len={seq_len})")

    is_code_file = _is_code_jsonl(jsonl_path)
    if is_code_file:
        print(f"  Code file detected — using function-level splitting")

    # Stream tokens to disk incrementally instead of accumulating in RAM.
    CHUNK_SIZE = 2_000_000  # Flush every 2M tokens (~4MB)
    buf = np.empty(CHUNK_SIZE, dtype=np.uint16)
    buf_pos = 0
    total_tokens = 0
    doc_count = 0
    total_chars = 0

    with open(output_path, 'wb') as fout:
        with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    text = json.loads(line)['text']
                except (json.JSONDecodeError, KeyError):
                    continue

                if not text.strip():
                    continue

                # For code files, split at function boundaries to avoid
                # truncating mid-function when packing into seq_len chunks
                text_chunks = [text]
                if is_code_file:
                    text_chunks = split_c_at_functions(text)

                for chunk_text in text_chunks:
                    if not chunk_text.strip():
                        continue

                    # Encode — tokenizer has NO post-processor, so this returns raw token IDs only.
                    ids = tokenizer.encode(chunk_text).ids

                    # Build token sequence: BOS + content + EOS
                    token_seq = [bos_id] + ids + [eos_id]
                    seq_len_actual = len(token_seq)

                    # Check if buffer needs flushing before adding this sequence
                    if buf_pos + seq_len_actual > CHUNK_SIZE:
                        # Flush buffer to disk
                        fout.write(buf[:buf_pos].tobytes())
                        total_tokens += buf_pos
                        buf_pos = 0

                    # Copy tokens into buffer
                    buf[buf_pos:buf_pos + seq_len_actual] = token_seq
                    buf_pos += seq_len_actual

                doc_count += 1
                total_chars += len(text)

                if doc_count % 100_000 == 0:
                    print(f"  {doc_count:,} docs, {total_tokens + buf_pos:,} tokens")

        # Flush remaining tokens
        if buf_pos > 0:
            fout.write(buf[:buf_pos].tobytes())
            total_tokens += buf_pos

    # Now re-read to trim to exact multiple of (seq_len + 1)
    total_tokens_final = total_tokens
    remainder = total_tokens_final % (seq_len + 1)
    if remainder:
        # Truncate the file by removing the last `remainder` tokens (2 bytes each)
        with open(output_path, 'r+b') as f:
            f.truncate((total_tokens_final - remainder) * 2)
        total_tokens_final -= remainder

    print(f"  Saved {total_tokens_final:,} tokens ({total_tokens_final//(seq_len+1):,} sequences)")
    print(f"  Docs: {doc_count:,} | Avg chars/doc: {total_chars/max(doc_count,1):.0f}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    train_path = "data_mixed/train.jsonl"
    val_path = "data_mixed/val.jsonl"

    if not os.path.exists(train_path):
        print("ERROR: data_mixed/train.jsonl not found. Run download.py → filter.py → mix.py first.")
        exit(1)

    # Train tokenizer if it doesn't exist
    if not os.path.exists("tokenizer.json"):
        print("\n=== Training Tokenizer ===")
        train_tokenizer([train_path])  # Train ONLY on train split — not val
    else:
        print("Tokenizer already exists. Delete tokenizer.json to retrain.")

    # Pack data
    print("\n=== Packing Training Data ===")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    pack_jsonl(train_path, "data/train.bin", tokenizer)

    print("\n=== Packing Validation Data ===")
    pack_jsonl(val_path, "data/val.bin", tokenizer)

    print("\nPreprocessing complete! Run: python train.py")
