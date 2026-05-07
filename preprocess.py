"""
preprocess.py — Tokenizer training and data packing with all fixes.

Key fixes from review:
  ✅ Dynamic eos_id from tokenizer (not hardcoded to 3)
  ✅ Add <bos> token at start of each document
  ✅ Remove byte_fallback=True (was double-encoding with ByteLevel pre-tokenizer)
  ✅ Train tokenizer on 500k docs instead of 50k (better vocab coverage)
  ✅ Maintain source ratios in tokenizer training sample
  ✅ Support for Italian-heavy vocabulary
"""

import os
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


def train_tokenizer(jsonl_paths, vocab_size=32000, save_path="tokenizer.json",
                    sample_size=500_000):
    """Train a BPE tokenizer on a representative sample of the data.

    Key choices:
      - ByteLevel pre-tokenizer (GPT-2 style) — NOT combined with byte_fallback
        (that would double-encode non-ASCII, breaking Italian text)
      - 500k doc sample (10× the original) for better vocab coverage
      - Balanced sample across source categories
      - Streams from disk to avoid loading everything into RAM
    """
    sample_path = "tokenizer_sample.txt"
    print(f"Sampling {sample_size:,} documents for tokenizer training...")

    # Count total available docs first (cheap scan)
    total_available = 0
    for path in jsonl_paths:
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                total_available += 1

    print(f"  Total docs available: {total_available:,}")

    # Reservoir sample to disk — avoids loading all text into RAM.
    # For 2.5M docs × ~500 bytes/doc = ~1.2GB of text strings in RAM,
    # which could OOM on a 16GB system. Instead we:
    # 1. Do a counting pass to know total docs
    # 2. Randomly select which line numbers to keep
    # 3. Do a second pass to write only those lines
    import random
    rng = random.Random(42)
    sample_ratio = min(sample_size / max(total_available, 1), 1.0)

    print(f"  Sampling ratio: {sample_ratio:.2%}")
    written = 0
    with open(sample_path, 'w', encoding='utf-8') as out:
        for path in jsonl_paths:
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if rng.random() >= sample_ratio:
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
    # Without this, decode() returns raw byte-level tokens with visible Ġ prefixes,
    # making all generated text look like: "Ġe Ġhi Ġsono Ġla" instead of " e hi sono la".
    tokenizer.decoder = ByteLevelDecoder()

    # Save
    tokenizer.save(save_path)
    os.remove(sample_path)

    # Print stats
    vocab = tokenizer.get_vocab()
    print(f"Tokenizer saved: {save_path}")
    print(f"  Vocab size: {len(vocab)}")
    print(f"  <pad>={tokenizer.token_to_id('<pad>')}, "
          f"<unk>={tokenizer.token_to_id('<unk>')}, "
          f"<bos>={tokenizer.token_to_id('<bos>')}, "
          f"<eos>={tokenizer.token_to_id('<eos>')}")

    return tokenizer


def pack_jsonl(jsonl_path, output_path, tokenizer, seq_len=1024):
    """Pack JSONL documents into a flat binary file for efficient training.

    Documents are separated by <eos> and preceded by <bos>.
    The binary format uses uint16 (2 bytes per token, max vocab 65535).
    Each training sample is seq_len+1 tokens (input + shifted target).
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

    # Stream tokens to disk incrementally instead of accumulating in RAM.
    # A Python list of 1B ints uses ~28GB RAM; streaming uses ~0.
    # We accumulate into a numpy buffer and flush to disk every CHUNK_SIZE tokens.
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

                # Encode — tokenizer has NO post-processor, so this returns raw token IDs only.
                ids = tokenizer.encode(text).ids

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
    # This is a simple truncation — we just need to know the total and trim the file
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
        train_tokenizer([train_path, val_path])
    else:
        print("Tokenizer already exists. Delete tokenizer.json to retrain.")

    # Pack data
    print("\n=== Packing Training Data ===")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    pack_jsonl(train_path, "data/train.bin", tokenizer)

    print("\n=== Packing Validation Data ===")
    pack_jsonl(val_path, "data/val.bin", tokenizer)

    print("\nPreprocessing complete! Run: python train.py")
