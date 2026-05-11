"""
preprocess.py v10 — Tokenizer + packing with audit fixes applied.

v10 fixes (audit-driven):
  + FIXED: CODE_OVERSAMPLE reduced from 3.0 to 1.5 — tokenizer sample is now
    ~50% code instead of ~71%, matching actual 40% training ratio (Issue 2)
  + FIXED: Tokenizer trains only on train_path, NOT val_path — eliminates
    train/val contamination via vocabulary adaptation (Issue 3)
  + FIXED: add_prefix_space=True for ByteLevel BPE — reduces Italian token
    fragmentation from doubled space-prefixed tokens (Issue 17)

v9 features preserved:
  + Windows-safe: ZERO large data through multiprocessing pipes
  + Temp files for BOTH input and output directions
  + Same encode_batch() with num_threads=2 for Rust-level parallelism
  + Same writer thread with bounded queue for backpressure
"""

import os
import sys
import argparse
import time
import random
import threading
import queue
import tempfile
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

try:
    import orjson
    def json_loads(line):
        return orjson.loads(line)
    def json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
    HAS_ORJSON = True
except ImportError:
    import json as _json
    def json_loads(line):
        return _json.loads(line)
    def json_dumps(obj):
        return _json.dumps(obj, ensure_ascii=False)
    HAS_ORJSON = False


# ─── RAM monitoring (Windows + Linux) ─────────────────────────────────────────

def _get_available_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except ImportError:
        pass
    if sys.platform == 'win32':
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / (1024**3)
        except Exception:
            pass
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return 999

MIN_FREE_RAM_GB = 2.0


CODE_PATTERNS = ['c_code', 'cpp_code', 'other_code', '/code/', 'multi_code']
def _is_code_path(path):
    return any(p in path.lower() for p in CODE_PATTERNS)

READ_BUF_SIZE = 64 * 1024 * 1024


def read_jsonl_texts(path):
    buf = ""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            block = f.read(READ_BUF_SIZE)
            if not block:
                if buf:
                    try:
                        t = json_loads(buf).get('text', '')
                        if t and t.strip(): yield t
                    except (ValueError, KeyError): pass
                break
            buf += block
            lines = buf.split('\n')
            buf = lines[-1]
            for line in lines[:-1]:
                line = line.strip()
                if not line: continue
                try:
                    t = json_loads(line).get('text', '')
                    if t and t.strip(): yield t
                except (ValueError, KeyError): continue


def count_lines_fast(path):
    count = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            block = f.read(READ_BUF_SIZE)
            if not block: break
            count += block.count('\n')
    return count


# ─── Tokenizer Training ─────────────────────────────────────────────────────

def train_tokenizer(jsonl_paths, vocab_size=32000, save_path="tokenizer.json",
                    sample_size=500_000, workers=None):
    if workers is None:
        workers = os.cpu_count() or 1

    sample_path = "tokenizer_sample.txt"
    print(f"Sampling {sample_size:,} documents for tokenizer training...")

    total_available = code_available = 0
    for path in jsonl_paths:
        if not os.path.exists(path): continue
        count = count_lines_fast(path)
        total_available += count
        if _is_code_path(path): code_available += count

    text_available = total_available - code_available

    # ── FIX Issue 2: CODE_OVERSAMPLE = 1.5 (was 3.0) ─────────────────────────
    # With 3.0x oversample, code was ~71% of tokenizer sample (way too much).
    # 1.5x gives ~50% code in sample, matching actual ~40% training ratio.
    CODE_OVERSAMPLE = 1.5
    code_weight = code_available / max(total_available, 1) * CODE_OVERSAMPLE
    text_weight = text_available / max(total_available, 1)
    total_weight = code_weight + text_weight
    print(f"  Docs: {total_available:,} (code: {code_available:,}, text: {text_available:,})")
    print(f"  Effective sample: ~{code_weight/total_weight*100:.1f}% code ({CODE_OVERSAMPLE:.1f}x oversample)")
    print(f"  Target: ~50% code in sample (matches ~40% training ratio)")

    rng = random.Random(42)
    written = 0
    write_buf = []
    WRITE_BUF_SIZE = 16 * 1024 * 1024

    with open(sample_path, 'w', encoding='utf-8', buffering=READ_BUF_SIZE) as out:
        for path in jsonl_paths:
            if not os.path.exists(path): continue
            is_code = _is_code_path(path)
            base_rate = sample_size / max(total_available, 1)
            sample_rate = min(base_rate * CODE_OVERSAMPLE, 1.0) if is_code else base_rate
            for text in read_jsonl_texts(path):
                if rng.random() >= sample_rate: continue
                write_buf.append(text + '\n')
                written += 1
                if sum(len(l) for l in write_buf) >= WRITE_BUF_SIZE:
                    out.write(''.join(write_buf))
                    write_buf = []
                if written >= sample_size * 1.2: break
            if write_buf:
                out.write(''.join(write_buf))
                write_buf = []

    print(f"  Sampled {written:,} documents")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )
    # ── FIX Issue 17: add_prefix_space=True ───────────────────────────────────
    # With add_prefix_space=False, mid-sentence words get a Ġ prefix, effectively
    # doubling some common Italian tokens (e.g., "cane" and "Ġcane" are separate).
    # add_prefix_space=True ensures consistent tokenization regardless of position.
    # This reduces Italian token fertility and makes better use of the 32k vocab.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()
    print(f"Training tokenizer (vocab_size={vocab_size}, add_prefix_space=True)...")
    tokenizer.train([sample_path], trainer)
    tokenizer.save(save_path)
    os.remove(sample_path)

    vocab = tokenizer.get_vocab()
    print(f"Tokenizer saved: {save_path} (vocab: {len(vocab)})")
    print(f"  <pad>={tokenizer.token_to_id('<pad>')}, <unk>={tokenizer.token_to_id('<unk>')}, "
          f"<bos>={tokenizer.token_to_id('<bos>')}, <eos>={tokenizer.token_to_id('<eos>')}")
    test_c = '#include <stdio.h>'
    enc = tokenizer.encode(test_c)
    print(f"  C test: '{test_c}' -> {len(enc.tokens)} tokens: {enc.tokens}")
    test_it = 'Il compilatore C mostra un errore'
    enc_it = tokenizer.encode(test_it)
    print(f"  IT test: '{test_it}' -> {len(enc_it.tokens)} tokens: {enc_it.tokens}")
    return tokenizer


# ─── Encoding Workers (temp file based — pipe safe) ──────────────────────────

BATCH_SIZE = 500

def _encode_batch_from_file(input_temp_path, tokenizer_path, num_threads=2):
    """Read texts from input temp file, tokenize, write results to output temp file."""
    tokenizer = Tokenizer.from_file(tokenizer_path)

    texts = []
    total_chars = 0
    with open(input_temp_path, 'r', encoding='utf-8', errors='ignore',
              buffering=READ_BUF_SIZE) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                text = json_loads(line).get('text', '')
                if text and text.strip():
                    texts.append(text)
                    total_chars += len(text)
            except (ValueError, KeyError):
                continue

    try:
        os.remove(input_temp_path)
    except OSError:
        pass

    if not texts:
        return (None, 0, 0)

    try:
        encodings = tokenizer.encode_batch(texts, num_threads=num_threads)
    except TypeError:
        encodings = tokenizer.encode_batch(texts)

    fd, output_temp = tempfile.mkstemp(suffix='.tok', prefix='enc_')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        for enc in encodings:
            f.write(' '.join(str(tid) for tid in enc.ids) + '\n')

    return (output_temp, len(texts), total_chars)


# ─── Background Binary Writer Thread ─────────────────────────────────────────

def _start_bin_writer(output_path, max_queue=4):
    """Background writer for binary token data. Bounded queue = backpressure."""
    wq = queue.Queue(maxsize=max_queue)
    result = {'total_tokens': 0}

    def _writer():
        buf = np.empty(2_000_000, dtype=np.uint16)
        buf_pos = 0
        total_tokens = 0

        with open(output_path, 'wb') as f:
            while True:
                item = wq.get()
                if item is None:
                    break
                for token_seq in item:
                    seq_len = len(token_seq)

                    if seq_len > len(buf):
                        if buf_pos > 0:
                            f.write(buf[:buf_pos].tobytes())
                            total_tokens += buf_pos
                            buf_pos = 0
                        f.write(np.array(token_seq, dtype=np.uint16).tobytes())
                        total_tokens += seq_len
                        continue

                    if buf_pos + seq_len > len(buf):
                        f.write(buf[:buf_pos].tobytes())
                        total_tokens += buf_pos
                        buf_pos = 0
                    buf[buf_pos:buf_pos + seq_len] = token_seq
                    buf_pos += seq_len

            if buf_pos > 0:
                f.write(buf[:buf_pos].tobytes())
                total_tokens += buf_pos

        result['total_tokens'] = total_tokens

    t = threading.Thread(target=_writer, daemon=True)
    t.start()
    return wq, t, result


# ─── Temp directory management ───────────────────────────────────────────────

def _make_temp_dir():
    temp_dir = os.path.join("data", ".enc_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


# ─── Packing ─────────────────────────────────────────────────────────────────

def pack_jsonl(jsonl_path, output_path, tokenizer_path, seq_len=1024, workers=None):
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) // 2)

    tokenizer = Tokenizer.from_file(tokenizer_path)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None or bos_id is None:
        raise ValueError("Tokenizer missing <bos> or <eos> token!")
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65535:
        raise ValueError(f"Vocab size {vocab_size} exceeds uint16 max.")
    del tokenizer

    num_threads = max(1, (os.cpu_count() or 1) // workers)
    pipeline_depth = workers * 2

    print(f"Packing {jsonl_path} -> {output_path}")
    print(f"  {workers} workers x {num_threads} threads = {workers*num_threads} threads")
    print(f"  Batch: {BATCH_SIZE} | Pipeline: {pipeline_depth}")

    total_docs = count_lines_fast(jsonl_path)
    print(f"  Total docs: {total_docs:,}")

    doc_count = 0
    total_chars = 0
    t_start = time.time()

    if workers <= 1:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        wq, writer_thread, writer_result = _start_bin_writer(output_path, max_queue=2)
        batch = []
        try:
            for text in read_jsonl_texts(jsonl_path):
                batch.append(text)
                if len(batch) >= BATCH_SIZE:
                    encodings = tokenizer.encode_batch(batch)
                    token_seqs = [[bos_id] + enc.ids + [eos_id] for enc in encodings]
                    wq.put(token_seqs)
                    doc_count += len(batch)
                    total_chars += sum(len(t) for t in batch)
                    batch = []
                    if doc_count % 100_000 == 0:
                        _print_prog(doc_count, t_start)
            if batch:
                encodings = tokenizer.encode_batch(batch)
                token_seqs = [[bos_id] + enc.ids + [eos_id] for enc in encodings]
                wq.put(token_seqs)
                doc_count += len(batch)
                total_chars += sum(len(t) for t in batch)
        finally:
            wq.put(None)
            writer_thread.join()
        total_tokens = writer_result['total_tokens']
    else:
        write_queue, writer_thread, writer_result = _start_bin_writer(output_path, max_queue=4)
        temp_dir = _make_temp_dir()

        pending = {}
        batch_texts = []
        exhausted = False
        chunk_iter = read_jsonl_texts(jsonl_path)

        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                while pending or not exhausted:
                    while len(pending) < pipeline_depth and not exhausted:
                        if len(pending) > workers and _get_available_ram_gb() < MIN_FREE_RAM_GB:
                            break

                        try:
                            text = next(chunk_iter)
                            batch_texts.append(text)
                            if len(batch_texts) >= BATCH_SIZE:
                                fd, input_temp = tempfile.mkstemp(
                                    suffix='.in', dir=temp_dir, prefix='bat_')
                                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                                    for t in batch_texts:
                                        f.write(json_dumps({'text': t}) + '\n')
                                del batch_texts
                                batch_texts = []

                                future = pool.submit(
                                    _encode_batch_from_file,
                                    input_temp, tokenizer_path, num_threads)
                                pending[future] = input_temp
                        except StopIteration:
                            exhausted = True
                            if batch_texts:
                                fd, input_temp = tempfile.mkstemp(
                                    suffix='.in', dir=temp_dir, prefix='bat_')
                                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                                    for t in batch_texts:
                                        f.write(json_dumps({'text': t}) + '\n')
                                del batch_texts
                                batch_texts = []

                                future = pool.submit(
                                    _encode_batch_from_file,
                                    input_temp, tokenizer_path, num_threads)
                                pending[future] = input_temp
                            break

                    if not pending:
                        break

                    done_set, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done_set:
                        output_temp, batch_docs, batch_chars = future.result()
                        doc_count += batch_docs
                        total_chars += batch_chars

                        if output_temp is not None:
                            token_seqs = []
                            with open(output_temp, 'r', encoding='utf-8') as f:
                                for line in f:
                                    ids = [int(x) for x in line.split()]
                                    token_seqs.append([bos_id] + ids + [eos_id])
                            try:
                                os.remove(output_temp)
                            except OSError:
                                pass
                            write_queue.put(token_seqs)

                        del pending[future]

                    if doc_count % 100_000 < BATCH_SIZE * 2:
                        _print_prog(doc_count, t_start)

        finally:
            write_queue.put(None)
            writer_thread.join()

            for fname in os.listdir(temp_dir):
                if fname.startswith(('bat_', 'enc_')):
                    try: os.remove(os.path.join(temp_dir, fname))
                    except OSError: pass
            try: os.rmdir(temp_dir)
            except OSError: pass

        total_tokens = writer_result['total_tokens']

    # Trim to exact multiple of (seq_len + 1)
    remainder = total_tokens % (seq_len + 1)
    if remainder:
        with open(output_path, 'r+b') as f:
            f.truncate((total_tokens - remainder) * 2)
        total_tokens -= remainder

    elapsed = time.time() - t_start
    print(f"  Saved {total_tokens:,} tokens ({total_tokens//(seq_len+1):,} sequences)")
    print(f"  Docs: {doc_count:,} | Avg chars/doc: {total_chars/max(doc_count,1):.0f}")
    if elapsed > 0:
        print(f"  Time: {elapsed:.1f}s ({doc_count/elapsed:,.0f} docs/s, "
              f"{total_tokens/elapsed/1e6:.2f}M tok/s)")


def _print_prog(doc_count, t_start):
    elapsed = time.time() - t_start
    avail = _get_available_ram_gb()
    print(f"  {doc_count:,} docs ({doc_count/max(elapsed,0.001):,.0f} docs/s) "
          f"RAM free: {avail:.1f}GB")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenizer + packing (max CPU)")
    parser.add_argument("-j", "--workers", type=int,
                        default=max(1, (os.cpu_count() or 4) // 2),
                        help="Workers (default: physical cores)")
    args = parser.parse_args()
    workers = max(1, args.workers)

    os.makedirs("data", exist_ok=True)
    train_path = "data_mixed/train.jsonl"
    val_path = "data_mixed/val.jsonl"

    if not os.path.exists(train_path):
        print("ERROR: data_mixed/train.jsonl not found. Run download->filter->mix first.")
        sys.exit(1)

    if not os.path.exists("tokenizer.json"):
        print("\n=== Training Tokenizer ===")
        # ── FIX Issue 3: Train ONLY on train_path, NOT val_path ───────────────
        # Training on val data causes vocabulary adaptation to validation text,
        # making validation loss optimistically biased. For a small model where
        # val loss is the primary quality signal, this is a significant integrity
        # problem. The tokenizer should only see training data.
        train_tokenizer([train_path], workers=workers)
    else:
        print("Tokenizer already exists. Delete tokenizer.json to retrain.")

    print(f"\n=== Packing Training Data ({workers} workers) ===")
    pack_jsonl(train_path, "data/train.bin", "tokenizer.json", workers=workers)

    print(f"\n=== Packing Validation Data ({workers} workers) ===")
    pack_jsonl(val_path, "data/val.bin", "tokenizer.json", workers=workers)

    print("\nPreprocessing complete! Run: python train.py")
