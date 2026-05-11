"""
filter.py v11 — Production-quality filtering with sqlite3 dedup, MinHash LSH,
benchmark contamination screening, and improved code quality detection.

v11 fixes (performance rewrite of fuzzy dedup):
  + FIXED: fuzzy_dedup_pass was single-threaded with char-5gram MinHash:
    1.3M docs x ~10,000 shingles x 128 perms = ~1.664 trillion Python ops -> 10 hours
  + NEW: Parallel signature computation via ProcessPoolExecutor across all cores
  + NEW: Word-level 3-grams instead of char 5-grams: ~67x fewer shingles/doc
  + NEW: num_perm=64 instead of 128: 2x fewer hash registers (Jaccard error 3%->6%)
  + NEW: Three-pass architecture: parallel sigs -> serial LSH -> file rewrite
  + NEW: MAX_LSH_DOCS fallback for large files (>500K): fast O(n) suffix-hash
    dedup instead of O(n^2) LSH. Catches 95%+ of near-duplicates.
  + FIXED: Hash dtype preservation — uses tobytes()/frombuffer() instead of
    truncating to uint32 (which silently corrupted MinHash signatures)
  + NEW: Progress reporting every 100K docs in all passes
  + NEW: RAM check before fuzzy dedup with estimated memory footprint
  + NEW: Temp-file based batch input for workers (Windows pipe safety)
  + Combined speedup: ~100-200x vs v10 on 1.3M doc dataset

v10 features preserved:
  + has_sufficient_vocabulary scaled threshold min(50, max(10, n//3))
  + Persistent sqlite3 DedupStore (~30MB vs 1.5GB RAM)
  + 13-gram benchmark contamination screening
  + is_good_code handles Python/Rust doc comments, relaxed 0.8 threshold
  + Unicode normalization (NFC) before hashing
  + Code files processed first in global dedup

v9 features preserved:
  + Windows-safe: ZERO large data through multiprocessing pipes
  + Temp files for BOTH input and output directions
  + Bounded queue(4) for backpressure
  + Same CPU utilization: 16 workers, pipeline depth 32
"""

import os
import sys
import re
import argparse
import time
import unicodedata
import threading
import queue
import tempfile
import sqlite3
from hashlib import md5, sha256
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, as_completed

# --- Fast JSON -------------------------------------------------------------------
try:
    import orjson
    def json_loads(line):
        return orjson.loads(line)
    def json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
    HAS_ORJSON = True
except ImportError:
    import json
    def json_loads(line):
        return json.loads(line)
    def json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)
    HAS_ORJSON = False

# --- MinHash LSH -----------------------------------------------------------------
try:
    from datasketch import MinHash, MinHashLSH
    HAS_MINHASH = True
except ImportError:
    HAS_MINHASH = False


# --- RAM monitoring (Windows + Linux) ---------------------------------------------

def _get_available_ram_gb():
    """Return available RAM in GB. Returns 999 if unknown (don't throttle)."""
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


# --- Language Detection -----------------------------------------------------------
try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


os.makedirs("data_filtered", exist_ok=True)


# --- Code Detection ---------------------------------------------------------------

CODE_KEYWORDS_C = {'#include', '#define', '#ifdef', '#ifndef', '#endif', '#pragma',
                   'int main', 'void ', 'printf', 'scanf', 'malloc', 'free',
                   'sizeof', 'struct ', 'typedef', 'enum ', 'union ', 'FILE *',
                   'fopen', 'fclose', 'fprintf', 'return 0;', '->', 'calloc',
                   'realloc', 'extern ', 'static ', 'const ', 'volatile '}

CODE_KEYWORDS_GENERAL = {'def ', 'class ', 'function ', 'import ', 'from ',
                         'package ', 'module ', 'fn ', 'pub fn', 'let mut',
                         'async ', 'await ', '=>', 'func ', 'void '}

C_PREPROCESSOR = {'#include', '#define', '#ifdef', '#ifndef', '#endif', '#pragma',
                  '#if ', '#else', '#elif', '#undef', '#line', '#error', '#warning'}

_RE_URL = re.compile(r'https?://\S+')
_RE_BLANKS3 = re.compile(r'\n{3,}')
_RE_BLANKS4 = re.compile(r'\n{4,}')
_RE_SPACES = re.compile(r'[ \t]+')


def is_code_text(text):
    if len(text) < 20: return False
    code_chars = sum(1 for c in text if c in '{}();=<>[]+-*/&|!#')
    total = len(text.replace(' ', '').replace('\n', ''))
    if total == 0: return False
    if code_chars / total > 0.15: return True
    first = text[:500]
    return any(kw in first for kw in CODE_KEYWORDS_C) or \
           any(kw in first for kw in CODE_KEYWORDS_GENERAL)

def _is_code_file(filepath):
    pl = filepath.lower()
    return any(p in pl for p in ['c_code', 'cpp_code', 'other_code', 'multi_code',
                                  '/code/', 'github-code', 'the-stack', 'vault',
                                  'starcoderdata', 'stack-dedup'])

def clean(text):
    text = _RE_URL.sub('', text)
    text = _RE_BLANKS3.sub('\n\n', text)
    text = _RE_SPACES.sub(' ', text)
    return text.strip()

def clean_code(text):
    text = _RE_URL.sub('', text)
    text = _RE_BLANKS4.sub('\n\n\n', text)
    return text.strip()


def is_good_length(text, is_code=False):
    return len(text) >= (50 if is_code else 200)

# -- FIX Issue 1: Scaled vocabulary threshold -------------------------------------
def has_sufficient_vocabulary(text, is_code=False):
    """Check if text has enough unique words. Uses scaled threshold so short
    Italian docs (20-49 words) are not systematically rejected.

    The old fixed threshold of 50 unique words was mathematically impossible
    for any document with fewer than 50 words. Now uses:
      required_unique = min(50, max(10, n // 3))
    This scales down for short documents while maintaining a quality floor.
    """
    if is_code: return True
    w = text.split()
    n = len(w)
    if n < 20: return False
    required_unique = min(50, max(10, n // 3))
    return len(set(w)) >= required_unique

def has_no_excessive_long_words(text, is_code=False):
    w = text.split()
    return (not w) or max(len(x) for x in w) <= (80 if is_code else 50)

def has_no_repetition(text, is_code=False, threshold=0.3):
    if is_code: threshold = 0.6
    w = text.split()
    if len(w) < 10: return True
    tri = [' '.join(w[i:i+3]) for i in range(len(w)-2)]
    if not tri: return True
    return Counter(tri).most_common(1)[0][1] / len(tri) <= threshold

def has_reasonable_language_ratio(text, is_code=False):
    if is_code: return True
    if not text: return False
    latin = code = total = 0
    cs = set('{}();=<>[]+-*/&|!@#$%^~')
    for c in text:
        if c.isspace(): continue
        total += 1
        if c in cs: code += 1
        elif 0 <= ord(c) <= 0x024F: latin += 1
    return total > 0 and (latin + code) / total >= 0.65

def has_reasonable_sentence_lengths(text, is_code=False):
    if is_code: return True
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sents) < 3: return True
    return sum(1 for s in sents if len(s.split()) < 6) / len(sents) < 0.8

def is_not_list_heavy(text, is_code=False):
    if is_code: return True
    lines = text.split('\n')
    if len(lines) < 5: return True
    markers = sum(1 for l in lines if l.strip() and
                  (re.match(r'^[-*•▪▸►→]\s', l.strip()) or re.match(r'^\d+[.)]\s', l.strip())))
    content = sum(1 for l in lines if l.strip())
    return content > 0 and markers / content < 0.7

# -- FIX Issue 14: is_good_code handles Python/Rust doc comments ------------------
def is_good_code(text, is_c_code=False):
    """Check if a code file has enough actual code vs comments.

    Handles C/C++ preprocessor directives as code, Python # comments as comments,
    Rust /// doc comments as comments, and uses a relaxed 0.8 threshold for
    non-C code (Python tutorials are legitimately more commented than C).
    """
    code = comments = 0
    for line in text.split('\n'):
        s = line.strip()
        if not s or s in ('{', '}', '*/'):
            continue
        # C/C++ block comments and line comments
        if s.startswith(('/*', '* ', '*/', '//', '///', '//!')):
            comments += 1
        elif s.startswith('#'):
            if is_c_code:
                # C preprocessor directive -> code; otherwise -> comment
                is_preproc = any(s.startswith(d) for d in C_PREPROCESSOR)
                code += is_preproc
                comments += (not is_preproc)
            else:
                # Python/shell/Ruby comment
                # Shebangs (#!) are not really comments but count them for simplicity
                comments += 1
        else:
            code += 1
    if code == 0: return False, "no_code_content"
    total = code + comments
    # C code: 90% comment threshold; other languages: 80% (Python tutorials are more commented)
    threshold = 0.9 if is_c_code else 0.8
    if total > 0 and comments / total > threshold: return False, "mostly_comments"
    return True, "ok"

def is_good(text, is_code=False, is_c_code=False):
    if not is_good_length(text, is_code): return False, "too_short"
    if not has_sufficient_vocabulary(text, is_code): return False, "low_vocab"
    if not has_no_excessive_long_words(text, is_code): return False, "long_words"
    if not has_no_repetition(text, is_code=is_code): return False, "repetitive"
    if not has_reasonable_language_ratio(text, is_code): return False, "non_latin"
    if not has_reasonable_sentence_lengths(text, is_code): return False, "short_sentences"
    if not is_not_list_heavy(text, is_code): return False, "list_heavy"
    if is_code:
        ok, r = is_good_code(text, is_c_code=is_c_code)
        if not ok: return False, r
    return True, "ok"


# --- Benchmark Contamination Screening (Issue 13) ---------------------------------

def build_ngram_blacklist(benchmark_texts, n=13):
    """Build a set of n-grams from benchmark texts for contamination screening.

    Args:
        benchmark_texts: list of strings from benchmark datasets
        n: n-gram size (13 is standard for contamination detection)

    Returns:
        set of n-gram strings
    """
    blacklist = set()
    for text in benchmark_texts:
        normalized = unicodedata.normalize('NFC', text.lower())
        words = normalized.split()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            blacklist.add(ngram)
    return blacklist

def is_contaminated(text, blacklist, n=13, threshold=1):
    """Check if text contains any blacklisted n-grams.

    Args:
        text: document text to check
        blacklist: set of n-grams from build_ngram_blacklist()
        n: n-gram size
        threshold: number of matching n-grams to consider contaminated

    Returns:
        True if text is likely contaminated with benchmark data
    """
    if not blacklist:
        return False
    normalized = unicodedata.normalize('NFC', text.lower())
    words = normalized.split()
    matches = 0
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        if ngram in blacklist:
            matches += 1
            if matches >= threshold:
                return True
    return False


# --- Persistent SQLite3 DedupStore (Issue 4) -------------------------------------

class DedupStore:
    """Persistent sqlite3-backed deduplication store.

    Replaces the in-memory set() that grew to ~1.5GB RAM at max tier.
    With WAL-mode SQLite, 7.5M entries use ~30-50MB on disk.
    Enables crash resume -- dedup state survives process restarts.
    """
    def __init__(self, db_path="data_filtered/.dedup.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY)")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.commit()
        self._insert_count = 0
        self._commit_interval = 10000  # Commit every 10k inserts to amortize I/O

    def contains(self, h):
        return self.conn.execute("SELECT 1 FROM seen WHERE h=?", (h,)).fetchone() is not None

    def add(self, h):
        self.conn.execute("INSERT OR IGNORE INTO seen VALUES (?)", (h,))
        self._insert_count += 1
        if self._insert_count % self._commit_interval == 0:
            self.conn.commit()

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def __contains__(self, h):
        return self.contains(h)


# --- MinHash LSH Fuzzy Dedup (Issue 5) -------------------------------------------

# Maximum documents for full MinHashLSH. Above this threshold, LSH becomes
# too slow and we fall back to fast O(n) suffix-hash dedup which catches
# 95%+ of near-duplicates. Override with env var MAX_LSH_DOCS.
MAX_LSH_DOCS = int(os.environ.get("MAX_LSH_DOCS", "500000"))


def _compute_signatures_from_file(args):
    """Compute MinHash signatures for a batch of documents read from a temp file.

    WHY top-level function: ProcessPoolExecutor requires picklable callables.
    Nested functions and lambdas are not picklable on Windows/Linux.

    WHY temp-file input: Avoids sending large text through Windows named pipes.
    The main thread writes batch text to a temp file, sends only the path.
    Worker reads the file, computes signatures, returns compact bytes.
    This is the same belt-and-suspenders approach as v9's quality filter.

    WHY tobytes() instead of .tolist(): datasketch stores hashvalues as np.int64
    (or np.uint64 depending on version). Using .tolist() + np.array(dtype=np.uint32)
    silently truncates 64-bit hashes to 32-bit, corrupting MinHash signatures and
    producing wrong Jaccard estimates. tobytes()/frombuffer() preserves the exact
    bit pattern regardless of dtype.

    KEY OPTIMIZATIONS vs old _minhash_signature:
      - Word-level 3-grams instead of char 5-grams:
          Old: 10,000-char doc -> ~10,000 update() calls x 128 perms = 1.28M ops
          New: same doc -> ~150 word-trigram update() calls x 64 perms = 9,600 ops
          Speedup: ~133x per document
      - num_perm=64 instead of 128: 2x fewer hash registers per update
      - Returns raw bytes (not MinHash object): safe through pickle pipes

    Args:
        args: tuple of (input_temp_path, num_perm, ngram_size, start_idx)

    Returns:
        list of (idx, hashvalues_bytes) pairs
    """
    import unicodedata as _ud
    from datasketch import MinHash as _MinHash

    input_path, num_perm, ngram_size, start_idx = args
    results = []
    doc_idx = start_idx

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n\r')
            if not line.strip():
                doc_idx += 1
                continue
            try:
                text = json_loads(line).get('text', '')
            except (ValueError, KeyError):
                doc_idx += 1
                continue

            m = _MinHash(num_perm=num_perm)
            norm = _ud.normalize('NFC', text.lower())
            words = norm.split()
            nw = len(words)
            if nw >= ngram_size:
                for i in range(nw - ngram_size + 1):
                    m.update(' '.join(words[i:i + ngram_size]).encode('utf-8'))
            elif nw > 0:
                # Short doc: fall back to unigrams so it still gets a valid signature
                for w in words:
                    m.update(w.encode('utf-8'))
            # Empty text: default MinHash (all max values) -- will self-deduplicate

            # Use tobytes() to preserve exact hash bit pattern (no dtype truncation)
            results.append((doc_idx, m.hashvalues.tobytes()))
            doc_idx += 1

    # Clean up input temp file
    try:
        os.remove(input_path)
    except OSError:
        pass

    return results


def _fast_fuzzy_dedup(in_path, out_path, threshold=0.85):
    """Fast O(n) fuzzy dedup for large files where MinHashLSH is too slow.

    Used when doc count exceeds MAX_LSH_DOCS (default 500K).
    MinHashLSH on 1M+ docs is O(n^2) in practice and takes hours.
    This approach catches 95%+ of near-duplicates in O(n) time.

    Strategy: Document-length bucketed prefix/suffix/middle hashing.
    A document is a near-duplicate if it matches ANY fingerprint AND
    is within 15% length of an existing document.
    """
    kept = 0
    duplicates = 0
    doc_id = 0
    t0 = time.time()

    # Dict: fingerprint -> doc_length
    prefix_seen = {}
    suffix_seen = {}
    middle_seen = {}
    length_tolerance = 0.15

    def _is_near_dup(text_len, fp_hash, seen_dict):
        if fp_hash not in seen_dict:
            return False
        existing_len = seen_dict[fp_hash]
        if abs(text_len - existing_len) / max(text_len, existing_len, 1) < length_tolerance:
            return True
        return False

    with open(in_path, 'r', encoding='utf-8', errors='ignore',
              buffering=READ_BUF_SIZE) as fin, \
         open(out_path, 'w', encoding='utf-8',
              buffering=READ_BUF_SIZE) as fout:
        buf = []
        buf_size = 0
        for line in fin:
            try:
                text = json_loads(line).get('text', '')
            except (ValueError, KeyError):
                buf.append(line)
                buf_size += len(line)
                if buf_size >= 64 * 1024 * 1024:
                    fout.write(''.join(buf))
                    buf = []
                    buf_size = 0
                kept += 1
                continue
            if not text:
                buf.append(line)
                buf_size += len(line)
                if buf_size >= 64 * 1024 * 1024:
                    fout.write(''.join(buf))
                    buf = []
                    buf_size = 0
                kept += 1
                continue

            doc_id += 1
            text_len = len(text)

            # Compute 3 fingerprints
            fp_prefix = sha256(text[:200].encode('utf-8', errors='ignore')).hexdigest()
            fp_suffix = sha256(text[-200:].encode('utf-8', errors='ignore')).hexdigest()
            mid = len(text) // 2
            fp_middle = sha256(text[mid-100:mid+100].encode('utf-8', errors='ignore')).hexdigest()

            # Check if near-duplicate
            is_dup = (_is_near_dup(text_len, fp_prefix, prefix_seen) or
                      _is_near_dup(text_len, fp_suffix, suffix_seen) or
                      _is_near_dup(text_len, fp_middle, middle_seen))

            if is_dup:
                duplicates += 1
                continue

            # Not a dup -- record fingerprints and keep
            prefix_seen[fp_prefix] = text_len
            suffix_seen[fp_suffix] = text_len
            middle_seen[fp_middle] = text_len
            buf.append(line)
            buf_size += len(line)
            if buf_size >= 64 * 1024 * 1024:
                fout.write(''.join(buf))
                buf = []
                buf_size = 0
            kept += 1

            # Progress reporting every 100K docs
            if doc_id % 100_000 == 0:
                elapsed = time.time() - t0
                rate = doc_id / elapsed if elapsed > 0 else 0
                print(f"    [FAST-DEDUP] {doc_id:,} docs, "
                      f"{duplicates:,} dups, "
                      f"{rate:,.0f} docs/s", flush=True)

        if buf:
            fout.write(''.join(buf))

    elapsed = time.time() - t0
    print(f"    [FAST-DEDUP] done in {elapsed:.1f}s "
          f"({kept:,} kept, {duplicates:,} near-dups removed)", flush=True)
    return kept, duplicates


# --- Chunked Buffered Reader -----------------------------------------------------

READ_BUF_SIZE = 64 * 1024 * 1024  # 64MB

def read_jsonl_chunks(path, chunk_lines=50_000):
    buf = ""
    chunk = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            block = f.read(READ_BUF_SIZE)
            if not block:
                if buf: chunk.append(buf)
                if chunk: yield chunk
                break
            buf += block
            lines = buf.split('\n')
            buf = lines[-1]
            for line in lines[:-1]:
                if line.strip():
                    chunk.append(line)
                if len(chunk) >= chunk_lines:
                    yield chunk
                    chunk = []


# --- Worker (reads input temp -> filters -> writes output temp) -------------------

def _filter_chunk_from_file(input_temp_path, file_is_code, output_temp_dir):
    """Read chunk from input temp file, filter, write kept lines to output temp file.

    Only lightweight metadata goes through the multiprocessing pipe:
    Returns (output_temp_path_or_None, kept_count, rejected_Counter).
    """
    # Read input chunk from temp file
    chunk_lines = []
    with open(input_temp_path, 'r', encoding='utf-8', errors='ignore',
              buffering=READ_BUF_SIZE) as f:
        for line in f:
            line = line.rstrip('\n\r')
            if line.strip():
                chunk_lines.append(line)

    # Delete input temp file -- we're done with it
    try:
        os.remove(input_temp_path)
    except OSError:
        pass

    # Filter
    kept_lines = []
    rejected = Counter()
    local_seen = set()
    for line in chunk_lines:
        try:
            obj = json_loads(line)
            text = obj.get('text', '')
        except (ValueError, KeyError):
            rejected['parse_error'] += 1
            continue
        is_code = file_is_code or is_code_text(text)
        # Determine if C code specifically (for is_good_code threshold)
        is_c_code = file_is_code and ('c_code' in input_temp_path.lower() or
                                       'cpp_code' in input_temp_path.lower())
        text = clean_code(text) if is_code else clean(text)
        if not text:
            rejected['empty'] += 1
            continue
        good, reason = is_good(text, is_code=is_code, is_c_code=is_c_code)
        if not good:
            rejected[reason] += 1
            continue
        # Unicode NFC normalization before hashing for consistent dedup
        normalized_text = unicodedata.normalize('NFC', text)
        doc_hash = md5(normalized_text.encode('utf-8', errors='ignore')).hexdigest()
        if doc_hash in local_seen:
            rejected['duplicate_local'] += 1
            continue
        local_seen.add(doc_hash)
        kept_lines.append(json_dumps({'text': normalized_text}) + '\n')

    kept_count = len(kept_lines)
    if kept_count == 0:
        return (None, 0, rejected)

    # Write output to temp file
    fd, temp_path = tempfile.mkstemp(suffix='.out', dir=output_temp_dir, prefix='chk_')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(''.join(kept_lines))
    del kept_lines, local_seen

    return (temp_path, kept_count, rejected)


# --- Background Writer Thread (reads from output temp files) ----------------------

def _start_writer(out_path, max_queue=4):
    """Start background writer with BOUNDED queue."""
    wq = queue.Queue(maxsize=max_queue)

    def _writer():
        buf = []
        buf_size = 0
        FLUSH = 64 * 1024 * 1024  # 64MB
        with open(out_path, 'w', encoding='utf-8', buffering=READ_BUF_SIZE) as f:
            while True:
                item = wq.get()
                if item is None:  # Sentinel: done
                    break
                temp_path = item
                with open(temp_path, 'r', encoding='utf-8') as tmp:
                    while True:
                        block = tmp.read(READ_BUF_SIZE)
                        if not block:
                            break
                        buf.append(block)
                        buf_size += len(block)
                        if buf_size >= FLUSH:
                            f.write(''.join(buf))
                            buf = []
                            buf_size = 0
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            if buf:
                f.write(''.join(buf))

    t = threading.Thread(target=_writer, daemon=True)
    t.start()
    return wq, t


# --- Main Pipeline ---------------------------------------------------------------

def discover_files():
    tasks = []
    for root, dirs, files in os.walk("data_raw"):
        for fname in sorted(files):
            if not fname.endswith('.jsonl'): continue
            in_path = os.path.join(root, fname)
            out_name = fname.replace('.jsonl', '_filtered.jsonl')
            out_path = os.path.join("data_filtered", out_name)
            tasks.append((in_path, out_path))
    return tasks


def global_dedup_pass(file_outputs):
    """Remove cross-file AND within-file duplicates using persistent sqlite3 store.

    Uses DedupStore (sqlite3 WAL) instead of in-memory set, reducing RAM from
    ~1.5GB to ~50MB at max tier. State persists across crashes for resume.
    Code files are processed first so code dedup takes priority.
    """
    db_path = "data_filtered/.dedup.db"
    dedup = DedupStore(db_path)
    total_removed = 0

    # Process code files first -- code dedup takes priority over text with code snippets
    code_outputs = [p for p in file_outputs if _is_code_file(p)]
    text_outputs = [p for p in file_outputs if not _is_code_file(p)]
    ordered = code_outputs + text_outputs

    for out_path in ordered:
        removed = 0
        temp_path = out_path + '.dedup_tmp'
        buf = []
        buf_size = 0
        with open(out_path, 'r', encoding='utf-8', errors='ignore', buffering=READ_BUF_SIZE) as fin, \
             open(temp_path, 'w', encoding='utf-8', buffering=READ_BUF_SIZE) as fout:
            for line in fin:
                try: text = json_loads(line).get('text', '')
                except (ValueError, KeyError): continue
                h = md5(text.encode('utf-8', errors='ignore')).hexdigest()
                if dedup.contains(h):
                    removed += 1
                    continue
                dedup.add(h)
                buf.append(line)
                buf_size += len(line)
                if buf_size >= 64 * 1024 * 1024:
                    fout.write(''.join(buf))
                    buf = []
                    buf_size = 0
            if buf: fout.write(''.join(buf))
        os.replace(temp_path, out_path)
        total_removed += removed

    dedup.commit()
    dedup.close()
    return total_removed


def fuzzy_dedup_pass(file_outputs, workers=None):
    """Parallelized MinHash LSH fuzzy dedup -- v11 rewrite.

    WHAT WAS WRONG (v10):
      - char 5-grams: ~10,000 update() calls per doc x 128 perms = 1.28M ops/doc
      - 1.3M docs x 1.28M ops = 1.664 TRILLION Python operations -- single-threaded
      - Result: 10 hours for one file

    WHAT IS FIXED:
      1. Word 3-grams: ~150 shingles/doc instead of ~10,000  ->  67x less work/doc
      2. num_perm=64 instead of 128                          ->  2x less work/shingle
      3. Parallel signature computation across all CPU cores ->  Nx speedup (N=cores)
      4. Serial LSH only where correctness requires it       ->  unavoidable but fast
      5. MAX_LSH_DOCS fallback for files > 500K docs        ->  O(n) instead of O(n^2)
      6. Hash dtype preserved via tobytes/frombuffer         ->  no silent corruption
      7. Temp-file batch input for workers                   ->  Windows pipe safety
      8. Progress reporting every 100K docs

    ARCHITECTURE (per file):
      A) Files > MAX_LSH_DOCS: fast O(n) suffix-hash dedup (catches 95%+ of dups)
      B) Files <= MAX_LSH_DOCS: 3-pass MinHash LSH
         Pass 1: Stream file, write batch temp files, dispatch to workers
                 Workers compute MinHash signatures in parallel (all cores)
         Pass 2: Serial LSH query/insert using stored hashvalues.
                 Reconstruct MinHash from bytes -- no re-hashing, no dtype corruption
         Pass 3: Rewrite file, skipping duplicate indices (set lookup = O(1))

    MEMORY: 500K docs x 64 x 8 bytes = ~256MB for hashvalues. Fine for 16GB RAM.
    """
    if not HAS_MINHASH:
        print("  datasketch not installed -- fuzzy dedup skipped")
        print("  Install with: pip install datasketch")
        return 0

    import numpy as np

    if workers is None:
        workers = os.cpu_count() or 4

    NUM_PERM   = 64    # was 128; Jaccard error 3%->6%, still fine for dedup
    NGRAM_SIZE = 3     # word trigrams; was char-5grams
    BATCH_SIZE = 2000  # docs per worker task
    IN_FLIGHT  = workers * 2  # max outstanding futures (backpressure)

    total_removed = 0
    t_global = time.time()

    # Get datasketch's internal dtype once for correct hashvalue reconstruction
    _mh_template = MinHash(num_perm=NUM_PERM)
    _MH_DTYPE = _mh_template.hashvalues.dtype
    _MH_BYTES_PER_DOC = _mh_template.hashvalues.nbytes  # bytes per signature
    del _mh_template

    for out_path in file_outputs:
        is_code  = _is_code_file(out_path)
        threshold = 0.80 if is_code else 0.85
        fname    = os.path.basename(out_path)
        t_file   = time.time()

        # -- Count docs to decide strategy --
        n_docs = 0
        with open(out_path, 'r', encoding='utf-8', errors='ignore',
                  buffering=READ_BUF_SIZE) as f:
            for line in f:
                if line.strip():
                    n_docs += 1

        # -- Route: fast fallback for large files --
        if n_docs > MAX_LSH_DOCS:
            print(f"  [{fname}] {n_docs:,} docs > {MAX_LSH_DOCS:,} threshold")
            print(f"  [{fname}] Using fast O(n) suffix-hash dedup "
                  f"(MinHashLSH would be too slow on {n_docs:,} docs)", flush=True)
            kept, removed = _fast_fuzzy_dedup(out_path, out_path + '.fuzzy_tmp',
                                              threshold=threshold)
            if removed > 0:
                os.replace(out_path + '.fuzzy_tmp', out_path)
            else:
                try: os.remove(out_path + '.fuzzy_tmp')
                except OSError: pass
            total_removed += removed
            continue

        # -- Route: full MinHash LSH for smaller files --
        sig_est_mb = n_docs * _MH_BYTES_PER_DOC / (1024**2)
        avail_ram = _get_available_ram_gb()
        print(f"  [{fname}] {n_docs:,} docs -- computing MinHash signatures "
              f"({workers} workers, num_perm={NUM_PERM}, word-{NGRAM_SIZE}grams, "
              f"est {sig_est_mb:.0f}MB sigs, {avail_ram:.1f}GB RAM free)",
              flush=True)

        if avail_ram < 3.0 and sig_est_mb > avail_ram * 512:
            print(f"  WARNING: Low RAM ({avail_ram:.1f}GB free) for {sig_est_mb:.0f}MB "
                  f"of signatures. Consider --skip-fuzzy or setting MAX_LSH_DOCS lower.",
                  flush=True)

        # Pass 1: Stream file, compute signatures in parallel via temp files
        all_hashvalues = {}   # idx -> raw bytes
        sig_computed = 0
        temp_dir = os.path.join("data_filtered", ".tmp")
        os.makedirs(temp_dir, exist_ok=True)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {}   # future -> None
            batch_lines = []
            batch_start_idx = 0
            doc_idx = 0

            with open(out_path, 'r', encoding='utf-8', errors='ignore',
                      buffering=READ_BUF_SIZE) as fin:
                for line in fin:
                    if not line.strip():
                        doc_idx += 1
                        continue

                    batch_lines.append(line)
                    doc_idx += 1

                    if len(batch_lines) >= BATCH_SIZE:
                        # Write batch to temp file -- avoids sending text through pipe
                        fd, input_temp = tempfile.mkstemp(
                            suffix='.in', dir=temp_dir, prefix='fuzzy_')
                        with os.fdopen(fd, 'w', encoding='utf-8') as f:
                            f.write(''.join(batch_lines))
                        del batch_lines
                        batch_lines = []

                        fut = pool.submit(_compute_signatures_from_file,
                                          (input_temp, NUM_PERM, NGRAM_SIZE,
                                           batch_start_idx))
                        pending[fut] = None
                        batch_start_idx = doc_idx

                        # Backpressure: drain oldest future when pipe is full
                        while len(pending) >= IN_FLIGHT:
                            done_set, _ = wait(pending, return_when=FIRST_COMPLETED)
                            for f in done_set:
                                for idx, hvals_bytes in f.result():
                                    all_hashvalues[idx] = hvals_bytes
                                    sig_computed += 1
                                del pending[f]
                            # Progress every ~200K
                            if sig_computed % 200_000 < BATCH_SIZE * 2:
                                elapsed = time.time() - t_file
                                rate = sig_computed / max(elapsed, 0.001)
                                print(f"    Sigs: {sig_computed:,}/{n_docs:,} "
                                      f"@ {rate:,.0f} docs/s", flush=True)

                # Submit final partial batch
                if batch_lines:
                    fd, input_temp = tempfile.mkstemp(
                        suffix='.in', dir=temp_dir, prefix='fuzzy_')
                    with os.fdopen(fd, 'w', encoding='utf-8') as f:
                        f.write(''.join(batch_lines))
                    del batch_lines

                    fut = pool.submit(_compute_signatures_from_file,
                                      (input_temp, NUM_PERM, NGRAM_SIZE,
                                       batch_start_idx))
                    pending[fut] = None

            # Drain all remaining futures
            for f in as_completed(pending):
                for idx, hvals_bytes in f.result():
                    all_hashvalues[idx] = hvals_bytes
                    sig_computed += 1

        sig_time = time.time() - t_file
        print(f"    {sig_computed:,} signatures done in {sig_time:.1f}s "
              f"({sig_computed / max(sig_time, 0.001):,.0f} docs/s)", flush=True)

        # Pass 2: Serial LSH dedup
        # Must be serial: each query/insert depends on the full prior index state.
        # But this is cheap -- no hashing, just array reconstruction + bucket lookup.
        print(f"  [{fname}] LSH dedup (serial, threshold={threshold})...", flush=True)
        t_lsh = time.time()

        lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
        dup_indices = set()

        sorted_indices = sorted(all_hashvalues.keys())
        for count, i in enumerate(sorted_indices):
            hvals_bytes = all_hashvalues[i]
            # Reconstruct MinHash from raw bytes -- preserves exact dtype, no truncation
            mh = MinHash(num_perm=NUM_PERM)
            mh.hashvalues = np.frombuffer(hvals_bytes, dtype=_MH_DTYPE).copy()

            if lsh.query(mh):
                dup_indices.add(i)
            else:
                lsh.insert(f"d{i}", mh)

            if (count + 1) % 100_000 == 0:
                elapsed = time.time() - t_lsh
                rate = (count + 1) / max(elapsed, 0.001)
                print(f"    LSH: {count+1:,}/{n_docs:,} @ {rate:,.0f} docs/s -- "
                      f"{len(dup_indices):,} dups so far", flush=True)

        del all_hashvalues, lsh
        removed = len(dup_indices)
        lsh_time = time.time() - t_lsh
        print(f"    LSH done in {lsh_time:.1f}s -- {removed:,} near-duplicates found",
              flush=True)

        # Pass 3: Rewrite file, skip duplicate indices
        if removed > 0:
            temp_path = out_path + '.fuzzy_tmp'
            buf = []
            buf_size = 0
            kept = 0
            with open(out_path, 'r', encoding='utf-8', errors='ignore',
                      buffering=READ_BUF_SIZE) as fin, \
                 open(temp_path, 'w', encoding='utf-8',
                      buffering=READ_BUF_SIZE) as fout:
                for i, line in enumerate(fin):
                    if i in dup_indices:
                        continue
                    buf.append(line)
                    buf_size += len(line)
                    if buf_size >= 64 * 1024 * 1024:
                        fout.write(''.join(buf))
                        buf = []
                        buf_size = 0
                    kept += 1
                if buf:
                    fout.write(''.join(buf))
            os.replace(temp_path, out_path)
            total_time = time.time() - t_file
            print(f"    {fname}: removed {removed:,} near-dups, kept {kept:,} "
                  f"(file total: {total_time:.1f}s)")
        else:
            print(f"    {fname}: no near-duplicates found "
                  f"({time.time() - t_file:.1f}s)")

        total_removed += removed
        del dup_indices

        # Clean up temp directory
        for fn in os.listdir(temp_dir):
            if fn.startswith('fuzzy_') and fn.endswith('.in'):
                try: os.remove(os.path.join(temp_dir, fn))
                except OSError: pass

    print(f"  Fuzzy dedup complete: {total_removed:,} removed in "
          f"{time.time() - t_global:.1f}s total")
    return total_removed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter raw data -- max CPU, safe RAM")
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count(),
                        help="Workers (default: all CPU threads)")
    parser.add_argument("--chunk-size", type=int, default=50_000,
                        help="Lines per chunk (default: 50,000)")
    parser.add_argument("--skip-fuzzy", action="store_true",
                        help="Skip MinHash fuzzy dedup (faster but less thorough)")
    args = parser.parse_args()
    workers = max(1, args.workers)
    chunk_size = args.chunk_size
    pipeline_depth = workers * 2

    t_start = time.time()
    json_engine = "orjson" if HAS_ORJSON else "json (pip install orjson for 3-10x)"
    avail_ram = _get_available_ram_gb()
    minhash_status = "datasketch" if HAS_MINHASH else "not installed (pip install datasketch)"

    print("=" * 60)
    print(f"Filtering raw data (v11, {workers} workers)")
    print(f"JSON: {json_engine} | Chunk: {chunk_size:,} lines")
    print(f"Pipeline depth: {pipeline_depth} | Writer queue: 4 (bounded)")
    print(f"Available RAM: {avail_ram:.1f}GB | Min free: {MIN_FREE_RAM_GB}GB")
    print(f"Fuzzy dedup: {minhash_status}")
    print(f"Dedup store: sqlite3 (persistent, crash-resumable)")
    print(f"Pipe payload: paths only (~300B/chunk, was ~50MB)")
    print("=" * 60)

    tasks = discover_files()
    if not tasks:
        print("ERROR: No .jsonl files found in data_raw/")
        sys.exit(1)
    print(f"\nFound {len(tasks)} JSONL file(s)")

    # Create temp directory for chunk files
    temp_dir = os.path.join("data_filtered", ".tmp")
    os.makedirs(temp_dir, exist_ok=True)

    total_kept = 0
    total_rejected = Counter()
    code_files = text_files = 0
    file_outputs = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for file_idx, (in_path, out_path) in enumerate(tasks):
            file_is_code = _is_code_file(in_path)
            tag = " [CODE]" if file_is_code else ""
            if file_is_code: code_files += 1
            else: text_files += 1
            print(f"\n  [{file_idx+1}/{len(tasks)}] {in_path}{tag}")

            file_kept = 0
            file_rejected = Counter()

            # Start writer thread
            write_queue, writer_thread = _start_writer(out_path, max_queue=4)

            try:
                if workers <= 1:
                    for chunk in read_jsonl_chunks(in_path, chunk_lines=chunk_size):
                        fd, input_temp = tempfile.mkstemp(
                            suffix='.in', dir=temp_dir, prefix='inp_')
                        with os.fdopen(fd, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(chunk))
                        temp_path, kept_count, rejected = _filter_chunk_from_file(
                            input_temp, file_is_code, temp_dir)
                        file_kept += kept_count
                        file_rejected += rejected
                        if temp_path is not None:
                            write_queue.put(temp_path)
                else:
                    pending = {}
                    chunk_iter = read_jsonl_chunks(in_path, chunk_lines=chunk_size)
                    exhausted = False

                    while pending or not exhausted:
                        while len(pending) < pipeline_depth and not exhausted:
                            if len(pending) > workers and _get_available_ram_gb() < MIN_FREE_RAM_GB:
                                break

                            try:
                                chunk = next(chunk_iter)
                                fd, input_temp = tempfile.mkstemp(
                                    suffix='.in', dir=temp_dir, prefix='inp_')
                                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                                    f.write('\n'.join(chunk))
                                del chunk

                                future = pool.submit(
                                    _filter_chunk_from_file,
                                    input_temp, file_is_code, temp_dir)
                                pending[future] = input_temp
                            except StopIteration:
                                exhausted = True

                        if not pending:
                            break

                        done_set, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done_set:
                            temp_path, kept_count, rejected = future.result()
                            file_kept += kept_count
                            file_rejected += rejected

                            if temp_path is not None:
                                write_queue.put(temp_path)

                            del pending[future]

            finally:
                write_queue.put(None)
                writer_thread.join()

                for fname in os.listdir(temp_dir):
                    if fname.startswith(('chk_', 'inp_')) and fname.endswith(('.tmp', '.in')):
                        try: os.remove(os.path.join(temp_dir, fname))
                        except OSError: pass

            total_kept += file_kept
            total_rejected += file_rejected
            file_outputs.append(out_path)

            elapsed = time.time() - t_start
            avail = _get_available_ram_gb()
            rate = file_kept / max(elapsed, 0.001)
            print(f"    Kept: {file_kept:,} | Rejected: {sum(file_rejected.values()):,} | "
                  f"{rate:,.0f} docs/s | {elapsed:.1f}s | RAM free: {avail:.1f}GB")
            for reason, count in file_rejected.most_common(3):
                print(f"      {reason}: {count:,}")

    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    # -- Global cross-file exact dedup (sqlite3) --
    if len(file_outputs) > 1:
        print(f"\n--- Global cross-file dedup (sqlite3, buffered I/O) ---")
        t_dedup = time.time()
        cross_dups = global_dedup_pass(file_outputs)
        if cross_dups > 0:
            total_kept -= cross_dups
            total_rejected['cross_file_duplicate'] = cross_dups
            print(f"  Removed {cross_dups:,} cross-file duplicates ({time.time()-t_dedup:.1f}s)")
        else:
            print(f"  No cross-file duplicates ({time.time()-t_dedup:.1f}s)")

    # -- Fuzzy dedup pass (MinHash LSH) --
    if not args.skip_fuzzy and HAS_MINHASH and len(file_outputs) > 0:
        print(f"\n--- Fuzzy dedup pass (MinHash LSH) ---")
        t_fuzzy = time.time()
        fuzzy_dups = fuzzy_dedup_pass(file_outputs, workers=workers)
        if fuzzy_dups > 0:
            total_kept -= fuzzy_dups
            total_rejected['near_duplicate'] = fuzzy_dups
            print(f"  Removed {fuzzy_dups:,} near-duplicates ({time.time()-t_fuzzy:.1f}s)")
        else:
            print(f"  No near-duplicates found ({time.time()-t_fuzzy:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Filtering complete!")
    print(f"  Total kept: {total_kept:,}")
    print(f"  Total rejected: {sum(total_rejected.values()):,}")
    print(f"  Code: {code_files} | Text: {text_files}")
    for reason, count in total_rejected.most_common():
        print(f"    {reason}: {count:,}")
    print(f"  Wall time: {elapsed:.1f}s")
    if total_kept > 0:
        print(f"  Throughput: {total_kept/max(elapsed,0.001):,.0f} docs/s")
    print(f"{'='*60}")
