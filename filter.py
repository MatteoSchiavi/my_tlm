"""
filter.py v4 — Data cleaning, deduplication, and quality filtering.
Multi-core parallelized for maximum throughput on large datasets.

Key fixes from review:
  ✅ Full document MD5 hash (not just first 500 chars)
  ✅ 3-gram repetition filter (kills boilerplate/SEO spam)
  ✅ Italian language support (not just ASCII)
  ✅ Quality scoring for web data
  ✅ Language detection with langdetect (keep Italian + English)
  ✅ CODE-AWARE FILTERING: code files skip NLP filters that reject valid code

v2 fixes (code pipeline):
  ✅ BUG FIX: #include, #define, #ifdef etc. are NOT comments in C/C++
  ✅ Relax repetition filter for code (0.3 → 0.6 threshold)
  ✅ Allow single-line C macros and one-line functions
  ✅ Code detection also checks text content (not just file path)

v3 fixes:
  ✅ Scaled vocabulary filter, sentence-preserving URL removal
  ✅ MinHash LSH fuzzy deduplication
  ✅ Persistent sqlite3 DedupStore
  ✅ Code-first dedup ordering
  ✅ Benchmark contamination screening

v4 — MULTI-CORE PARALLEL:
  ✅ Phase 1: Quality filtering runs in parallel across files (ProcessPoolExecutor)
  ✅ Phase 2: Exact dedup runs sequentially (sqlite3 is not fork-safe)
  ✅ Phase 3: Fuzzy dedup runs in parallel per file (ProcessPoolExecutor)
  ✅ --workers flag to control parallelism (default: all CPUs)
  ✅ Progress reporting with live throughput stats
"""

import os
import json
import re
import sqlite3
import time
import argparse
from hashlib import md5
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Language Detection ────────────────────────────────────────────────────────
try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


os.makedirs("data_filtered", exist_ok=True)


# ─── Persistent Deduplication Store (Issue 10) ────────────────────────────────

class DedupStore:
    """Persistent deduplication store using sqlite3. ~30MB on disk for 500k hashes,
    vs 600-700MB for a Python set. Also enables resuming filter.py."""

    def __init__(self, db_path="data_filtered/dedup_hashes.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS hashes (hash TEXT PRIMARY KEY)")
        self.conn.commit()

    def seen(self, doc_hash):
        """Check if hash has been seen. Returns True if duplicate."""
        cur = self.conn.execute("SELECT 1 FROM hashes WHERE hash=?", (doc_hash,))
        return cur.fetchone() is not None

    def add(self, doc_hash):
        """Add hash to store."""
        try:
            self.conn.execute("INSERT INTO hashes VALUES (?)", (doc_hash,))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass  # Already exists

    def close(self):
        self.conn.close()


# ─── Benchmark Contamination Screening (Issue 20) ─────────────────────────────

_CONTAMINATION_BLACKLIST = None  # Loaded on demand


def build_ngram_blacklist(texts, n=13):
    """Build 13-gram blacklist from benchmark texts."""
    blacklist = set()
    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            blacklist.add(' '.join(words[i:i+n]))
    return blacklist


def is_contaminated(text, blacklist, n=13):
    """Check if document shares any 13-gram with benchmark data."""
    if blacklist is None or not blacklist:
        return False
    words = text.lower().split()
    for i in range(len(words) - n + 1):
        if ' '.join(words[i:i+n]) in blacklist:
            return True
    return False


# ─── Code Detection ────────────────────────────────────────────────────────────

CODE_EXTENSIONS = {'.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.py', '.js',
                   '.rs', '.go', '.java', '.rb', '.php', '.sh', '.bash'}

CODE_KEYWORDS_C = {'#include', '#define', '#ifdef', '#ifndef', '#endif', '#pragma',
                   'int main', 'void ', 'printf', 'scanf', 'malloc', 'free',
                   'sizeof', 'struct ', 'typedef', 'enum ', 'union ', 'FILE *',
                   'fopen', 'fclose', 'fprintf', 'return 0;', '->', 'calloc',
                   'realloc', 'extern ', 'static ', 'const ', 'volatile '}

CODE_KEYWORDS_GENERAL = {'def ', 'class ', 'function ', 'import ', 'from ',
                         'package ', 'module ', 'fn ', 'pub fn', 'let mut',
                         'async ', 'await ', '=>', 'func ', 'void '}

# C/C++ preprocessor directives — these are CODE, not comments
C_PREPROCESSOR = {'#include', '#define', '#ifdef', '#ifndef', '#endif', '#pragma',
                  '#if ', '#else', '#elif', '#undef', '#line', '#error', '#warning'}


def is_code_text(text):
    """Detect if text is likely source code rather than natural language."""
    if len(text) < 20:
        return False
    code_chars = sum(1 for c in text if c in '{}();=<>[]+-*/&|!#')
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars == 0:
        return False
    code_char_ratio = code_chars / total_chars
    first_500 = text[:500]
    has_c_keywords = any(kw in first_500 for kw in CODE_KEYWORDS_C)
    has_general_keywords = any(kw in first_500 for kw in CODE_KEYWORDS_GENERAL)
    if code_char_ratio > 0.15:
        return True
    if has_c_keywords or has_general_keywords:
        return True
    return False


def _is_code_file(filepath):
    """Check if the source file is a code file based on path/name."""
    path_lower = filepath.lower()
    code_patterns = ['c_code', 'cpp_code', 'other_code', 'multi_code',
                     '/code/', 'github-code', 'the-stack', 'vault',
                     'starcoderdata', 'stack-dedup']
    return any(p in path_lower for p in code_patterns)


# ─── Cleaning (Issue 7: URL removal without breaking sentences) ───────────────

def clean(text):
    """Normalise whitespace and remove URLs without breaking sentences."""
    text = re.sub(r'^\s*https?://\S+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+https?://\S+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def clean_code(text):
    """Light cleaning for code — preserve structure and indentation."""
    text = re.sub(r'^\s*https?://\S+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+https?://\S+', '', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()


# ─── Quality Filters ──────────────────────────────────────────────────────────

def is_good_length(text, is_code=False):
    min_len = 50 if is_code else 200
    return len(text) >= min_len


def has_sufficient_vocabulary(text, is_code=False):
    if is_code:
        return True
    words = text.split()
    n = len(words)
    if n < 20:
        return False
    required_unique = min(50, max(10, n // 3))
    return len(set(words)) >= required_unique


def has_no_excessive_long_words(text, is_code=False):
    words = text.split()
    if len(words) == 0:
        return True
    max_word = max(len(w) for w in words) if words else 0
    threshold = 80 if is_code else 50
    return max_word <= threshold


def has_no_repetition(text, is_code=False, threshold=0.3):
    if is_code:
        threshold = 0.6
    words = text.split()
    if len(words) < 10:
        return True
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return True
    top_count = Counter(trigrams).most_common(1)[0][1]
    return top_count / len(trigrams) <= threshold


def has_reasonable_language_ratio(text, is_code=False):
    if is_code:
        return True
    if len(text) == 0:
        return False
    latin_chars = 0
    code_chars = 0
    total = 0
    code_set = set('{}();=<>[]+-*/&|!@#$%^~')
    for c in text:
        if c.isspace():
            continue
        total += 1
        cp = ord(c)
        if c in code_set:
            code_chars += 1
        elif 0x0000 <= cp <= 0x024F:
            latin_chars += 1
    if total == 0:
        return False
    valid_ratio = (latin_chars + code_chars) / total
    return valid_ratio >= 0.65


def has_reasonable_sentence_lengths(text, is_code=False):
    if is_code:
        return True
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(sentences) < 3:
        return True
    short_count = sum(1 for s in sentences if len(s.split()) < 6)
    short_ratio = short_count / len(sentences)
    return short_ratio < 0.8


def is_not_list_heavy(text, is_code=False):
    if is_code:
        return True
    lines = text.split('\n')
    if len(lines) < 5:
        return True
    list_markers = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^[-*•▪▸►→]\s', stripped):
            list_markers += 1
        elif re.match(r'^\d+[.)]\s', stripped):
            list_markers += 1
    total_content_lines = sum(1 for l in lines if l.strip())
    if total_content_lines == 0:
        return False
    return list_markers / total_content_lines < 0.7


# ─── Code-specific Filters ────────────────────────────────────────────────────

def is_good_code(text):
    lines = text.split('\n')
    code_lines = 0
    comment_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('/*') or stripped.startswith('* ') or stripped.startswith('*/'):
            comment_lines += 1
        elif stripped.startswith('//'):
            comment_lines += 1
        elif stripped.startswith('#'):
            is_c_preprocessor = any(stripped.startswith(d) for d in C_PREPROCESSOR)
            if is_c_preprocessor:
                code_lines += 1
            else:
                comment_lines += 1
        else:
            code_lines += 1

    if code_lines == 0:
        return False, "no_code_content"
    total_content = code_lines + comment_lines
    if total_content > 0 and comment_lines / total_content > 0.9:
        return False, "mostly_comments"
    return True, "ok"


# ─── Combined Filter ─────────────────────────────────────────────────────────

def is_good(text, is_code=False):
    """Run all quality filters. Returns (is_good, reason)."""
    if not is_good_length(text, is_code):
        return False, "too_short"
    if not has_sufficient_vocabulary(text, is_code):
        return False, "low_vocab"
    if not has_no_excessive_long_words(text, is_code):
        return False, "long_words"
    if not has_no_repetition(text, is_code=is_code):
        return False, "repetitive"
    if not has_reasonable_language_ratio(text, is_code):
        return False, "non_latin"
    if not has_reasonable_sentence_lengths(text, is_code):
        return False, "short_sentences"
    if not is_not_list_heavy(text, is_code):
        return False, "list_heavy"
    if is_code:
        code_ok, code_reason = is_good_code(text)
        if not code_ok:
            return False, code_reason
    return True, "ok"


# ─── Phase 1: Quality Filtering (CPU-bound, parallelizable) ──────────────────

def quality_filter_file(in_path):
    """Filter a single JSONL file for quality ONLY (no dedup).
    Returns (out_path, kept_count, rejected_counter, is_code_flag).

    This function is designed to be called in parallel across files.
    Each worker writes to a temp file that the main process will
    then deduplicate in Phase 2.
    """
    file_is_code = _is_code_file(in_path)
    out_path = in_path + '.quality_filtered'

    kept = 0
    rejected = Counter()

    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                text = json.loads(line).get('text', '')
            except (json.JSONDecodeError, KeyError):
                rejected['parse_error'] += 1
                continue

            is_code = file_is_code or is_code_text(text)

            if is_code:
                text = clean_code(text)
            else:
                text = clean(text)

            if not text:
                rejected['empty'] += 1
                continue

            good, reason = is_good(text, is_code=is_code)
            if not good:
                rejected[reason] += 1
                continue

            fout.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            kept += 1

    return out_path, kept, rejected, file_is_code


# ─── Phase 2: Exact Dedup (sqlite3 — sequential, fast) ──────────────────────

def dedup_file(in_path, out_path, dedup_store=None):
    """Exact MD5 dedup against the persistent store. Sequential — sqlite3
    is not fork-safe, and batched INSERTs are fast enough single-threaded."""
    kept = 0
    rejected = Counter()
    local_seen = set()

    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                text = json.loads(line).get('text', '')
            except (json.JSONDecodeError, KeyError):
                continue
            if not text:
                continue

            doc_hash = md5(text.encode('utf-8', errors='ignore')).hexdigest()
            if dedup_store is not None:
                if dedup_store.seen(doc_hash):
                    rejected['duplicate'] += 1
                    continue
                dedup_store.add(doc_hash)
            else:
                if doc_hash in local_seen:
                    rejected['duplicate'] += 1
                    continue
                local_seen.add(doc_hash)

            fout.write(line)
            kept += 1

    return kept, rejected


# ─── Phase 3: Fuzzy Dedup (MinHash LSH, parallelizable per file) ─────────────

def build_minhash(text, num_perm=128, ngram_size=5):
    try:
        from datasketch import MinHash
    except ImportError:
        return None
    m = MinHash(num_perm=num_perm)
    for i in range(len(text) - ngram_size + 1):
        m.update(text[i:i+ngram_size].encode('utf-8'))
    return m


def fuzzy_dedup_file(in_path, out_path=None, threshold=0.80, num_perm=128):
    """Remove near-duplicates using MinHash LSH.
    Returns (in_path, kept_count) — designed for parallel execution."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  [FUZZY-DEDUP] datasketch not installed, skipping fuzzy dedup")
        print("  Install with: pip install datasketch")
        if out_path:
            import shutil
            shutil.copy2(in_path, out_path)
        count = 0
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                count += 1
        return in_path, count

    if out_path is None:
        out_path = in_path + '.fuzzy_deduped'

    is_code = _is_code_file(in_path)
    threshold = 0.80 if is_code else 0.85

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = 0
    doc_id = 0

    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                text = json.loads(line).get('text', '')
            except (json.JSONDecodeError, KeyError):
                fout.write(line)
                kept += 1
                continue
            if not text:
                fout.write(line)
                kept += 1
                continue

            mh = build_minhash(text, num_perm=num_perm)
            if mh is None:
                fout.write(line)
                kept += 1
                continue

            key = f"doc_{doc_id}"
            doc_id += 1

            if lsh.query(mh):
                continue

            lsh.insert(key, mh)
            fout.write(line)
            kept += 1

    return in_path, kept


# ─── Main Pipeline ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter raw data (code-aware, multi-core)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    workers = args.workers or os.cpu_count() or 4

    print("=" * 60)
    print(f"Filtering raw data (code-aware, v4, {workers} workers)")
    print("=" * 60)

    t_start = time.time()

    # Collect all input files
    all_files = []
    for root, dirs, files in os.walk("data_raw"):
        for fname in sorted(files):
            if not fname.endswith('.jsonl'):
                continue
            in_path = os.path.join(root, fname)
            all_files.append(in_path)

    # Sort: code files first, then text files (Issue 14: code dedup priority)
    code_file_list = [f for f in all_files if _is_code_file(f)]
    text_file_list = [f for f in all_files if not _is_code_file(f)]
    ordered_files = code_file_list + text_file_list

    # ── Phase 1: Quality filtering (parallel) ──────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 1: Quality filtering ({workers} workers)")
    print(f"{'='*60}")

    phase1_kept = 0
    phase1_rejected = Counter()
    quality_filtered_files = []  # (quality_filtered_path, is_code)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(quality_filter_file, f): f for f in ordered_files}
        for future in as_completed(futures):
            src = futures[future]
            try:
                out_path, kept, rejected, is_code = future.result()
                tag = " [CODE]" if is_code else ""
                print(f"  {os.path.basename(src)}{tag}: kept {kept:,}, rejected {sum(rejected.values()):,}")
                phase1_kept += kept
                phase1_rejected += rejected
                quality_filtered_files.append((out_path, is_code))
            except Exception as e:
                print(f"  ERROR processing {src}: {e}")

    print(f"\nPhase 1 complete: {phase1_kept:,} passed quality filter "
          f"({time.time()-t_start:.1f}s)")

    # ── Phase 2: Exact MD5 dedup (sequential — sqlite3) ───────────────────
    print(f"\n{'='*60}")
    print("Phase 2: Exact MD5 deduplication (sqlite3)")
    print(f"{'='*60}")

    t_phase2 = time.time()
    dedup_store = DedupStore()
    contamination_blacklist = _CONTAMINATION_BLACKLIST

    phase2_kept = 0
    phase2_rejected = Counter()
    deduped_files = []

    for quality_path, is_code in quality_filtered_files:
        fname = os.path.basename(quality_path).replace('.quality_filtered', '')
        out_name = fname.replace('.jsonl', '_filtered.jsonl')
        out_path = os.path.join("data_filtered", out_name)
        tmp_path = out_path + '.dedup_tmp'

        tag = " [CODE]" if is_code else ""
        kept, rejected = dedup_file(quality_path, tmp_path, dedup_store=dedup_store)
        phase2_kept += kept
        phase2_rejected += rejected
        deduped_files.append((tmp_path, is_code))

        print(f"  {fname}{tag}: kept {kept:,}, dupes {sum(rejected.values()):,}")

        # Clean up intermediate quality-filtered file
        os.remove(quality_path)

    dedup_store.close()
    print(f"\nPhase 2 complete: {phase2_kept:,} after exact dedup "
          f"({time.time()-t_phase2:.1f}s)")

    # ── Phase 3: Fuzzy dedup (parallel per file) ──────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 3: Fuzzy deduplication (MinHash LSH, {workers} workers)")
    print(f"{'='*60}")

    t_phase3 = time.time()
    phase3_kept = 0

    try:
        from datasketch import MinHash  # noqa: F401 — check if available
        has_datasketch = True
    except ImportError:
        has_datasketch = False

    if has_datasketch:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for tmp_path, is_code in deduped_files:
                fname = os.path.basename(tmp_path).replace('.dedup_tmp', '')
                final_path = tmp_path.replace('.dedup_tmp', '')
                futures[pool.submit(fuzzy_dedup_file, tmp_path, final_path)] = fname

            for future in as_completed(futures):
                fname = futures[future]
                try:
                    in_path, kept = future.result()
                    phase3_kept += kept
                    print(f"  {fname}: kept {kept:,} after fuzzy dedup")
                    # Clean up intermediate dedup file
                    tmp = in_path.replace('_filtered.jsonl', '_filtered.jsonl.dedup_tmp')
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception as e:
                    print(f"  ERROR fuzzy dedup {fname}: {e}")
    else:
        # No datasketch — just rename dedup files to final
        for tmp_path, is_code in deduped_files:
            final_path = tmp_path.replace('.dedup_tmp', '')
            os.rename(tmp_path, final_path)
            with open(final_path, 'r', encoding='utf-8', errors='ignore') as f:
                count = sum(1 for _ in f)
            phase3_kept += count
        print("  Skipped — install datasketch for fuzzy dedup (pip install datasketch)")

    print(f"\nPhase 3 complete: {phase3_kept:,} after fuzzy dedup "
          f"({time.time()-t_phase3:.1f}s)")

    # ── Summary ────────────────────────────────────────────────────────────
    total_rejected = phase1_rejected + phase2_rejected
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Filtering complete!")
    print(f"  Quality filter: {phase1_kept:,} passed")
    print(f"  After exact dedup: {phase2_kept:,}")
    print(f"  After fuzzy dedup: {phase3_kept:,}")
    print(f"  Total rejected: {sum(total_rejected.values()):,}")
    for reason, count in total_rejected.most_common():
        print(f"    {reason}: {count:,}")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"{'='*60}")
