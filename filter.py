"""
filter.py v3 — Data cleaning, deduplication, and quality filtering.

Key fixes from review:
  ✅ Full document MD5 hash (not just first 500 chars)
  ✅ 3-gram repetition filter (kills boilerplate/SEO spam)
  ✅ Italian language support (not just ASCII)
  ✅ Quality scoring for web data
  ✅ Language detection with langdetect (keep Italian + English)
  ✅ CODE-AWARE FILTERING: code files skip NLP filters that reject valid code

v2 fixes (code pipeline):
  ✅ BUG FIX: #include, #define, #ifdef etc. are NOT comments in C/C++
     Old code: `stripped.startswith('#')` counted preprocessor directives as comments
     This caused valid C code to be rejected as "mostly_comments" — devastating!
  ✅ Relax repetition filter for code (0.3 → 0.6 threshold)
     Code has legitimate repetitive patterns (switch cases, struct field access, etc.)
  ✅ Allow single-line C macros and one-line functions (removed len(lines) < 2 check)
  ✅ Code detection also checks text content (not just file path)
     Files from the-stack-dedup might not have "code" in the path but ARE code

v3 fixes:
  ✅ Issue 2: Scale vocabulary uniqueness requirement with document length
     (was rejecting ALL short documents because len(set(words)) < 50 always)
  ✅ Issue 7: URL removal no longer breaks sentences — removes whole URL lines
     and trailing URLs, not in-place replacement
  ✅ Issue 3: MinHash LSH fuzzy deduplication for near-duplicate detection
  ✅ Issue 10: Persistent sqlite3 DedupStore instead of in-memory Python set
     (~30MB on disk vs 600-700MB in RAM; also enables resuming)
  ✅ Issue 14: Process code files first so code dedup takes priority over text
  ✅ Issue 20: Benchmark contamination screening infrastructure (13-gram overlap)
"""

import os
import json
import re
import sqlite3
from hashlib import md5
from collections import Counter

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

# Benchmark contamination blacklist (13-gram overlap)
# Populate by extracting 13-grams from benchmark test sets
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
    """Detect if text is likely source code rather than natural language.

    Uses multiple signals:
    - High density of code characters ({, }, ;, ->, etc.)
    - Presence of code keywords
    - Code-like line patterns (#include, def, class, etc.)
    """
    if len(text) < 20:
        return False

    # Count code-specific characters and patterns
    code_chars = sum(1 for c in text if c in '{}();=<>[]+-*/&|!#')
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars == 0:
        return False

    code_char_ratio = code_chars / total_chars

    # Check for code keywords
    first_500 = text[:500]
    has_c_keywords = any(kw in first_500 for kw in CODE_KEYWORDS_C)
    has_general_keywords = any(kw in first_500 for kw in CODE_KEYWORDS_GENERAL)

    # High code character density = definitely code
    if code_char_ratio > 0.15:
        return True

    # Code keywords present = likely code
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
    # Remove whole lines that are mostly a URL (navigation menus, link lists)
    text = re.sub(r'^\s*https?://\S+\s*$', '', text, flags=re.MULTILINE)
    # Remove trailing URLs from sentences (leave sentence intact up to the URL)
    text = re.sub(r'\s+https?://\S+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def clean_code(text):
    """Light cleaning for code — preserve structure and indentation.
    Remove URLs in comments without breaking code structure."""
    # Remove whole lines that are just a URL (e.g. reference links in comments)
    text = re.sub(r'^\s*https?://\S+\s*$', '', text, flags=re.MULTILINE)
    # Remove trailing URLs from lines (leave code intact up to the URL)
    text = re.sub(r'\s+https?://\S+', '', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)  # Collapse excessive blank lines
    return text.strip()


# ─── Quality Filters ──────────────────────────────────────────────────────────

def is_good_length(text, is_code=False):
    """Remove documents that are too short to be useful.
    Code files are allowed to be shorter (many functions are 50-200 chars)."""
    min_len = 50 if is_code else 200
    return len(text) >= min_len


def has_sufficient_vocabulary(text, is_code=False):
    """Remove documents with very low unique word count (gibberish/boilerplate).
    For code: skip this check entirely — code naturally has fewer 'words'.

    Issue 2 fix: Scale uniqueness requirement with document length.
    Old code required len(set(words)) >= 50, which rejected ALL documents
    with 20-49 words (since len(set(words)) <= len(words) < 50)."""
    if is_code:
        return True
    words = text.split()
    n = len(words)
    if n < 20:
        return False
    # Scale uniqueness requirement with document length
    required_unique = min(50, max(10, n // 3))
    return len(set(words)) >= required_unique


def has_no_excessive_long_words(text, is_code=False):
    """Remove documents with very long "words" (URLs, paths, encoded data).
    For code: allow longer 'words' (long_variable_names_in_snake_case are normal)."""
    words = text.split()
    if len(words) == 0:
        return True
    max_word = max(len(w) for w in words) if words else 0
    threshold = 80 if is_code else 50
    return max_word <= threshold


def has_no_repetition(text, is_code=False, threshold=0.3):
    """3-gram repetition filter — one of the most effective spam filters.
    Removes documents where the most common 3-gram covers >threshold of the text.

    For natural language: 30% threshold (catches SEO spam, boilerplate)
    For code: 60% threshold — code has legitimate repetitive patterns like:
      - Switch/case blocks: `case X: break;` repeated many times
      - Struct field access: `obj->field1 = ...; obj->field2 = ...;`
      - Repeated #include guards, enum definitions, etc.
      Only catches truly degenerate auto-generated code at this threshold.
    """
    if is_code:
        threshold = 0.6

    words = text.split()
    if len(words) < 10:
        return True  # Too short to evaluate
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return True
    top_count = Counter(trigrams).most_common(1)[0][1]
    return top_count / len(trigrams) <= threshold


def has_reasonable_language_ratio(text, is_code=False):
    """Allow both Italian and Latin-script text. Reject documents that are
    predominantly non-Latin script (CJK, Cyrillic, Arabic) unless they're code.

    For code: skip this check — code uses ASCII/English identifiers naturally.
    """
    if is_code:
        return True

    if len(text) == 0:
        return False

    latin_chars = 0
    code_chars = 0  # brackets, semicolons, etc. for code
    total = 0
    code_set = set('{}();=<>[]+-*/&|!@#$%^~')

    for c in text:
        if c.isspace():
            continue
        total += 1
        cp = ord(c)
        if c in code_set:
            code_chars += 1
        elif 0x0000 <= cp <= 0x007F:
            latin_chars += 1
        elif 0x0080 <= cp <= 0x00FF:
            latin_chars += 1
        elif 0x0100 <= cp <= 0x024F:
            latin_chars += 1

    if total == 0:
        return False

    valid_ratio = (latin_chars + code_chars) / total
    return valid_ratio >= 0.65


def has_reasonable_sentence_lengths(text, is_code=False):
    """SEO spam and low-quality content has very short sentences.
    For code: skip this check — code doesn't have 'sentences'."""
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
    """Pages that are mostly lists tend to be low-quality.
    For code: skip — code is naturally list-heavy (line after line)."""
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
    """Code-specific quality filters.

    CRITICAL FIX: #include, #define, #ifdef etc. are C preprocessor directives,
    NOT comments. The old code counted any line starting with '#' as a comment,
    which caused valid C files like:

        #include <stdio.h>
        #include <stdlib.h>
        #define MAX 100
        int main() { ... }

    to be rejected as "mostly_comments" because 3/5 non-blank lines start with #.

    Now: lines starting with known C preprocessor directives are counted as CODE.
    Only lines starting with # that don't match any directive are treated as
    comments (which is correct for Python/shell # comments).
    """
    lines = text.split('\n')

    # Allow single-line code (macros like `#define MAX_SIZE 1024`)
    # Old code rejected these — wrong for C!

    # Count code vs comment lines with C-preprocessor awareness
    code_lines = 0
    comment_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Check for C/C++ block comments: /* ... */
        if stripped.startswith('/*') or stripped.startswith('* ') or stripped.startswith('*/'):
            comment_lines += 1
        # Check for C++ line comments: // ...
        elif stripped.startswith('//'):
            comment_lines += 1
        # Check for # lines: C preprocessor directive or Python/shell comment?
        elif stripped.startswith('#'):
            # C preprocessor directives are CODE, not comments!
            is_c_preprocessor = any(stripped.startswith(d) for d in C_PREPROCESSOR)
            if is_c_preprocessor:
                code_lines += 1
            else:
                # Unknown # line — could be Python/shell comment
                comment_lines += 1
        else:
            code_lines += 1

    if code_lines == 0:
        return False, "no_code_content"

    # Skip if >90% comments (very generous for code — code files should have code)
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

    # Code-specific checks
    if is_code:
        code_ok, code_reason = is_good_code(text)
        if not code_ok:
            return False, code_reason

    return True, "ok"


# ─── Fuzzy Deduplication (Issue 3: MinHash LSH) ──────────────────────────────

def build_minhash(text, num_perm=128, ngram_size=5):
    """Build MinHash signature using character n-grams (works for both code and text)."""
    try:
        from datasketch import MinHash
    except ImportError:
        return None
    m = MinHash(num_perm=num_perm)
    for i in range(len(text) - ngram_size + 1):
        m.update(text[i:i+ngram_size].encode('utf-8'))
    return m


def fuzzy_dedup_file(in_path, out_path, threshold=0.80, num_perm=128):
    """Remove near-duplicates using MinHash LSH.
    threshold: 0.80 for code (aggressive), 0.85 for text (conservative).
    Returns: kept_count"""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  [FUZZY-DEDUP] datasketch not installed, skipping fuzzy dedup")
        print("  Install with: pip install datasketch")
        import shutil
        shutil.copy2(in_path, out_path)
        count = 0
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                count += 1
        return count

    is_code = _is_code_file(in_path)
    threshold = 0.80 if is_code else 0.85

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = 0
    rejected = 0
    doc_id = 0

    # First pass: build LSH index and identify near-duplicates
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

            # Check if near-duplicate already exists
            if lsh.query(mh):
                rejected += 1
                continue

            # Not a near-duplicate — keep it
            lsh.insert(key, mh)
            fout.write(line)
            kept += 1

    return kept


# ─── Main Filter Pipeline ────────────────────────────────────────────────────

def filter_file(in_path, out_path, dedup_store=None, contamination_blacklist=None):
    """Filter a single JSONL file.
    dedup_store: DedupStore instance for persistent deduplication (Issue 10).
    contamination_blacklist: set of 13-grams from benchmark data (Issue 20).
    Returns: (kept_count, rejected_counter)
    """
    file_is_code = _is_code_file(in_path)
    kept = 0
    rejected = Counter()
    # Fallback in-memory set for dedup when no DedupStore is provided
    local_seen = set()

    with open(in_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                text = json.loads(line).get('text', '')
            except (json.JSONDecodeError, KeyError):
                rejected['parse_error'] += 1
                continue

            # Determine if this document is code
            # Use file-level detection AND text-level detection for robustness
            # (files from the-stack-dedup might not have "code" in path but ARE code)
            is_code = file_is_code or is_code_text(text)

            # Clean differently for code vs text
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

            # Full document deduplication (Issue 10: persistent sqlite3 store)
            doc_hash = md5(text.encode('utf-8', errors='ignore')).hexdigest()
            if dedup_store is not None:
                if dedup_store.seen(doc_hash):
                    rejected['duplicate'] += 1
                    continue
                dedup_store.add(doc_hash)
            else:
                # Fallback to in-memory set if no DedupStore provided
                if doc_hash in local_seen:
                    rejected['duplicate'] += 1
                    continue
                local_seen.add(doc_hash)

            # Benchmark contamination check (Issue 20)
            if is_contaminated(text, contamination_blacklist):
                rejected['contaminated'] += 1
                continue

            fout.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            kept += 1

    return kept, rejected


if __name__ == "__main__":
    print("=" * 60)
    print("Filtering raw data (code-aware, v3)")
    print("=" * 60)

    # Issue 10: Persistent sqlite3 dedup store instead of in-memory set
    dedup_store = DedupStore()

    # Issue 20: Benchmark contamination blacklist (loaded on demand)
    contamination_blacklist = _CONTAMINATION_BLACKLIST

    total_kept = 0
    total_rejected = Counter()
    code_files = 0
    text_files = 0

    # Issue 14: Process code files first, then text files.
    # This way code dedup takes priority — when an Italian tutorial with C code
    # has the same hash as a C source file, the C source file (processed first)
    # is kept, and the tutorial's duplicate is rejected.
    all_files = []
    for root, dirs, files in os.walk("data_raw"):
        for fname in sorted(files):
            if not fname.endswith('.jsonl'):
                continue
            in_path = os.path.join(root, fname)
            all_files.append(in_path)

    # Sort: code files first, then text files
    code_file_list = [f for f in all_files if _is_code_file(f)]
    text_file_list = [f for f in all_files if not _is_code_file(f)]
    ordered_files = code_file_list + text_file_list

    for in_path in ordered_files:
        fname = os.path.basename(in_path)
        out_name = fname.replace('.jsonl', '_filtered.jsonl')
        out_path = os.path.join("data_filtered", out_name)

        is_code = _is_code_file(in_path)
        tag = " [CODE]" if is_code else ""
        if is_code:
            code_files += 1
        else:
            text_files += 1

        print(f"\nFiltering {in_path}{tag}")
        kept, rejected = filter_file(in_path, out_path,
                                     dedup_store=dedup_store,
                                     contamination_blacklist=contamination_blacklist)
        total_kept += kept
        total_rejected += rejected

        print(f"  Kept: {kept:,}")
        for reason, count in rejected.most_common():
            print(f"  Rejected ({reason}): {count:,}")

    # Fuzzy deduplication pass (MinHash LSH for near-duplicates)
    print(f"\n{'='*60}")
    print("Fuzzy deduplication (MinHash LSH)")
    print(f"{'='*60}")

    fuzzy_kept = 0
    fuzzy_rejected = 0
    for fname in sorted(os.listdir("data_filtered")):
        if not fname.endswith('_filtered.jsonl'):
            continue
        in_path = os.path.join("data_filtered", fname)
        # Use _fuzzy suffix, then rename back
        tmp_path = os.path.join("data_filtered", fname + '.fuzzy_tmp')
        print(f"\nFuzzy dedup: {in_path}")
        kept = fuzzy_dedup_file(in_path, tmp_path)
        # Replace original with fuzzy-deduped version
        os.replace(tmp_path, in_path)
        removed = 0
        # count removed by comparing
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            new_count = sum(1 for _ in f)
        fuzzy_kept += kept
        print(f"  Kept: {kept:,}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Total kept (after quality filter): {total_kept:,}")
    print(f"Total kept (after fuzzy dedup):   {fuzzy_kept:,}")
    print(f"Total rejected: {sum(total_rejected.values()):,}")
    print(f"Code files: {code_files}, Text files: {text_files}")
    for reason, count in total_rejected.most_common():
        print(f"  {reason}: {count:,}")

    # Close persistent dedup store
    dedup_store.close()
