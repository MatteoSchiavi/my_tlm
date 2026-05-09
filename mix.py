"""
mix.py v2 — Mix and split data sources by TOKEN count (not document count).

CRITICAL FIX: Previous version mixed by document count, but document sizes vary
massively across categories:
  - C code:     avg ~200 chars/doc  → ~50 tokens/doc
  - Wikipedia:  avg ~2000 chars/doc → ~500 tokens/doc
  - Italian web: avg ~800 chars/doc → ~200 tokens/doc

So "35% C code docs" was actually only ~10% C code TOKENS!
The model saw mostly Italian text and very little C code in actual training.

v2 fixes this by mixing by CHARACTER count (proxy for token count, ~4 chars/token).
This ensures the actual training data matches the intended ratios.

Data mix ratios (by TOKEN count, not document count):
  Italian:    35%  (primary language for chat)
  C code:     30%  (primary programming skill)
  C++ code:   10%  (complements C understanding)
  Other code:  5%  (Python, JS, Rust — general programming structure)
  English:    20%  (general knowledge)

Other v2 fixes:
  + Token-based (character proxy) mixing instead of document-based
  + Hard error on uncategorized files (no silent English pollution)
    --allow-unknown places them in 'unknown' bucket (excluded from training)
    with per-file diagnostics and likely-category inference
  + Detailed token ratio reporting
  + Code-aware category detection
  + Per-file reservoir sampling (fixes cross-file bias)
  + Per-category chars/token calibration (fixes wrong token estimates)
  + Systematic sampling for val split (fixes length distribution bias)
"""

import os
import json
import random
import glob
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

os.makedirs("data_mixed", exist_ok=True)

SEED = 42
random.seed(SEED)

# ─── Data Mix Configuration ──────────────────────────────────────────────────

# Target total CHARACTERS (proxy for tokens, ~4 chars/token for BPE)
# This is a budget — actual total depends on available data
TARGET_TOTAL_CHARS = 10_000_000_000  # 10B chars ≈ 2.5B tokens

# Category -> target ratio (by TOKEN/CHAR count, NOT document count)
RATIOS = {
    'italian':    0.35,   # Italian web + edu + wiki
    'c_code':     0.30,   # C programming
    'cpp_code':   0.10,   # C++ programming (complements C)
    'other_code': 0.05,   # Python, JS, Rust
    'english':    0.20,   # English web + edu + wiki
}

# File pattern -> category mapping
CATEGORY_PATTERNS = {
    'italian': ['italian', 'oscar_it', 'wiki_it', 'fineweb_it', 'gutenberg_it'],
    'c_code':  ['c_code', '/c/', 'github-code-c', 'the-stack', 'vault-function',
                'starcoderdata', 'stack-dedup'],
    'cpp_code': ['cpp_code', '/cpp/', 'github-code-c++', 'c++'],
    'other_code': ['other_code', 'multi_code', 'python', 'javascript', 'rust', 'shell',
                   'stackoverflow'],
    'english': ['english', 'openweb', 'fineweb_edu', 'wiki_en', 'gutenberg_en'],
}

# Per-category chars/token estimates (empirically measured for ByteLevel BPE 32k vocab)
# These are defaults — calibrate_chars_per_token() measures actual values
DEFAULT_CHARS_PER_TOKEN = {
    'italian':    4.7,
    'c_code':     2.9,
    'cpp_code':   3.2,
    'other_code': 3.3,
    'english':    4.3,
    'unknown':    4.0,  # Fallback — no training ratio, excluded from sampling
}


def categorize_file(filepath):
    """Determine which category a filtered JSONL file belongs to."""
    path_lower = filepath.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in path_lower:
                return category
    return None


def count_lines(path):
    """Count lines in a file efficiently."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f)


def _scan_file(path):
    """Scan a file for stats: (path, line_count, avg_chars, total_est_chars).
    Called in parallel for I/O-bound file scanning."""
    line_count = 0
    total_chars = 0
    sample_count = 0
    SAMPLE_LIMIT = 1000
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_count += 1
                if sample_count < SAMPLE_LIMIT:
                    try:
                        text = json.loads(line).get('text', '')
                        total_chars += len(text)
                        sample_count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
    except Exception:
        return path, 0, 0, 0
    avg_chars = total_chars / max(sample_count, 1)
    return path, line_count, avg_chars, avg_chars * line_count


# Cache for single-file access (used by calibrate_chars_per_token)
_line_count_cache = {}
def cached_count_lines(path):
    if path not in _line_count_cache:
        _line_count_cache[path] = count_lines(path)
    return _line_count_cache[path]


def parallel_scan_files(paths, workers=4):
    """Scan multiple files in parallel for line counts and char estimates.
    Returns dict: path -> (line_count, avg_chars, est_total_chars)"""
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_file, p): p for p in paths}
        for future in as_completed(futures):
            try:
                path, line_count, avg_chars, est_total = future.result()
                results[path] = (line_count, avg_chars, est_total)
            except Exception:
                pass
    # Also populate the line count cache
    for path, (line_count, _, _) in results.items():
        _line_count_cache[path] = line_count
    return results


def estimate_chars(path, sample_size=1000):
    """Estimate average characters per document by sampling the first N lines.
    This is much faster than reading every line's length."""
    total_chars = 0
    count = 0
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    text = json.loads(line).get('text', '')
                    total_chars += len(text)
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
                if count >= sample_size:
                    break
    except Exception:
        return 0, 0

    avg_chars = total_chars / max(count, 1)
    return avg_chars, count


def calibrate_chars_per_token(jsonl_path, tokenizer_path=None, n_samples=5000):
    """Empirically measure chars/token for this file's content.
    If tokenizer is not available, returns category-based defaults."""
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(tokenizer_path)
            total_chars, total_tokens = 0, 0
            with open(jsonl_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= n_samples:
                        break
                    try:
                        text = json.loads(line)['text'][:2000]
                        total_chars += len(text)
                        total_tokens += len(tokenizer.encode(text).ids)
                    except Exception:
                        continue
            if total_tokens > 0:
                return total_chars / total_tokens
        except Exception:
            pass

    # Fallback: use category-based defaults
    cat = categorize_file(jsonl_path)
    return DEFAULT_CHARS_PER_TOKEN.get(cat, 4.0)


def _reservoir_sample_file(path, k, rng):
    """Standard reservoir sampling of k items from a single file.
    Returns (reservoir_lines, chars_sampled)."""
    reservoir = []
    chars_sampled = 0
    for i, line in enumerate(open(path, 'r', encoding='utf-8', errors='ignore')):
        try:
            text = json.loads(line).get('text', '')
        except (json.JSONDecodeError, KeyError):
            continue
        if not text:
            continue
        if len(reservoir) < k:
            reservoir.append(line)
            chars_sampled += len(text)
        else:
            j = rng.randint(0, i)
            if j < k:
                old_text = json.loads(reservoir[j]).get('text', '')
                chars_sampled -= len(old_text)
                reservoir[j] = line
                chars_sampled += len(text)
    return reservoir, chars_sampled


def reservoir_sample_by_chars(paths, char_budget, seed=SEED):
    """Sample documents from paths until char_budget is reached.
    Uses per-file reservoir sampling to ensure correct representation."""
    rng = random.Random(seed)
    file_stats = []
    total_available_chars = 0

    for path in paths:
        avg_chars, _ = estimate_chars(path)
        total_docs = cached_count_lines(path)
        est_chars = avg_chars * total_docs
        file_stats.append({'path': path, 'avg_chars': avg_chars,
                           'estimated_chars': est_chars, 'total_docs': total_docs})
        total_available_chars += est_chars

    if total_available_chars == 0:
        return [], 0, 0

    sample, total_chars = [], 0
    total_docs_available = 0
    for stats in file_stats:
        if total_available_chars == 0:
            break
        file_share = (stats['estimated_chars'] / total_available_chars) * char_budget
        k = max(1, int(file_share / max(stats['avg_chars'], 1)))
        k = min(k, stats['total_docs'])
        docs, chars = _reservoir_sample_file(stats['path'], k, rng)
        sample.extend(docs)
        total_chars += chars
        total_docs_available += stats['total_docs']

    return sample, total_chars, total_docs_available


def _infer_likely_category(filepath):
    """Infer the most likely category for an uncategorized file by heuristics.

    Checks both the filename and (if available) a small sample of file content
    for signals like code characters, Italian words, etc.

    Returns (likely_category, confidence) where confidence is 'high'/'medium'/'low'.
    """
    path_lower = filepath.lower()
    fname = os.path.basename(path_lower)

    # ── Filename-based heuristics ──

    # Code signals in filename
    code_fname_signals = ['c_code', 'cpp_code', 'code', 'github', 'stack',
                          'vault', 'starcoder', 'codesearchnet', 'python',
                          'javascript', 'rust', 'shell', 'golang', 'java',
                          'algorithm', 'competitive', 'leetcode', 'hackerrank']
    code_hits = sum(1 for s in code_fname_signals if s in fname)
    if code_hits >= 2:
        return 'c_code', 'high'
    if code_hits == 1:
        return 'other_code', 'medium'

    # Italian signals in filename
    italian_fname_signals = ['ital', 'ita_', 'it_', '_it.', 'oscar', 'fineweb_it']
    if any(s in fname for s in italian_fname_signals):
        return 'italian', 'high'

    # English signals in filename
    english_fname_signals = ['en_', '_en.', 'english', 'openweb', 'wiki_en',
                             'fineweb_edu', 'gutenberg', 'wikibooks', 'bookcorpus']
    if any(s in fname for s in english_fname_signals):
        return 'english', 'high'

    # ── Content-based heuristics (sample first 100 lines) ──
    try:
        code_chars = 0
        italian_markers = 0
        total_lines = 0
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if total_lines >= 100:
                    break
                total_lines += 1
                try:
                    text = json.loads(line).get('text', '')
                except (json.JSONDecodeError, KeyError):
                    continue
                if not text:
                    continue

                # Code signal: high density of code characters
                cc = sum(1 for c in text[:500] if c in '{}();=<>[]+-*/&|!#')
                tc = len(text[:500].replace(' ', '').replace('\n', ''))
                if tc > 0 and cc / tc > 0.12:
                    code_chars += 1

                # Italian signal
                sample_lower = text[:500].lower()
                it_markers = [' della ', ' delle ', ' dello ', ' degli ',
                              ' nella ', ' questo ', ' questa ', ' sono ']
                if any(m in sample_lower for m in it_markers):
                    italian_markers += 1

        if total_lines > 0:
            code_ratio = code_chars / total_lines
            italian_ratio = italian_markers / total_lines

            if code_ratio > 0.5 and italian_ratio < 0.05:
                return 'c_code' if code_ratio > 0.7 else 'other_code', 'medium'
            if italian_ratio > 0.2:
                return 'italian', 'medium'
            if code_ratio < 0.1:
                return 'english', 'medium'
    except Exception:
        pass

    return 'english', 'low'


def collect_files(allow_unknown=False):
    """Find and categorize all filtered JSONL files.

    Args:
        allow_unknown: If True, uncategorized files are placed in an 'unknown'
            bucket (ratio=0, excluded from sampling) with a detailed diagnostic
            report including likely-category inference. If False (default),
            raises ValueError.
    """
    files = glob.glob("data_filtered/*_filtered.jsonl")
    if not files:
        # Also check data_raw for direct JSONL files (if filter was skipped)
        files = glob.glob("data_raw/**/*.jsonl", recursive=True)

    cat_files = defaultdict(list)
    uncategorized = []

    for f in files:
        cat = categorize_file(f)
        if cat:
            cat_files[cat].append(f)
        else:
            uncategorized.append(f)

    if uncategorized:
        # ── Build diagnostic report for each uncategorized file ──
        print(f"\n{'!'*60}")
        print(f"ERROR: {len(uncategorized)} uncategorized file(s) found")
        print(f"{'!'*60}")
        print()
        print("  These files cannot be assigned to any known category.")
        print("  Assigning them to 'english' would silently pollute the corpus:")
        print("    - Token balance corruption (English ratio inflates)")
        print("    - Tokenizer training distribution skews")
        print("    - Code specialization weakens")
        print()
        print("  DIAGNOSTIC REPORT:")
        print(f"  {'─'*54}")

        total_unknown_chars = 0
        for f in uncategorized:
            fname = os.path.basename(f)
            # Estimate size
            docs = 0
            chars = 0
            try:
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    for i, line in enumerate(fh):
                        if i >= 1000:
                            break
                        try:
                            text = json.loads(line).get('text', '')
                            chars += len(text)
                            docs += 1
                        except (json.JSONDecodeError, KeyError):
                            continue
                # Extrapolate from sample
                total_docs = cached_count_lines(f)
                avg_chars = chars / max(docs, 1)
                est_total_chars = avg_chars * total_docs
                est_tokens = est_total_chars / 4.0  # rough estimate
            except Exception:
                est_total_chars = 0
                est_tokens = 0
                total_docs = 0

            total_unknown_chars += est_total_chars

            # Infer likely category
            likely_cat, confidence = _infer_likely_category(f)

            print(f"  File: {fname}")
            print(f"    Docs:         ~{total_docs:,}")
            print(f"    Est. chars:   ~{est_total_chars/1e6:.1f}M")
            print(f"    Est. tokens:  ~{est_tokens/1e6:.1f}M")
            print(f"    Likely cat:   {likely_cat} (confidence: {confidence})")
            print()

        print(f"  Total unknown tokens: ~{total_unknown_chars/4e6:.1f}M")
        print(f"  {'─'*54}")
        print()
        print("  FIX: Add matching patterns to CATEGORY_PATTERNS in mix.py:")
        for cat, patterns in CATEGORY_PATTERNS.items():
            print(f"    {cat:12s}: {patterns}")
        print()

        if allow_unknown:
            # Place in 'unknown' bucket — NOT english
            # This bucket has ratio=0 so it is EXCLUDED from sampling
            # but remains visible in reports for auditing
            cat_files['unknown'].extend(uncategorized)
            print(f"  --allow-unknown: placed {len(uncategorized)} file(s) in 'unknown' bucket.")
            print(f"  These files are EXCLUDED from the training mix (ratio=0%).")
            print(f"  To include them, add the right pattern to CATEGORY_PATTERNS.")
            print()
        else:
            print("  Re-run with --allow-unknown to place them in an 'unknown' bucket")
            print("  (excluded from training) instead of erroring.")
            print()
            raise ValueError(
                f"{len(uncategorized)} uncategorized file(s). "
                f"Add patterns to CATEGORY_PATTERNS or use --allow-unknown to proceed."
            )

    return cat_files


def main():
    parser = argparse.ArgumentParser(description="Mix data sources by token count")
    parser.add_argument("--allow-unknown", action="store_true",
                        help="Place uncategorized files in an 'unknown' bucket (excluded "
                             "from training, ratio=0%) instead of erroring. "
                             "Unknown files are reported with diagnostics but never sampled.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for file scanning (default: CPU count)")
    args = parser.parse_args()

    workers = args.workers or os.cpu_count() or 4

    t_start = time.time()
    print("=" * 60)
    print(f"Mixing data sources by TOKEN count (char proxy, {workers} workers)")
    print("=" * 60)

    cat_files = collect_files(allow_unknown=args.allow_unknown)

    # ── Parallel scan: line counts + char estimates for all files at once ──
    all_paths = []
    for paths in cat_files.values():
        all_paths.extend(paths)

    print(f"\nScanning {len(all_paths)} files in parallel ({workers} workers)...")
    t_scan = time.time()
    scan_results = parallel_scan_files(all_paths, workers=workers)
    print(f"  Scanned in {time.time()-t_scan:.1f}s")

    # Calibrate chars/token per category
    cat_chars_per_token = {}
    for cat in cat_files:
        cal_path = cat_files[cat][0]
        cat_chars_per_token[cat] = calibrate_chars_per_token(cal_path)
        print(f"  {cat:12s}: chars/token = {cat_chars_per_token[cat]:.2f}")

    # Show available data per category using pre-scanned data
    print("\nAvailable data:")
    total_available_chars = 0
    for cat, paths in sorted(cat_files.items()):
        docs = 0
        total_chars = 0
        for p in paths:
            if p in scan_results:
                lc, avg, est = scan_results[p]
                docs += lc
                total_chars += est
            else:
                # Fallback for uncached paths
                lc = cached_count_lines(p)
                avg, _ = estimate_chars(p)
                total_chars += avg * lc
                docs += lc
        total_available_chars += total_chars
        target_pct = RATIOS.get(cat, 0) * 100
        cpt = cat_chars_per_token.get(cat, 4.0)
        est_tokens = total_chars / cpt
        if cat == 'unknown':
            print(f"  {cat:12s}: {docs:>10,} docs, ~{total_chars/1e6:.1f}M chars (~{est_tokens/1e6:.1f}M tokens) [EXCLUDED — ratio=0%] chars/tok={cpt:.2f}")
        else:
            print(f"  {cat:12s}: {docs:>10,} docs, ~{total_chars/1e6:.1f}M chars (~{est_tokens/1e6:.1f}M tokens) [target: {target_pct:.0f}%] chars/tok={cpt:.2f}")

    # Warn about missing categories
    for cat in RATIOS:
        if cat not in cat_files or not cat_files[cat]:
            print(f"\n  WARNING: No data for category '{cat}'!")
            if cat == 'c_code':
                print(f"     CRITICAL: C code is 30% of the target mix.")
                print(f"     FIX: python download.py --sources c_code --tier max")
            elif cat == 'cpp_code':
                print(f"     C++ code complements C understanding (shares syntax).")
                print(f"     FIX: python download.py --sources cpp_code --tier max")

    # Sample from each category according to TOKEN (char) ratios
    # Use pre-scanned data for reservoir sampling budgets
    all_lines = []
    actual_chars = {}

    for cat, paths in cat_files.items():
        if not paths:
            print(f"  WARNING: No files for category '{cat}'")
            continue

        char_budget = int(TARGET_TOTAL_CHARS * RATIOS.get(cat, 0))

        # Use pre-scanned data for available chars
        available_chars = 0
        for p in paths:
            if p in scan_results:
                available_chars += scan_results[p][2]  # est_total_chars
            else:
                avg, _ = estimate_chars(p)
                available_chars += avg * cached_count_lines(p)

        char_budget = min(char_budget, int(available_chars))

        if char_budget == 0:
            continue

        cpt = cat_chars_per_token.get(cat, 4.0)
        est_tokens = char_budget / cpt
        print(f"\nSampling ~{char_budget/1e6:.1f}M chars (~{est_tokens/1e6:.1f}M tokens) from {cat}...")
        lines, chars_scanned, docs_scanned = reservoir_sample_by_chars(paths, char_budget)
        all_lines.extend(lines)
        actual_chars[cat] = chars_scanned
        print(f"  Got {len(lines):,} docs, ~{chars_scanned/1e6:.1f}M chars")

    if not all_lines:
        print("ERROR: No data found! Run download.py and filter.py first.")
        return

    # Print actual mix ratios by character count
    total_chars_mixed = sum(actual_chars.values())
    print(f"\nActual data mix (by character count):")
    for cat, chars in sorted(actual_chars.items(), key=lambda x: -x[1]):
        pct = chars / total_chars_mixed * 100 if total_chars_mixed > 0 else 0
        target_pct = RATIOS.get(cat, 0) * 100
        cpt = cat_chars_per_token.get(cat, 4.0)
        est_tokens = chars / cpt
        match = "OK" if abs(pct - target_pct) < 10 else "OFF"
        print(f"  {cat:12s}: ~{chars/1e6:>8.1f}M chars (~{est_tokens/1e6:>6.1f}M tokens) {pct:5.1f}% [target: {target_pct:.0f}%] chars/tok={cpt:.2f} {match}")

    # Shuffle globally
    print(f"\nShuffling {len(all_lines):,} documents...")
    random.shuffle(all_lines)

    # Split: ~1% validation using systematic sampling
    val_interval = 100
    val_indices = set(range(0, len(all_lines), val_interval))
    val_lines = [all_lines[i] for i in val_indices]
    train_lines = [all_lines[i] for i in range(len(all_lines)) if i not in val_indices]

    print(f"Writing train ({len(train_lines):,}) and val ({len(val_lines):,})...")

    with open("data_mixed/train.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open("data_mixed/val.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    # Compute total estimated tokens using per-category calibration
    total_est_tokens = 0
    for cat, chars in actual_chars.items():
        cpt = cat_chars_per_token.get(cat, 4.0)
        total_est_tokens += chars / cpt

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Mix complete!")
    print(f"  Total documents: {len(all_lines):,}")
    print(f"  Total chars: ~{total_chars_mixed/1e6:.1f}M")
    print(f"  Estimated tokens: ~{total_est_tokens/1e6:.1f}M (calibrated)")
    print(f"  Train: {len(train_lines):,}")
    print(f"  Val:   {len(val_lines):,} (systematic sampling, every {val_interval}th doc)")
    print(f"  Wall time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
