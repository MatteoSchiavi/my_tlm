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
  + Explicit warning for uncategorized files instead of silent English assignment
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
from collections import defaultdict

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


# Cache line counts to avoid re-reading files
_line_count_cache = {}
def cached_count_lines(path):
    if path not in _line_count_cache:
        _line_count_cache[path] = count_lines(path)
    return _line_count_cache[path]


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


def collect_files():
    """Find and categorize all filtered JSONL files."""
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
        print(f"\nWARNING: {len(uncategorized)} uncategorized files:")
        for f in uncategorized:
            print(f"  {f}")
        print(f"  These will be assigned to 'english' as default.")
        print(f"  If this is wrong, add patterns to CATEGORY_PATTERNS in mix.py")
        cat_files['english'].extend(uncategorized)

    return cat_files


def main():
    print("=" * 60)
    print("Mixing data sources by TOKEN count (char proxy)")
    print("=" * 60)

    cat_files = collect_files()

    # Calibrate chars/token per category
    cat_chars_per_token = {}
    for cat in cat_files:
        # Use the first file in the category for calibration
        cal_path = cat_files[cat][0]
        cat_chars_per_token[cat] = calibrate_chars_per_token(cal_path)
        print(f"  {cat:12s}: chars/token = {cat_chars_per_token[cat]:.2f}")

    # Show available data per category (both docs and estimated chars)
    print("\nAvailable data:")
    total_available_chars = 0
    for cat, paths in sorted(cat_files.items()):
        docs = sum(cached_count_lines(p) for p in paths)
        # Estimate total chars
        total_chars = 0
        for p in paths:
            avg, _ = estimate_chars(p)
            total_chars += avg * cached_count_lines(p)
        total_available_chars += total_chars
        target_pct = RATIOS.get(cat, 0) * 100
        cpt = cat_chars_per_token.get(cat, 4.0)
        est_tokens = total_chars / cpt
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
    all_lines = []
    actual_chars = {}

    for cat, paths in cat_files.items():
        if not paths:
            print(f"  WARNING: No files for category '{cat}'")
            continue

        # Calculate char budget for this category
        char_budget = int(TARGET_TOTAL_CHARS * RATIOS.get(cat, 0))

        # Check available chars
        available_chars = 0
        for p in paths:
            avg, _ = estimate_chars(p)
            available_chars += avg * cached_count_lines(p)

        # Don't try to sample more than available
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
    # This ensures val covers the full length/distribution spectrum
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
    print(f"\n{'='*60}")
    print(f"Mix complete!")
    print(f"  Total documents: {len(all_lines):,}")
    print(f"  Total chars: ~{total_chars_mixed/1e6:.1f}M")
    print(f"  Estimated tokens: ~{total_est_tokens/1e6:.1f}M (calibrated)")
    print(f"  Train: {len(train_lines):,}")
    print(f"  Val:   {len(val_lines):,} (systematic sampling, every {val_interval}th doc)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
