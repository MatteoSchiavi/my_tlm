"""
mix.py — Mix and split data sources, optimised for Italian + C programming.

Data mix ratios (Italian + C priority):
  Italian web + edu:    35%  (primary language)
  Italian Wikipedia:    10%  (high-quality Italian)
  C code:               20%  (primary programming language)
  C++ code:              5%  (complements C understanding)
  Other code:           10%  (Python, JS, Bash, Rust — general programming structure)
  English web + edu:    15%  (general knowledge, English understanding)
  English Wikipedia:    10%  (encyclopedic knowledge)

This gives: ~45% Italian, ~30% code (67% C), ~25% English

Key improvements:
  + Italian-priority data mix
  + C programming as dominant code language
  + C++ code support (complements C)
  + Proper reservoir sampling across categories
  + File naming convention matching for category detection
  + Shuffle with seed for reproducibility
  + Support for new data sources (gutenberg, stackoverflow, cpp)
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

# Target total documents (overridden by what's actually available)
TARGET_TOTAL = 7_000_000

# Category -> target ratio (must sum to 1.0)
RATIOS = {
    'italian':    0.45,   # Italian web + edu + wiki + gutenberg
    'c_code':     0.20,   # C programming
    'cpp_code':   0.05,   # C++ programming (complements C)
    'other_code': 0.05,   # Python, JS, Bash, Rust, StackOverflow
    'english':    0.25,   # English web + edu + wiki + gutenberg
}

# File pattern -> category mapping (more robust than basename matching)
CATEGORY_PATTERNS = {
    'italian': ['italian', 'oscar_it', 'wiki_it', 'fineweb_it', 'gutenberg_it'],
    'c_code':  ['c_code', '/c/', 'github-code-c', 'the-stack'],
    'cpp_code': ['cpp_code', '/cpp/', 'github-code-c++'],
    'other_code': ['other_code', 'multi_code', 'python', 'javascript', 'rust', 'shell',
                   'stackoverflow'],
    'english': ['english', 'openweb', 'fineweb_edu', 'wiki_en', 'gutenberg_en'],
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


def reservoir_sample(paths, k, seed=SEED):
    """Reservoir sampling: select k items from a stream of unknown size.
    Guarantees uniform probability of selection for each item."""
    rng = random.Random(seed)
    sample = []
    total = 0

    for path in paths:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                total += 1
                if len(sample) < k:
                    sample.append(line)
                else:
                    j = rng.randint(0, total - 1)
                    if j < k:
                        sample[j] = line

    return sample, total


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
        print(f"Warning: {len(uncategorized)} uncategorized files:")
        for f in uncategorized:
            print(f"  {f}")
        # Assign uncategorized to 'english' as default
        cat_files['english'].extend(uncategorized)

    return cat_files


def main():
    print("=" * 60)
    print("Mixing data sources (Italian + C priority)")
    print("=" * 60)

    cat_files = collect_files()

    # Show available data per category
    print("\nAvailable data:")
    for cat, paths in sorted(cat_files.items()):
        available = sum(cached_count_lines(p) for p in paths)
        target = int(TARGET_TOTAL * RATIOS.get(cat, 0))
        print(f"  {cat:12s}: {available:>10,} available, target: {target:>10,}")

    # Sample from each category according to ratios
    all_lines = []
    for cat, paths in cat_files.items():
        if not paths:
            print(f"  WARNING: No files for category '{cat}'")
            continue

        available = sum(cached_count_lines(p) for p in paths)
        target = min(int(TARGET_TOTAL * RATIOS.get(cat, 0)), available)

        if target == 0:
            continue

        print(f"\nSampling {target:,} from {cat}...")
        lines, total = reservoir_sample(paths, target)
        all_lines.extend(lines)
        print(f"  Got {len(lines):,} lines (from {total:,} available)")

    if not all_lines:
        print("ERROR: No data found! Run download.py and filter.py first.")
        return

    # Shuffle globally
    print(f"\nShuffling {len(all_lines):,} documents...")
    random.shuffle(all_lines)

    # Split: 99% train, 1% validation
    split = int(len(all_lines) * 0.99)
    train_lines = all_lines[:split]
    val_lines = all_lines[split:]

    print(f"Writing train ({len(train_lines):,}) and val ({len(val_lines):,})...")

    with open("data_mixed/train.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open("data_mixed/val.jsonl", 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    # Summary
    print(f"\n{'='*60}")
    print(f"Mix complete!")
    print(f"  Total documents: {len(all_lines):,}")
    print(f"  Train: {len(train_lines):,}")
    print(f"  Val:   {len(val_lines):,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
