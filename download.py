"""
download.py - Download training data with tiered levels and Italian + C focus.

Tier system (use --tier flag):
  smoke    - ~28k docs total, ~5 min   (pipeline testing only)
  quick    - ~250k docs total, ~30 min  (first real training)
  standard - ~1.5M docs total, ~4 hrs   (production quality)
  full     - ~3.1M docs total, ~10 hrs  (near Chinchilla-optimal)
  max      - ~7.2M docs total, ~28 hrs  (maximum intelligence)

Data mix is optimised for Italian language + C programming:
  - Italian: mc4 Italian + Italian Wikipedia + Italian FineWeb
  - C Code:  The Vault (Fsoft-AIC) + CodeSearchNet C fallback
  - Other Code: CodeSearchNet + The Vault (Python, JS, Shell, etc.)
  - English: OpenWebText + English Wikipedia + FineWeb-Edu

Key fixes in v3.3:
  + Replaced ALL BigCode datasets (now gated) with truly open alternatives:
    - bigcode/starcoderdata -> Fsoft-AIC/the-vault (OPEN, no auth needed)
    - bigcode/the-stack-smol -> code_search_net (OPEN, no auth needed)
    - codeparrot/github-code-clean -> removed (deprecated loading script)
  + Fsoft-AIC/the-vault: 1.9M C files, 1.5M Python files, etc.
  + code_search_net: Python, JS, Ruby, Go, Java, PHP (no C, but complements)
  + Sequential download on Windows by default
  + Windows symlink warning suppression
  + Better error messages with actionable fixes
"""

import os
import sys
import json
import argparse
import platform
import warnings
from pathlib import Path

# Suppress Windows symlink warnings from huggingface_hub
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

IS_WINDOWS = platform.system() == 'Windows'


# ─── Tier Definitions ─────────────────────────────────────────────────────────

TIERS = {
    'smoke': {
        'italian_oscar':      3_000,
        'italian_wiki':       1_500,
        'italian_fineweb':    1_500,
        'english_web':        6_000,
        'english_wiki':       3_000,
        'english_edu':        3_000,
        'c_code':             5_000,
        'other_code':         5_000,
    },
    'quick': {
        'italian_oscar':      30_000,
        'italian_wiki':       15_000,
        'italian_fineweb':    15_000,
        'english_web':        60_000,
        'english_wiki':       30_000,
        'english_edu':        30_000,
        'c_code':             40_000,
        'other_code':         30_000,
    },
    'standard': {
        'italian_oscar':      200_000,
        'italian_wiki':       60_000,
        'italian_fineweb':    150_000,
        'english_web':        400_000,
        'english_wiki':       120_000,
        'english_edu':        250_000,
        'c_code':             200_000,
        'other_code':         120_000,
    },
    'full': {
        'italian_oscar':      500_000,
        'italian_wiki':       60_000,
        'italian_fineweb':    250_000,
        'english_web':        800_000,
        'english_wiki':       300_000,
        'english_edu':        600_000,
        'c_code':             400_000,
        'other_code':         200_000,
    },
    'max': {
        'italian_oscar':      1_000_000,
        'italian_wiki':       60_000,
        'italian_fineweb':    600_000,
        'english_web':        2_000_000,
        'english_wiki':       600_000,
        'english_edu':        1_200_000,
        'c_code':             800_000,
        'other_code':         400_000,
    },
}

# Directory structure
DIRS = {
    'italian_oscar':   'data_raw/italian/oscar',
    'italian_wiki':    'data_raw/italian/wiki',
    'italian_fineweb': 'data_raw/italian/fineweb',
    'english_web':     'data_raw/english/web',
    'english_wiki':    'data_raw/english/wiki',
    'english_edu':     'data_raw/english/edu',
    'c_code':          'data_raw/code/c',
    'other_code':      'data_raw/code/other',
}


def ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)


def already_downloaded(path, expected_lines):
    """Check if file exists and has approximately the expected number of lines."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            count = sum(1 for _ in f)
        return count >= expected_lines * 0.8
    except Exception:
        return False


def _progress_log(source, count, interval=25000):
    """Log progress at regular intervals."""
    if count > 0 and count % interval == 0:
        print(f"  [{source}] {count:,} docs")


# ─── Language Detection ────────────────────────────────────────────────────────

try:
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

ITALIAN_MARKERS = [
    ' della ', ' delle ', ' dello ', ' degli ',
    ' nella ', ' nelle ', ' nello ', ' negli ',
    ' questo ', ' questa ', ' questi ', ' queste ',
    ' sono ', ' siamo ', ' essere ', ' avere ',
    ' possono ', ' deve ', ' stato ',
]

def is_likely_italian(text):
    """Check if text is likely Italian."""
    sample = text[:500]
    if HAS_LANGDETECT:
        try:
            return detect(sample) == 'it'
        except Exception:
            return False
    else:
        text_lower = sample.lower()
        return any(w in text_lower for w in ITALIAN_MARKERS)


# ─── Download Functions ────────────────────────────────────────────────────────

def download_italian_oscar(limit):
    """Italian web crawl data. Uses mc4 (open access) instead of OSCAR (gated)."""
    out_path = os.path.join(DIRS['italian_oscar'], 'oscar_it.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[IT-OSCAR] Already downloaded, skipping")
        return

    print(f"[IT-OSCAR] Starting ({limit:,} docs)...")
    count = 0

    # Primary: mc4 Italian (open access)
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "it", split="train", streaming=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for ex in ds:
                text = ex.get('text', '')
                if text and len(text) > 100:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('IT-OSCAR', count)
        print(f"[IT-OSCAR] mc4: {count:,} docs")
    except Exception as e:
        print(f"[IT-OSCAR] mc4 failed: {e}")

    # Fallback: OSCAR-2301 (gated - needs HF auth)
    if count < limit:
        try:
            from datasets import load_dataset
            print(f"[IT-OSCAR] Trying OSCAR-2301 (gated - requires HF auth)...")
            ds = load_dataset("oscar-corpus/OSCAR-2301", "it", split="train", streaming=True)
            mode = 'a' if count > 0 else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    text = ex.get('text', '')
                    if text and len(text) > 100:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('IT-OSCAR/OSCAR', count)
            print(f"[IT-OSCAR] With OSCAR fallback: {count:,} docs")
        except Exception as e2:
            print(f"[IT-OSCAR] OSCAR-2301 also failed (gated dataset).")
            print(f"  FIX: huggingface-cli login, then visit:")
            print(f"  https://huggingface.co/datasets/oscar-corpus/OSCAR-2301")

    print(f"[IT-OSCAR] Done: {count:,} docs")


def download_italian_wiki(limit):
    """Italian Wikipedia - high-quality encyclopedic Italian."""
    out_path = os.path.join(DIRS['italian_wiki'], 'wiki_it.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[IT-WIKI] Already downloaded, skipping")
        return

    print(f"[IT-WIKI] Starting ({limit:,} docs)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.it", split="train",
                         streaming=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            count = 0
            for ex in ds:
                text = ex.get('text', '')
                if text:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('IT-WIKI', count, 10000)
        print(f"[IT-WIKI] Done: {count:,} docs")
    except Exception as e:
        print(f"[IT-WIKI] Failed: {e}")


def download_italian_fineweb(limit):
    """FineWeb Italian subset - educational quality Italian web data."""
    out_path = os.path.join(DIRS['italian_fineweb'], 'fineweb_it.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[IT-FINEWEB] Already downloaded, skipping")
        return

    print(f"[IT-FINEWEB] Starting ({limit:,} docs)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-2", "ita_Latn", split="train",
                         streaming=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            count = 0
            for ex in ds:
                text = ex.get('text', '')
                if text and len(text) > 100:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('IT-FINEWEB', count)
        print(f"[IT-FINEWEB] Done: {count:,} docs")
    except Exception as e:
        print(f"[IT-FINEWEB] FineWeb-2 failed: {e}. Trying FineWeb-Edu Italian filter...")
        if not HAS_LANGDETECT:
            print("  TIP: pip install langdetect for better Italian filtering")
        try:
            from datasets import load_dataset
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "default", split="train",
                             streaming=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                count = 0
                for ex in ds:
                    text = ex.get('text', '')
                    if text and is_likely_italian(text):
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('IT-FINEWEB-fallback', count, 10000)
            print(f"[IT-FINEWEB] Fallback: {count:,} docs")
        except Exception as e2:
            print(f"[IT-FINEWEB] Fallback also failed: {e2}")


def download_english_web(limit):
    """OpenWebText - general English web text."""
    out_path = os.path.join(DIRS['english_web'], 'openwebtext.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[EN-WEB] Already downloaded, skipping")
        return

    print(f"[EN-WEB] Starting ({limit:,} docs)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            count = 0
            for ex in ds:
                text = ex.get('text', '')
                if text:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('EN-WEB', count, 50000)
        print(f"[EN-WEB] Done: {count:,} docs")
    except Exception as e:
        print(f"[EN-WEB] Failed: {e}")


def download_english_wiki(limit):
    """English Wikipedia - encyclopedic English."""
    out_path = os.path.join(DIRS['english_wiki'], 'wiki_en.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[EN-WIKI] Already downloaded, skipping")
        return

    print(f"[EN-WIKI] Starting ({limit:,} docs)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                         streaming=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            count = 0
            for ex in ds:
                text = ex.get('text', '')
                if text:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('EN-WIKI', count, 50000)
        print(f"[EN-WIKI] Done: {count:,} docs")
    except Exception as e:
        print(f"[EN-WIKI] Failed: {e}")


def download_english_edu(limit):
    """FineWeb-Edu - educational quality English web data."""
    out_path = os.path.join(DIRS['english_edu'], 'fineweb_edu.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[EN-EDU] Already downloaded, skipping")
        return

    print(f"[EN-EDU] Starting ({limit:,} docs)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "default", split="train",
                         streaming=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            count = 0
            for ex in ds:
                text = ex.get('text', '')
                if text:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('EN-EDU', count, 50000)
        print(f"[EN-EDU] Done: {count:,} docs")
    except Exception as e:
        print(f"[EN-EDU] Failed: {e}")


def download_c_code(limit):
    """C programming code. Tries truly open-access sources:

    1. Fsoft-AIC/the-vault (train/c subset) — OPEN, no auth needed, ~1.9M C files
    2. bigcode/starcoderdata (C subset) — gated but high quality (needs HF auth)
    3. bigcode/the-stack-smol (data/c) — gated, smaller (needs HF auth)

    NOTE: As of 2025, ALL BigCode datasets require authentication.
    The Vault is the only truly open large-scale code dataset.
    """
    out_path = os.path.join(DIRS['c_code'], 'c_code.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[C-CODE] Already downloaded, skipping")
        return

    print(f"[C-CODE] Starting ({limit:,} docs)...")
    count = 0

    # Source 1: Fsoft-AIC/the-vault — truly open access, ~1.9M C files
    try:
        from datasets import load_dataset
        print(f"  [C-CODE] Trying Fsoft-AIC/the-vault (C, open access)...")
        ds = load_dataset("Fsoft-AIC/the-vault", "language=c", split="train",
                         streaming=True, trust_remote_code=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for ex in ds:
                # The Vault uses 'code' field, may also have 'content'
                text = ex.get('code', ex.get('content', ex.get('text', '')))
                if text and len(text) > 50:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('C-CODE/Vault', count)
        if count > 0:
            print(f"  [C-CODE/Vault] Got {count:,} files")
    except Exception as e:
        err_msg = str(e)
        if 'trust_remote_code' in err_msg or 'loading script' in err_msg.lower():
            print(f"  [C-CODE/Vault] Needs trust_remote_code, trying with it...")
            try:
                from datasets import load_dataset
                ds = load_dataset("Fsoft-AIC/the-vault", split="train",
                                 streaming=True, trust_remote_code=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    for ex in ds:
                        text = ex.get('code', ex.get('content', ex.get('text', '')))
                        if text and len(text) > 50:
                            # Filter for C code by checking language field if available
                            lang = ex.get('language', ex.get('lang', ''))
                            if lang and lang.lower() not in ('c',):
                                continue
                            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                            count += 1
                        if count >= limit:
                            break
                        _progress_log('C-CODE/Vault-noconfig', count)
                if count > 0:
                    print(f"  [C-CODE/Vault] Got {count:,} files (no config)")
            except Exception as e2:
                print(f"  [C-CODE/Vault] Also failed: {e2}")
        else:
            print(f"  [C-CODE/Vault] Failed: {e}")

    # Source 2: bigcode/starcoderdata (gated — needs HF auth)
    if count < limit:
        remaining = limit - count
        print(f"  [C-CODE] Need {remaining:,} more. Trying bigcode/starcoderdata (gated)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("bigcode/starcoderdata", "c", split="train",
                             streaming=True)
            mode = 'a' if count > 0 else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    text = ex.get('content', ex.get('text', ''))
                    if text and len(text) > 50:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('C-CODE/StarCoder', count)
            if count > 0:
                print(f"  [C-CODE/StarCoder] Total: {count:,} files")
        except Exception as e:
            print(f"  [C-CODE/StarCoder] Failed (gated): {e}")
            if 'gated' in str(e).lower() or 'access' in str(e).lower():
                print(f"    FIX: huggingface-cli login + visit https://huggingface.co/datasets/bigcode/starcoderdata")

    # Source 3: bigcode/the-stack-smol (gated)
    if count < limit:
        remaining = limit - count
        print(f"  [C-CODE] Need {remaining:,} more. Trying the-stack-smol (gated)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("bigcode/the-stack-smol", data_dir="data/c",
                             split="train", streaming=True)
            mode = 'a' if count > 0 else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    text = ex.get('content', ex.get('text', ''))
                    if text and len(text) > 50:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('C-CODE/StackSmol', count)
            if count > 0:
                print(f"  [C-CODE/StackSmol] Total: {count:,} files")
        except Exception as e:
            print(f"  [C-CODE/StackSmol] Failed (gated): {e}")

    # Summary
    if count == 0:
        print(f"[C-CODE] FAILED: No code data downloaded!")
        print(f"  CRITICAL: The model needs C code to meet its design goals.")
        print(f"  Easiest fix: huggingface-cli login")
        print(f"  Then visit https://huggingface.co/datasets/bigcode/starcoderdata")
        print(f"  Accept the terms, then re-run: python download.py --sources c_code")
    elif count < limit:
        print(f"[C-CODE] Partial: {count:,}/{limit:,} files. Training will work with less code data.")
    else:
        print(f"[C-CODE] Done: {count:,} files")


def download_other_code(limit):
    """Multi-language code (Python, JavaScript, Shell, Rust).
    Tries truly open-access sources first, then gated ones.

    CodeSearchNet: Python, JavaScript, Ruby, Go, Java, PHP (OPEN, no auth)
    The Vault: Python, JavaScript, etc. (OPEN, no auth)
    """
    out_path = os.path.join(DIRS['other_code'], 'multi_code.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[OTHER-CODE] Already downloaded, skipping")
        return

    print(f"[OTHER-CODE] Starting ({limit:,} docs)...")
    count = 0
    per_lang = limit // 4

    languages = [
        ("python", "Python"),
        ("javascript", "JavaScript"),
        ("shell", "Shell/Bash"),
        ("rust", "Rust"),
    ]

    with open(out_path, 'w', encoding='utf-8') as f:
        for lang_key, lang_name in languages:
            lang_count = 0

            # Source 1: The Vault (truly open access)
            if lang_key in ('python', 'javascript', 'rust'):
                vault_lang = lang_key
                try:
                    from datasets import load_dataset
                    print(f"  [OTHER-CODE/{lang_name}] Trying the-vault...")
                    ds = load_dataset("Fsoft-AIC/the-vault",
                                     f"language={vault_lang}",
                                     split="train", streaming=True,
                                     trust_remote_code=True)
                    for ex in ds:
                        text = ex.get('code', ex.get('content', ex.get('text', '')))
                        if text and len(text) > 50:
                            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                            lang_count += 1
                            count += 1
                        if lang_count >= per_lang:
                            break
                        _progress_log(f'OTHER-CODE/{lang_name}/Vault', lang_count, 10000)
                except Exception as e:
                    # Try without config if the config syntax failed
                    print(f"  [OTHER-CODE/{lang_name}/Vault] Failed: {e}")

            # Source 2: CodeSearchNet (open access, has Python/JS/Ruby/Go/Java/PHP)
            if lang_count < per_lang and lang_key in ('python', 'javascript'):
                csn_lang = lang_key
                try:
                    from datasets import load_dataset
                    print(f"  [OTHER-CODE/{lang_name}] Trying code_search_net...")
                    ds = load_dataset("code_search_net", csn_lang,
                                     split="train", streaming=True)
                    for ex in ds:
                        text = ex.get('code', ex.get('func_code_string',
                                    ex.get('content', ex.get('text', ''))))
                        if text and len(text) > 50:
                            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                            lang_count += 1
                            count += 1
                        if lang_count >= per_lang:
                            break
                        _progress_log(f'OTHER-CODE/{lang_name}/CSN', lang_count, 10000)
                except Exception as e:
                    print(f"  [OTHER-CODE/{lang_name}/CSN] Failed: {e}")

            # Source 3: bigcode/starcoderdata (gated - needs HF auth)
            if lang_count < per_lang:
                try:
                    from datasets import load_dataset
                    print(f"  [OTHER-CODE/{lang_name}] Trying starcoderdata (gated)...")
                    ds = load_dataset("bigcode/starcoderdata", lang_key,
                                     split="train", streaming=True)
                    for ex in ds:
                        text = ex.get('content', ex.get('text', ''))
                        if text and len(text) > 50:
                            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                            lang_count += 1
                            count += 1
                        if lang_count >= per_lang:
                            break
                        _progress_log(f'OTHER-CODE/{lang_name}/StarCoder', lang_count, 10000)
                except Exception as e:
                    print(f"  [OTHER-CODE/{lang_name}/StarCoder] Failed (gated): {e}")

            print(f"  [OTHER-CODE] {lang_name}: {lang_count:,} files")

    if count == 0:
        print(f"[OTHER-CODE] FAILED: No code data downloaded!")
        print(f"  FIX: huggingface-cli login, then visit:")
        print(f"  https://huggingface.co/datasets/bigcode/starcoderdata")
    else:
        print(f"[OTHER-CODE] Done: {count:,} files")


# ─── Download Dispatcher ──────────────────────────────────────────────────────

DOWNLOADERS = {
    'italian_oscar':   download_italian_oscar,
    'italian_wiki':    download_italian_wiki,
    'italian_fineweb': download_italian_fineweb,
    'english_web':     download_english_web,
    'english_wiki':    download_english_wiki,
    'english_edu':     download_english_edu,
    'c_code':          download_c_code,
    'other_code':      download_other_code,
}


def run_downloads(tier, sources=None, parallel=True):
    """Download all data sources for the given tier."""
    limits = TIERS[tier]
    total = sum(limits.values())

    print(f"\n{'='*60}")
    print(f"Download Tier: {tier.upper()}")
    print(f"Target: ~{total:,} documents")
    print(f"Mode: {'Sequential' if not parallel else 'Parallel'}")
    if not HAS_LANGDETECT:
        print(f"NOTE: pip install langdetect for better Italian filtering")
    print(f"{'='*60}\n")

    if sources:
        limits = {k: v for k, v in limits.items() if k in sources}

    if parallel:
        from multiprocessing import Process
        processes = []
        for source, limit in limits.items():
            downloader = DOWNLOADERS[source]
            p = Process(target=downloader, args=(limit,))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
    else:
        for source, limit in limits.items():
            DOWNLOADERS[source](limit)

    print(f"\n{'='*60}")
    print("All downloads complete!")

    # Print summary
    print(f"\nDownload summary:")
    any_empty = False
    for source, dir_path in DIRS.items():
        if source not in limits:
            continue
        for fname in sorted(os.listdir(dir_path) if os.path.exists(dir_path) else []):
            if fname.endswith('.jsonl'):
                fpath = os.path.join(dir_path, fname)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
                status = "OK" if lines > 0 else "EMPTY"
                if lines == 0:
                    any_empty = True
                print(f"  {source:20s}: {lines:>10,} docs  [{status}]")

    if any_empty:
        print(f"\nWARNING: Some sources downloaded 0 files!")
        print(f"  FIX: huggingface-cli login")
        print(f"  Then accept terms at: https://huggingface.co/datasets/bigcode/starcoderdata")
        print(f"  Re-run: python download.py --tier {tier} --sequential")

    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download training data for Italian+C SLM")
    parser.add_argument('--tier', choices=['smoke', 'quick', 'standard', 'full', 'max'],
                        default='quick',
                        help='Download tier: smoke (~28k), quick (~250k), standard (~1.5M), full (~3.1M), max (~7.2M)')
    parser.add_argument('--sources', nargs='*', default=None,
                        help='Specific sources to download (e.g., --sources italian_oscar c_code)')
    parser.add_argument('--sequential', action='store_true',
                        help='Download sequentially (recommended on Windows)')
    args = parser.parse_args()

    ensure_dirs()
    # Default to sequential on Windows (multiprocessing issues with HF datasets)
    use_parallel = not args.sequential and not IS_WINDOWS
    run_downloads(args.tier, args.sources, parallel=use_parallel)
