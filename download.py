"""
download.py v3.5 - Download training data with tiered levels and Italian + C focus.

Tier system (use --tier flag):
  smoke    - ~30k docs total, ~5 min   (pipeline testing only)
  quick    - ~300k docs total, ~30 min  (first real training)
  standard - ~1.5M docs total, ~4 hrs   (production quality)
  full     - ~3.3M docs total, ~10 hrs  (near Chinchilla-optimal)
  max      - ~7.5M docs total, ~28 hrs  (maximum intelligence)

Data mix is optimised for Italian language + C programming:
  - Italian: mc4 Italian + Italian Wikipedia + Italian FineWeb
  - C Code:  StarCoderData (gated) + The-Stack-dedup (gated) + GitHub-Code (open)
  - C++ Code: StarCoderData (gated) + GitHub-Code (open)
  - Other Code: CodeSearchNet (Python, JS) + GitHub-Code (Rust)
  - English: OpenWebText + English Wikipedia + FineWeb-Edu

Key fixes in v3.5:
  + FIXED: C code download was completely broken (got 0.8% of target)
  + FIXED: Fsoft-AIC/the-vault with language=c config is BROKEN (only 'default' exists)
  + FIXED: bigcode/starcoderdata now PRIMARY C source (Parquet-based, no loading script)
  + FIXED: Added HF auth check with --check-auth flag
  + FIXED: Multiple fallback sources for C code (gated -> open -> small)
  + ADDED: C++ as separate download category (shares syntax with C)
  + ADDED: codeparrot/github-code as open-access fallback
  + ADDED: Fsoft-AIC/the-vault "default" config with language filtering
  + REMOVED: bigcode/the-stack-smol for C (only has 9,975 C files - useless)
  + Clear instructions for huggingface-cli login

IMPORTANT: For C/C++ code, you MUST authenticate with HuggingFace:
  1. Go to https://huggingface.co/settings/tokens and create a token
  2. Run: huggingface-cli login
  3. Accept terms at: https://huggingface.co/datasets/bigcode/starcoderdata
  4. (Optional) Also accept: https://huggingface.co/datasets/bigcode/the-stack-dedup
  5. Re-run this script

  Without auth, C code download falls back to smaller open sources only.
"""

import os
import sys
import json
import argparse
import platform
import warnings
import multiprocessing as mp
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
        'cpp_code':           2_000,
        'other_code':         3_000,
    },
    'quick': {
        'italian_oscar':      30_000,
        'italian_wiki':       15_000,
        'italian_fineweb':    15_000,
        'english_web':        60_000,
        'english_wiki':       30_000,
        'english_edu':        30_000,
        'c_code':             40_000,
        'cpp_code':           15_000,
        'other_code':         15_000,
    },
    'standard': {
        'italian_oscar':      200_000,
        'italian_wiki':       60_000,
        'italian_fineweb':    150_000,
        'english_web':        400_000,
        'english_wiki':       120_000,
        'english_edu':        250_000,
        'c_code':             200_000,
        'cpp_code':           80_000,
        'other_code':         80_000,
    },
    'full': {
        'italian_oscar':      500_000,
        'italian_wiki':       60_000,
        'italian_fineweb':    250_000,
        'english_web':        800_000,
        'english_wiki':       300_000,
        'english_edu':        600_000,
        'c_code':             400_000,
        'cpp_code':           150_000,
        'other_code':         150_000,
    },
    'max': {
        'italian_oscar':      1_000_000,
        'italian_wiki':       60_000,
        'italian_fineweb':    600_000,
        'english_web':        1_200_000,  # Moderate — English not primary goal
        'english_wiki':       400_000,
        'english_edu':        800_000,
        'c_code':             800_000,    # BOOSTED — primary skill
        'cpp_code':           300_000,    # NEW — shares syntax with C
        'other_code':         300_000,    # Python, JS, Rust — general code
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
    'cpp_code':        'data_raw/code/cpp',
    'other_code':      'data_raw/code/other',
}


def ensure_dirs():
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)


def already_downloaded(path, expected_lines):
    """Check if file exists and has approximately the expected number of lines.
    Also verifies last line is valid JSON to catch truncated downloads."""
    if not os.path.exists(path):
        return False
    try:
        count = 0
        last_line = None
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                last_line = line
                count += 1
        if count < expected_lines * 0.8:
            return False
        # Verify last line is valid JSON (catches truncated files)
        if last_line:
            json.loads(last_line)
        return True
    except (json.JSONDecodeError, Exception):
        print(f"  WARNING: {path} appears truncated, re-downloading")
        return False


def _progress_log(source, count, interval=25000):
    """Log progress at regular intervals."""
    if count > 0 and count % interval == 0:
        print(f"  [{source}] {count:,} docs")


# ─── HF Auth Check ─────────────────────────────────────────────────────────────

def check_hf_auth():
    """Check if HuggingFace authentication is available.
    Returns True if auth token is found.

    Supports both old (HfFolder) and new (get_token/HfApi) huggingface_hub APIs.
    """
    token = None

    # Method 1: New API (huggingface_hub >= 1.0)
    try:
        from huggingface_hub.utils import get_token as _get_token
        token = _get_token()
    except (ImportError, AttributeError):
        pass

    # Method 2: Old API (huggingface_hub < 1.0)
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except (ImportError, AttributeError):
            pass

    # Method 3: Direct file check (works regardless of library version)
    if not token:
        # Windows: C:\Users\<user>\.cache\huggingface\token
        # Linux: ~/.cache/huggingface/token  or  ~/.huggingface/token
        for token_path in [
            os.path.expanduser("~/.cache/huggingface/token"),
            os.path.expanduser("~/.huggingface/token"),
        ]:
            if os.path.exists(token_path):
                try:
                    with open(token_path, 'r') as f:
                        token = f.read().strip()
                    if token:
                        break
                except Exception:
                    pass

    if token:
        print("HF auth: Token found (authenticated)")
        return True
    else:
        print("HF auth: No token found (not authenticated)")
        return False


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
    """Italian web crawl data. Sources ordered by quality:
    1. FineWeb-2 ita_Latn (best quality, has language label)
    2. Italian Wikipedia (high-quality encyclopedic)
    3. OSCAR-2301 (gated but clean)
    4. mc4 (last resort only — severe quality issues)
    """
    out_path = os.path.join(DIRS['italian_oscar'], 'oscar_it.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[IT-OSCAR] Already downloaded, skipping")
        return

    print(f"[IT-OSCAR] Starting ({limit:,} docs)...")
    count = 0

    # Primary: FineWeb-2 ita_Latn (best quality, has language label)
    try:
        from datasets import load_dataset
        print(f"[IT-OSCAR] Trying FineWeb-2 ita_Latn (best quality)...")
        ds = load_dataset("HuggingFaceFW/fineweb-2", "ita_Latn", split="train",
                         streaming=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for ex in ds:
                if ex.get('score', 1.0) < 0.7:
                    continue
                text = ex.get('text', '')
                if text and len(text) > 100:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('IT-OSCAR/FineWeb2', count)
        print(f"[IT-OSCAR] FineWeb-2: {count:,} docs")
    except Exception as e:
        print(f"[IT-OSCAR] FineWeb-2 failed: {e}")

    # Fallback 1: Italian Wikipedia (high-quality encyclopedic)
    if count < limit:
        try:
            from datasets import load_dataset
            print(f"[IT-OSCAR] Trying Italian Wikipedia...")
            ds = load_dataset("wikimedia/wikipedia", "20231101.it", split="train",
                             streaming=True)
            mode = 'a' if count > 0 else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    text = ex.get('text', '')
                    if text and len(text) > 100:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('IT-OSCAR/Wiki', count)
            print(f"[IT-OSCAR] With Wikipedia fallback: {count:,} docs")
        except Exception as e:
            print(f"[IT-OSCAR] Italian Wikipedia failed: {e}")

    # Fallback 2: OSCAR-2301 (gated - needs HF auth)
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

    # Fallback 3 (last resort): mc4 Italian (severe quality issues)
    if count < limit:
        try:
            from datasets import load_dataset
            print(f"[IT-OSCAR] Trying mc4 (LAST RESORT — severe quality issues)...")
            ds = load_dataset("allenai/c4", "it", split="train", streaming=True)
            mode = 'a' if count > 0 else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    text = ex.get('text', '')
                    if text and len(text) > 100:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('IT-OSCAR/mc4', count)
            print(f"[IT-OSCAR] With mc4 fallback: {count:,} docs")
        except Exception as e:
            print(f"[IT-OSCAR] mc4 also failed: {e}")

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
            print("  SKIP: langdetect is REQUIRED for FineWeb-Edu Italian fallback.")
            print("  The FineWeb-Edu dataset has no language label, so we must detect Italian.")
            print("  Install with: pip install langdetect")
            print(f"[IT-FINEWEB] Done: 0 docs (langdetect not installed, fallback skipped)")
            return
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


# ─── C Code Download (FIXED - multiple working sources) ───────────────────────

def download_c_code(limit):
    """C programming code from multiple sources with fallbacks.

    Source priority (tries each in order until limit is met):
      1. bigcode/starcoderdata config "c"   — GATED, ~1.7M C files, Parquet (BEST)
      2. bigcode/the-stack-dedup data/c     — GATED, ~500k C files, high quality
      3. codeparrot/github-code languages=C  — OPEN, ~14M C files (may need trust_remote_code)
      4. Fsoft-AIC/the-vault "default"       — OPEN, filter for C language (slow streaming)
      5. bigcode/the-stack-smol data/c       — GATED, ~10k C files (last resort)

    For sources 1-2: Run `huggingface-cli login` and accept terms at:
      https://huggingface.co/datasets/bigcode/starcoderdata
      https://huggingface.co/datasets/bigcode/the-stack-dedup
    """
    out_path = os.path.join(DIRS['c_code'], 'c_code.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[C-CODE] Already downloaded, skipping")
        return

    print(f"[C-CODE] Starting ({limit:,} docs)...")
    count = 0

    # ── Source 1: bigcode/starcoderdata (GATED — Parquet, no loading script) ──
    # This is the BEST source: millions of C files, Parquet-based (no loading script issues),
    # properly curated. REQUIRES: huggingface-cli login + accept T&C
    try:
        from datasets import load_dataset
        print(f"  [C-CODE] Trying bigcode/starcoderdata (gated, Parquet, ~1.7M C files)...")
        ds = load_dataset("bigcode/starcoderdata", "c", split="train", streaming=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for ex in ds:
                text = ex.get('content', ex.get('code', ex.get('text', '')))
                if text and len(text) > 50:
                    # Filter out auto-generated files
                    first_line = text.split('\n')[0] if text else ''
                    if any(skip in first_line.lower() for skip in
                           ['auto-generated', 'generated by', 'do not edit', 'automatically generated']):
                        continue
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('C-CODE/StarCoder', count)
        if count > 0:
            print(f"  [C-CODE/StarCoder] Got {count:,} files")
    except Exception as e:
        err_msg = str(e).lower()
        if 'gated' in err_msg or 'access' in err_msg:
            print(f"  [C-CODE/StarCoder] GATED — needs HF authentication")
            print(f"    FIX: huggingface-cli login")
            print(f"    Then accept terms at: https://huggingface.co/datasets/bigcode/starcoderdata")
        else:
            print(f"  [C-CODE/StarCoder] Failed: {e}")

    # ── Source 2: bigcode/the-stack-dedup (GATED — high quality, permissive license) ──
    if count < limit:
        remaining = limit - count
        print(f"  [C-CODE] Need {remaining:,} more. Trying the-stack-dedup (gated)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/c",
                             split="train", streaming=True, trust_remote_code=True)
            mode = 'a' if count > 0 else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    text = ex.get('content', ex.get('code', ex.get('text', '')))
                    if text and len(text) > 50:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('C-CODE/StackDedup', count)
            if count > 0:
                print(f"  [C-CODE/StackDedup] Total: {count:,} files")
        except Exception as e:
            err_msg = str(e).lower()
            if 'gated' in err_msg or 'access' in err_msg:
                print(f"  [C-CODE/StackDedup] GATED — needs HF authentication")
                print(f"    FIX: Accept terms at: https://huggingface.co/datasets/bigcode/the-stack-dedup")
            elif 'trust_remote_code' in err_msg or 'loading script' in err_msg:
                # Try without trust_remote_code
                print(f"  [C-CODE/StackDedup] Loading script issue, trying without trust_remote_code...")
                try:
                    from datasets import load_dataset
                    ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/c",
                                     split="train", streaming=True)
                    mode = 'a' if count > 0 else 'w'
                    with open(out_path, mode, encoding='utf-8') as f:
                        for ex in ds:
                            text = ex.get('content', ex.get('code', ex.get('text', '')))
                            if text and len(text) > 50:
                                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                                count += 1
                            if count >= limit:
                                break
                            _progress_log('C-CODE/StackDedup2', count)
                    if count > 0:
                        print(f"  [C-CODE/StackDedup] Total: {count:,} files")
                except Exception as e2:
                    print(f"  [C-CODE/StackDedup] Also failed: {e2}")
            else:
                print(f"  [C-CODE/StackDedup] Failed: {e}")

    # ── Source 3: codeparrot/github-code (OPEN — huge but uses loading script) ──
    if count < limit:
        remaining = limit - count
        print(f"  [C-CODE] Need {remaining:,} more. Trying codeparrot/github-code (OPEN)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("codeparrot/github-code", streaming=True, split="train",
                             languages=["C"], trust_remote_code=True)
            mode = 'a' if count > 0 else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    text = ex.get('code', ex.get('content', ex.get('text', '')))
                    if text and len(text) > 50:
                        first_line = text.split('\n')[0] if text else ''
                        if any(skip in first_line.lower() for skip in
                               ['auto-generated', 'generated by', 'do not edit']):
                            continue
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('C-CODE/GitHub', count)
            if count > 0:
                print(f"  [C-CODE/GitHub] Total: {count:,} files")
        except Exception as e:
            err_msg = str(e)
            print(f"  [C-CODE/GitHub] Failed: {e}")
            # Try with config name instead of languages parameter
            if 'languages' in err_msg or 'trust_remote_code' in err_msg.lower():
                print(f"  [C-CODE/GitHub] Trying with config 'C' instead of languages param...")
                try:
                    from datasets import load_dataset
                    ds = load_dataset("codeparrot/github-code", "C",
                                     split="train", streaming=True)
                    mode = 'a' if count > 0 else 'w'
                    with open(out_path, mode, encoding='utf-8') as f:
                        for ex in ds:
                            text = ex.get('code', ex.get('content', ex.get('text', '')))
                            if text and len(text) > 50:
                                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                                count += 1
                            if count >= limit:
                                break
                            _progress_log('C-CODE/GitHub2', count)
                    if count > 0:
                        print(f"  [C-CODE/GitHub] Total: {count:,} files (config mode)")
                except Exception as e2:
                    print(f"  [C-CODE/GitHub2] Also failed: {e2}")

    # ── Source 4: Fsoft-AIC/the-vault "default" config (OPEN — filter for C) ──
    # The old `language=c` config is BROKEN. But `default` config exists and contains
    # all languages. We stream through and keep only C code.
    # SLOW but reliable — streams through all languages to find C code.
    if count < limit:
        remaining = limit - count
        print(f"  [C-CODE] Need {remaining:,} more. Trying the-vault 'default' config (OPEN, slow)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Fsoft-AIC/the-vault", "default", split="train",
                             streaming=True, trust_remote_code=True)
            mode = 'a' if count > 0 else 'w'
            scanned = 0
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    scanned += 1
                    # Filter for C language
                    lang = ex.get('language', ex.get('lang', '')).lower()
                    if lang != 'c':
                        continue
                    text = ex.get('code', ex.get('content', ex.get('text', '')))
                    if text and len(text) > 50:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    if scanned % 100_000 == 0:
                        print(f"  [C-CODE/Vault] Scanned {scanned:,} files, found {count:,} C files")
            if count > 0:
                print(f"  [C-CODE/Vault] Total: {count:,} files (scanned {scanned:,})")
        except Exception as e:
            print(f"  [C-CODE/Vault] Failed: {e}")

    # ── Source 5: bigcode/the-stack-smol (GATED — only ~10k C files, last resort) ──
    if count < limit:
        remaining = limit - count
        print(f"  [C-CODE] Need {remaining:,} more. Trying the-stack-smol (only ~10k C files)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("bigcode/the-stack-smol", data_dir="data/c",
                             split="train", streaming=True)
            mode = 'a' if count > 0 else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    text = ex.get('content', ex.get('code', ex.get('text', '')))
                    if text and len(text) > 50:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('C-CODE/StackSmol', count)
            if count > 0:
                print(f"  [C-CODE/StackSmol] Total: {count:,} files")
        except Exception as e:
            print(f"  [C-CODE/StackSmol] Failed: {e}")

    # Summary
    if count == 0:
        print(f"\n[C-CODE] CRITICAL: No C code downloaded!")
        print(f"  The model CANNOT learn C programming without C code data.")
        print(f"")
        print(f"  EASIEST FIX (5 minutes):")
        print(f"    1. Go to https://huggingface.co/settings/tokens")
        print(f"    2. Create a Read token")
        print(f"    3. Run: huggingface-cli login")
        print(f"    4. Visit https://huggingface.co/datasets/bigcode/starcoderdata")
        print(f"    5. Accept the terms")
        print(f"    6. Re-run: python download.py --sources c_code --tier max")
    elif count < limit:
        pct = count / limit * 100
        print(f"[C-CODE] Partial: {count:,}/{limit:,} files ({pct:.0f}%)")
        if pct < 50:
            print(f"  WARNING: Less than 50% of target C code. Model C quality will be limited.")
            print(f"  FIX: huggingface-cli login + accept terms at bigcode/starcoderdata")
    else:
        print(f"[C-CODE] Done: {count:,} files")


# ─── C++ Code Download (NEW — shares syntax with C) ───────────────────────────

def download_cpp_code(limit):
    """C++ programming code (complements C understanding, shares syntax).

    C++ code shares: #include, preprocessor directives, types, pointers,
    memory management, and low-level system programming patterns.
    Having C++ data dramatically improves C generation quality.

    Sources (tried in order until limit is met):
      1. bigcode/the-stack-dedup data/cpp    — GATED, ~500k C++ files (WORKS — same as C source)
      2. bigcode/the-stack-dedup data/c++    — GATED, alt directory name
      3. bigcode/starcoderdata default+filter — GATED, stream + filter by language (SLOW)
      4. Fsoft-AIC/the-vault default+filter   — OPEN, stream + filter by language (SLOW)

    NOTE: bigcode/starcoderdata per-language configs ("c++") NO LONGER EXIST.
    The dataset only has a "default" config now. To get C++ from starcoderdata,
    we must stream the "default" config and filter by language field — very slow.

    NOTE: codeparrot/github-code uses a loading script (deprecated in datasets>=3.0).
    It cannot be used with current versions of the `datasets` library.
    """
    out_path = os.path.join(DIRS['cpp_code'], 'cpp_code.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[C++-CODE] Already downloaded, skipping")
        return

    print(f"[C++-CODE] Starting ({limit:,} docs)...")
    count = 0

    # Source 1: bigcode/the-stack-dedup data/cpp (GATED — same source that worked for C!)
    # The directory is "cpp" not "c++" in the-stack-dedup
    try:
        from datasets import load_dataset
        print(f"  [C++-CODE] Trying the-stack-dedup data/cpp (gated)...")
        ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/cpp",
                         split="train", streaming=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for ex in ds:
                text = ex.get('content', ex.get('code', ex.get('text', '')))
                if text and len(text) > 50:
                    f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                    count += 1
                if count >= limit:
                    break
                _progress_log('C++-CODE/StackDedup-cpp', count)
        if count > 0:
            print(f"  [C++-CODE/StackDedup-cpp] Got {count:,} files")
    except Exception as e:
        err_msg = str(e).lower()
        if 'does not contain any data' in err_msg or 'not found' in err_msg:
            print(f"  [C++-CODE/StackDedup-cpp] Directory 'data/cpp' not found, trying 'data/c++'...")
        elif 'gated' in err_msg or 'access' in err_msg:
            print(f"  [C++-CODE/StackDedup-cpp] GATED — needs HF authentication")
            print(f"    FIX: Accept terms at: https://huggingface.co/datasets/bigcode/the-stack-dedup")
        else:
            print(f"  [C++-CODE/StackDedup-cpp] Failed: {e}")

    # Source 2: bigcode/the-stack-dedup data/c++ (alternative directory name)
    if count < limit:
        remaining = limit - count
        print(f"  [C++-CODE] Need {remaining:,} more. Trying the-stack-dedup data/c++...")
        try:
            from datasets import load_dataset
            ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/c++",
                             split="train", streaming=True)
            mode = 'a' if count > 0 else 'w'
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    text = ex.get('content', ex.get('code', ex.get('text', '')))
                    if text and len(text) > 50:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    _progress_log('C++-CODE/StackDedup-c++', count)
            if count > 0:
                print(f"  [C++-CODE/StackDedup-c++] Total: {count:,} files")
        except Exception as e:
            err_msg = str(e).lower()
            if 'does not contain any data' in err_msg or 'not found' in err_msg:
                print(f"  [C++-CODE/StackDedup-c++] Directory 'data/c++' also not found")
            else:
                print(f"  [C++-CODE/StackDedup-c++] Failed: {e}")

    # Source 3: bigcode/the-stack-dedup — try listing available subdirectories
    if count < limit:
        remaining = limit - count
        print(f"  [C++-CODE] Need {remaining:,} more. Trying to find C++ in the-stack-dedup...")
        # Try common directory names for C++ in the-stack-dedup
        for dir_name in ['data/cplusplus', 'data/C++', 'data/CPlusPlus']:
            if count >= limit:
                break
            try:
                from datasets import load_dataset
                print(f"  [C++-CODE] Trying the-stack-dedup {dir_name}...")
                ds = load_dataset("bigcode/the-stack-dedup", data_dir=dir_name,
                                 split="train", streaming=True)
                mode = 'a' if count > 0 else 'w'
                with open(out_path, mode, encoding='utf-8') as f:
                    for ex in ds:
                        text = ex.get('content', ex.get('code', ex.get('text', '')))
                        if text and len(text) > 50:
                            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                            count += 1
                        if count >= limit:
                            break
                        _progress_log(f'C++-CODE/StackDedup-{dir_name}', count)
                if count > 0:
                    print(f"  [C++-CODE/StackDedup-{dir_name}] Total: {count:,} files")
                    break  # Found it!
            except Exception:
                continue  # Try next directory name

    # Source 4: bigcode/starcoderdata "default" config with language filtering (SLOW)
    # starcoderdata per-language configs no longer exist. Must stream "default" and filter.
    if count < limit:
        remaining = limit - count
        print(f"  [C++-CODE] Need {remaining:,} more. Trying starcoderdata 'default' + language filter (SLOW)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("bigcode/starcoderdata", "default", split="train", streaming=True)
            mode = 'a' if count > 0 else 'w'
            scanned = 0
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    scanned += 1
                    # Filter for C++ language
                    lang = ex.get('language', ex.get('lang', '')).lower()
                    if lang not in ('c++', 'cpp', 'cplusplus'):
                        continue
                    text = ex.get('content', ex.get('code', ex.get('text', '')))
                    if text and len(text) > 50:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    if scanned % 500_000 == 0:
                        print(f"  [C++-CODE/StarCoder] Scanned {scanned:,} files, found {count:,} C++ files")
            if count > 0:
                print(f"  [C++-CODE/StarCoder] Total: {count:,} files (scanned {scanned:,})")
        except Exception as e:
            print(f"  [C++-CODE/StarCoder] Failed: {e}")

    # Source 5: Fsoft-AIC/the-vault "default" config with language filtering (SLOW)
    if count < limit:
        remaining = limit - count
        print(f"  [C++-CODE] Need {remaining:,} more. Trying the-vault 'default' + language filter (SLOW)...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Fsoft-AIC/the-vault", "default", split="train",
                             streaming=True, trust_remote_code=True)
            mode = 'a' if count > 0 else 'w'
            scanned = 0
            with open(out_path, mode, encoding='utf-8') as f:
                for ex in ds:
                    scanned += 1
                    lang = ex.get('language', ex.get('lang', '')).lower()
                    if lang not in ('c++', 'cpp', 'cplusplus'):
                        continue
                    text = ex.get('code', ex.get('content', ex.get('text', '')))
                    if text and len(text) > 50:
                        f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                        count += 1
                    if count >= limit:
                        break
                    if scanned % 100_000 == 0:
                        print(f"  [C++-CODE/Vault] Scanned {scanned:,} files, found {count:,} C++ files")
            if count > 0:
                print(f"  [C++-CODE/Vault] Total: {count:,} files (scanned {scanned:,})")
        except Exception as e:
            print(f"  [C++-CODE/Vault] Failed: {e}")

    if count == 0:
        print(f"\n[C++-CODE] FAILED: No C++ data downloaded!")
        print(f"  C++ is not critical but improves C quality (shares syntax).")
        print(f"  The model will still train — just with less C/C++ synergy.")
        print(f"  If you want C++: try `python download.py --sources cpp_code --tier standard`")
    elif count < limit:
        pct = count / limit * 100
        print(f"[C++-CODE] Partial: {count:,}/{limit:,} files ({pct:.0f}%)")
    else:
        print(f"[C++-CODE] Done: {count:,} files")


# ─── Other Code Download ──────────────────────────────────────────────────────

def download_other_code(limit):
    """Multi-language code (Python, JavaScript, Rust).
    Tries truly open-access sources first, then gated ones.

    Sources:
    - code_search_net: Python, JavaScript (OPEN, curated)
    - codeparrot/github-code: Rust (OPEN)
    - bigcode/starcoderdata: all languages (GATED, fallback)
    """
    out_path = os.path.join(DIRS['other_code'], 'multi_code.jsonl')
    if already_downloaded(out_path, limit):
        print(f"[OTHER-CODE] Already downloaded, skipping")
        return

    print(f"[OTHER-CODE] Starting ({limit:,} docs)...")
    count = 0
    per_lang = limit // 3

    languages = [
        ("python", "Python"),
        ("javascript", "JavaScript"),
        ("rust", "Rust"),
    ]

    with open(out_path, 'w', encoding='utf-8') as f:
        for lang_key, lang_name in languages:
            lang_count = 0

            # Source 1: CodeSearchNet (OPEN, curated — Python & JavaScript only)
            if lang_key in ('python', 'javascript'):
                try:
                    from datasets import load_dataset
                    print(f"  [OTHER-CODE/{lang_name}] Trying code_search_net...")
                    ds = load_dataset("code_search_net", lang_key,
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

            # Source 2: codeparrot/github-code (OPEN — for Rust and overflow)
            if lang_count < per_lang:
                try:
                    from datasets import load_dataset
                    print(f"  [OTHER-CODE/{lang_name}] Trying github-code (OPEN)...")
                    # Map lang_key to github-code language name
                    gh_lang = lang_name  # "Python", "JavaScript", "Rust"
                    ds = load_dataset("codeparrot/github-code", streaming=True, split="train",
                                     languages=[gh_lang], trust_remote_code=True)
                    for ex in ds:
                        text = ex.get('code', ex.get('content', ex.get('text', '')))
                        if text and len(text) > 50:
                            f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                            lang_count += 1
                            count += 1
                        if lang_count >= per_lang:
                            break
                        _progress_log(f'OTHER-CODE/{lang_name}/GitHub', lang_count, 10000)
                except Exception as e:
                    print(f"  [OTHER-CODE/{lang_name}/GitHub] Failed: {e}")
                    # Try with config name
                    try:
                        from datasets import load_dataset
                        ds = load_dataset("codeparrot/github-code", gh_lang,
                                         split="train", streaming=True)
                        for ex in ds:
                            text = ex.get('code', ex.get('content', ex.get('text', '')))
                            if text and len(text) > 50:
                                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                                lang_count += 1
                                count += 1
                            if lang_count >= per_lang:
                                break
                            _progress_log(f'OTHER-CODE/{lang_name}/GitHub2', lang_count, 10000)
                    except Exception as e2:
                        print(f"  [OTHER-CODE/{lang_name}/GitHub2] Also failed: {e2}")

            # Source 3: bigcode/starcoderdata (GATED — fallback)
            if lang_count < per_lang:
                try:
                    from datasets import load_dataset
                    print(f"  [OTHER-CODE/{lang_name}] Trying starcoderdata (gated)...")
                    ds = load_dataset("bigcode/starcoderdata", lang_key,
                                     split="train", streaming=True)
                    for ex in ds:
                        text = ex.get('content', ex.get('code', ex.get('text', '')))
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
        print(f"  FIX: pip install --upgrade datasets")
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
    'cpp_code':        download_cpp_code,
    'other_code':      download_other_code,
}


def run_downloads(tier, sources=None, parallel=False):
    """Download all data sources for the given tier."""
    limits = TIERS[tier]
    total = sum(limits.values())

    print(f"\n{'='*60}")
    print(f"Download Tier: {tier.upper()}")
    print(f"Target: ~{total:,} documents")
    print(f"Mode: {'Sequential' if not parallel else 'Parallel'}")

    # Check HF auth status
    has_auth = check_hf_auth()
    if not has_auth:
        print(f"\nWARNING: No HuggingFace authentication detected!")
        print(f"  C/C++ code download will be SEVERELY limited without auth.")
        print(f"  FIX: huggingface-cli login")
        print(f"  Then accept terms at: https://huggingface.co/datasets/bigcode/starcoderdata")
        print()

    if not HAS_LANGDETECT:
        print(f"NOTE: pip install langdetect for better Italian filtering")
    print(f"{'='*60}\n")

    if sources:
        limits = {k: v for k, v in limits.items() if k in sources}

    if parallel:
        ctx = mp.get_context('spawn')
        processes = []
        for source, limit in limits.items():
            downloader = DOWNLOADERS[source]
            p = ctx.Process(target=downloader, args=(limit,))
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
    any_below_target = False
    for source, dir_path in DIRS.items():
        if source not in limits:
            continue
        for fname in sorted(os.listdir(dir_path) if os.path.exists(dir_path) else []):
            if fname.endswith('.jsonl'):
                fpath = os.path.join(dir_path, fname)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = sum(1 for _ in f)
                target = limits[source]
                status = "OK" if lines > 0 else "EMPTY"
                if lines == 0:
                    any_empty = True
                elif lines < target * 0.5:
                    any_below_target = True
                    status = f"LOW ({lines/target*100:.0f}%)"
                print(f"  {source:20s}: {lines:>10,} docs  [{status}] (target: {target:,})")

    if any_empty:
        print(f"\nWARNING: Some sources downloaded 0 files!")
        print(f"  FIX: huggingface-cli login")
        print(f"  Then accept terms at: https://huggingface.co/datasets/bigcode/starcoderdata")
    if any_below_target:
        print(f"\nNOTE: Some sources are below 50% of target.")
        print(f"  Training will still work but model quality may be limited.")

    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download training data for Italian+C SLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download.py --tier quick          # Quick test (~300k docs)
  python download.py --tier max            # Full dataset (~7.5M docs)
  python download.py --sources c_code cpp_code --tier max  # C/C++ only
  python download.py --check-auth          # Check HF authentication

IMPORTANT: For C/C++ code, authenticate first:
  1. huggingface-cli login
  2. Accept terms at: https://huggingface.co/datasets/bigcode/starcoderdata
""")
    parser.add_argument('--tier', choices=['smoke', 'quick', 'standard', 'full', 'max'],
                        default='quick',
                        help='Download tier (default: quick)')
    parser.add_argument('--sources', nargs='*', default=None,
                        help='Specific sources to download (e.g., --sources c_code cpp_code)')
    parser.add_argument('--sequential', action='store_true',
                        help='Force sequential download (overrides --parallel)')
    parser.add_argument('--parallel', action='store_true',
                        help='Enable parallel download (uses spawn start method)')
    parser.add_argument('--check-auth', action='store_true',
                        help='Check HuggingFace authentication status and exit')
    args = parser.parse_args()

    if args.check_auth:
        has_auth = check_hf_auth()
        if has_auth:
            print("\nYou are authenticated. C/C++ code sources will work.")
            print("Make sure you've accepted terms at:")
            print("  https://huggingface.co/datasets/bigcode/starcoderdata")
        else:
            print("\nYou are NOT authenticated. C/C++ code will be limited.")
            print("Fix:")
            print("  1. Go to https://huggingface.co/settings/tokens")
            print("  2. Create a Read token")
            print("  3. Run: huggingface-cli login")
            print("  4. Accept terms at: https://huggingface.co/datasets/bigcode/starcoderdata")
        sys.exit(0 if has_auth else 1)

    ensure_dirs()
    # Default to sequential (safer with HF streaming datasets; use --parallel to override)
    use_parallel = not args.sequential and args.parallel
    run_downloads(args.tier, args.sources, parallel=use_parallel)
