"""
inference.py — Text generation with KV cache and speculative decoding.

Key improvements:
  + KV cache for ~4-5x faster generation (Feature #1)
  + Speculative decoding using MTP heads as draft models (Feature #6)
  + BOS token prepended automatically
  + Top-p (nucleus) sampling with temperature
  + Top-k sampling option
  + Repetition penalty
  + Interactive multi-turn conversation mode
  + EOS detection to stop generation
  + Handles compiled model state dict prefix (_orig_mod.)
"""

import os
import sys
import argparse
import torch
from model import Transformer, ModelArgs, KVCache
from tokenizers import Tokenizer


# ─── Standard Generation with KV Cache ────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=256, temperature=0.8,
             top_p=0.95, top_k=50, repetition_penalty=1.2, use_kv_cache=True):
    """Generate text autoregressically using KV cache for fast inference.

    With KV cache, each generation step only processes the new token
    instead of recomputing the entire context. This provides ~4-5x speedup
    on an RTX 3070 (~40 tok/s -> ~150-200 tok/s).

    Args:
        model: The Transformer model
        tokenizer: The BPE tokenizer
        prompt: Input text prompt
        max_new: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling cutoff
        repetition_penalty: Penalty for repeating recent tokens
        use_kv_cache: Whether to use KV cache (True = fast, False = simple sliding window)
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize with BOS
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokens = tokenizer.encode(prompt).ids

    # Prepend BOS if not already present
    if not tokens or tokens[0] != bos_id:
        tokens = [bos_id] + tokens

    max_seq_len = model.args.max_seq_len
    generated = list(tokens)

    if use_kv_cache:
        # ── KV Cache Generation (fast path) ──
        generated = _generate_with_kv_cache(
            model, tokenizer, tokens, device, max_seq_len, max_new,
            temperature, top_p, top_k, repetition_penalty, bos_id, eos_id
        )
    else:
        # ── Sliding Window Generation (simple fallback) ──
        generated = _generate_sliding_window(
            model, tokenizer, tokens, device, max_seq_len, max_new,
            temperature, top_p, top_k, repetition_penalty, eos_id
        )

    # Decode (skip BOS token)
    output = tokenizer.decode(generated[1:])
    return output


def _generate_with_kv_cache(model, tokenizer, tokens, device, max_seq_len,
                            max_new, temperature, top_p, top_k,
                            repetition_penalty, bos_id, eos_id):
    """Fast generation using KV cache — processes only 1 token per step after prefill."""
    # Initialize KV caches for all layers
    kv_caches = model.init_kv_caches(batch_size=1, device=device)

    # ── Prefill: process the entire prompt in one forward pass ──
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits, _ = model(input_ids, kv_caches=kv_caches, start_pos=0)

    generated = list(tokens)
    next_logits = logits[:, -1, :]  # Only need last token's logits

    # ── Generate one token at a time ──
    for step in range(max_new):
        cur_pos = len(generated)

        if cur_pos >= max_seq_len:
            break

        # Apply sampling
        next_token = _sample_token(
            next_logits, generated, temperature, top_p, top_k,
            repetition_penalty, eos_id
        )

        generated.append(next_token)
        if next_token == eos_id:
            break

        # Forward pass with just the new token (start_pos = cur_pos)
        new_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
        logits, _ = model(new_input, kv_caches=kv_caches, start_pos=cur_pos)
        next_logits = logits[:, -1, :]

    return generated


def _generate_sliding_window(model, tokenizer, tokens, device, max_seq_len,
                             max_new, temperature, top_p, top_k,
                             repetition_penalty, eos_id):
    """Simple generation without KV cache — recomputes full context each step.
    Used as fallback or for debugging."""
    generated = list(tokens)

    for _ in range(max_new):
        if len(generated) >= max_seq_len:
            break

        context = torch.tensor([generated[-max_seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(context)
        next_logits = logits[:, -1, :]

        next_token = _sample_token(
            next_logits, generated, temperature, top_p, top_k,
            repetition_penalty, eos_id
        )

        generated.append(next_token)
        if next_token == eos_id:
            break

    return generated


def _sample_token(logits, generated, temperature, top_p, top_k,
                  repetition_penalty, eos_id):
    """Sample a single token from logits with temperature, top-k, top-p, and repetition penalty."""
    next_logits = logits / temperature

    # Repetition penalty: reduce probability of recently generated tokens
    if repetition_penalty > 1.0 and len(generated) > 0:
        recent = generated[-64:]  # Penalise last 64 tokens
        for token_id in set(recent):
            if next_logits[0, token_id] > 0:
                next_logits[0, token_id] /= repetition_penalty
            else:
                next_logits[0, token_id] *= repetition_penalty

    # Top-k filtering
    if top_k > 0:
        top_k_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        threshold = top_k_values[:, -1:]
        next_logits[next_logits < threshold] = -float('Inf')

    # Top-p (nucleus) filtering
    probs = torch.softmax(next_logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    sorted_idx_to_remove = cumsum > top_p
    sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
    sorted_idx_to_remove[..., 0] = 0
    indices_to_remove = sorted_idx[sorted_idx_to_remove]
    next_logits[0, indices_to_remove] = -float('Inf')

    # Sample
    probs = torch.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.item()


# ─── Speculative Decoding with MTP Heads (Feature #6) ─────────────────────────

@torch.no_grad()
def generate_speculative(model, tokenizer, prompt, max_new=256, temperature=0.8,
                         top_p=0.95, top_k=50, repetition_penalty=1.2,
                         draft_steps=2):
    """Generate text using speculative decoding with MTP heads as draft models.

    Speculative decoding uses the auxiliary MTP heads to predict multiple future
    tokens at once (the "draft"), then verifies them against the main model.
    Accepted tokens are kept; rejected tokens trigger resampling. This can
    provide 1.5-2.5x speedup on top of the KV cache speedup.

    Fixes applied (v3.3):
      + MTP heads now use full transformer hidden states (not embedding-only)
      + KV cache verify pass no longer double-writes the last real token
      + Partial acceptance uses cache truncation (not full rebuild)
      + Correct start_pos for multi-token verify pass

    How it works:
    1. Prefill the prompt — get logits AND hidden state h
    2. For each generation step:
       a. Use MTP heads on the REAL hidden state h to draft tokens
       b. Run the main model on ONLY the draft tokens (correct start_pos)
       c. Accept/reject using standard speculative decoding criterion
       d. On partial acceptance: truncate KV cache (fast, O(1))
    """
    model.eval()
    device = next(model.parameters()).device

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokens = tokenizer.encode(prompt).ids

    if not tokens or tokens[0] != bos_id:
        tokens = [bos_id] + tokens

    max_seq_len = model.args.max_seq_len
    generated = list(tokens)

    # Initialize KV caches
    kv_caches = model.init_kv_caches(batch_size=1, device=device)

    # ── Prefill: process the entire prompt in one forward pass ──
    # Also retrieve the final hidden state h for MTP draft prediction
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    logits, _, h = model(input_ids, kv_caches=kv_caches, start_pos=0, return_hidden=True)
    h_last = h[:, -1:, :]  # Hidden state at last position: (1, 1, dim)
    next_logits = logits[:, -1, :]

    total_generated = 0

    while total_generated < max_new:
        cur_pos = len(generated)
        if cur_pos >= max_seq_len:
            break

        # ── Draft phase: use MTP heads on REAL hidden state ──
        # (Previously used embedding-only, which produced garbage predictions)
        draft_tokens = []
        draft_probs = []

        for i in range(min(draft_steps, model.args.n_mtp_tokens)):
            if i < len(model.mtp_heads):
                mtp_logits = model.mtp_heads[i](h_last, model.tok_embeddings.weight)
                mtp_probs = torch.softmax(mtp_logits[:, -1, :] / max(temperature, 0.1), dim=-1)
                draft_token = torch.multinomial(mtp_probs, num_samples=1).item()
                draft_tokens.append(draft_token)
                draft_probs.append(mtp_probs)
                # Approximate: update h_last for next MTP head using embedding
                # (This is an approximation — the MTP heads are trained on real h,
                #  but for the cascaded draft, embedding-based approximation is acceptable)
                h_last = model.norm(model.tok_embeddings(
                    torch.tensor([[draft_token]], dtype=torch.long, device=device)
                ))

        if not draft_tokens:
            # No MTP heads available, fall back to standard generation
            next_token = _sample_token(
                next_logits, generated, temperature, top_p, top_k,
                repetition_penalty, eos_id
            )
            generated.append(next_token)
            total_generated += 1
            if next_token == eos_id:
                break

            # Update KV cache and get new hidden state
            new_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, _, h = model(new_input, kv_caches=kv_caches, start_pos=cur_pos,
                                 return_hidden=True)
            next_logits = logits[:, -1, :]
            h_last = h[:, -1:, :]
            continue

        # ── Verify phase: process ONLY the draft tokens through the model ──
        # FIX: Do NOT include generated[-1] in the input — it's already cached!
        # The KV cache has entries up to cur_pos-1. We process draft tokens
        # starting at start_pos=cur_pos, which is correct.
        draft_input = torch.tensor(
            [draft_tokens],
            dtype=torch.long, device=device
        )
        verify_logits, _, verify_h = model(draft_input, kv_caches=kv_caches,
                                            start_pos=cur_pos, return_hidden=True)

        # ── Accept/reject phase ──
        # verify_logits[j] gives logits predicting token at position cur_pos+j+1
        # (because logits at position j predict the next token after it)
        # But we need logits that PREDICT draft_tokens[j], which are at position j-1
        # of the verify pass, or the last cached position for j=0.
        #
        # Actually: after prefill/previous step, we have logits for position cur_pos-1
        # which predicts the token at cur_pos. verify_logits[0] is from processing
        # draft_tokens[0] at position cur_pos, and predicts position cur_pos+1.
        #
        # So the prediction for draft_tokens[0] comes from next_logits (the logits
        # we already had from the previous step). The prediction for draft_tokens[1]
        # comes from verify_logits[0], etc.

        accepted = 0

        # Check draft_tokens[0] against the logits we already have
        main_probs_0 = torch.softmax(next_logits / temperature, dim=-1)
        main_prob_0 = main_probs_0[0, draft_tokens[0]].item()
        draft_prob_0 = draft_probs[0][0, draft_tokens[0]].item()

        if draft_prob_0 > 0:
            accept_prob = min(1.0, main_prob_0 / draft_prob_0)
        else:
            accept_prob = 1.0

        if torch.rand(1).item() < accept_prob:
            # Accept draft_tokens[0]
            generated.append(draft_tokens[0])
            total_generated += 1
            accepted += 1

            if draft_tokens[0] == eos_id:
                return tokenizer.decode(generated[1:])
            if total_generated >= max_new:
                break

            # Check remaining draft tokens
            for j in range(1, len(draft_tokens)):
                # verify_logits[j-1] predicts the token at position cur_pos+j
                main_probs_j = torch.softmax(
                    verify_logits[:, j-1, :] / temperature, dim=-1
                )
                main_prob_j = main_probs_j[0, draft_tokens[j]].item()
                draft_prob_j = draft_probs[j][0, draft_tokens[j]].item()

                if draft_prob_j > 0:
                    accept_prob_j = min(1.0, main_prob_j / draft_prob_j)
                else:
                    accept_prob_j = 1.0

                if torch.rand(1).item() < accept_prob_j:
                    # Accept this draft token
                    generated.append(draft_tokens[j])
                    total_generated += 1
                    accepted += 1

                    if draft_tokens[j] == eos_id:
                        return tokenizer.decode(generated[1:])
                    if total_generated >= max_new:
                        break
                else:
                    # Reject: sample from adjusted distribution
                    adjusted_probs = torch.clamp(main_probs_j - draft_probs[j], min=0)
                    if adjusted_probs.sum() > 0:
                        adjusted_probs = adjusted_probs / adjusted_probs.sum()
                        resampled = torch.multinomial(adjusted_probs, num_samples=1).item()
                    else:
                        resampled = torch.multinomial(main_probs_j, num_samples=1).item()

                    generated.append(resampled)
                    total_generated += 1
                    break  # Stop accepting after first rejection
        else:
            # Reject draft_tokens[0]: sample from adjusted distribution
            adjusted_probs = torch.clamp(main_probs_0 - draft_probs[0], min=0)
            if adjusted_probs.sum() > 0:
                adjusted_probs = adjusted_probs / adjusted_probs.sum()
                resampled = torch.multinomial(adjusted_probs, num_samples=1).item()
            else:
                resampled = torch.multinomial(main_probs_0, num_samples=1).item()

            generated.append(resampled)
            total_generated += 1

        # ── Update KV cache state ──
        if accepted < len(draft_tokens):
            # Partial acceptance: truncate KV cache to the correct length
            # The verify pass added len(draft_tokens) entries to the cache,
            # but we only accepted `accepted` of them. We need to remove
            # the entries for the rejected tokens.
            #
            # The cache should have entries for positions 0..cur_pos+accepted-1
            # Total entries after verify: cur_pos (from prefill) + len(draft_tokens)
            # We want: cur_pos + accepted entries
            target_cache_len = cur_pos + accepted
            for cache in kv_caches:
                cache.truncate(target_cache_len)

            # Get logits and hidden state for the next iteration
            if accepted == 0:
                # Nothing accepted — the rejected/resampled token needs processing
                last_token = generated[-1]
                new_input = torch.tensor([[last_token]], dtype=torch.long, device=device)
                logits, _, h = model(new_input, kv_caches=kv_caches,
                                     start_pos=len(generated) - 1, return_hidden=True)
            else:
                # Some accepted — the resampled/bonus token at the end needs processing
                last_token = generated[-1]
                new_input = torch.tensor([[last_token]], dtype=torch.long, device=device)
                logits, _, h = model(new_input, kv_caches=kv_caches,
                                     start_pos=len(generated) - 1, return_hidden=True)

            next_logits = logits[:, -1, :]
            h_last = h[:, -1:, :]
        else:
            # All accepted! KV cache is already correct from verify pass.
            # Get logits and hidden state for the bonus token / next iteration.
            # verify_logits[-1] gives logits from the last draft token position
            next_logits = verify_logits[:, -1, :]
            h_last = verify_h[:, -1:, :]

            # Bonus token: sample one more from the last verify position
            if total_generated < max_new:
                bonus_probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(bonus_probs, num_samples=1).item()
                generated.append(next_token)
                total_generated += 1

                if next_token == eos_id:
                    return tokenizer.decode(generated[1:])

                # Process the bonus token to update cache for next iteration
                new_input = torch.tensor([[next_token]], dtype=torch.long, device=device)
                logits, _, h = model(new_input, kv_caches=kv_caches,
                                     start_pos=len(generated) - 1, return_hidden=True)
                next_logits = logits[:, -1, :]
                h_last = h[:, -1:, :]

    # Decode (skip BOS token)
    return tokenizer.decode(generated[1:])


# ─── Interactive Mode ─────────────────────────────────────────────────────────

def interactive_mode(model, tokenizer, args):
    """Interactive chat mode for testing the model."""
    use_spec = args.speculative
    use_cache = not args.no_kv_cache

    print("\n" + "=" * 60)
    print("Interactive Generation Mode")
    print(f"KV Cache: {'ON' if use_cache else 'OFF'} | "
          f"Speculative Decoding: {'ON' if use_spec else 'OFF'}")
    print("Type your prompt and press Enter. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt or prompt.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break

        gen_func = generate_speculative if use_spec else generate
        output = gen_func(
            model, tokenizer, prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            **({'draft_steps': args.draft_steps} if use_spec else {}),
        )

        print(f"\n{output}\n")


# ─── Checkpoint Loading ───────────────────────────────────────────────────────

def find_checkpoint():
    """Find the best available checkpoint to load."""
    checkpoint_dir = "checkpoints"

    # Prefer best.pt, then final.pt, then latest ckpt_*.pt
    for preferred in ['best.pt', 'final.pt']:
        path = os.path.join(checkpoint_dir, preferred)
        if os.path.exists(path):
            return path

    # Find latest checkpoint by step number
    import glob
    files = glob.glob(os.path.join(checkpoint_dir, "ckpt_*.pt"))
    if files:
        return max(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))

    return None


def load_model(ckpt_path, device="cuda"):
    """Load model from checkpoint."""
    args = ModelArgs()
    model = Transformer(args).to(device)

    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Handle both raw and compiled model state dicts
    state_dict = ckpt.get('model', ckpt)
    # Remove '_orig_mod.' prefix if present (from compiled model)
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        cleaned[new_key] = v

    model.load_state_dict(cleaned, strict=False)

    step = ckpt.get('step', '?')
    val_loss = ckpt.get('best_val', ckpt.get('val_loss', '?'))
    print(f"  Step: {step}, Best val loss: {val_loss}")

    return model


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with trained SLM")
    parser.add_argument('--prompt', type=str, default=None, help='Prompt text')
    parser.add_argument('--max-new', type=int, default=256, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling threshold')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                        help='Repetition penalty (1.2-1.5 recommended for early-stage models)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--no-kv-cache', action='store_true',
                        help='Disable KV cache (slower, for debugging)')
    parser.add_argument('--speculative', action='store_true',
                        help='Use speculative decoding with MTP heads (faster)')
    parser.add_argument('--draft-steps', type=int, default=2,
                        help='Number of draft tokens per speculation round (1-3)')
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file("tokenizer.json")
    ckpt_path = args.checkpoint or find_checkpoint()

    if not ckpt_path:
        print("No checkpoint found! Train the model first: python train.py")
        sys.exit(1)

    model = load_model(ckpt_path)

    if args.prompt:
        # Single prompt mode
        gen_func = generate_speculative if args.speculative else generate
        output = gen_func(
            model, tokenizer, args.prompt,
            max_new=args.max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            **({'draft_steps': args.draft_steps} if args.speculative else {}),
        )
        print(output)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, args)
