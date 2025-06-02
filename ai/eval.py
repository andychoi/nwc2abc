# ai/eval.py

"""
Usage:

python eval.py examples/melody.abc \
  --device cuda \
  --max_len 256 \
  --temperature 1.0 \
  --top_k 50 \
  --top_p 0.9 \
  --out generated_tokens.txt \
  --export_abc prompt_vs_gen.abc \
  --export_musicxml prompt_vs_gen.xml
"""

import argparse
from pathlib import Path
import torch
from music21 import converter, stream, metadata, instrument, key as m21key, interval, meter

from remi_tokenizer import REMIABCTokenizer
from remi_detokenizer import remi_tokens_to_score
from train import DecoderOnlyMusicGenModel


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    From: https://github.com/huggingface/transformers/blob/main/src/transformers/generation_logits_process.py
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove tokens with a probability less than the top-k largest probabilities
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def generate_sequence(
    model, tokenizer, prime_tokens, max_len=200, temperature=1.0, top_k=0, top_p=0.0, device="cpu"
):
    """
    Autoregressive generation with optional top-k and/or nucleus (top-p) sampling.
    """
    model.eval()
    input_ids = tokenizer.encode(prime_tokens)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            outputs = model(input_tensor)  # (1, seq_len, vocab_size)
            next_logits = outputs[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_logits.clone(), top_k=top_k, top_p=top_p)
            next_probs = torch.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(next_probs, num_samples=1).item()

        input_ids.append(next_token_id)
        if tokenizer.rev_vocab.get(next_token_id) == "<EOS>":
            break
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    return tokenizer.decode(input_ids)


def merge_prompt_and_generated(prompt_score: stream.Score, generated_score: stream.Score) -> stream.Score:
    """
    Side-by-side layout: keep all original staves, then append 'Generated Continuation' as a piano part.
    """
    merged = stream.Score()
    merged.insert(0, metadata.Metadata())
    merged.metadata.title = "Prompt + AI Generated"
    merged.metadata.composer = "AI Composer"

    # Keep each original part (SATB, etc.)
    for i, part in enumerate(prompt_score.parts):
        part_copy = part.flat.notesAndRests.stream()
        part_copy.id = part.id or f"Voice_{i}"
        part_copy.partName = part.partName or part.id or f"Voice_{i}"

        label = part_copy.partName.lower()
        if "sop" in label or label == "s":
            part_copy.insert(0, instrument.Soprano())
        elif "alt" in label or label == "a":
            part_copy.insert(0, instrument.Alto())
        elif "ten" in label or label == "t":
            part_copy.insert(0, instrument.Tenor())
        elif "bass" in label or label == "b":
            part_copy.insert(0, instrument.Bass())
        else:
            part_copy.insert(0, instrument.Vocalist())

        merged.insert(i, part_copy)

    # Add generated part (Piano)
    gen_part = stream.Part()
    gen_part.id = "Generated"
    gen_part.partName = "Generated Continuation"
    gen_part.insert(0, instrument.Piano())
    gen_part.append(generated_score.flat.notesAndRests)
    merged.insert(len(prompt_score.parts), gen_part)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Generate music with REMI model: transpose input→C, generate→C, then retranspose→original key/time"
    )
    parser.add_argument("prompt_file", type=Path, help="ABC file as prompt")
    parser.add_argument("--model", type=Path, default="musicgen_remi_model.pt", help="Path to trained model")
    parser.add_argument("--device", default="cpu", help="Device (cuda/mps/cpu)")
    parser.add_argument("--max_len", type=int, default=200, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0 → disabled)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling (0 → disabled)")
    parser.add_argument("--out_tokens", type=Path, help="Save generated tokens to file")
    parser.add_argument("--export_abc", type=Path, help="Save side-by-side as ABC")
    parser.add_argument("--export_musicxml", type=Path, help="Save side-by-side as MusicXML")
    args = parser.parse_args()

    # 1) Load original prompt
    prompt_score = converter.parse(str(args.prompt_file))
    orig_ts = prompt_score.recurse().getElementsByClass(meter.TimeSignature).first()
    orig_key_obj = prompt_score.analyze("key")

    # 2) Transpose prompt to C major/minor
    if orig_key_obj.mode == "major":
        target_key = m21key.Key("C")
    else:
        target_key = m21key.Key("C", "minor")
    iv_to_C = interval.Interval(orig_key_obj.tonic, target_key.tonic)
    prompt_in_C = prompt_score.transpose(iv_to_C)

    # 3) Tokenize in-C
    tokenizer = REMIABCTokenizer()
    tokens = tokenizer.tokenize(prompt_in_C)
    print(f"🎼 Prompt tokens (in C):\n{' '.join(tokens)}")

    # 4) Load model & vocab
    checkpoint = torch.load(str(args.model), map_location=args.device)
    model = DecoderOnlyMusicGenModel(vocab_size=len(checkpoint["vocab"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    # Replace tokenizer vocab with training-time vocab
    tokenizer.vocab = checkpoint["vocab"]
    tokenizer.rev_vocab = {v: k for k, v in tokenizer.vocab.items()}

    # 5) Generate continuation (in C) with top-k/top-p
    print("🎹 Generating continuation (in C)...")
    generated_tokens = generate_sequence(
        model=model,
        tokenizer=tokenizer,
        prime_tokens=tokens,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
    )

    print("\n🎵 Generated Tokens:\n")
    print(" ".join(generated_tokens))

    if args.out_tokens:
        args.out_tokens.write_text(" ".join(generated_tokens), encoding="utf-8")
        print(f"✅ Tokens saved to {args.out_tokens}")

    # 6) Detokenize → Score in C
    score_in_C = remi_tokens_to_score(generated_tokens)

    # 7) Reverse-transpose back to original key
    iv_to_orig = interval.Interval(target_key.tonic, orig_key_obj.tonic)
    score_in_orig = score_in_C.transpose(iv_to_orig)

    # 8) Re-insert original time signature & key at offset 0
    if orig_ts:
        score_in_orig.insert(0, orig_ts)
    score_in_orig.insert(
        0,
        m21key.Key(
            orig_key_obj.tonic.name
            + ("major" if orig_key_obj.mode == "major" else "minor")
        ),
    )

    # 9) Side-by-side merge & show
    combined_score = merge_prompt_and_generated(prompt_score, score_in_orig)
    combined_score.show()

    # 10) Exports
    if args.export_abc:
        combined_score.write("abc", fp=str(args.export_abc))
        print(f"✅ ABC exported to {args.export_abc}")
    if args.export_musicxml:
        combined_score.write("musicxml", fp=str(args.export_musicxml))
        print(f"✅ MusicXML exported to {args.export_musicxml}")


if __name__ == "__main__":
    main()
