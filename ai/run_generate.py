#!/usr/bin/env python3
"""
run_generate.py

Given:
  • A folder of ABC training data (already tokenized by REMIABCTokenizer during train) 
  • A trained REMI‐Transformer checkpoint
  • A single ABC prompt file

Produce:
  • A generated continuation (in ABC and MusicXML)
  • Writes everything into a user‐specified output directory

Usage:
    python run_generate.py \
      --abc-input-dir /path/to/abc_corpus \
      --model-checkpoint /path/to/musicgen_remi_model.pt \
      --prompt-file /path/to/prompt.abc \
      --output-dir /path/to/output_folder \
      --device cuda \
      --max-len 256 \
      --temperature 1.0 \
      --top-k 50 \
      --top-p 0.9
"""

import argparse
import shutil
import sys
from pathlib import Path
import torch
from music21 import converter, stream, metadata, instrument, key as m21key, interval, meter

from remi_tokenizer import REMIABCTokenizer
from remi_detokenizer import remi_tokens_to_score
from train import DecoderOnlyMusicGenModel


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits before sampling.
    """
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Zero out everything not in top_k
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits[logits < threshold] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Tokens to remove: cumulative_probs > top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Always keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original indices
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits


def generate_sequence(
    model,
    tokenizer: REMIABCTokenizer,
    prime_tokens: list,
    max_len: int = 200,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    device: str = "cpu",
) -> list:
    """
    Autoregressively generate a token stream given:
      • model: a trained DecoderOnlyMusicGenModel
      • tokenizer: REMIABCTokenizer (vocab already replaced by train-vocab)
      • prime_tokens: list[str] from tokenizer.tokenize(prompt_in_C)
      • max_len: maximum number of new tokens (break on <EOS>)
      • temperature, top_k, top_p: sampling hyperparameters
      • device: 'cpu' or 'cuda' or 'mps'
    """
    model.eval()
    input_ids = tokenizer.encode(prime_tokens)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    for _ in range(max_len):
        with torch.no_grad():
            out = model(input_tensor)  # shape: (1, seq_len, vocab_size)
            logits = out[0, -1] / temperature
            filtered = top_k_top_p_filtering(logits.clone(), top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        input_ids.append(next_id)
        if tokenizer.rev_vocab.get(next_id) == "<EOS>":
            break
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    return tokenizer.decode(input_ids)


def merge_prompt_and_generated(prompt_score: stream.Score, generated_score: stream.Score) -> stream.Score:
    """
    Build a side-by-side Score:
      - Keep every part in prompt_score as separate staves (SATB, etc.).
      - Append a new “Generated Continuation” part at the end (Piano).
    """
    merged = stream.Score()
    merged.insert(0, metadata.Metadata())
    merged.metadata.title = "Prompt + AI Generation"
    merged.metadata.composer = "AI Composer"

    # Preserve original parts
    for i, part in enumerate(prompt_score.parts):
        clone = part.flat.notesAndRests.stream()
        clone.id = part.id or f"Voice_{i}"
        clone.partName = part.partName or part.id or f"Voice_{i}"

        lname = clone.partName.lower()
        if "sop" in lname or lname == "s":
            clone.insert(0, instrument.Soprano())
        elif "alt" in lname or lname == "a":
            clone.insert(0, instrument.Alto())
        elif "ten" in lname or lname == "t":
            clone.insert(0, instrument.Tenor())
        elif "bass" in lname or lname == "b":
            clone.insert(0, instrument.Bass())
        else:
            clone.insert(0, instrument.Vocalist())

        merged.insert(i, clone)

    # Append generated continuation as a Piano part
    gen_part = stream.Part()
    gen_part.id = "Generated"
    gen_part.partName = "Generated Continuation"
    gen_part.insert(0, instrument.Piano())
    gen_part.append(generated_score.flat.notesAndRests)
    merged.insert(len(prompt_score.parts), gen_part)

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Load ABC corpus + trained REMI model, then generate new score from a prompt."
    )
    parser.add_argument(
        "--abc-input-dir",
        type=Path,
        required=True,
        help="Root directory of ABC files (training not run here)",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (musicgen_remi_model.pt)",
    )
    parser.add_argument(
        "--prompt-file", type=Path, required=True, help="Single ABC file to use as prompt"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated ABC & MusicXML will be saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for inference",
    )
    parser.add_argument(
        "--max-len", type=int, default=200, help="Max number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling (0 to disable)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling (0.0 to disable)"
    )
    args = parser.parse_args()

    abc_dir = args.abc_input_dir.resolve()
    model_ckpt = args.model_checkpoint.resolve()
    prompt_path = args.prompt_file.resolve()
    out_dir = args.output_dir.resolve()
    device = args.device
    max_len = args.max_len
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p

    if not abc_dir.is_dir():
        print(f"Error: ABC input directory not found: {abc_dir}")
        sys.exit(1)
    if not model_ckpt.is_file():
        print(f"Error: Model checkpoint not found: {model_ckpt}")
        sys.exit(1)
    if not prompt_path.is_file():
        print(f"Error: Prompt file not found: {prompt_path}")
        sys.exit(1)

    # 1) Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Load & parse prompt
    prompt_score = converter.parse(str(prompt_path))
    orig_ts = prompt_score.recurse().getElementsByClass(meter.TimeSignature).first()
    orig_key_obj = prompt_score.analyze("key")

    # 3) Transpose prompt → C major/minor
    if orig_key_obj.mode == "major":
        target_key = m21key.Key("C")
    else:
        target_key = m21key.Key("C", "minor")
    iv_to_C = interval.Interval(orig_key_obj.tonic, target_key.tonic)
    prompt_in_C = prompt_score.transpose(iv_to_C)

    # 4) Tokenize prompt_in_C
    tokenizer = REMIABCTokenizer()
    tokens = tokenizer.tokenize(prompt_in_C)
    print("Prompt tokens (in C):")
    print(" ".join(tokens))

    # 5) Load trained model & checkpoint
    checkpoint = torch.load(str(model_ckpt), map_location=device)
    model = DecoderOnlyMusicGenModel(vocab_size=len(checkpoint["vocab"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 6) Restore tokenizer vocab from checkpoint
    tokenizer.vocab = checkpoint["vocab"]
    tokenizer.rev_vocab = {v: k for k, v in tokenizer.vocab.items()}

    # 7) Generate continuation (in C)
    print("\nGenerating continuation (in C)...")
    generated_tokens = generate_sequence(
        model=model,
        tokenizer=tokenizer,
        prime_tokens=tokens,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )

    # Save generated tokens to a file
    gen_tokens_file = out_dir / "generated_tokens.txt"
    gen_tokens_file.write_text(" ".join(generated_tokens), encoding="utf-8")
    print(f"Saved generated tokens → {gen_tokens_file}\n")

    # 8) Detokenize → Score in C
    score_in_C = remi_tokens_to_score(generated_tokens)

    # 9) Reverse-transpose back to original key
    iv_to_orig = interval.Interval(target_key.tonic, orig_key_obj.tonic)
    score_in_orig = score_in_C.transpose(iv_to_orig)

    # 10) Re-insert original time signature & key
    if orig_ts:
        score_in_orig.insert(0, orig_ts)
    score_in_orig.insert(
        0,
        m21key.Key(
            orig_key_obj.tonic.name
            + ("major" if orig_key_obj.mode == "major" else "minor")
        ),
    )

    # 11) Merge prompt and generated side-by-side
    combined = merge_prompt_and_generated(prompt_score, score_in_orig)

    # 12) Save combined as ABC & MusicXML
    out_abc = out_dir / "prompt_vs_gen.abc"
    out_mxl = out_dir / "prompt_vs_gen.xml"
    combined.write("abc", fp=str(out_abc))
    combined.write("musicxml", fp=str(out_mxl))
    print(f"Saved side-by-side ABC      → {out_abc}")
    print(f"Saved side-by-side MusicXML → {out_mxl}")

    print("\nGeneration complete. All files are in:", out_dir)


if __name__ == "__main__":
    main()
