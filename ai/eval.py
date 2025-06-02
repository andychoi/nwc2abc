# ai/eval.py
import argparse
from pathlib import Path
import torch
from music21 import converter, key, stream
from train import MusicGenModel, StructuredABCTokenizer, generate_sequence


def detect_key(score):
    k = score.analyze('key')
    return k


def get_transposition_interval(original_key, target_key):
    interval = key.transposeInterval(original_key, target_key)
    return interval


def transpose_score(score, target_key):
    original_key = detect_key(score)
    interval = original_key.tonic.intervalBetween(target_key.tonic)
    return score.transpose(interval), interval


def reverse_transpose_abc(abc_str, interval):
    # Reverse transposition using music21 (approximate):
    try:
        s = converter.parse(abc_str, format='abc')
        s_trans = s.transpose(-interval)
        return s_trans.write('abc')
    except Exception as e:
        print(f"❌ Reverse transposition failed: {e}")
        return abc_str


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained music generation model on ABC input")
    parser.add_argument("abc_file", type=Path, help="Input ABC file as melody prompt")
    parser.add_argument("--model", type=Path, default=Path("musicgen_model.pt"), help="Trained model path")
    parser.add_argument("--device", default="cpu", help="Device to run model")
    parser.add_argument("--out", type=Path, help="Output ABC file")
    parser.add_argument("--transpose_key", default="C", help="Transpose input to this key (e.g., C, Am)")
    parser.add_argument("--max_len", type=int, default=200, help="Max generation length")
    args = parser.parse_args()

    # Load input ABC and parse
    abc_text = args.abc_file.read_text(encoding="utf-8")
    score = converter.parse(abc_text, format='abc')

    # Detect key and transpose to target
    orig_key = detect_key(score)
    target_key = key.Key(args.transpose_key)
    print(f"🎼 Transposing from {orig_key} to {target_key}")
    transposed_score, interval = transpose_score(score, target_key)

    # Convert transposed score to ABC text
    abc_transposed = transposed_score.write('abc')
    prompt_text = Path(abc_transposed).read_text(encoding='utf-8')

    # Load model and tokenizer
    checkpoint = torch.load(args.model, map_location=args.device)
    model = MusicGenModel(vocab_size=len(checkpoint['vocab']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)

    tokenizer = StructuredABCTokenizer()
    tokenizer.vocab = checkpoint['vocab']
    tokenizer.rev_vocab = {v: k for k, v in tokenizer.vocab.items()}

    # Generate
    print("🎹 Generating continuation...")
    gen_abc = generate_sequence(
        model=model,
        tokenizer=tokenizer,
        prime_tokens=prompt_text,
        max_len=args.max_len,
        temperature=1.0,
        device=args.device
    )

    # Reverse transpose
    print("🔄 Transposing back to original key")
    reversed = reverse_transpose_abc(gen_abc, interval)

    # Output
    if args.out:
        args.out.write_text(reversed, encoding="utf-8")
        print(f"✅ Saved output to: {args.out}")
    else:
        print("\n🎼 Generated Music (in original key):\n")
        print(reversed)


if __name__ == "__main__":
    main()