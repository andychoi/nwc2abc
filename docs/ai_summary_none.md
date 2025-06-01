# Python Project Summary

## `eval.py`
```python
# eval.py
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
```

## `eval_analysis_tool.py`
```python
# eval_analysis_tool.py
import argparse
from pathlib import Path
from music21 import converter, roman, interval, stream, chord
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def plot_chord_flow(score, output_path):
    chords = []
    key = score.analyze('key')

    for part in score.parts:
        for m in part.getElementsByClass('Measure'):
            rm = roman.romanNumeralFromChord(m.chordify().closedPosition().root(), key)
            chords.append(str(rm.figure))

    plt.figure(figsize=(12, 2))
    plt.plot(chords, marker='o', linestyle='-', color='darkblue')
    plt.xticks(rotation=90)
    plt.title("Chord Flow (Roman Numerals)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Chord flow saved to {output_path}")


def analyze_voice_leading(score, output_path):
    intervals = []
    if len(score.parts) < 2:
        print("⚠️ Not enough parts for voice leading analysis.")
        return

    top, bottom = score.parts[0], score.parts[1]
    for n1, n2 in zip(top.flat.notes, bottom.flat.notes):
        if n1.isNote and n2.isNote:
            iv = interval.Interval(n2, n1)
            intervals.append(iv.semiSimpleName)

    counts = Counter(intervals)
    names, freqs = zip(*counts.items()) if counts else ([], [])

    plt.figure(figsize=(8, 4))
    plt.bar(names, freqs, color='forestgreen')
    plt.title("Voice Leading Interval Histogram")
    plt.xlabel("Interval")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Voice leading analysis saved to {output_path}")


def interactive_review(score):
    print("📖 Launching interactive score viewer...")
    score.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated ABC/MusicXML for musical structure.")
    parser.add_argument("input", type=Path, help="Input ABC or MusicXML file")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"❌ File not found: {args.input}")
        return

    try:
        score = converter.parse(str(args.input))
    except Exception as e:
        print(f"❌ Failed to parse score: {e}")
        return

    chord_flow_img = args.input.with_suffix("_chord_flow.png")
    vl_img = args.input.with_suffix("_voice_leading.png")

    plot_chord_flow(score, chord_flow_img)
    analyze_voice_leading(score, vl_img)
    interactive_review(score)


if __name__ == "__main__":
    main()
```

## `train.py`
```python
# train.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from music21 import converter, key, interval
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Any
import matplotlib.pyplot as plt

class StructuredABCTokenizer:
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '|': 4}
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def fit(self, sequences: List[str]):
        for seq in sequences:
            for tok in seq.split():
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(tok, self.vocab['<UNK>']) for tok in text.split()]

    def decode(self, tokens: List[int]) -> str:
        return ' '.join(self.rev_vocab.get(tok, '<UNK>') for tok in tokens)

def extract_parts_from_folder(folder_path: str) -> List[Any]:
    scores = []
    for file in Path(folder_path).rglob("*.abc"):
        try:
            score = converter.parse(str(file))
            scores.append(score)
        except Exception as e:
            print(f"Failed: {file} => {e}")
    return scores

class MusicGenModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embed(x)
        x = self.norm(x)
        x = self.transformer(x)
        return self.fc(x)

def tokenize_part(part, part_name: str) -> str:
    tokens = [f"<part={part_name}>"]
    for n in part.flat.notesAndRests:
        dur = int(n.quarterLength * 4)

        if n.isRest:
            tokens.append(f"z{dur}")
        elif n.isChord:
            chord_notes = " ".join(p.nameWithOctave for p in n.pitches)
            tokens.append(f"[{chord_notes}]/{dur}")
        else:
            note = n.nameWithOctave
            dyn = ''
            if n.expressions:
                dyn = str(n.expressions[0])
            tokens.append(f"{note}/{dur}{' ' + dyn if dyn else ''}")
    return ' '.join(tokens)

def transpose_to_c(score):
    try:
        original_key = score.analyze('key')
        i = interval.Interval(original_key.tonic, key.Key('C').tonic)
        return score.transpose(i)
    except Exception as e:
        print(f"⚠️ Transposition error: {e}")
        return score

def generate_sequence(model, tokenizer, prime_tokens, max_len=200, temperature=1.0, device="cpu") -> str:
    model.eval()
    input_ids = tokenizer.encode(prime_tokens)
    generated = input_ids.copy()
    x = torch.tensor(generated, dtype=torch.long).to(device).unsqueeze(0)
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(x)
        logits = logits[0, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        if tokenizer.rev_vocab.get(next_token) == '<EOS>':
            break
        x = torch.tensor(generated, dtype=torch.long).to(device).unsqueeze(0)
    return tokenizer.decode(generated)

class MusicDataset(Dataset):
    def __init__(self, tokenizer, scores, augment=True):
        self.samples = []
        self.tokenizer = tokenizer

        for score in scores:
            score = transpose_to_c(score)
            for part in score.parts:
                seq = tokenize_part(part, part.partName or part.id or "Part")
                self.samples.append(seq)
                if augment:
                    for _ in range(2):
                        interval_semitones = random.choice(range(-3, 4))
                        transposed = score.transpose(interval_semitones)
                        for p in transposed.parts:
                            aug_seq = tokenize_part(p, p.partName or p.id or "Part")
                            self.samples.append(aug_seq)
        tokenizer.fit(self.samples)
        self.encoded = [tokenizer.encode(s) for s in self.samples]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        x = self.encoded[idx][:-1]
        y = self.encoded[idx][1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xs, ys

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    loss_values = []
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        out = out.view(-1, out.size(-1))
        y = y.view(-1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
    return loss_values

def plot_loss(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Batch")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_loss.png")
    print("📈 Saved loss plot to training_loss.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    default_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--input', type=str, required=True, help="Path to input folder containing ABC files")
    parser.add_argument('--augment', action="store_true", help="Enable key transposition augmentation")
    args = parser.parse_args()

    scores = extract_parts_from_folder(args.input)
    if not scores:
        print("No scores loaded. Please check the input folder.")
        return
    print(f"Loaded {len(scores)} scores from {args.input}")
    print(f"Using device: {args.device}")

    tokenizer = StructuredABCTokenizer()
    dataset = MusicDataset(tokenizer, scores, augment=args.augment)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = MusicGenModel(vocab_size=len(tokenizer.vocab)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    all_losses = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}")
        epoch_losses = train(model, dataloader, optimizer, criterion, args.device)
        all_losses.extend(epoch_losses)

    plot_loss(all_losses)

    save_path = Path("musicgen_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': tokenizer.vocab
    }, save_path)
    print(f"✅ Model saved to {save_path}")

    prime = "<part=Piano-RH> C4/4 E4/4 G4/4 | <part=Piano-LH> C2/1 |"
    generated_music = generate_sequence(model, tokenizer, prime, device=args.device)
    print("🎵 Generated music:")
    print(generated_music)

if __name__ == "__main__":
    main()
```

