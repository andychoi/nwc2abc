# ai/train.py

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
from music21 import converter, stream, key as m21key, interval, meter

import matplotlib.pyplot as plt

from remi_tokenizer import REMIABCTokenizer  # updated tokenizer
from relative_transformer import RelativeTransformerDecoder

# === Model definition ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class DecoderOnlyMusicGenModel(nn.Module):
    """
    Even if you keep a single‐stage model, you can add a small “chord head” on top of the transformer’s 
    hidden states at bar boundaries. During training, you force it to predict the next bar’s chord.
    In your training loop:

    1. Collect chord_positions:
        - While tokenizing each C‐transposed score, record the indices of every <BarStart> that’s immediately followed by <Chord_...> in the token stream.
        - Create a tensor of shape (batch, n_bars) containing those indices.

    2. Compute two losses:
        - token_loss: cross‐entropy over next‐token at each position.
        - chord_loss: cross‐entropy between chord_logits[i, j, :] and the ground‐truth chord ID at bar j + 1.

    3. Combine them: loss = token_loss + λ * chord_loss (e.g. λ = 0.5).

    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_rel_dist: int = 1024,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=4096)

        self.decoder = RelativeTransformerDecoder( ... )
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Chord vocabulary: a separate small embedding & linear
        self.chord_vocab_size = len([tok for tok in tokenizer.vocab if tok.startswith("<Chord_")])
        self.chord_indices = {
            tok: idx for idx, tok in enumerate(tokenizer.vocab) if tok.startswith("<Chord_")
        }
        self.chord_head = nn.Linear(d_model, self.chord_vocab_size)

    def forward(self, x, chord_positions=None):
        """
        chord_positions: a list of token‐indices at which <BarStart> occurs.
        We'll extract the hidden state at those positions to predict the next chord.
        """
        emb = self.embedding(x)
        emb = self.positional(emb)
        tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        hidden = self.decoder(emb, tgt_mask=tgt_mask)  # (batch, seq_len, d_model)
        logits = self.fc_out(hidden)                  # (batch, seq_len, vocab_size)

        # If chord_positions is provided, we compute chord_logits:
        chord_logits = None
        if chord_positions is not None:
            # hidden_at_bar = hidden[:, chord_positions, :]  # gather at each bar boundary
            gathered = []
            for i, positions in enumerate(chord_positions):
                # positions: list of indices (e.g., [3, 10, 17, ...])
                # We want hidden[i, pos, :]
                gathered.append(hidden[i, positions, :])        # tensor (n_bars, d_model)
            # Pad to max bars, or stack if uniform length
            # For simplicity, assume each batch sample has the same number of bars:
            hidden_at_bar = torch.stack(gathered, dim=0)       # (batch, n_bars, d_model)
            chord_logits = self.chord_head(hidden_at_bar)      # (batch, n_bars, chord_vocab_size)

        return logits, chord_logits

# === Dataset, with transposition to C ===
class MusicREMI_Dataset(Dataset):
    def __init__(self, tokenizer: REMIABCTokenizer, scores: List[stream.Score]):
        self.tokenizer = tokenizer
        self.samples = []

        for score in scores:
            try:
                # 1) Analyze key of the original
                orig_key: m21key.Key = score.analyze("key")
                if orig_key.mode == "major":
                    target_key = m21key.Key("C")
                else:
                    target_key = m21key.Key("C", "minor")
                iv = interval.Interval(orig_key.tonic, target_key.tonic)
                score_C = score.transpose(iv)

                # 2) Tokenize the C‐transposed score
                tokens = tokenizer.tokenize(score_C)
                encoded = tokenizer.encode(tokens)
                self.samples.append(encoded)
            except Exception as e:
                print(f"❌ Failed to tokenize/transposition: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = seq[:-1]
        y = seq[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xs, ys


def extract_scores(folder_path: str) -> List[stream.Score]:
    scores = []
    for file in Path(folder_path).rglob("*.abc"):
        try:
            score = converter.parse(file)
            scores.append(score)
        except Exception as e:
            print(f"❌ Failed to parse {file}: {e}")
    return scores


# === Training & Evaluation Functions ===
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    losses = []
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)  # (batch, seq_len, vocab_size)
        out_flat = out.view(-1, out.size(-1))
        y_flat = y.view(-1)
        loss = criterion(out_flat, y_flat)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            out_flat = out.view(-1, out.size(-1))
            targets = y.view(-1)
            loss = criterion(out_flat, targets)
            total_loss += loss.item() * targets.size(0)
            total_tokens += targets.size(0)

            preds = out_flat.argmax(dim=-1)
            mask = targets != 0
            correct += (preds[mask] == targets[mask]).sum().item()

    perplexity = np.exp(total_loss / total_tokens)
    accuracy = correct / total_tokens
    return perplexity, accuracy


def plot_losses(losses):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("📈 Saved training_loss.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing ABC files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    print(f"📁 Loading scores from: {args.input}")
    scores = extract_scores(args.input)
    print(f"✅ Loaded {len(scores)} scores.")

    # Split into train / validation (80% / 20%)
    torch.manual_seed(0)
    n_total = len(scores)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val
    train_scores, val_scores = torch.utils.data.random_split(scores, [n_train, n_val])

    tokenizer = REMIABCTokenizer()
    train_dataset = MusicREMI_Dataset(tokenizer, train_scores)
    val_dataset = MusicREMI_Dataset(tokenizer, val_scores)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # At this point, tokenizer.vocab has been populated
    model = DecoderOnlyMusicGenModel(vocab_size=len(tokenizer.vocab)).to(args.device)

    # Use CrossEntropyLoss with label smoothing = 0.1
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    all_losses = []
    for epoch in range(args.epochs):
        print(f"\n🌀 Epoch {epoch+1}/{args.epochs}")
        train_losses = train_epoch(model, train_loader, optimizer, criterion, args.device)
        all_losses.extend(train_losses)

        train_ppl, train_acc = evaluate(model, train_loader, criterion, args.device)
        val_ppl, val_acc = evaluate(model, val_loader, criterion, args.device)
        print(f"   ▶️ Train  — Perplexity: {train_ppl:.2f}, Accuracy: {train_acc:.2%}")
        print(f"   ▶️ Valid  — Perplexity: {val_ppl:.2f}, Accuracy: {val_acc:.2%}")

    plot_losses(all_losses)

    # Save model + vocabulary
    torch.save(
        {"model_state_dict": model.state_dict(), "vocab": tokenizer.vocab},
        "musicgen_remi_model.pt",
    )
    print("✅ Model and vocab saved to musicgen_remi_model.pt")


if __name__ == "__main__":
    main()
