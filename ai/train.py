
# ai/train.py
import argparse
import math
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from music21 import converter, stream, key as m21key, interval

import matplotlib.pyplot as plt

from remi_tokenizer import REMIABCTokenizer  # your tokenizer
from relative_transformer import RelativeTransformerDecoder  # your decoder


# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1)]


# =========================
# Decoder-Only Model (with optional chord head)
# =========================
class DecoderOnlyMusicGenModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        chord_token_ids: Optional[List[int]] = None,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_rel_dist: int = 1024,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.positional = PositionalEncoding(d_model, max_len=4096)

        # Your relative transformer decoder should mirror nn.TransformerDecoder behavior
        self.decoder = RelativeTransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_rel_dist=max_rel_dist,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Optional chord head
        self.chord_token_ids = chord_token_ids or []
        self.chord_vocab_size = len(self.chord_token_ids)
        self.chord_head = nn.Linear(d_model, self.chord_vocab_size) if self.chord_vocab_size > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        chord_positions: Optional[torch.Tensor] = None,  # (B, M) with -1 for "no position"
    ):
        """
        x: (B, T) LongTensor of token ids
        chord_positions: Optional (B, M) LongTensor of time indices; -1 indicates no position
        Returns:
          logits: (B, T, V)
          chord_logits: (B, M, C) or None
        """
        B, T = x.size()
        device = x.device

        # Masks
        tgt_mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)  # True masks FUTURE
        key_padding_mask = (x == self.padding_idx)  # (B, T) True masks PAD

        # Embedding + pos + (optional) scaling
        emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        emb = self.positional(emb)  # (B, T, D)

        # Decoder forward (assumes decoder accepts tgt_mask & key_padding_mask)
        hidden = self.decoder(emb, tgt_mask=tgt_mask, key_padding_mask=key_padding_mask)  # (B, T, D)
        logits = self.fc_out(hidden)  # (B, T, V)

        chord_logits = None
        if self.chord_head is not None and chord_positions is not None:
            # chord_positions: (B, M) with -1 for "no position"
            B, M = chord_positions.size()
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, M)
            safe_pos = chord_positions.clamp_min(0)
            gathered = hidden[batch_idx, safe_pos, :]  # (B, M, D)
            chord_logits = self.chord_head(gathered)   # (B, M, C)

        return logits, chord_logits


# =========================
# Dataset with C-transposition and chord supervision
# =========================
class MusicREMI_Dataset(Dataset):
    """
    Produces:
      - x (LongTensor): input token ids (T-1)
      - y (LongTensor): next-token ids (T-1)
      - bar_pos (LongTensor): positions of <BarStart> predicting the next bar chord (length M)
      - bar_chords (List[str]): chord token strings (length M)
    """
    def __init__(self, tokenizer: REMIABCTokenizer, scores: List[stream.Score]):
        self.tok = tokenizer
        self.samples: List[List[int]] = []
        self.chord_pos: List[List[int]] = []
        self.chord_lbl: List[List[str]] = []

        for score in scores:
            try:
                score_C = self._transpose_to_c(score)
                tokens = self.tok.tokenize(score_C)       # list[str]
                encoded = self.tok.encode(tokens)         # list[int]
                bar_positions, chord_labels = self._extract_bar_chord_pairs(tokens)

                if len(encoded) >= 2:  # need at least 2 tokens to form x,y
                    self.samples.append(encoded)
                    self.chord_pos.append(bar_positions)
                    self.chord_lbl.append(chord_labels)
            except Exception as e:
                print(f"❌ Failed to tokenize/transposition: {e}")

    def _transpose_to_c(self, score: stream.Score) -> stream.Score:
        try:
            k = score.analyze("key")
            tonic = getattr(k, "tonic", None)
            mode = getattr(k, "mode", "major")
            if tonic is None:
                # Fallback: assume C major
                return score
            target = m21key.Key("C") if mode == "major" else m21key.Key("C", "minor")
            iv = interval.Interval(tonic, target.tonic)
            return score.transpose(iv)
        except Exception:
            return score

    def _extract_bar_chord_pairs(self, tokens: List[str]):
        bar_positions: List[int] = []
        chord_labels: List[str] = []
        # For each <BarStart>, find the first <Chord_...> token that follows shortly
        for i, tok in enumerate(tokens[:-1]):
            if tok == "<BarStart>":
                for j in range(i + 1, min(i + 16, len(tokens))):  # small lookahead window
                    if tokens[j].startswith("<Chord_"):
                        bar_positions.append(i)
                        chord_labels.append(tokens[j])
                        break
        return bar_positions, chord_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)

        bar_pos = torch.tensor(self.chord_pos[idx], dtype=torch.long) if self.chord_pos[idx] else torch.empty(0, dtype=torch.long)
        bar_chords = self.chord_lbl[idx]  # keep as list[str]; mapped in collate
        return {"x": x, "y": y, "bar_pos": bar_pos, "bar_chords": bar_chords}


# =========================
# Collate function
# =========================
def collate_fn(
    batch: List[Dict],
    padding_idx: int,
    chord_token_to_class: Optional[Dict[str, int]] = None,
):
    xs = [b["x"] for b in batch]
    ys = [b["y"] for b in batch]
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=padding_idx)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=padding_idx)

    max_m = max((len(b["bar_pos"]) for b in batch), default=0)
    if max_m > 0 and chord_token_to_class is not None and len(chord_token_to_class) > 0:
        B = len(batch)
        pos_pad = torch.full((B, max_m), -1, dtype=torch.long)  # -1 => "no position"
        lbl_pad = torch.full((B, max_m), -1, dtype=torch.long)  # -1 => "no label"
        for i, b in enumerate(batch):
            m = len(b["bar_pos"])
            if m:
                pos_pad[i, :m] = b["bar_pos"]
                mapped = []
                for t in b["bar_chords"]:
                    if t in chord_token_to_class:
                        mapped.append(chord_token_to_class[t])
                    else:
                        mapped.append(-1)
                if len(mapped):
                    lbl_pad[i, :len(mapped)] = torch.tensor(mapped, dtype=torch.long)
        return xs, ys, pos_pad, lbl_pad

    return xs, ys, None, None


# =========================
# File loader
# =========================
def extract_scores(folder_path: str) -> List[stream.Score]:
    exts = ("*.abc", "*.musicxml", "*.xml", "*.mxl")
    scores: List[stream.Score] = []
    for pattern in exts:
        for file in Path(folder_path).rglob(pattern):
            try:
                s = converter.parse(str(file))
                scores.append(s)
            except Exception as e:
                print(f"❌ Failed to parse {file}: {e}")
    return scores


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str):
    model.eval()
    loss_sum = 0.0
    tok_count = 0
    correct = 0

    for batch in dataloader:
        # batch may be (x, y, pos, lbl) or (x, y, None, None)
        if len(batch) == 4:
            x, y, _, _ = batch
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)

        logits, _ = model(x)  # (B, T, V)
        B, T, V = logits.size()
        flat_logits = logits.reshape(B * T, V)
        flat_targets = y.reshape(B * T)

        mask = (flat_targets != 0)
        # Sum-reduction CE for correct perplexity
        loss = nn.functional.cross_entropy(flat_logits, flat_targets, ignore_index=0, reduction="sum")
        loss_sum += float(loss.item())
        tok_count += int(mask.sum().item())

        preds = flat_logits.argmax(dim=-1)
        correct += int((preds[mask] == flat_targets[mask]).sum().item())

    perplexity = float(np.exp(loss_sum / max(1, tok_count))) if tok_count > 0 else float("inf")
    accuracy = float(correct / max(1, tok_count)) if tok_count > 0 else 0.0
    return perplexity, accuracy


# =========================
# Plot
# =========================
def plot_losses(losses: List[float]):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss (per non-pad token)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("📈 Saved training_loss.png")


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder containing ABC/MusicXML files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    default_device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    print(f"📁 Loading scores from: {args.input}")
    scores = extract_scores(args.input)
    print(f"✅ Loaded {len(scores)} scores.")
    if len(scores) < 2:
        print("⚠️ Need at least 2 scores for a train/val split. Add more data.")
        # Continue anyway: put 1 in train, 1 in val if possible

    # Split into train / validation (80% / 20%)
    torch.manual_seed(0)
    n_total = len(scores)
    n_val = max(1, int(0.2 * max(2, n_total)))
    n_train = max(1, n_total - n_val)

    # Random split indices then build lists (more predictable than Subset of music21 Scores)
    all_indices = torch.randperm(n_total).tolist()
    train_idx = all_indices[:n_train]
    val_idx = all_indices[n_train:n_train + n_val]
    train_scores = [scores[i] for i in train_idx]
    val_scores = [scores[i] for i in val_idx]

    tokenizer = REMIABCTokenizer()

    # Build datasets (tokenizer.vocab will be populated during tokenization/encoding)
    train_dataset = MusicREMI_Dataset(tokenizer, train_scores)
    val_dataset = MusicREMI_Dataset(tokenizer, val_scores)

    # Token maps
    # If tokenizer exposes token_to_id, use it; otherwise build from vocab
    token_to_id = getattr(tokenizer, "token_to_id", {tok: i for i, tok in enumerate(tokenizer.vocab)})

    # Chord tokens & ids for chord head
    chord_tokens = [t for t in tokenizer.vocab if t.startswith("<Chord_")]
    chord_token_ids = [token_to_id[t] for t in chord_tokens]
    chord_token_to_class = {t: i for i, t in enumerate(chord_tokens)}

    padding_idx = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, padding_idx=padding_idx, chord_token_to_class=chord_token_to_class),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, padding_idx=padding_idx, chord_token_to_class=chord_token_to_class),
    )

    model = DecoderOnlyMusicGenModel(
        vocab_size=len(tokenizer.vocab),
        chord_token_ids=chord_token_ids,
        padding_idx=padding_idx,
    ).to(args.device)

    # Losses
    # - Token CE uses ignore_index=0 (pad), SUM reduction for correct perplexity later
    criterion_tok = nn.CrossEntropyLoss(ignore_index=padding_idx, reduction="sum", label_smoothing=0.1)
    # - Chord CE uses -1 as ignore_index (we pad with -1 in collate), SUM reduction
    criterion_chd = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    lambda_chord = 0.5  # weight for chord head loss

    all_losses: List[float] = []

    for epoch in range(args.epochs):
        model.train()
        step = 0
        epoch_tok_loss_sum = 0.0
        epoch_tok_count = 0

        pbar = tqdm(train_loader, desc=f"Training E{epoch+1}/{args.epochs}")
        for batch in pbar:
            x, y, bar_pos, bar_lbl = batch  # bar_pos/lbl may be None
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()

            logits, chord_logits = model(x, chord_positions=bar_pos.to(args.device) if bar_pos is not None else None)

            # Token loss
            B, T, V = logits.size()
            flat_logits = logits.reshape(B * T, V)
            flat_targets = y.reshape(B * T)
            loss_tok = criterion_tok(flat_logits, flat_targets)
            valid_tok = int((flat_targets != padding_idx).sum().item())

            # Chord loss (optional)
            loss_chd = torch.tensor(0.0, device=x.device)
            if chord_logits is not None and bar_lbl is not None:
                Bc, Mc, C = chord_logits.size()
                flat_chord_logits = chord_logits.reshape(Bc * Mc, C)
                flat_chord_labels = bar_lbl.to(args.device).reshape(Bc * Mc)
                loss_chd = criterion_chd(flat_chord_logits, flat_chord_labels)

            loss = loss_tok + lambda_chord * loss_chd
            loss.backward()
            optimizer.step()

            # Track per-token loss
            if valid_tok > 0:
                all_losses.append(loss.item() / valid_tok)
                epoch_tok_loss_sum += loss_tok.item()
                epoch_tok_count += valid_tok

            step += 1
            if valid_tok > 0:
                pbar.set_postfix({"tok_loss_per_token": f"{(loss_tok.item()/valid_tok):.4f}"})

        # Evaluation
        train_ppl, train_acc = evaluate(model, train_loader, args.device)
        val_ppl, val_acc = evaluate(model, val_loader, args.device)
        print(f"\n🌀 Epoch {epoch+1}/{args.epochs}")
        print(f"   ▶️ Train  — Perplexity: {train_ppl:.2f}, Accuracy: {train_acc:.2%}")
        print(f"   ▶️ Valid  — Perplexity: {val_ppl:.2f}, Accuracy: {val_acc:.2%}")

    plot_losses(all_losses)

    # Save model + vocabulary + chord tokens + token maps
    ckpt = {
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "chord_tokens": chord_tokens,
    }
    # Save token maps if present
    if hasattr(tokenizer, "token_to_id"):
        ckpt["token_to_id"] = tokenizer.token_to_id
    if hasattr(tokenizer, "id_to_token"):
        ckpt["id_to_token"] = tokenizer.id_to_token

    torch.save(ckpt, "musicgen_remi_model.pt")
    print("✅ Model and vocab saved to musicgen_remi_model.pt")


if __name__ == "__main__":
    main()       

