# ai/train_two_stage.py

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import numpy as np
import random

from music21 import converter, stream, key as m21key, interval, meter

import matplotlib.pyplot as plt

from remi_tokenizer import REMIABCTokenizer
from relative_transformer import RelativeTransformerDecoder  # your relative‐attention decoder
from remi_detokenizer import remi_tokens_to_score  # not used here, but for completeness


# ----------------------------------
# 1) Model Definitions
# ----------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BarPlannerModel(nn.Module):
    """
    Stage 1: BarPlanner predicts a sequence of <Chord_...> tokens (plus global tokens like <time>, <key>, <Tempo>, <PhraseStart>).
    We feed it only the chord‐sequence portion of each score. It is a 6-layer, d_model=512 Transformer‐Decoder.
    """
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=1024)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                   dropout=dropout, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len_chord)
        returns: logits over full vocab, but only chord‐positions will be evaluated.
        """
        emb = self.embedding(x)            # (batch, seq_len_chord, d_model)
        emb = self.positional(emb)         # add positional encodings
        seq_len = x.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1).bool()
        memory = torch.zeros_like(emb)     # dummy memory
        out = self.decoder(emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)            # (batch, seq_len_chord, vocab_size)


class DetailGeneratorModel(nn.Module):
    """
    Stage 2: DetailGenerator sees the entire full-token stream, including chord tokens at the start of each bar,
    and must predict the next token (notes, durations, velocities, barstarts, etc.). We build a large
    12-layer, d_model=1024 Transformer‐Decoder with relative attention.
    """
    def __init__(self, vocab_size: int, d_model: int = 1024, nhead: int = 16, num_layers: int = 12,
                 dim_feedforward: int = 4096, dropout: float = 0.1, max_rel_dist: int = 2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=8192)
        self.decoder = RelativeTransformerDecoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_rel_dist=max_rel_dist
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len_full)
        returns: (batch, seq_len_full, vocab_size)
        """
        emb = self.embedding(x)            # (batch, seq_len_full, d_model)
        emb = self.positional(emb)
        seq_len = x.size(1)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), 1).bool()
        out = self.decoder(emb, tgt_mask=tgt_mask)
        return self.fc_out(out)            # (batch, seq_len_full, vocab_size)


# ----------------------------------
# 2) Data Processing / Dataset Classes
# ----------------------------------

def extract_scores(folder_path: str) -> List[stream.Score]:
    """
    Recursively read all .abc files in folder_path and parse them with music21.
    Returns a list of music21.Score objects.
    """
    scores = []
    for file in Path(folder_path).rglob("*.abc"):
        try:
            score = converter.parse(str(file))
            scores.append(score)
        except Exception as e:
            print(f"❌ Failed to parse {file}: {e}")
    return scores


def extract_chord_sequence(full_tokens: List[str]) -> List[str]:
    """
    Given the full REMI token sequence for a score, filter out only:
      - <time=...>, <key=...>, <Tempo=...>, <PhraseStart>
      - <BarStart> <Chord_xxx> pairs
      - <EOS> at the end
    Ensures that every <BarStart> is immediately followed by its <Chord_...> token.
    """
    chord_seq = []
    saw_bos = False
    for i, tok in enumerate(full_tokens):
        if tok == "<BOS>":
            saw_bos = True
            chord_seq.append(tok)
            continue
        if not saw_bos:
            continue

        # Global tokens (only once at beginning)
        if tok.startswith("<time=") or tok.startswith("<key=") or tok.startswith("<Tempo=") or tok == "<PhraseStart>":
            chord_seq.append(tok)
            continue

        # BarStart → expect next token to be a Chord_...
        if tok == "<BarStart>":
            chord_seq.append(tok)
            # look ahead one token
            if i + 1 < len(full_tokens) and full_tokens[i + 1].startswith("<Chord_"):
                chord_seq.append(full_tokens[i + 1])
            else:
                chord_seq.append("<Chord_unk>")  # fallback if missing
            continue

        # <Chord_...> if it doesn’t directly follow <BarStart> is ignored
        # <EOS> at end
        if tok == "<EOS>":
            chord_seq.append(tok)
            break

    # Ensure <EOS> is present
    if chord_seq[-1] != "<EOS>":
        chord_seq.append("<EOS>")
    return chord_seq


def extract_full_sequence(full_tokens: List[str]) -> List[str]:
    """
    Just returns the full token list again (we assume remi_tokenizer.tokenize already produced
    a well‐formed, single <BOS> … <EOS> sequence).
    """
    return full_tokens.copy()


class BarChordDataset(Dataset):
    """
    Dataset for Stage 1: each item is a sequence of chord‐only tokens.
    We convert them to integer IDs here.
    """
    def __init__(self, tokenizer: REMIABCTokenizer, chord_token_lists: List[List[str]]):
        self.tokenizer = tokenizer
        self.encoded = []
        for chord_seq in chord_token_lists:
            ids = tokenizer.encode(chord_seq)
            self.encoded.append(ids)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        seq = self.encoded[idx]
        x = seq[:-1]
        y = seq[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class FullDetailDataset(Dataset):
    """
    Dataset for Stage 2: each item is the full REMI token sequence (chords + notes + everything).
    We convert them to integer IDs here.
    """
    def __init__(self, tokenizer: REMIABCTokenizer, full_token_lists: List[List[str]]):
        self.tokenizer = tokenizer
        self.encoded = []
        for full_seq in full_token_lists:
            ids = tokenizer.encode(full_seq)
            self.encoded.append(ids)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int):
        seq = self.encoded[idx]
        x = seq[:-1]
        y = seq[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    xs, ys = zip(*batch)
    xs_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=0)
    return xs_padded, ys_padded


# ----------------------------------
# 3) Training Loops
# ----------------------------------

def train_bar_planner(
    model: BarPlannerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: str,
    num_epochs: int = 5,
):
    """
    Standard training loop for the BarPlannerModel. Only uses chord sequences.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for x, y in tqdm(dataloader, desc=f"BarPlanner Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)             # (batch, seq_len_chord, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"    → BarPlanner Epoch {epoch+1} Loss: {avg_loss:.4f}")


def train_detail_generator(
    model: DetailGeneratorModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: str,
    num_epochs: int = 5,
):
    """
    Standard training loop for the DetailGeneratorModel. Uses full token sequences.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for x, y in tqdm(dataloader, desc=f"DetailGenerator Epoch {epoch+1}/{num_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)             # (batch, seq_len_full, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"    → DetailGenerator Epoch {epoch+1} Loss: {avg_loss:.4f}")


# ----------------------------------
# 4) Main: Putting It All Together
# ----------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Training: BarPlanner → DetailGenerator")
    parser.add_argument("--input", required=True, help="Path to folder containing ABC files")
    parser.add_argument("--bar_epochs", type=int, default=5, help="Epochs for BarPlanner")
    parser.add_argument("--detail_epochs", type=int, default=5, help="Epochs for DetailGenerator")
    parser.add_argument("--batch_size", type=int, default=8)
    default_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    args = parser.parse_args()

    print(f"📁 Loading scores from: {args.input}")
    all_scores = extract_scores(args.input)
    print(f"✅ Loaded {len(all_scores)} scores.")

    # 1) Build a unified vocabulary by tokenizing each score once
    tokenizer = REMIABCTokenizer()
    all_full_tokens: List[List[str]] = []

    print("🔨 Building full token lists (to grow vocab)…")
    for score in tqdm(all_scores, desc="Tokenizing Scores"):
        # 1a) Transpose to C major/minor
        orig_key = score.analyze("key")
        if orig_key.mode == "major":
            target_key = m21key.Key("C")
        else:
            target_key = m21key.Key("C", "minor")
        iv = interval.Interval(orig_key.tonic, target_key.tonic)
        score_C = score.transpose(iv)

        # 1b) Fully tokenize into REMI tokens
        full_tokens = tokenizer.tokenize(score_C)  # this also grows tokenizer.vocab
        all_full_tokens.append(full_tokens)

    print(f"🔑 Vocabulary size after scanning: {len(tokenizer.vocab)} tokens.")

    # 2) Extract chord-only sequences
    all_chord_sequences = [extract_chord_sequence(full) for full in all_full_tokens]

    # 3) Extract full detail sequences (just copy)
    all_detail_sequences = [extract_full_sequence(full) for full in all_full_tokens]

    # 4) Split into train/validation (80/20) by index
    random.seed(0)
    n = len(all_scores)
    idxs = list(range(n))
    random.shuffle(idxs)
    n_val = max(1, int(0.2 * n))
    train_idxs, val_idxs = idxs[n_val:], idxs[:n_val]

    # 5) Build BarChordDataset (train + val)
    train_chords = [all_chord_sequences[i] for i in train_idxs]
    val_chords = [all_chord_sequences[i] for i in val_idxs]
    train_chord_dataset = BarChordDataset(tokenizer, train_chords)
    val_chord_dataset   = BarChordDataset(tokenizer, val_chords)

    # 6) Dataloaders for BarPlanner
    bar_train_loader = DataLoader(train_chord_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    bar_val_loader   = DataLoader(val_chord_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 7) Instantiate BarPlannerModel
    bar_model = BarPlannerModel(vocab_size=len(tokenizer.vocab), d_model=512, nhead=8, num_layers=6, dropout=0.1)
    bar_optimizer = torch.optim.AdamW(bar_model.parameters(), lr=1e-4, weight_decay=1e-2)
    bar_criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 8) Pretrain BarPlannerModel
    print("\n🎬 Stage 1: Pretraining BarPlannerModel (chord sequences)")
    train_bar_planner(bar_model, bar_train_loader, bar_optimizer, bar_criterion, args.device, num_epochs=args.bar_epochs)

    # Optionally: evaluate on validation chord sequences
    bar_model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_tokens = 0
        for x, y in bar_val_loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = bar_model(x)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss = bar_criterion(logits_flat, y_flat)
            total_loss += loss.item() * y_flat.size(0)
            total_tokens += y_flat.size(0)
        val_perplexity = np.exp(total_loss / total_tokens)
        print(f"    ▶️ BarPlanner Validation Perplexity: {val_perplexity:.2f}")

    # 9) Build FullDetailDataset (train + val)
    train_details = [all_detail_sequences[i] for i in train_idxs]
    val_details   = [all_detail_sequences[i] for i in val_idxs]
    train_detail_dataset = FullDetailDataset(tokenizer, train_details)
    val_detail_dataset   = FullDetailDataset(tokenizer, val_details)

    detail_train_loader = DataLoader(train_detail_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    detail_val_loader   = DataLoader(val_detail_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 10) Instantiate DetailGeneratorModel
    detail_model = DetailGeneratorModel(
        vocab_size=len(tokenizer.vocab),
        d_model=1024,
        nhead=16,
        num_layers=12,
        dim_feedforward=4096,
        dropout=0.1,
        max_rel_dist=2048
    )
    detail_optimizer = torch.optim.AdamW(detail_model.parameters(), lr=1e-4, weight_decay=1e-2)
    detail_criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 11) Train DetailGeneratorModel
    print("\n🎬 Stage 2: Training DetailGeneratorModel (full detail sequences)")
    train_detail_generator(detail_model, detail_train_loader, detail_optimizer, detail_criterion, args.device, num_epochs=args.detail_epochs)

    # Optionally: evaluate on validation detail sequences
    detail_model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_tokens = 0
        correct = 0
        for x, y in detail_val_loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = detail_model(x)
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            loss = detail_criterion(logits_flat, y_flat)
            total_loss += loss.item() * y_flat.size(0)
            total_tokens += y_flat.size(0)
            preds = logits_flat.argmax(dim=-1)
            mask = y_flat != 0
            correct += (preds[mask] == y_flat[mask]).sum().item()

        val_perplexity = np.exp(total_loss / total_tokens)
        val_accuracy = correct / total_tokens
        print(f"    ▶️ DetailGenerator Validation Perplexity: {val_perplexity:.2f}, Accuracy: {val_accuracy:.2%}")

    # 12) Save both models and the shared vocabulary
    torch.save({
        "bar_model_state_dict": bar_model.state_dict(),
        "detail_model_state_dict": detail_model.state_dict(),
        "vocab": tokenizer.vocab
    }, "two_stage_musicgen.pt")
    print("✅ Saved two-stage checkpoint to two_stage_musicgen.pt")


if __name__ == "__main__":
    main()
