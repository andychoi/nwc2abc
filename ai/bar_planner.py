# ai/bar_planner.py

import torch.nn as nn
import torch
from relative_transformer import PositionalEncoding

class BarPlannerModel(nn.Module):
    """
    Given a prefix of <time=...>, <key=...>, <BarStart>, <Chord_...>, predict next <Chord_...>.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model, max_len=1024)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        # We'll only decode positions that correspond to <BarStart> events.

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.positional(emb)
        tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        memory = torch.zeros_like(emb)
        out = self.decoder(emb, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(out)
        return logits  # (batch, seq_len, vocab_size)
