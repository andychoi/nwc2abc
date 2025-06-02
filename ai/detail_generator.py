# ai/detail_generator.py

import torch.nn as nn
import torch
from relative_transformer import PositionalEncoding, PositionalEncoding


class DetailGeneratorModel(nn.Module):
    """
    Given a chord plan (e.g. <Chord_I>, <Chord_IV>, ...), we generate a sequence of
    `<voice=...>`, `<BarStart>`, `<RelPos_...>`, `<Note-On_...>`, `<Duration_...>`, `<Velocity_...>`, `<BarStart>`, ...
    for each bar, conditioned on the chord plan tokens.
    """
    def __init__(
        self,
        vocab_size,
        d_model=1024,
        nhead=16,
        num_layers=12,
        dim_feedforward=4096,
        dropout=0.1,
        max_rel_dist=2048
    ):
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

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.positional(emb)
        tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        out = self.decoder(emb, tgt_mask=tgt_mask)
        return self.fc_out(out)
