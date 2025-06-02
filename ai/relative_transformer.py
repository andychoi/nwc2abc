# ai/relative_transformer.py

import torch
import torch.nn as nn


class RelativeMultiheadAttention(nn.Module):
    """
    A drop‐in replacement for nn.MultiheadAttention that adds
    relative positional biases. This follows the Music Transformer approach:
    we compute a learned R^{(length, length)} bias for each head, indexing
    by (i-j) relative distance.
    """
    def __init__(self, d_model, nhead, max_rel_dist=1024, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # relative bias table: for distances in [-max_rel_dist, +max_rel_dist]
        self.max_rel_dist = max_rel_dist
        self.rel_bias = nn.Parameter(
            torch.zeros((2 * max_rel_dist + 1, nhead))
        )

    def forward(self, x, attn_mask=None):
        # x: (batch, seq_len, d_model)
        bsz, seq_len, d_model = x.shape
        # 1) Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*d_model)
        qkv = qkv.view(bsz, seq_len, 3, self.nhead, self.d_k)
        q, k, v = qkv.unbind(dim=2)  # each is (batch, seq_len, nhead, d_k)

        # 2) reshape for multihead: (batch*nhead, seq_len, d_k)
        q = q.transpose(1, 2).reshape(bsz * self.nhead, seq_len, self.d_k)
        k = k.transpose(1, 2).reshape(bsz * self.nhead, seq_len, self.d_k)
        v = v.transpose(1, 2).reshape(bsz * self.nhead, seq_len, self.d_k)

        # 3) Scaled dot‐product attention
        scores = torch.bmm(q, k.transpose(1, 2))  # (bsz*nhead, seq_len, seq_len)
        scores = scores / (self.d_k ** 0.5)

        # 4) Add relative position bias
        #    Compute a matrix R where R_{i,j} = rel_bias[i-j + max_rel_dist]
        #    Clip distances outside [-max_rel_dist, max_rel_dist].
        device = x.device
        idxs = torch.arange(seq_len, device=device)
        rel_pos = idxs.unsqueeze(1) - idxs.unsqueeze(0)  # (seq_len, seq_len)
        rel_pos_clamped = torch.clamp(
            rel_pos + self.max_rel_dist, 0, 2 * self.max_rel_dist
        )  # shift into [0, 2*max_rel_dist]
        # rel_bias_table: (2*max_rel_dist+1, nhead)
        bias = self.rel_bias[rel_pos_clamped]  # (seq_len, seq_len, nhead)
        bias = bias.permute(2, 0, 1).contiguous()  # (nhead, seq_len, seq_len)
        bias = bias.view(self.nhead * 1, seq_len, seq_len)  # repeat for each batch
        # Since scores is (bsz*nhead, seq_len, seq_len), tile bias
        bias = bias.repeat(bsz, 1, 1)  # (bsz*nhead, seq_len, seq_len)
        scores = scores + bias

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == True, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)  # (bsz*nhead, seq_len, d_k)
        out = out.view(bsz, self.nhead, seq_len, self.d_k)
        out = out.transpose(1, 2).reshape(bsz, seq_len, d_model)
        out = self.out_proj(out)
        return out


class RelativeTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, max_rel_dist=1024):
        super().__init__()
        self.self_attn = RelativeMultiheadAttention(d_model, nhead, max_rel_dist, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, tgt_mask=None):
        # Self‐attention with relative bias
        sa = self.self_attn(x, attn_mask=tgt_mask)
        x = x + self.dropout1(sa)
        x = self.norm1(x)

        # Feed‐forward
        ff = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ff)
        x = self.norm2(x)
        return x


class RelativeTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout, max_rel_dist=1024):
        super().__init__()
        self.layers = nn.ModuleList([
            RelativeTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_rel_dist=max_rel_dist
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, tgt_mask):
        out = x
        for layer in self.layers:
            out = layer(out, tgt_mask=tgt_mask)
        return out
