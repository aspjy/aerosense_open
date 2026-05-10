from typing import Optional

import torch
import torch.nn as nn


class SharedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        bsz, n_aircraft, feat_dim = x.shape
        h = self.net(x.reshape(-1, feat_dim))
        return h.reshape(bsz, n_aircraft, -1)


class MaskedSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor, return_attention: bool = False):
        # mask: True for valid aircraft, False for padding.
        key_padding_mask = ~mask if mask is not None else None
        attn_out, attn_weights = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.norm(h + attn_out)
        if return_attention:
            return h, attn_weights
        return h, None


class SumPooling(nn.Module):
    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return torch.sum(h, dim=1)
        return torch.sum(h * mask.unsqueeze(-1).float(), dim=1)


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
