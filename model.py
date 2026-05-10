from typing import Optional

import torch
import torch.nn as nn

from module import MaskedSelfAttention, PredictionHead, SharedMLP, SumPooling


class AeroSense(nn.Module):
    """Aircraft-level state-to-flow model.

    Input:
        x:    Tensor [batch, num_aircraft, input_dim]
        mask: Bool tensor [batch, num_aircraft], True for valid aircraft.

    Output:
        Tensor [batch, 2], target order [AP, AR].
    """

    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_decoupled_heads: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_decoupled_heads = use_decoupled_heads

        self.encoder = SharedMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        if use_attention:
            self.interaction = MaskedSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.pooling = SumPooling()

        if use_decoupled_heads:
            self.ap_head = PredictionHead(hidden_dim=hidden_dim, dropout=dropout)
            self.ar_head = PredictionHead(hidden_dim=hidden_dim, dropout=dropout)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False):
        h = self.encoder(x)
        attn_weights = None

        if self.use_attention:
            h, attn_weights = self.interaction(h, mask, return_attention=return_attention)

        z = self.pooling(h, mask)

        if self.use_decoupled_heads:
            y_ap = self.ap_head(z)
            y_ar = self.ar_head(z)
            y_hat = torch.cat([y_ap, y_ar], dim=-1)
        else:
            y_hat = self.head(z)

        if return_attention:
            return y_hat, attn_weights
        return y_hat
