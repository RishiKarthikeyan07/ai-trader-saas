from __future__ import annotations

import torch
from torch import nn


class StockFormer(nn.Module):
    """Minimal StockFormer inference/training stub.

    Expects:
      x_price: (B,120,5)
      x_kronos: (B,512)
      x_context: (B,29)
    Returns dict with ret (B,3), up_logits (B,3), up_prob (B,3).
    """

    def __init__(self, lookback: int = 120, price_dim: int = 5, kronos_dim: int = 512, context_dim: int = 29, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, ffn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.price_dim = price_dim
        self.price_encoder = nn.Sequential(nn.Flatten(), nn.Linear(lookback * price_dim, d_model), nn.ReLU())
        self.kronos_proj = nn.Linear(kronos_dim, d_model)
        self.context_proj = nn.Linear(context_dim, d_model)
        self.head_ret = nn.Linear(d_model, 3)
        self.head_logits = nn.Linear(d_model, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_price: torch.Tensor, x_kronos: torch.Tensor, x_context: torch.Tensor):
        price_feat = self.price_encoder(x_price)
        kronos_feat = self.kronos_proj(x_kronos)
        context_feat = self.context_proj(x_context)
        fused = self.dropout(price_feat + kronos_feat + context_feat)
        ret_pred = self.head_ret(fused)
        up_logits = self.head_logits(fused)
        up_prob = torch.sigmoid(up_logits)
        return {"ret": ret_pred, "up_logits": up_logits, "up_prob": up_prob}
