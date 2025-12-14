from __future__ import annotations

import torch
from torch import nn


class TFT(nn.Module):
    """Minimal TFT inference/training stub for bands/volatility.

    Expects:
      x_price: (B,120,5)
      x_kronos: (B,512)
      x_context: (B,29)
    Returns dict with ret (B,3), vol_10d (B,1), upper_10d (B,1), lower_10d (B,1).
    """

    def __init__(self, lookback: int = 120, price_dim: int = 5, kronos_dim: int = 512, context_dim: int = 29, emb_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.price_encoder = nn.Sequential(nn.Flatten(), nn.Linear(lookback * price_dim, emb_dim), nn.ReLU())
        self.kronos_proj = nn.Linear(kronos_dim, emb_dim)
        self.context_proj = nn.Linear(context_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.head_ret = nn.Linear(emb_dim, 3)
        self.head_vol = nn.Sequential(nn.Linear(emb_dim, 1), nn.ReLU())
        self.head_upper = nn.Linear(emb_dim, 1)
        self.head_lower = nn.Linear(emb_dim, 1)

    def forward(self, x_price: torch.Tensor, x_kronos: torch.Tensor, x_context: torch.Tensor):
        price_feat = self.price_encoder(x_price)
        kronos_feat = self.kronos_proj(x_kronos)
        context_feat = self.context_proj(x_context)
        fused = self.dropout(price_feat + kronos_feat + context_feat)
        ret_pred = self.head_ret(fused)
        vol_10d = self.head_vol(fused)
        upper_10d = self.head_upper(fused)
        lower_10d = self.head_lower(fused)
        return {
            "ret": ret_pred,
            "vol_10d": vol_10d,
            "upper_10d": upper_10d,
            "lower_10d": lower_10d,
        }
