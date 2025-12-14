#!/usr/bin/env python
"""Build training dataset using yfinance daily OHLCV + Kronos embeddings.

Outputs training_data/v1/dataset.parquet with columns:
- symbol, asof
- ohlcv_norm (120x5), kronos_emb (512), context (29)
- y_ret (3), y_up (3)
- raw_features (dict)

Usage:
  PYTHONPATH=.. python backend/scripts/build_dataset_yf.py \
      --tickers-file nifty200_symbols.txt \
      --start 2020-01-01 --end 2025-12-13
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from tqdm import tqdm

from app.ml.preprocess.normalize import (
    normalize_ohlcv_120,
    build_tf_align_vec,
    build_smc_vec,
    build_ta_vec,
)
from app.services.model_registry import KronosEncoder
from app.services.feature_engine import compute_ta_features, compute_smc_features

LOOKBACK = 120
HORIZONS = [3, 5, 10]
BATCH = 64


def fetch_daily(sym: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(sym, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "date"}, inplace=True)
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        df[f"ret_{h}"] = (df["Close"].shift(-h) / df["Close"]) - 1.0
        df[f"up_{h}"] = (df[f"ret_{h}"] > 0).astype(np.int32)
    return df


def build_alignment(feature_df: pd.DataFrame) -> dict:
    latest = feature_df.iloc[-1]
    ema_fast = latest.get("ema_fast", 0)
    ema_slow = latest.get("ema_slow", 0)
    bias = 1.0 if ema_fast > ema_slow else -1.0
    return {
        "monthly_bias": bias,
        "weekly_bias": bias,
        "daily_bias": bias,
        "h4_align": bias,
        "h1_align": bias,
    }


def compute_feature_vectors(window_df: pd.DataFrame):
    ta_df = compute_ta_features(window_df.set_index("date"))
    smc_df = compute_smc_features(ta_df)
    latest = smc_df.iloc[-1].to_dict()
    alignment = build_alignment(smc_df)
    tf_align = build_tf_align_vec(alignment).astype(np.float32)
    smc_vec = build_smc_vec(latest).astype(np.float32)
    ta_vec = build_ta_vec(latest).astype(np.float32)
    return tf_align, smc_vec, ta_vec, latest


def kronos_encoder(device: str = "cpu") -> KronosEncoder:
    enc = KronosEncoder(artifact_path=None)
    return enc


def kronos_encode_batch(enc: KronosEncoder, batch: np.ndarray, device: str) -> np.ndarray:
    x = torch.tensor(batch, dtype=torch.float32)
    if device != "cpu":
        x = x.to(device)
    with torch.no_grad():
        emb = enc.encode(x)
    return emb if isinstance(emb, np.ndarray) else emb.cpu().numpy()


def build_dataset(tickers: List[str], start: str, end: str, device: str) -> pd.DataFrame:
    enc = kronos_encoder(device=device)
    rows = []
    for sym in tqdm(tickers):
        try:
            df = fetch_daily(sym, start=start, end=end)
            df = add_labels(df)
            if len(df) < LOOKBACK + max(HORIZONS) + 10:
                continue
            batch_norm, batch_meta = [], []
            for i in range(LOOKBACK - 1, len(df) - max(HORIZONS)):
                window = df.loc[i - LOOKBACK + 1 : i].copy()
                ohlcv = window[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
                if ohlcv.shape[0] != LOOKBACK:
                    continue
                ohlcv_norm = normalize_ohlcv_120(ohlcv)
                tf_align, smc_vec, ta_vec, latest_feat = compute_feature_vectors(window)
                context = np.concatenate([tf_align, smc_vec, ta_vec]).astype(np.float32)
                y_ret = np.array([df.loc[i, f"ret_{h}"] for h in HORIZONS], dtype=np.float32)
                y_up = np.array([df.loc[i, f"up_{h}"] for h in HORIZONS], dtype=np.float32)
                if np.any(np.isnan(y_ret)):
                    continue
                batch_norm.append(ohlcv_norm)
                batch_meta.append((sym, df.loc[i, "date"], context, y_ret, y_up, latest_feat))
                if len(batch_norm) >= BATCH:
                    emb = kronos_encode_batch(enc, np.stack(batch_norm, axis=0), device)
                    for meta, m_emb, m_ohlcv in zip(batch_meta, emb, batch_norm):
                        m_sym, m_date, m_ctx, m_ret, m_up, m_feat = meta
                        rows.append(
                            {
                                "symbol": m_sym,
                                "asof": pd.to_datetime(m_date),
                                "ohlcv_norm": m_ohlcv,
                                "kronos_emb": m_emb,
                                "context": m_ctx,
                                "raw_features": m_feat,
                                "y_ret": m_ret,
                                "y_up": m_up,
                            }
                        )
                    batch_norm, batch_meta = [], []
            if batch_norm:
                emb = kronos_encode_batch(enc, np.stack(batch_norm, axis=0), device)
                for meta, m_emb, m_ohlcv in zip(batch_meta, emb, batch_norm):
                    m_sym, m_date, m_ctx, m_ret, m_up, m_feat = meta
                    rows.append(
                        {
                            "symbol": m_sym,
                            "asof": pd.to_datetime(m_date),
                            "ohlcv_norm": m_ohlcv,
                            "kronos_emb": m_emb,
                            "context": m_ctx,
                            "raw_features": m_feat,
                            "y_ret": m_ret,
                            "y_up": m_up,
                        }
                    )
        except Exception as exc:
            print(f"ERR {sym}: {exc}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers-file", type=Path, help="Path to tickers file (one per line)")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-13")
    parser.add_argument("--out", default=Path("training_data/v1/dataset.parquet"), type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.tickers_file and args.tickers_file.exists():
        tickers = [t.strip() for t in args.tickers_file.read_text().splitlines() if t.strip()]
    else:
        tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS", "LT.NS", "KOTAKBANK.NS"]

    df = build_dataset(tickers, start=args.start, end=args.end, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved dataset to {args.out} with {len(df)} samples")


if __name__ == "__main__":
    main()
