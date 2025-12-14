#!/usr/bin/env python
"""Single-symbol replay to sanity check pipeline outputs over recent history.

Usage: PYTHONPATH=.. python scripts/replay_single_symbol.py --symbol RELIANCE --mode stub
In prod mode, ensure model artifacts are configured.
"""
import argparse
from datetime import datetime, timedelta
import sys

import pandas as pd

from app.core.config import get_settings, Settings
from app.services.feature_engine import build_feature_set
from app.services.model_registry import ModelRegistry


def replay(symbol: str, settings: Settings, days: int = 180):
    registry = ModelRegistry(settings)
    feature_set = build_feature_set(symbol, settings=settings)
    daily = feature_set.get("daily")
    if daily is None or daily.empty:
        print(f"No data for {symbol}")
        return
    cutoff = datetime.utcnow() - timedelta(days=days)
    daily = daily[daily.index >= cutoff]
    for ts, row in daily.iterrows():
        latest = row.to_dict()
        out = registry.infer(symbol, latest)
        print(
            f"{ts.date()} {symbol} dir_prob={out.direction_prob:.3f} exp_ret={out.expected_return:.4f} vol={out.expected_volatility:.4f} veto={out.veto_score:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--days", type=int, default=180)
    args = parser.parse_args()
    settings = get_settings()
    replay(args.symbol, settings, days=args.days)


if __name__ == "__main__":
    sys.exit(main())
