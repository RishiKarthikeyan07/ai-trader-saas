from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import requests
import pandas as pd
import yfinance as yf
try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

from app.core.config import Settings

TIMEFRAME_MAP = {
    "1h": "60m",
    "4h": "60m",  # fetch hourly then resample to 4H
    "1d": "1d",
    "1w": "1wk",
    "1m": "1mo",
}


def _alpha_vantage_fetch(symbol: str, output_size: str, settings: Settings) -> Optional[pd.DataFrame]:
    if not settings.alphavantage_api_key:
        return None

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": f"{symbol}.BSE" if not symbol.endswith(".NS") else symbol,
        "outputsize": output_size,
        "apikey": settings.alphavantage_api_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        return None
    payload = resp.json()
    series = payload.get("Time Series (Daily)")
    if not series:
        return None
    records = []
    for ts, row in series.items():
        records.append(
            {
                "timestamp": pd.to_datetime(ts),
                "open": float(row["1. open"]),
                "high": float(row["2. high"]),
                "low": float(row["3. low"]),
                "close": float(row["4. close"]),
                "volume": float(row["6. volume"]),
            }
        )
    df = pd.DataFrame(records).sort_values("timestamp").set_index("timestamp")
    return df


def _yfinance_fetch(symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol if symbol.endswith(".NS") else f"{symbol}.NS")
    df = ticker.history(period=period, interval=interval, auto_adjust=False)
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    df = df.rename(columns=str.lower)
    df.index = df.index.tz_convert(None)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_ohlcv(symbol: str, timeframe: str, lookback_days: int, settings: Settings) -> pd.DataFrame:
    cache_path = _cache_path(settings.data_cache_dir, symbol, timeframe)
    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            if not cached.empty:
                last_ts = pd.to_datetime(cached.index.max())
                if datetime.utcnow() - last_ts.to_pydatetime() < timedelta(hours=2):
                    return cached
        except Exception:
            pass

    if timeframe == "1d":
        df = _alpha_vantage_fetch(symbol, "full", settings) or pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty:
        interval = TIMEFRAME_MAP.get(timeframe, "1d")
        period = f"{max(lookback_days, 60)}d" if "min" not in interval else "60d"
        df = _yfinance_fetch(symbol, period=period, interval=interval)

    if df.empty:
        return df

    if timeframe == "4h":
        df = df.resample("4H").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    df = df[df.index >= cutoff]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    _upsert_duckdb(symbol, timeframe, df, settings.duckdb_path)
    return df


def _cache_path(base: Path, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "_")
    return base / f"{safe_symbol}_{timeframe}.parquet"


def _upsert_duckdb(symbol: str, timeframe: str, df: pd.DataFrame, duckdb_path: Path) -> None:
    if df.empty or duckdb is None:
        return
    con = duckdb.connect(str(duckdb_path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT,
            timeframe TEXT,
            ts TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
        """
    )
    index_name = df.index.name or "index"
    df_reset = df.reset_index().rename(columns={index_name: "ts"})
    df_reset["symbol"] = symbol
    df_reset["timeframe"] = timeframe
    df_reset = df_reset[["symbol", "timeframe", "ts", "open", "high", "low", "close", "volume"]]
    con.execute("BEGIN TRANSACTION")
    con.execute("DELETE FROM ohlcv WHERE symbol=? AND timeframe=?", [symbol, timeframe])
    con.register("df_reset", df_reset)
    con.execute("INSERT INTO ohlcv SELECT * FROM df_reset")
    con.execute("COMMIT")
    con.close()


def load_cached(symbol: str, timeframe: str, settings: Settings) -> pd.DataFrame:
    cache_path = _cache_path(settings.data_cache_dir, symbol, timeframe)
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def list_duckdb_symbols(settings: Settings) -> list[str]:
    if duckdb is None:
        return []
    con = duckdb.connect(str(settings.duckdb_path))
    try:
        rows = con.execute("SELECT DISTINCT symbol FROM ohlcv").fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []
    finally:
        con.close()


def run_pk_stock_screener(universe: str = "NSE", top_n: int = 150) -> list[dict]:
    try:
        proc = subprocess.run(
            ["pk-stock-screener", "-e", universe, "--output", "json", "--top", str(top_n)],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            return json.loads(proc.stdout)
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []
    return []


def bhavcopy_ingest(file_path: Path, settings: Settings) -> pd.DataFrame:
    if duckdb is None:
        raise ImportError("duckdb is required for bhavcopy ingestion")
    df = pd.read_csv(file_path)
    required = {"SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "TOTTRDQTY", "TIMESTAMP"}
    if not required.issubset(set(df.columns)):
        raise ValueError("Invalid bhavcopy schema")
    df = df.rename(
        columns={
            "SYMBOL": "symbol",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "TOTTRDQTY": "volume",
            "TIMESTAMP": "timestamp",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    grouped = df.groupby("symbol")
    con = duckdb.connect(str(settings.duckdb_path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT,
            timeframe TEXT,
            ts TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE
        )
        """
    )
    for symbol, g in grouped:
        g = g.sort_values("timestamp")
        con.execute("DELETE FROM ohlcv WHERE symbol=? AND timeframe='1d'", [symbol])
        con.execute(
            "INSERT INTO ohlcv VALUES (?, '1d', ?, ?, ?, ?, ?, ?)",
            [
                [symbol] * len(g),
                g["timestamp"].to_list(),
                g["open"].to_list(),
                g["high"].to_list(),
                g["low"].to_list(),
                g["close"].to_list(),
                g["volume"].to_list(),
            ],
        )
        cache_path = _cache_path(settings.data_cache_dir, symbol, "1d")
        g.set_index("timestamp")[["open", "high", "low", "close", "volume"]].to_parquet(cache_path)
    con.close()
    return df
