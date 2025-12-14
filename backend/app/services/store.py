from __future__ import annotations

import json
from datetime import datetime
from typing import List
try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

from app.core.config import Settings
from app.models.signal import Signal


def _ensure_tables(con):
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id TEXT,
            symbol TEXT,
            signal_type TEXT,
            entry_zone_low DOUBLE,
            entry_zone_high DOUBLE,
            stop_loss DOUBLE,
            target_1 DOUBLE,
            target_2 DOUBLE,
            confidence DOUBLE,
            expected_return DOUBLE,
            expected_volatility DOUBLE,
            tf_alignment JSON,
            smc_score DOUBLE,
            smc_flags JSON,
            model_versions JSON,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            ready_state TEXT,
            notes TEXT
        )
        """
    )


def save_signals(signals: List[Signal], settings: Settings) -> None:
    if not signals or duckdb is None:
        return
    con = duckdb.connect(str(settings.duckdb_path))
    _ensure_tables(con)
    con.execute("BEGIN TRANSACTION")
    for sig in signals:
        con.execute(
            "DELETE FROM signals WHERE id=?",
            [sig.id],
        )
        con.execute(
            """
            INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                sig.id,
                sig.symbol,
                sig.signal_type,
                sig.entry_zone_low,
                sig.entry_zone_high,
                sig.stop_loss,
                sig.target_1,
                sig.target_2,
                sig.confidence,
                sig.expected_return,
                sig.expected_volatility,
                json.dumps(sig.tf_alignment or {}),
                sig.smc_score,
                json.dumps(sig.smc_flags or {}),
                json.dumps(sig.model_versions or {}),
                sig.created_at,
                sig.updated_at,
                sig.ready_state,
                sig.notes,
            ],
        )
    con.execute("COMMIT")
    con.close()


def fetch_latest(limit: int, settings: Settings) -> List[Signal]:
    if duckdb is None:
        return []
    con = duckdb.connect(str(settings.duckdb_path))
    _ensure_tables(con)
    rows = con.execute(
        "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
        [limit],
    ).fetchall()
    con.close()
    signals: List[Signal] = []
    for row in rows:
        (
            id_,
            symbol,
            signal_type,
            entry_zone_low,
            entry_zone_high,
            stop_loss,
            target_1,
            target_2,
            confidence,
            expected_return,
            expected_volatility,
            tf_alignment,
            smc_score,
            smc_flags,
            model_versions,
            created_at,
            updated_at,
            ready_state,
            notes,
        ) = row
        signals.append(
            Signal(
                id=id_,
                symbol=symbol,
                signal_type=signal_type,
                entry_zone_low=entry_zone_low,
                entry_zone_high=entry_zone_high,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                confidence=confidence,
                expected_return=expected_return,
                expected_volatility=expected_volatility,
                tf_alignment=json.loads(tf_alignment) if isinstance(tf_alignment, str) else tf_alignment,
                smc_score=smc_score,
                smc_flags=json.loads(smc_flags) if isinstance(smc_flags, str) else smc_flags,
                model_versions=json.loads(model_versions) if isinstance(model_versions, str) else model_versions,
                created_at=created_at,
                updated_at=updated_at,
                ready_state=ready_state,
                notes=notes,
            )
        )
    return signals


def fetch_by_id(signal_id: str, settings: Settings) -> Signal | None:
    if duckdb is None:
        return None
    con = duckdb.connect(str(settings.duckdb_path))
    _ensure_tables(con)
    row = con.execute("SELECT * FROM signals WHERE id=? LIMIT 1", [signal_id]).fetchone()
    con.close()
    if not row:
        return None
    (
        id_,
        symbol,
        signal_type,
        entry_zone_low,
        entry_zone_high,
        stop_loss,
        target_1,
        target_2,
        confidence,
        expected_return,
        expected_volatility,
        tf_alignment,
        smc_score,
        smc_flags,
        model_versions,
        created_at,
        updated_at,
        ready_state,
        notes,
    ) = row
    return Signal(
        id=id_,
        symbol=symbol,
        signal_type=signal_type,
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        confidence=confidence,
        expected_return=expected_return,
        expected_volatility=expected_volatility,
        tf_alignment=json.loads(tf_alignment) if isinstance(tf_alignment, str) else tf_alignment,
        smc_score=smc_score,
        smc_flags=json.loads(smc_flags) if isinstance(smc_flags, str) else smc_flags,
        model_versions=json.loads(model_versions) if isinstance(model_versions, str) else model_versions,
        created_at=created_at,
        updated_at=updated_at,
        ready_state=ready_state,
        notes=notes,
    )


def update_ready_state(signal_id: str, ready_state: str, settings: Settings) -> None:
    if duckdb is None:
        return
    con = duckdb.connect(str(settings.duckdb_path))
    _ensure_tables(con)
    con.execute(
        "UPDATE signals SET ready_state=?, updated_at=? WHERE id=?",
        [ready_state, datetime.utcnow(), signal_id],
    )
    con.close()


def signal_counts(settings: Settings) -> dict:
    if duckdb is None:
        return {"total": 0, "buy": 0, "sell": 0, "hold": 0}
    con = duckdb.connect(str(settings.duckdb_path))
    _ensure_tables(con)
    total = con.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    buy = con.execute("SELECT COUNT(*) FROM signals WHERE signal_type='BUY'").fetchone()[0]
    sell = con.execute("SELECT COUNT(*) FROM signals WHERE signal_type='SELL'").fetchone()[0]
    hold = con.execute("SELECT COUNT(*) FROM signals WHERE signal_type='HOLD'").fetchone()[0]
    con.close()
    return {"total": total, "buy": buy, "sell": sell, "hold": hold}
