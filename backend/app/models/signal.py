from __future__ import annotations

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

SignalType = Literal["BUY", "SELL", "HOLD"]


class Signal(BaseModel):
    id: str
    symbol: str
    signal_type: SignalType
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    target_1: float
    target_2: float
    confidence: float = Field(ge=0.0, le=1.0)
    expected_return: float | None = None
    expected_volatility: float | None = None
    tf_alignment: dict[str, float] | None = None
    smc_score: float | None = None
    smc_flags: dict[str, float] | None = None
    model_versions: dict[str, str] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    ready_state: Literal["READY_TO_ENTER", "WAIT"] | None = None
    notes: str | None = None


class TierGatedSignals(BaseModel):
    tier: Literal["basic", "pro", "elite"]
    signals: list[Signal]
