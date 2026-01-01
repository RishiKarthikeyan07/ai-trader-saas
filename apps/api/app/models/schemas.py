from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class UserTier(str, Enum):
    BASIC = "basic"
    PRO = "pro"
    ELITE = "elite"


class SignalStatus(str, Enum):
    WAIT = "wait"
    READY = "ready"
    FILLED = "filled"
    TP1_HIT = "tp1_hit"
    TP2_HIT = "tp2_hit"
    SL_HIT = "sl_hit"
    EXITED = "exited"
    EXPIRED = "expired"


class SignalHorizon(str, Enum):
    INTRADAY = "intraday"
    SWING = "swing"
    POSITIONAL = "positional"


class RiskGrade(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SignalType(str, Enum):
    MOMENTUM_BREAKOUT = "momentum_breakout"
    SQUEEZE_EXPANSION = "squeeze_expansion"
    PULLBACK_CONTINUATION = "pullback_continuation"
    LIQUIDITY_SWEEP_REVERSAL = "liquidity_sweep_reversal"
    RELATIVE_STRENGTH = "relative_strength"


class SignalResponse(BaseModel):
    id: str
    symbol: str
    signal_type: SignalType
    rank: int
    score: float
    confidence: int
    risk_grade: RiskGrade
    horizon: SignalHorizon
    status: SignalStatus
    entry_min: float
    entry_max: float
    stop_loss: float
    target_1: float
    target_2: float
    setup_tags: List[str]
    created_at: datetime
    updated_at: datetime
    confirmed_at: Optional[datetime] = None


class SignalExplanationResponse(BaseModel):
    signal_id: str
    why_now: str
    key_factors: List[str]
    risk_notes: List[str]
    mtf_alignment: Optional[List[dict]] = None


class ScannerPackResponse(BaseModel):
    id: str
    name: str
    description: str
    icon: str
    signal_type: SignalType


class ScannerCandidateResponse(BaseModel):
    id: str
    scanner_run_id: str
    symbol: str
    pack_score: float
    tags: List[str]
    created_at: datetime


class PositionResponse(BaseModel):
    id: str
    user_id: str
    signal_id: str
    symbol: str
    entry_price: float
    quantity: int
    stop_loss: float
    target_1: float
    target_2: float
    status: str
    pnl: Optional[float] = None
    is_paper: bool
    opened_at: datetime
    closed_at: Optional[datetime] = None


class OrderResponse(BaseModel):
    id: str
    user_id: str
    signal_id: str
    symbol: str
    side: str
    type: str
    quantity: int
    price: Optional[float] = None
    status: str
    is_paper: bool
    created_at: datetime
    filled_at: Optional[datetime] = None


class AlertResponse(BaseModel):
    id: str
    user_id: str
    signal_id: Optional[str] = None
    type: str
    message: str
    read: bool
    created_at: datetime


class PipelineRunResponse(BaseModel):
    id: str
    type: str
    status: str
    signals_generated: Optional[int] = None
    signals_updated: Optional[int] = None
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None


class UserProfileResponse(BaseModel):
    id: str
    email: str
    tier: UserTier
    pro_mode_enabled: bool
    cinematic_enabled: bool
    auto_trading_enabled: bool
    created_at: datetime
    updated_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
