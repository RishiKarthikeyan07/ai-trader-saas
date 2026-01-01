from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List
from app.models.schemas import ScannerPackResponse, ScannerCandidateResponse
from app.core.security import get_current_user
from app.core.supabase import supabase

router = APIRouter()

# Scanner pack definitions
SCANNER_PACKS = [
    {
        "id": "momentum-breakout",
        "name": "Momentum Breakout",
        "description": "Stocks breaking out with strong volume and momentum",
        "icon": "TrendingUp",
        "signal_type": "momentum_breakout"
    },
    {
        "id": "squeeze-expansion",
        "name": "Squeeze → Expansion",
        "description": "Volatility squeeze releasing into expansion phase",
        "icon": "Zap",
        "signal_type": "squeeze_expansion"
    },
    {
        "id": "pullback-continuation",
        "name": "Pullback Continuation",
        "description": "Healthy pullbacks in strong trends",
        "icon": "ArrowDownUp",
        "signal_type": "pullback_continuation"
    },
    {
        "id": "liquidity-sweep",
        "name": "Liquidity Sweep Reversal",
        "description": "Liquidity grabs with reversal setups",
        "icon": "Waves",
        "signal_type": "liquidity_sweep_reversal"
    },
    {
        "id": "relative-strength",
        "name": "Relative Strength Leaders",
        "description": "Sector and market outperformers",
        "icon": "Award",
        "signal_type": "relative_strength"
    }
]


@router.get("/packs", response_model=List[ScannerPackResponse])
async def get_scanner_packs(user_id: str = Depends(get_current_user)):
    """Get available scanner packs"""
    return SCANNER_PACKS


@router.get("/run/latest")
async def get_latest_scanner_run(
    pack_id: str = Query(...),
    user_id: str = Depends(get_current_user)
):
    """Get latest scanner run for a pack"""
    response = supabase.table('scanner_runs') \
        .select('*') \
        .eq('pack_id', pack_id) \
        .order('started_at', desc=True) \
        .limit(1) \
        .execute()

    if not response.data:
        raise HTTPException(status_code=404, detail="No scanner runs found")

    return response.data[0]


@router.get("/run/{run_id}/candidates", response_model=List[ScannerCandidateResponse])
async def get_scanner_candidates(
    run_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get candidates from a scanner run"""
    response = supabase.table('scanner_candidates') \
        .select('*') \
        .eq('scanner_run_id', run_id) \
        .order('pack_score', desc=True) \
        .execute()

    return response.data
