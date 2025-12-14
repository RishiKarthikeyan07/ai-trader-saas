from fastapi import APIRouter, Depends
from app.services.pipeline import run_daily_pipeline, run_hourly_refinement
from app.core.config import get_settings, Settings

router = APIRouter()


@router.post("/pipeline/run-daily")
async def run_daily(settings: Settings = Depends(get_settings)):
    result = run_daily_pipeline(settings=settings)
    return {"status": "ok", "summary": result}


@router.post("/pipeline/run-hourly")
async def run_hourly(settings: Settings = Depends(get_settings)):
    result = run_hourly_refinement(settings=settings)
    return {"status": "ok", "summary": result}
