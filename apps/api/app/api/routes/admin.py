from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
from app.models.schemas import PipelineRunResponse
from app.core.security import get_current_user
from app.core.supabase import supabase
from app.services.pipeline import run_daily_pipeline, run_hourly_confirmer
from app.services.performance import compute_outcomes_job, compute_daily_performance_job
from datetime import datetime

router = APIRouter()


async def is_admin(user_id: str = Depends(get_current_user)) -> str:
    """Check if user is admin"""
    response = supabase.table('profiles') \
        .select('is_admin') \
        .eq('user_id', user_id) \
        .single() \
        .execute()

    if not response.data or not response.data.get('is_admin'):
        raise HTTPException(status_code=403, detail="Admin access required")

    return user_id


@router.post("/pipeline/run-daily")
async def trigger_daily_pipeline(
    background_tasks: BackgroundTasks,
    user_id: str = Depends(is_admin)
):
    """Trigger daily pipeline (Admin only)"""
    # Create pipeline run record
    run_data = {
        'type': 'daily',
        'status': 'running',
        'started_at': datetime.utcnow().isoformat()
    }

    response = supabase.table('pipeline_runs').insert(run_data).execute()
    run_id = response.data[0]['id']

    # Run pipeline in background
    background_tasks.add_task(run_daily_pipeline, run_id)

    return {"status": "started", "run_id": run_id}


@router.post("/pipeline/run-hourly")
async def trigger_hourly_pipeline(
    background_tasks: BackgroundTasks,
    user_id: str = Depends(is_admin)
):
    """Trigger hourly confirmer (Admin only)"""
    # Create pipeline run record
    run_data = {
        'type': 'hourly',
        'status': 'running',
        'started_at': datetime.utcnow().isoformat()
    }

    response = supabase.table('pipeline_runs').insert(run_data).execute()
    run_id = response.data[0]['id']

    # Run confirmer in background
    background_tasks.add_task(run_hourly_confirmer, run_id)

    return {"status": "started", "run_id": run_id}


@router.get("/logs")
async def get_logs(
    limit: int = 100,
    user_id: str = Depends(is_admin)
):
    """Get audit logs (Admin only)"""
    response = supabase.table('audit_logs') \
        .select('*') \
        .order('created_at', desc=True) \
        .limit(limit) \
        .execute()

    return response.data


@router.get("/pipeline/runs", response_model=List[PipelineRunResponse])
async def get_pipeline_runs(
    limit: int = 20,
    user_id: str = Depends(is_admin)
):
    """Get pipeline run history (Admin only)"""
    response = supabase.table('pipeline_runs') \
        .select('*') \
        .order('started_at', desc=True) \
        .limit(limit) \
        .execute()

    return response.data


@router.post("/performance/compute-outcomes")
async def trigger_compute_outcomes(
    background_tasks: BackgroundTasks,
    lookback_days: int = 10,
    user_id: str = Depends(is_admin)
):
    """
    Trigger signal outcome computation (Admin only)

    Computes whether signals hit TP or SL within their horizon.
    Should run daily after market close.
    """
    background_tasks.add_task(compute_outcomes_job, lookback_days)
    return {"status": "started", "lookback_days": lookback_days}


@router.post("/performance/compute-daily")
async def trigger_compute_daily(
    background_tasks: BackgroundTasks,
    lookback_days: int = 30,
    user_id: str = Depends(is_admin)
):
    """
    Trigger daily performance aggregation (Admin only)

    Aggregates resolved signals into daily metrics for heatmap.
    Should run daily after outcome computation.
    """
    background_tasks.add_task(compute_daily_performance_job, lookback_days)
    return {"status": "started", "lookback_days": lookback_days}
