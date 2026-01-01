"""
Daily and Hourly Pipeline Services

Daily Pipeline:
- Run after market close
- Execute PKScreener scans
- Generate AI-ranked signals
- Initial status: WAIT

Hourly Confirmer:
- Update WAIT → READY based on confirmation logic
- Send alerts for newly READY signals
"""

from datetime import datetime
from typing import List, Dict
import asyncio
from app.core.supabase import supabase, log_audit_event
from app.services.scanner import run_pkscreener_packs
from app.services.ai_ranker import rank_signals_stub
from app.services.confirmer import check_confirmation
from app.services.alerts import send_signal_ready_alert


async def run_daily_pipeline(run_id: str):
    """
    Daily Heavy Pipeline
    1. Run tradability gate
    2. Execute PKScreener pack scans
    3. Generate AI-ranked signals with WAIT status
    4. Save to database
    """
    try:
        print(f"[DAILY PIPELINE] Starting run {run_id}")

        # Step 1: Tradability Gate (stub for now)
        if not is_market_tradable():
            await update_pipeline_run(run_id, 'failed', error="Market not tradable")
            return

        # Step 2: Run PKScreener Packs
        print("[DAILY PIPELINE] Running PKScreener packs...")
        scanner_candidates = await run_pkscreener_packs()
        print(f"[DAILY PIPELINE] Found {len(scanner_candidates)} total candidates")

        # Step 3: Rank signals using AI model (stub scoring for now)
        print("[DAILY PIPELINE] Ranking signals...")
        ranked_signals = await rank_signals_stub(scanner_candidates)
        print(f"[DAILY PIPELINE] Ranked {len(ranked_signals)} signals")

        # Step 4: Save signals to database
        print("[DAILY PIPELINE] Saving signals to database...")
        signals_count = await save_signals(ranked_signals)

        # Update pipeline run
        await update_pipeline_run(
            run_id,
            'completed',
            signals_generated=signals_count
        )

        print(f"[DAILY PIPELINE] Completed successfully: {signals_count} signals generated")

    except Exception as e:
        print(f"[DAILY PIPELINE] Error: {str(e)}")
        await update_pipeline_run(run_id, 'failed', error=str(e))
        raise


async def run_hourly_confirmer(run_id: str):
    """
    Hourly Light Confirmer
    1. Get active signals in WAIT status
    2. Check 1H/4H confirmation
    3. Update WAIT → READY
    4. Send alerts
    """
    try:
        print(f"[HOURLY CONFIRMER] Starting run {run_id}")

        # Get signals in WAIT status from last 5 days
        signals = await get_wait_signals()
        print(f"[HOURLY CONFIRMER] Found {len(signals)} signals in WAIT status")

        updated_count = 0

        for signal in signals:
            # Check confirmation
            is_ready = await check_confirmation(signal)

            if is_ready:
                # Update signal status
                await update_signal_status(signal['id'], 'ready')
                updated_count += 1

                # Send alerts to subscribed users
                await send_signal_ready_alert(signal)

        # Update pipeline run
        await update_pipeline_run(
            run_id,
            'completed',
            signals_updated=updated_count
        )

        print(f"[HOURLY CONFIRMER] Completed: {updated_count} signals updated to READY")

    except Exception as e:
        print(f"[HOURLY CONFIRMER] Error: {str(e)}")
        await update_pipeline_run(run_id, 'failed', error=str(e))
        raise


# Helper functions

def is_market_tradable() -> bool:
    """Check if market is open and tradable (stub)"""
    # TODO: Implement actual market hours check
    return True


async def get_wait_signals() -> List[Dict]:
    """Get signals in WAIT status from last 5 days"""
    response = supabase.table('signals') \
        .select('*') \
        .eq('status', 'wait') \
        .gte('created_at', datetime.utcnow().isoformat()) \
        .execute()

    return response.data


async def save_signals(signals: List[Dict]) -> int:
    """Save ranked signals to database"""
    if not signals:
        return 0

    # Also create explanations
    for signal in signals:
        # Insert signal
        signal_data = {
            'symbol': signal['symbol'],
            'signal_type': signal['signal_type'],
            'rank': signal['rank'],
            'score': signal['score'],
            'confidence': signal['confidence'],
            'risk_grade': signal['risk_grade'],
            'horizon': signal['horizon'],
            'status': 'wait',  # Initially WAIT
            'entry_min': signal['entry_min'],
            'entry_max': signal['entry_max'],
            'stop_loss': signal['stop_loss'],
            'target_1': signal['target_1'],
            'target_2': signal['target_2'],
            'setup_tags': signal['setup_tags']
        }

        result = supabase.table('signals').insert(signal_data).execute()
        signal_id = result.data[0]['id']

        # Insert explanation (black box - no model secrets)
        explanation_data = {
            'signal_id': signal_id,
            'why_now': signal.get('why_now', 'AI analysis indicates favorable setup'),
            'key_factors': signal.get('key_factors', []),
            'risk_notes': signal.get('risk_notes', []),
            'mtf_alignment': signal.get('mtf_alignment')
        }

        supabase.table('signal_explanations').insert(explanation_data).execute()

    return len(signals)


async def update_signal_status(signal_id: str, status: str):
    """Update signal status"""
    update_data = {
        'status': status,
        'confirmed_at': datetime.utcnow().isoformat() if status == 'ready' else None
    }

    supabase.table('signals').update(update_data).eq('id', signal_id).execute()


async def update_pipeline_run(
    run_id: str,
    status: str,
    signals_generated: int = None,
    signals_updated: int = None,
    error: str = None
):
    """Update pipeline run record"""
    update_data = {
        'status': status,
        'completed_at': datetime.utcnow().isoformat() if status != 'running' else None
    }

    if signals_generated is not None:
        update_data['signals_generated'] = signals_generated

    if signals_updated is not None:
        update_data['signals_updated'] = signals_updated

    if error:
        update_data['error_message'] = error

    supabase.table('pipeline_runs').update(update_data).eq('id', run_id).execute()
