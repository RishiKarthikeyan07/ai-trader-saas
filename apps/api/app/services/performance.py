"""
Performance Computation Service

Computes signal outcomes and daily performance metrics for Top-K heatmap
"""

from datetime import datetime, timedelta
from typing import List, Dict
from app.core.supabase import supabase


async def compute_outcomes_job(lookback_days: int = 10):
    """
    Compute outcomes for signals from last N days

    This runs after market close (or next morning) to allow signals
    to resolve within their horizon (5d or 10d).
    """
    print(f"[PERFORMANCE] Computing outcomes for last {lookback_days} days")

    # Get signals from lookback period
    start_date = (datetime.utcnow() - timedelta(days=lookback_days)).isoformat()

    signals = supabase.table('signals') \
        .select('id, created_at') \
        .gte('created_at', start_date) \
        .execute()

    if not signals.data:
        print("[PERFORMANCE] No signals to process")
        return

    # For each signal, compute outcome for both horizons
    for signal in signals.data:
        signal_id = signal['id']

        # Check if enough time has passed for each horizon
        signal_date = datetime.fromisoformat(signal['created_at'].replace('Z', '+00:00'))
        days_since = (datetime.utcnow().replace(tzinfo=signal_date.tzinfo) - signal_date).days

        # Compute 5-day outcome if >= 5 days old
        if days_since >= 5:
            await compute_signal_outcome(signal_id, horizon=5)

        # Compute 10-day outcome if >= 10 days old
        if days_since >= 10:
            await compute_signal_outcome(signal_id, horizon=10)

    print(f"[PERFORMANCE] Processed {len(signals.data)} signals")


async def compute_signal_outcome(signal_id: str, horizon: int):
    """
    Compute outcome for a single signal at a given horizon

    In production, this would:
    1. Fetch actual price data for the symbol
    2. Check if TP or SL was hit within horizon days
    3. Calculate realized R multiple

    For now, uses signal status as proxy.
    """
    # Call Supabase function
    supabase.rpc('compute_signal_outcome', {
        'p_signal_id': signal_id,
        'p_horizon': horizon
    }).execute()


async def compute_daily_performance_job(lookback_days: int = 30):
    """
    Compute daily performance metrics for all K buckets and horizons

    This aggregates resolved signals into daily metrics for the heatmap.
    """
    print(f"[PERFORMANCE] Computing daily performance for last {lookback_days} days")

    # Compute for each day in lookback period
    for days_ago in range(lookback_days):
        target_date = datetime.utcnow().date() - timedelta(days=days_ago)

        # Compute all combinations
        for k_bucket in ['K3', 'K10', 'KALL']:
            for horizon_bucket in ['H5', 'H10', 'BOTH']:
                await compute_daily_performance(target_date, k_bucket, horizon_bucket)

    print(f"[PERFORMANCE] Computed performance for {lookback_days} days")


async def compute_daily_performance(date, k_bucket: str, horizon_bucket: str):
    """
    Compute performance metrics for a specific date/K/horizon combination
    """
    # Call Supabase function
    supabase.rpc('compute_daily_performance', {
        'p_date': date.isoformat(),
        'p_k_bucket': k_bucket,
        'p_horizon_bucket': horizon_bucket
    }).execute()


async def get_performance_heatmap(days: int = 30) -> List[Dict]:
    """
    Get heatmap data for last N days

    Returns:
    [
        {
            "date": "2025-12-26",
            "K3_H5_win_rate": 65.5,
            "K3_H5_avg_r": 2.1,
            "K10_H5_win_rate": 58.2,
            ...
        }
    ]
    """
    start_date = (datetime.utcnow().date() - timedelta(days=days)).isoformat()

    # Fetch all daily performance records
    response = supabase.table('performance_daily') \
        .select('*') \
        .gte('date', start_date) \
        .order('date', desc=True) \
        .execute()

    if not response.data:
        return []

    # Reshape into heatmap format
    heatmap = {}

    for row in response.data:
        date = row['date']
        k = row['k_bucket']
        h = row['horizon_bucket']

        if date not in heatmap:
            heatmap[date] = {'date': date}

        key_prefix = f"{k}_{h}"
        heatmap[date][f"{key_prefix}_win_rate"] = float(row['win_rate'])
        heatmap[date][f"{key_prefix}_avg_r"] = float(row['avg_r'])
        heatmap[date][f"{key_prefix}_total"] = row['total_signals']
        heatmap[date][f"{key_prefix}_profit_factor"] = float(row['profit_factor']) if row['profit_factor'] else None

    # Convert to list sorted by date
    result = sorted(heatmap.values(), key=lambda x: x['date'], reverse=True)

    return result


async def get_performance_summary(days: int = 30) -> Dict:
    """
    Get aggregated performance summary for last N days

    Returns:
    {
        "K3": {
            "H5": {"win_rate": 65.5, "avg_r": 2.1, "total": 90},
            "H10": {...}
        },
        "K10": {...},
        "KALL": {...}
    }
    """
    start_date = (datetime.utcnow().date() - timedelta(days=days)).isoformat()

    response = supabase.table('performance_daily') \
        .select('*') \
        .gte('date', start_date) \
        .execute()

    if not response.data:
        return {}

    # Aggregate by K and H
    summary = {}

    for row in response.data:
        k = row['k_bucket']
        h = row['horizon_bucket']

        if k not in summary:
            summary[k] = {}
        if h not in summary[k]:
            summary[k][h] = {
                'total_signals': 0,
                'total_wins': 0,
                'total_losses': 0,
                'total_r': 0.0,
                'count_days': 0
            }

        summary[k][h]['total_signals'] += row['total_signals']
        summary[k][h]['total_wins'] += row['winning_signals']
        summary[k][h]['total_losses'] += row['losing_signals']
        summary[k][h]['total_r'] += float(row['avg_r']) * row['total_signals']
        summary[k][h]['count_days'] += 1

    # Calculate final metrics
    result = {}

    for k, horizons in summary.items():
        result[k] = {}
        for h, data in horizons.items():
            total = data['total_signals']
            result[k][h] = {
                'win_rate': (data['total_wins'] / total * 100) if total > 0 else 0,
                'avg_r': (data['total_r'] / total) if total > 0 else 0,
                'total_signals': total,
                'total_wins': data['total_wins'],
                'total_losses': data['total_losses'],
                'days': data['count_days']
            }

    return result


async def get_top_k_comparison() -> Dict:
    """
    Get comparison metrics across K buckets for upsell messaging

    Shows Basic users what they're missing by not upgrading.
    """
    summary = await get_performance_summary(days=30)

    if not summary:
        return {}

    comparison = {}

    for k in ['K3', 'K10', 'KALL']:
        if k in summary and 'H5' in summary[k]:
            comparison[k] = {
                'win_rate': summary[k]['H5']['win_rate'],
                'avg_r': summary[k]['H5']['avg_r'],
                'total_signals': summary[k]['H5']['total_signals']
            }

    return comparison
