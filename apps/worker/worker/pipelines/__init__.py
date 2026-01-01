"""
AutoPilot Trading Pipelines

This module contains all background pipelines for the AutoPilot trading system:

- daily_brain: Daily stock selection and trade intention generation (runs at 7 AM IST)
- executor: Trade execution and position management (runs every 15 min during market hours)
- position_monitor: Position monitoring and notifications (runs every 5 min)
"""

from .daily_brain import DailyBrainPipeline, main as daily_brain_main
from .executor import ExecutorPipeline, main as executor_main
from .position_monitor import PositionMonitorPipeline, main as position_monitor_main

__all__ = [
    "DailyBrainPipeline",
    "ExecutorPipeline",
    "PositionMonitorPipeline",
    "daily_brain_main",
    "executor_main",
    "position_monitor_main",
]
