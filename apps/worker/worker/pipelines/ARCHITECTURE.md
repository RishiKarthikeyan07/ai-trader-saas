# AutoPilot Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      AUTOPILOT TRADING SYSTEM                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE ORCHESTRATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ Daily Brain  │      │  Executor    │      │  Position    │  │
│  │              │      │              │      │  Monitor     │  │
│  │  7:00 AM     │      │  Every 15m   │      │  Every 5m    │  │
│  │  (IST)       │      │  (9:15-15:30)│      │  (Continuous)│  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                     │           │
└─────────┼─────────────────────┼─────────────────────┼───────────┘
          │                     │                     │
          ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SUPABASE DATABASE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ trade_intentions│  │   positions     │  │  notifications  │ │
│  │                 │  │                 │  │                 │ │
│  │ - Date          │  │ - User ID       │  │ - Type          │ │
│  │ - Symbol        │  │ - Symbol        │  │ - Title         │ │
│  │ - Entry zones   │  │ - Quantity      │  │ - Message       │ │
│  │ - SL/TP levels  │  │ - Entry price   │  │ - Severity      │ │
│  │ - Confidence    │  │ - Current price │  │                 │ │
│  │ - Risk grade    │  │ - Unrealized P&L│  │                 │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │    orders       │  │  risk_limits    │  │  pipeline_runs  │ │
│  │                 │  │                 │  │                 │ │
│  │ - Order type    │  │ - Max positions │  │ - Type          │ │
│  │ - Side (B/S)    │  │ - Max loss      │  │ - Status        │ │
│  │ - Quantity      │  │ - Max exposure  │  │ - Duration      │ │
│  │ - Price         │  │ - Risk %/trade  │  │ - Metadata      │ │
│  │ - Status        │  │ - Capital       │  │ - Errors        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Pipeline Execution Flow

### 1. Daily Brain Pipeline (7:00 AM IST)

```
START
  │
  ├─► CREATE pipeline_run (type: daily_brain, status: RUNNING)
  │
  ├─► FETCH tradable instruments
  │    └─► SELECT * FROM instruments WHERE is_tradable = true
  │
  ├─► PKSCREENER stub (select random 50)
  │    └─► TODO: Replace with actual PKScreener logic
  │
  ├─► AI RANKING stub (assign confidence scores 0.6-0.9)
  │    └─► TODO: Replace with trained AI models
  │
  ├─► GENERATE trade intentions (top 20)
  │    ├─► Calculate entry zones (price ± 2%)
  │    ├─► Calculate SL (entry - 3%)
  │    ├─► Calculate TP1 (entry + 5%), TP2 (entry + 8%)
  │    ├─► Assign random risk grade (LOW/MEDIUM/HIGH)
  │    └─► Assign random horizon (INTRADAY/SWING/POSITIONAL)
  │
  ├─► INSERT into trade_intentions
  │    └─► UPSERT (on_conflict: date + canonical_symbol)
  │
  ├─► UPDATE pipeline_run (status: SUCCESS)
  │
END
```

### 2. Executor Pipeline (Every 15 min, 9:15 AM - 3:30 PM IST)

```
START
  │
  ├─► CHECK market hours
  │    └─► IF outside hours → SKIP
  │
  ├─► CREATE pipeline_run (type: executor, status: RUNNING)
  │
  ├─► GET autopilot users
  │    └─► SELECT * FROM profiles
  │         WHERE autopilot_enabled = true
  │         AND is_active_subscriber = true
  │
  ├─► FOR EACH user:
  │    │
  │    ├─► GET risk_limits
  │    │
  │    ├─► CHECK risk limits
  │    │    ├─► Current exposure < max_exposure?
  │    │    └─► Daily P&L > -max_daily_loss?
  │    │
  │    ├─► IF risk limits OK:
  │    │    │
  │    │    ├─► GET today's trade_intentions
  │    │    │
  │    │    ├─► GET user's current positions
  │    │    │
  │    │    ├─► FOR EACH intention (top confidence first):
  │    │    │    │
  │    │    │    ├─► IF symbol not in current positions
  │    │    │    │    AND positions < max_positions:
  │    │    │    │    │
  │    │    │    │    ├─► CALCULATE position size
  │    │    │    │    │    └─► qty = (capital × risk%) / (entry - SL)
  │    │    │    │    │
  │    │    │    │    ├─► CREATE BUY order
  │    │    │    │    │    └─► INSERT INTO orders (is_paper_trade = true)
  │    │    │    │    │
  │    │    │    │    ├─► CREATE position
  │    │    │    │    │    └─► INSERT INTO positions (status = OPEN)
  │    │    │    │    │
  │    │    │    │    └─► SEND notification (POSITION_OPENED)
  │    │    │    │
  │    │    │    └─► ELSE: skip (already have position or max reached)
  │    │    │
  │    │    └─► END FOR
  │    │
  │    ├─► PROCESS exits (always run, even if risk breach):
  │    │    │
  │    │    ├─► GET all OPEN positions
  │    │    │
  │    │    ├─► FOR EACH position:
  │    │    │    │
  │    │    │    ├─► GET current price (stub: use entry price)
  │    │    │    │
  │    │    │    ├─► CHECK exit conditions:
  │    │    │    │    ├─► current_price <= SL → EXIT (SL_HIT)
  │    │    │    │    ├─► current_price >= TP2 → EXIT (TP2_HIT)
  │    │    │    │    └─► current_price >= TP1 → EXIT (TP1_HIT)
  │    │    │    │
  │    │    │    ├─► IF exit triggered:
  │    │    │    │    │
  │    │    │    │    ├─► CREATE SELL order
  │    │    │    │    │
  │    │    │    │    ├─► UPDATE position (status = CLOSED)
  │    │    │    │    │
  │    │    │    │    └─► SEND notification (POSITION_CLOSED)
  │    │    │    │
  │    │    │    └─► ELSE: continue monitoring
  │    │    │
  │    │    └─► END FOR
  │    │
  │    └─► END user processing
  │
  ├─► UPDATE pipeline_run (status: SUCCESS, metadata: stats)
  │
END
```

### 3. Position Monitor Pipeline (Every 5 min)

```
START
  │
  ├─► CREATE pipeline_run (type: position_monitor, status: RUNNING)
  │
  ├─► GET all OPEN positions
  │    └─► SELECT * FROM positions WHERE status = 'OPEN'
  │
  ├─► FOR EACH position:
  │    │
  │    ├─► GET current price
  │    │    └─► TODO: yfinance.Ticker(symbol).info['currentPrice']
  │    │    └─► STUB: Simulate random movement (±5%)
  │    │
  │    ├─► CALCULATE unrealized P&L
  │    │    ├─► P&L = (current_price - entry_price) × quantity
  │    │    └─► P&L% = (current_price - entry_price) / entry_price
  │    │
  │    ├─► UPDATE position
  │    │    └─► UPDATE positions SET
  │    │         current_price = ...,
  │    │         unrealized_pnl = ...,
  │    │         unrealized_pnl_percent = ...
  │    │
  │    ├─► IF trailing_stop_enabled:
  │    │    │
  │    │    ├─► GET trailing_stop_data
  │    │    │    └─► {highest_price, initial_sl, trail_percent}
  │    │    │
  │    │    ├─► UPDATE highest_price
  │    │    │    └─► highest = max(current, highest)
  │    │    │
  │    │    ├─► CALCULATE new SL
  │    │    │    └─► new_sl = highest × (1 - trail_percent)
  │    │    │
  │    │    ├─► IF new_sl > current_sl:
  │    │    │    └─► UPDATE position (sl = new_sl)
  │    │    │
  │    │    └─► END IF
  │    │
  │    ├─► CHECK notification conditions:
  │    │    │
  │    │    ├─► IF current_price >= TP1:
  │    │    │    └─► SEND notification (TP1_HIT)
  │    │    │
  │    │    ├─► IF current_price >= TP2:
  │    │    │    └─► SEND notification (TP2_HIT)
  │    │    │
  │    │    ├─► IF current_price <= SL:
  │    │    │    └─► SEND notification (SL_HIT, severity: WARNING)
  │    │    │
  │    │    ├─► IF P&L% < -10%:
  │    │    │    └─► SEND notification (LARGE_DRAWDOWN, severity: WARNING)
  │    │    │
  │    │    └─► IF P&L% > +15%:
  │    │         └─► SEND notification (LARGE_PROFIT, severity: INFO)
  │    │
  │    └─► END position processing
  │
  ├─► UPDATE pipeline_run (status: SUCCESS, metadata: stats)
  │
END
```

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    DAILY BRAIN PIPELINE                       │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Generates 20 trade intentions
                         │
                         ▼
              ┌──────────────────────┐
              │ trade_intentions     │
              │                      │
              │ [RELIANCE, BUY, ...]│
              │ [TCS, BUY, ...]     │
              │ [INFY, BUY, ...]    │
              │ ...                 │
              └──────────┬───────────┘
                         │
                         │ Read by executor
                         │
                         ▼
              ┌──────────────────────────────────┐
              │      EXECUTOR PIPELINE            │
              │                                   │
              │  For each autopilot user:        │
              │  - Check risk limits             │
              │  - Create positions (BUY orders) │
              │  - Monitor exits (SELL orders)   │
              └──────────┬───────────────────────┘
                         │
                         │ Creates positions & orders
                         │
          ┌──────────────┴──────────────┐
          │                              │
          ▼                              ▼
┌──────────────────┐          ┌──────────────────┐
│   positions      │          │     orders       │
│                  │          │                  │
│ [User A, LONG,  │          │ [User A, BUY,   │
│  RELIANCE, 100] │          │  RELIANCE, 100] │
│                  │          │                  │
│ [User A, LONG,  │          │ [User A, BUY,   │
│  TCS, 50]       │          │  TCS, 50]       │
│                  │          │                  │
└────────┬─────────┘          └──────────────────┘
         │
         │ Monitored by position_monitor
         │
         ▼
┌──────────────────────────────────┐
│   POSITION MONITOR PIPELINE       │
│                                   │
│  - Fetch current prices           │
│  - Update P&L                     │
│  - Check trailing stops           │
│  - Send notifications             │
└──────────┬───────────────────────┘
           │
           │ Sends notifications
           │
           ▼
┌──────────────────┐
│  notifications   │
│                  │
│ [TP1 Hit: TCS]  │
│ [SL Hit: INFY]  │
│ [Large Profit]  │
└──────────────────┘
```

## Component Interactions

```
┌────────────────────────────────────────────────────────────────┐
│                         COMPONENTS                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐                                              │
│  │ Supabase     │◄─────────────┐                               │
│  │ Config       │              │                               │
│  │              │              │                               │
│  │ - URL        │              │                               │
│  │ - Service Key│              │                               │
│  │ - Settings   │              │                               │
│  └──────────────┘              │                               │
│         │                      │                               │
│         │ get_supabase_client()│                               │
│         │                      │                               │
│         ▼                      │                               │
│  ┌──────────────────────────────────────────┐                 │
│  │         Pipeline Classes                 │                 │
│  ├──────────────────────────────────────────┤                 │
│  │                                          │                 │
│  │  DailyBrainPipeline                     │                 │
│  │  ├─ __init__(supabase)                  │                 │
│  │  ├─ run()                                │                 │
│  │  ├─ _fetch_tradable_instruments()       │                 │
│  │  ├─ _pkscreener_stub()                  │                 │
│  │  ├─ _ai_ranking_stub()                  │                 │
│  │  ├─ _generate_trade_intentions()        │                 │
│  │  └─ _insert_trade_intentions()          │                 │
│  │                                          │                 │
│  │  ExecutorPipeline                       │                 │
│  │  ├─ __init__(supabase)                  │                 │
│  │  ├─ run()                                │                 │
│  │  ├─ _get_autopilot_users()              │                 │
│  │  ├─ _process_user()                     │                 │
│  │  ├─ _check_risk_limits()                │                 │
│  │  ├─ _process_entries()                  │                 │
│  │  ├─ _create_entry_order()               │                 │
│  │  ├─ _process_exits()                    │                 │
│  │  └─ _create_exit_order()                │                 │
│  │                                          │                 │
│  │  PositionMonitorPipeline                │                 │
│  │  ├─ __init__(supabase)                  │                 │
│  │  ├─ run()                                │                 │
│  │  ├─ _get_open_positions()               │                 │
│  │  ├─ _monitor_position()                 │                 │
│  │  ├─ _get_current_price()                │                 │
│  │  ├─ _update_position_price()            │                 │
│  │  ├─ _check_trailing_stop()              │                 │
│  │  └─ _check_position_events()            │                 │
│  │                                          │                 │
│  └──────────────────────────────────────────┘                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Error Handling & Logging

```
┌─────────────────────────────────────────────────────────────┐
│                   ERROR HANDLING FLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Pipeline.run()                                             │
│      │                                                       │
│      ├─► try:                                               │
│      │    │                                                  │
│      │    ├─► Create pipeline_run (status: RUNNING)         │
│      │    │                                                  │
│      │    ├─► Execute pipeline logic                        │
│      │    │    │                                             │
│      │    │    ├─► Log: INFO/DEBUG messages                 │
│      │    │    │                                             │
│      │    │    └─► Collect statistics                       │
│      │    │                                                  │
│      │    ├─► Update pipeline_run (status: SUCCESS)         │
│      │    │                                                  │
│      │    └─► Return result                                 │
│      │                                                       │
│      └─► except Exception as e:                             │
│           │                                                  │
│           ├─► Log: ERROR with traceback                     │
│           │                                                  │
│           ├─► Update pipeline_run (status: FAILED)          │
│           │    └─► error_message = str(e)                   │
│           │                                                  │
│           └─► raise (propagate to scheduler)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Logging Format:
2025-12-26 18:00:00 - daily_brain - INFO - Step 1: Found 100 tradable instruments
2025-12-26 18:00:01 - daily_brain - DEBUG - Price for RELIANCE: $2500.00
2025-12-26 18:00:05 - daily_brain - WARNING - Symbol XYZ not found in instruments
2025-12-26 18:00:10 - daily_brain - ERROR - Failed to insert trade intentions: ...
```

## Production Deployment

```
┌──────────────────────────────────────────────────────────────┐
│                    SCHEDULER (APScheduler)                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Cron Jobs:                                                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Daily Brain                                          │    │
│  │ Schedule: cron(hour=7, minute=0)                    │    │
│  │ Trigger: daily_brain_main()                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Executor                                             │    │
│  │ Schedule: cron(minute='*/15', hour='9-15')          │    │
│  │ Trigger: executor_main()                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Position Monitor                                     │    │
│  │ Schedule: cron(minute='*/5')                        │    │
│  │ Trigger: position_monitor_main()                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────┐
│                      SUPABASE DATABASE                        │
│                                                               │
│  - All pipeline state                                        │
│  - All trading data                                          │
│  - Audit logs                                                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Future Enhancements

1. Model Integration
2. Real Broker APIs
3. Advanced Risk Management
4. PPO-based Trailing Stops
5. Multi-timeframe Analysis
6. Portfolio Optimization
7. Performance Analytics
