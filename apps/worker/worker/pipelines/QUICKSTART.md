# Pipeline Quickstart Guide

This guide will help you get the AutoPilot pipelines running quickly.

## Prerequisites

1. **Supabase Setup**
   - Database migrations applied (autopilot schema)
   - Environment variables configured

2. **Python Environment**
   ```bash
   cd /Users/rishi/Downloads/AI_TRADER
   pip install supabase-py pydantic-settings python-dotenv yfinance
   ```

3. **Environment Variables**
   Create `.env` file in project root:
   ```bash
   SUPABASE_URL=your-supabase-url
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   REDIS_URL=redis://localhost:6379
   ENVIRONMENT=development
   ```

## Quick Test Run

### 1. Prepare Test Data

First, add some test instruments to your database:

```sql
-- Insert test instruments
INSERT INTO instruments (canonical_symbol, exchange, name, sector, is_tradable) VALUES
('RELIANCE', 'NSE', 'Reliance Industries Ltd', 'Energy', true),
('TCS', 'NSE', 'Tata Consultancy Services', 'IT', true),
('INFY', 'NSE', 'Infosys Ltd', 'IT', true),
('HDFC', 'NSE', 'HDFC Bank Ltd', 'Banking', true),
('ICICIBANK', 'NSE', 'ICICI Bank Ltd', 'Banking', true),
('SBIN', 'NSE', 'State Bank of India', 'Banking', true),
('BHARTIARTL', 'NSE', 'Bharti Airtel Ltd', 'Telecom', true),
('WIPRO', 'NSE', 'Wipro Ltd', 'IT', true),
('HCLTECH', 'NSE', 'HCL Technologies Ltd', 'IT', true),
('MARUTI', 'NSE', 'Maruti Suzuki India Ltd', 'Automobile', true);

-- Create a test user profile (replace with your actual user_id from auth.users)
INSERT INTO profiles (user_id, email, full_name, autopilot_enabled, is_active_subscriber)
VALUES
('00000000-0000-0000-0000-000000000001', 'test@example.com', 'Test User', true, true);

-- Set risk limits for test user
INSERT INTO risk_limits (user_id, max_positions, max_daily_loss, max_exposure, risk_per_trade_percent, capital_allocated)
VALUES
('00000000-0000-0000-0000-000000000001', 5, 10000, 100000, 1.0, 100000);
```

### 2. Run Daily Brain Pipeline

Generate today's trade intentions:

```bash
cd /Users/rishi/Downloads/AI_TRADER
python -m apps.worker.worker.pipelines.daily_brain
```

**Expected Output**:
```
================================================================================
DAILY BRAIN PIPELINE STARTED
================================================================================
Step 1: Found 10 tradable instruments
Step 2: PKScreener selected 10 candidates
Step 3: AI ranking completed for 10 symbols
Step 4: Generated 10 trade intentions
Step 5: Inserted 10 trade intentions into database
================================================================================
DAILY BRAIN PIPELINE COMPLETED SUCCESSFULLY in 2s
================================================================================
```

**Verify in Database**:
```sql
SELECT canonical_symbol, confidence, entry_zone_low, entry_zone_high, sl, tp1, tp2
FROM trade_intentions
WHERE date = CURRENT_DATE
ORDER BY confidence DESC
LIMIT 10;
```

### 3. Run Executor Pipeline

Execute trades for autopilot users:

```bash
python -m apps.worker.worker.pipelines.executor
```

**Expected Output**:
```
================================================================================
EXECUTOR PIPELINE STARTED
================================================================================
Found 1 users with autopilot enabled
Processing user: test@example.com (00000000-0000-0000-0000-000000000001)
User 00000000-0000-0000-0000-000000000001 risk check: exposure $0.00 / $100000.00, daily P&L $0.00 / -$10000.00
Created entry order for RELIANCE
Created entry order for TCS
Created entry order for INFY
Created entry order for HDFC
Created entry order for ICICIBANK
User 00000000-0000-0000-0000-000000000001 stats: {'entries': 5, 'exits': 0, 'orders': 5, 'risk_breaches': 0}
================================================================================
EXECUTOR PIPELINE COMPLETED in 3s
Stats: {'users_processed': 1, 'entries_executed': 5, 'exits_executed': 0, 'orders_placed': 5, 'risk_limit_breaches': 0}
================================================================================
```

**Verify in Database**:
```sql
-- Check created positions
SELECT canonical_symbol, quantity, avg_entry_price, sl, tp1, tp2, status
FROM positions
WHERE user_id = '00000000-0000-0000-0000-000000000001'
AND status = 'OPEN';

-- Check created orders
SELECT canonical_symbol, side, quantity, price, status, is_paper_trade
FROM orders
WHERE user_id = '00000000-0000-0000-0000-000000000001'
ORDER BY created_at DESC;

-- Check notifications
SELECT type, title, message
FROM notifications
WHERE user_id = '00000000-0000-0000-0000-000000000001'
ORDER BY created_at DESC;
```

### 4. Run Position Monitor Pipeline

Monitor open positions:

```bash
python -m apps.worker.worker.pipelines.position_monitor
```

**Expected Output**:
```
================================================================================
POSITION MONITOR PIPELINE STARTED
================================================================================
Found 5 open positions to monitor
Monitoring position ...: RELIANCE
Monitoring position ...: TCS
Monitoring position ...: INFY
Monitoring position ...: HDFC
Monitoring position ...: ICICIBANK
================================================================================
POSITION MONITOR PIPELINE COMPLETED in 2s
Stats: {'positions_monitored': 5, 'prices_updated': 5, 'notifications_sent': 0, 'trailing_stops_updated': 0, 'errors': 0}
================================================================================
```

**Verify in Database**:
```sql
-- Check updated positions with P&L
SELECT
    canonical_symbol,
    quantity,
    avg_entry_price,
    current_price,
    unrealized_pnl,
    unrealized_pnl_percent,
    status
FROM positions
WHERE user_id = '00000000-0000-0000-0000-000000000001'
AND status = 'OPEN'
ORDER BY unrealized_pnl_percent DESC;
```

## Monitoring Pipeline Runs

Check pipeline execution history:

```sql
-- Recent pipeline runs
SELECT
    type,
    status,
    started_at,
    duration_seconds,
    metadata
FROM pipeline_runs
ORDER BY started_at DESC
LIMIT 20;

-- Failed runs (for debugging)
SELECT
    type,
    status,
    started_at,
    error_message
FROM pipeline_runs
WHERE status = 'FAILED'
ORDER BY started_at DESC;

-- Pipeline statistics
SELECT
    type,
    COUNT(*) as total_runs,
    COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) as successful,
    COUNT(CASE WHEN status = 'FAILED' THEN 1 END) as failed,
    AVG(duration_seconds) as avg_duration_sec
FROM pipeline_runs
GROUP BY type;
```

## Simulate a Full Trading Day

Run all pipelines in sequence to simulate a complete trading day:

```bash
#!/bin/bash
# simulate_trading_day.sh

echo "🌅 Starting trading day simulation..."

# Step 1: Daily brain (morning 7 AM)
echo ""
echo "📊 Step 1: Running daily brain pipeline..."
python -m apps.worker.worker.pipelines.daily_brain

# Step 2: Executor (market hours)
echo ""
echo "💼 Step 2: Running executor pipeline (market open)..."
python -m apps.worker.worker.pipelines.executor

# Step 3: Position monitor
echo ""
echo "👁️  Step 3: Running position monitor..."
python -m apps.worker.worker.pipelines.position_monitor

# Step 4: Executor again (simulate 15 min later)
echo ""
echo "💼 Step 4: Running executor again (simulate intraday)..."
python -m apps.worker.worker.pipelines.executor

# Step 5: Position monitor again
echo ""
echo "👁️  Step 5: Running position monitor again..."
python -m apps.worker.worker.pipelines.position_monitor

echo ""
echo "✅ Trading day simulation complete!"
```

Run it:
```bash
chmod +x simulate_trading_day.sh
./simulate_trading_day.sh
```

## Troubleshooting

### Issue: "Failed to create Supabase client"

**Solution**: Check your `.env` file has correct Supabase credentials:
```bash
echo $SUPABASE_URL
echo $SUPABASE_SERVICE_ROLE_KEY
```

### Issue: "No tradable instruments found"

**Solution**: Add instruments to the database (see step 1 above)

### Issue: "No autopilot users found"

**Solution**: Enable autopilot for a test user:
```sql
UPDATE profiles
SET autopilot_enabled = true, is_active_subscriber = true
WHERE email = 'your-test-email@example.com';
```

### Issue: "Outside market hours"

The executor pipeline only runs during market hours (9:15 AM - 3:30 PM IST). To test outside market hours, temporarily modify `config.py`:

```python
# For testing only
MARKET_OPEN_HOUR: int = 0
MARKET_CLOSE_HOUR: int = 23
```

### Issue: "Module not found"

Ensure you're running from the project root:
```bash
cd /Users/rishi/Downloads/AI_TRADER
python -m apps.worker.worker.pipelines.daily_brain
```

## Next Steps

1. **Production Deployment**: Set up a scheduler (APScheduler, Celery, or Temporal)
2. **Real Data Integration**: Replace stubs with actual market data APIs
3. **Broker Integration**: Connect to real brokers via BrokerHub
4. **Model Integration**: Replace AI ranking stub with trained models
5. **Monitoring**: Set up alerts for failed pipeline runs

## Reference

- Full documentation: `README.md`
- Database schema: `/Users/rishi/Downloads/AI_TRADER/supabase/migrations/003_autopilot_schema.sql`
- Configuration: `../config.py`
