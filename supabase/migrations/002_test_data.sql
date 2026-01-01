-- Test data for Supabase schema
-- Run this AFTER 001_foundation_schema.sql to verify tier gating works

-- Note: In production, users will be created via auth.users automatically
-- This is just for testing the schema

-- Insert test signals (simulate daily pipeline output)
insert into public.signals_daily (
  trade_date,
  symbol,
  signal_type,
  status,
  entry_zone_low,
  entry_zone_high,
  recommended_limit,
  stop_loss,
  target_1,
  target_2,
  confidence,
  rank_score,
  rank_position,
  smc_score,
  tf_alignment,
  smc_flags,
  model_versions
) values
  -- Rank 1-3: Basic tier can see
  (
    current_date,
    'RELIANCE',
    'LONG',
    'READY',
    2450.00,
    2465.00,
    2460.00,
    2430.00,
    2485.00,
    2510.00,
    0.87,
    0.92,
    1,
    0.85,
    '{"m15": "bullish", "h1": "bullish", "h4": "bullish", "d1": "bullish"}'::jsonb,
    '{"fvg_present": true, "order_block": true, "breaker": false}'::jsonb,
    '{"veto": "v1.2.0", "smc": "v1.0.0", "kronos": "v1.1.0"}'::jsonb
  ),
  (
    current_date,
    'TCS',
    'LONG',
    'WAIT',
    3720.00,
    3735.00,
    3730.00,
    3695.00,
    3760.00,
    3790.00,
    0.82,
    0.88,
    2,
    0.78,
    '{"m15": "bullish", "h1": "bullish", "h4": "neutral", "d1": "bullish"}'::jsonb,
    '{"fvg_present": true, "order_block": false, "breaker": false}'::jsonb,
    '{"veto": "v1.2.0", "smc": "v1.0.0", "kronos": "v1.1.0"}'::jsonb
  ),
  (
    current_date,
    'INFY',
    'SHORT',
    'READY',
    1520.00,
    1515.00,
    1517.00,
    1535.00,
    1495.00,
    1475.00,
    0.79,
    0.85,
    3,
    0.72,
    '{"m15": "bearish", "h1": "bearish", "h4": "bearish", "d1": "neutral"}'::jsonb,
    '{"fvg_present": false, "order_block": true, "breaker": true}'::jsonb,
    '{"veto": "v1.2.0", "smc": "v1.0.0", "kronos": "v1.1.0"}'::jsonb
  ),

  -- Rank 4-10: Pro tier can see (basic cannot)
  (
    current_date,
    'HDFCBANK',
    'LONG',
    'WAIT',
    1680.00,
    1690.00,
    1685.00,
    1665.00,
    1705.00,
    1720.00,
    0.75,
    0.81,
    4,
    0.68,
    '{"m15": "bullish", "h1": "neutral", "h4": "bullish", "d1": "bullish"}'::jsonb,
    '{"fvg_present": true, "order_block": false, "breaker": false}'::jsonb,
    '{"veto": "v1.2.0", "smc": "v1.0.0", "kronos": "v1.1.0"}'::jsonb
  ),
  (
    current_date,
    'ICICIBANK',
    'LONG',
    'NEW',
    1025.00,
    1035.00,
    1030.00,
    1010.00,
    1050.00,
    1065.00,
    0.71,
    0.78,
    5,
    0.65,
    '{"m15": "bullish", "h1": "bullish", "h4": "neutral", "d1": "neutral"}'::jsonb,
    '{"fvg_present": false, "order_block": true, "breaker": false}'::jsonb,
    '{"veto": "v1.2.0", "smc": "v1.0.0", "kronos": "v1.1.0"}'::jsonb
  ),

  -- Rank 11+: Elite only
  (
    current_date,
    'TATASTEEL',
    'SHORT',
    'WAIT',
    140.00,
    138.50,
    139.00,
    142.00,
    135.00,
    132.00,
    0.65,
    0.70,
    11,
    0.58,
    '{"m15": "bearish", "h1": "bearish", "h4": "neutral", "d1": "neutral"}'::jsonb,
    '{"fvg_present": true, "order_block": false, "breaker": true}'::jsonb,
    '{"veto": "v1.2.0", "smc": "v1.0.0", "kronos": "v1.1.0"}'::jsonb
  ),
  (
    current_date,
    'WIPRO',
    'NO_TRADE',
    'INVALIDATED',
    null,
    null,
    null,
    null,
    null,
    null,
    0.45,
    0.48,
    15,
    0.42,
    '{"m15": "neutral", "h1": "neutral", "h4": "neutral", "d1": "neutral"}'::jsonb,
    '{"fvg_present": false, "order_block": false, "breaker": false}'::jsonb,
    '{"veto": "v1.2.0", "smc": "v1.0.0", "kronos": "v1.1.0"}'::jsonb
  )
on conflict (trade_date, symbol) do nothing;

-- Verification queries (run these after creating test users)
-- These should be run in the Supabase SQL editor while authenticated as different users

-- Example: Check tier limit function
-- select public.current_tier_limit();

-- Example: Verify RLS works - Basic user should see only rank 1-3
-- select * from public.signals_daily where trade_date = current_date order by rank_position;

-- Example: Admin query to see all signals
-- select
--   rank_position,
--   symbol,
--   signal_type,
--   confidence,
--   case
--     when rank_position <= 3 then 'Basic, Pro, Elite'
--     when rank_position <= 10 then 'Pro, Elite'
--     else 'Elite only'
--   end as accessible_by
-- from public.signals_daily
-- where trade_date = current_date
-- order by rank_position;
