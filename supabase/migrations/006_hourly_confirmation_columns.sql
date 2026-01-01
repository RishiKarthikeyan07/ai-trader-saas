/**
 * Hourly Confirmation - Observability Columns
 *
 * Adds columns to track hourly pipeline checks and entry conditions.
 */

-- Add observability columns for hourly confirmation
ALTER TABLE public.signals_daily
ADD COLUMN IF NOT EXISTS last_checked_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS last_price NUMERIC,
ADD COLUMN IF NOT EXISTS ready_reason JSONB;

-- Index for hourly pipeline query (active signals)
CREATE INDEX IF NOT EXISTS signals_daily_active_idx
  ON public.signals_daily(trade_date, status, signal_type)
  WHERE status IN ('WAIT', 'READY') AND signal_type IN ('LONG', 'SHORT');

COMMENT ON COLUMN public.signals_daily.last_checked_at IS 'Last time hourly confirmation pipeline checked this signal';
COMMENT ON COLUMN public.signals_daily.last_price IS 'Latest price from hourly confirmation check';
COMMENT ON COLUMN public.signals_daily.ready_reason IS 'JSON object explaining why signal became READY (or stayed WAIT)';
COMMENT ON INDEX signals_daily_active_idx IS 'Optimizes hourly pipeline query for active signals';
