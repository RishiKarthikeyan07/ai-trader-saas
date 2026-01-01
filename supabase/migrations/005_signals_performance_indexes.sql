/**
 * Performance Indexes for Signals Explorer
 *
 * Optimizes queries for the admin signals explorer page.
 */

-- Index for date + type filtering
CREATE INDEX IF NOT EXISTS signals_daily_date_type_idx
  ON public.signals_daily(trade_date, signal_type);

-- Index for date + status filtering
CREATE INDEX IF NOT EXISTS signals_daily_date_status_idx
  ON public.signals_daily(trade_date, status);

-- Index for confidence sorting
CREATE INDEX IF NOT EXISTS signals_daily_date_conf_idx
  ON public.signals_daily(trade_date, confidence DESC);

-- Index for symbol search
CREATE INDEX IF NOT EXISTS signals_daily_symbol_idx
  ON public.signals_daily(symbol);

-- Composite index for common query pattern (date + rank sorting)
CREATE INDEX IF NOT EXISTS signals_daily_date_rank_idx
  ON public.signals_daily(trade_date, rank_position);

COMMENT ON INDEX signals_daily_date_type_idx IS 'Speeds up signals explorer type filter';
COMMENT ON INDEX signals_daily_date_status_idx IS 'Speeds up signals explorer status filter';
COMMENT ON INDEX signals_daily_date_conf_idx IS 'Speeds up confidence sorting';
COMMENT ON INDEX signals_daily_symbol_idx IS 'Speeds up symbol search';
COMMENT ON INDEX signals_daily_date_rank_idx IS 'Speeds up rank-based pagination';
