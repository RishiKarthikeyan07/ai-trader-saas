/**
 * Pipeline Automation & Monitoring
 *
 * Adds tables for:
 * - pipeline_runs: Track execution status of daily/hourly jobs
 * - global_settings: Kill switch and global controls
 */

-- Pipeline run tracking for admin monitoring
CREATE TABLE IF NOT EXISTS public.pipeline_runs (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_name TEXT NOT NULL CHECK (job_name IN ('daily','hourly')),
  status TEXT NOT NULL CHECK (status IN ('started','success','failed')),
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  details JSONB
);

CREATE INDEX IF NOT EXISTS pipeline_runs_job_time_idx
  ON public.pipeline_runs(job_name, started_at DESC);

-- Global settings for kill switches and system-wide controls
CREATE TABLE IF NOT EXISTS public.global_settings (
  id INT PRIMARY KEY DEFAULT 1,
  auto_trading_globally_enabled BOOLEAN NOT NULL DEFAULT true,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Initialize with default settings
INSERT INTO public.global_settings (id)
VALUES (1)
ON CONFLICT (id) DO NOTHING;

-- Row-level security
ALTER TABLE public.pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.global_settings ENABLE ROW LEVEL SECURITY;

-- Admin-only access to pipeline runs
CREATE POLICY "pipeline_runs_admin_only"
ON public.pipeline_runs FOR SELECT
TO authenticated
USING (public.is_admin());

-- Admin-only read access to global settings
CREATE POLICY "global_settings_admin_only"
ON public.global_settings FOR SELECT
TO authenticated
USING (public.is_admin());

-- Admin-only update access to global settings
CREATE POLICY "global_settings_admin_update"
ON public.global_settings FOR UPDATE
TO authenticated
USING (public.is_admin())
WITH CHECK (public.is_admin());

-- Service role needs insert access for pipeline logging
CREATE POLICY "pipeline_runs_service_insert"
ON public.pipeline_runs FOR INSERT
TO service_role
WITH CHECK (true);

CREATE POLICY "pipeline_runs_service_update"
ON public.pipeline_runs FOR UPDATE
TO service_role
USING (true)
WITH CHECK (true);

COMMENT ON TABLE public.pipeline_runs IS 'Tracks execution of automated daily/hourly pipeline jobs';
COMMENT ON TABLE public.global_settings IS 'Global system settings including kill switches';
COMMENT ON COLUMN public.global_settings.auto_trading_globally_enabled IS 'Master kill switch - disables all auto-trading when false';
