-- Admin System Setup
-- Adds admin flag, user blocking, and enhanced RLS policies

-- 1. Add admin and blocking columns
ALTER TABLE public.profiles
ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT false,
ADD COLUMN IF NOT EXISTS is_blocked BOOLEAN NOT NULL DEFAULT false;

-- 2. Create index for admin queries
CREATE INDEX IF NOT EXISTS profiles_is_admin_idx ON public.profiles(is_admin) WHERE is_admin = true;
CREATE INDEX IF NOT EXISTS profiles_is_blocked_idx ON public.profiles(is_blocked) WHERE is_blocked = true;

-- 3. Update is_admin helper function
CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN
LANGUAGE SQL
STABLE
SECURITY DEFINER
AS $$
  SELECT COALESCE(
    (SELECT is_admin FROM public.profiles WHERE user_id = auth.uid()),
    false
  );
$$;

-- 4. Enhanced update policy (prevents self-promotion)
DROP POLICY IF EXISTS "profiles_update_own" ON public.profiles;
DROP POLICY IF EXISTS "profiles_update_own_safe" ON public.profiles;

CREATE POLICY "profiles_update_own_safe"
ON public.profiles
FOR UPDATE
TO authenticated
USING (
  -- Can update own profile OR is admin
  user_id = auth.uid() OR public.is_admin()
)
WITH CHECK (
  -- Same as USING, plus restrictions:
  (user_id = auth.uid() OR public.is_admin())
  AND (
    -- Admins can change anything
    public.is_admin()
    OR (
      -- Regular users cannot change these sensitive fields
      is_admin = (SELECT is_admin FROM public.profiles WHERE user_id = auth.uid())
      AND is_blocked = (SELECT is_blocked FROM public.profiles WHERE user_id = auth.uid())
      AND tier = (SELECT tier FROM public.profiles WHERE user_id = auth.uid())
    )
  )
);

-- 5. Block access for blocked users
CREATE OR REPLACE FUNCTION public.is_user_blocked()
RETURNS BOOLEAN
LANGUAGE SQL
STABLE
SECURITY DEFINER
AS $$
  SELECT COALESCE(
    (SELECT is_blocked FROM public.profiles WHERE user_id = auth.uid()),
    false
  );
$$;

-- 6. Add blocking check to signals policy
DROP POLICY IF EXISTS "signals_select_by_tier" ON public.signals_daily;

CREATE POLICY "signals_select_by_tier"
ON public.signals_daily
FOR SELECT
TO authenticated
USING (
  NOT public.is_user_blocked()
  AND (
    public.is_admin()
    OR rank_position <= public.current_tier_limit()
  )
);

-- 7. Admin-only policy for all signals (bypass tier limits)
CREATE POLICY "admin_full_access_signals"
ON public.signals_daily
FOR ALL
TO authenticated
USING (public.is_admin())
WITH CHECK (public.is_admin());

-- 8. Grant execute permissions
GRANT EXECUTE ON FUNCTION public.is_admin() TO authenticated;
GRANT EXECUTE ON FUNCTION public.is_user_blocked() TO authenticated;

-- 9. Create admin audit view
CREATE OR REPLACE VIEW public.admin_audit_summary AS
SELECT
  DATE(created_at) as date,
  action,
  COUNT(*) as count,
  array_agg(DISTINCT user_id) FILTER (WHERE user_id IS NOT NULL) as user_ids
FROM public.audit_logs
GROUP BY DATE(created_at), action
ORDER BY date DESC, count DESC;

-- 10. Admin stats view
CREATE OR REPLACE VIEW public.admin_stats AS
SELECT
  (SELECT COUNT(*) FROM public.profiles) as total_users,
  (SELECT COUNT(*) FROM public.profiles WHERE tier = 'basic') as basic_users,
  (SELECT COUNT(*) FROM public.profiles WHERE tier = 'pro') as pro_users,
  (SELECT COUNT(*) FROM public.profiles WHERE tier = 'elite') as elite_users,
  (SELECT COUNT(*) FROM public.profiles WHERE is_admin = true) as admin_users,
  (SELECT COUNT(*) FROM public.profiles WHERE is_blocked = true) as blocked_users,
  (SELECT COUNT(*) FROM public.profiles WHERE auto_trade_enabled = true) as auto_trade_users,
  (SELECT COUNT(*) FROM public.signals_daily WHERE trade_date = CURRENT_DATE) as signals_today,
  (SELECT COUNT(*) FROM public.orders WHERE DATE(created_at) = CURRENT_DATE) as orders_today,
  (SELECT COUNT(*) FROM public.positions WHERE status = 'OPEN') as open_positions;

COMMENT ON VIEW public.admin_stats IS 'Real-time admin dashboard statistics';
COMMENT ON VIEW public.admin_audit_summary IS 'Aggregated audit log summary for admin';
