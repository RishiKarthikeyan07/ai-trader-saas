-- ============================================================================
-- AutoPilot AI Trading Bot - Row Level Security Policies
-- Migration 004: RLS for AutoPilot Architecture
-- Date: 2025-12-26
-- Description: Secure all tables with proper RLS policies
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE broker_connections ENABLE ROW LEVEL SECURITY;
ALTER TABLE instruments ENABLE ROW LEVEL SECURITY;
ALTER TABLE broker_instruments ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_intentions ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE fills ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_limits ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_versions ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- PROFILES
-- ============================================================================

-- Users can read their own profile
CREATE POLICY "Users can view own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = user_id);

-- Users can update their own profile (except is_admin, is_active_subscriber)
CREATE POLICY "Users can update own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = user_id);

-- Admins can view all profiles
CREATE POLICY "Admins can view all profiles"
    ON profiles FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- ============================================================================
-- SUBSCRIPTIONS
-- ============================================================================

-- Users can view their own subscriptions
CREATE POLICY "Users can view own subscriptions"
    ON subscriptions FOR SELECT
    USING (auth.uid() = user_id);

-- Admins can view all subscriptions
CREATE POLICY "Admins can view all subscriptions"
    ON subscriptions FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- Service role can insert/update subscriptions (webhook handler)
-- Note: This is handled by service_role key, not RLS

-- ============================================================================
-- BROKER CONNECTIONS
-- ============================================================================

-- Users can view their own broker connections
CREATE POLICY "Users can view own broker connections"
    ON broker_connections FOR SELECT
    USING (auth.uid() = user_id);

-- Users can insert their own broker connections
CREATE POLICY "Users can insert own broker connections"
    ON broker_connections FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Users can update their own broker connections
CREATE POLICY "Users can update own broker connections"
    ON broker_connections FOR UPDATE
    USING (auth.uid() = user_id);

-- Users can delete their own broker connections
CREATE POLICY "Users can delete own broker connections"
    ON broker_connections FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================================================
-- INSTRUMENTS (Public Read-Only)
-- ============================================================================

-- Everyone can read instruments (public reference data)
CREATE POLICY "Instruments are publicly readable"
    ON instruments FOR SELECT
    USING (true);

-- Only admins can modify instruments
CREATE POLICY "Admins can modify instruments"
    ON instruments FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- ============================================================================
-- BROKER INSTRUMENTS (Public Read-Only)
-- ============================================================================

-- Everyone can read broker instruments (public reference data)
CREATE POLICY "Broker instruments are publicly readable"
    ON broker_instruments FOR SELECT
    USING (true);

-- Only admins can modify broker instruments
CREATE POLICY "Admins can modify broker instruments"
    ON broker_instruments FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- ============================================================================
-- TRADE INTENTIONS (INTERNAL - Admin Only)
-- ============================================================================

-- Trade intentions are INTERNAL and NOT exposed to users
-- Only admins and backend services can access

CREATE POLICY "Admins can view trade intentions"
    ON trade_intentions FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

CREATE POLICY "Admins can modify trade intentions"
    ON trade_intentions FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- Note: Backend services use service_role key to bypass RLS

-- ============================================================================
-- POSITIONS
-- ============================================================================

-- Users can view their own positions
CREATE POLICY "Users can view own positions"
    ON positions FOR SELECT
    USING (auth.uid() = user_id);

-- Admins can view all positions
CREATE POLICY "Admins can view all positions"
    ON positions FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- Backend services can insert/update positions (service_role)
-- Users cannot directly insert/update positions

-- ============================================================================
-- ORDERS
-- ============================================================================

-- Users can view their own orders
CREATE POLICY "Users can view own orders"
    ON orders FOR SELECT
    USING (auth.uid() = user_id);

-- Admins can view all orders
CREATE POLICY "Admins can view all orders"
    ON orders FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- Backend services can insert/update orders (service_role)
-- Users cannot directly insert/update orders

-- ============================================================================
-- FILLS
-- ============================================================================

-- Users can view fills for their own orders
CREATE POLICY "Users can view own fills"
    ON fills FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM orders
            WHERE orders.id = fills.order_id
            AND orders.user_id = auth.uid()
        )
    );

-- Admins can view all fills
CREATE POLICY "Admins can view all fills"
    ON fills FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- ============================================================================
-- RISK LIMITS
-- ============================================================================

-- Users can view their own risk limits
CREATE POLICY "Users can view own risk limits"
    ON risk_limits FOR SELECT
    USING (auth.uid() = user_id);

-- Users can update their own risk limits
CREATE POLICY "Users can update own risk limits"
    ON risk_limits FOR UPDATE
    USING (auth.uid() = user_id);

-- Users can insert their own risk limits (first-time setup)
CREATE POLICY "Users can insert own risk limits"
    ON risk_limits FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Admins can view all risk limits
CREATE POLICY "Admins can view all risk limits"
    ON risk_limits FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- ============================================================================
-- PERFORMANCE DAILY
-- ============================================================================

-- Users can view their own performance data
CREATE POLICY "Users can view own performance"
    ON performance_daily FOR SELECT
    USING (auth.uid() = user_id);

-- Admins can view all performance data
CREATE POLICY "Admins can view all performance"
    ON performance_daily FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- Backend services can insert/update performance (service_role)

-- ============================================================================
-- NOTIFICATIONS
-- ============================================================================

-- Users can view their own notifications
CREATE POLICY "Users can view own notifications"
    ON notifications FOR SELECT
    USING (auth.uid() = user_id);

-- Users can update their own notifications (mark as read)
CREATE POLICY "Users can update own notifications"
    ON notifications FOR UPDATE
    USING (auth.uid() = user_id);

-- Backend services can insert notifications (service_role)

-- ============================================================================
-- AUDIT LOGS
-- ============================================================================

-- Users can view their own audit logs
CREATE POLICY "Users can view own audit logs"
    ON audit_logs FOR SELECT
    USING (auth.uid() = user_id);

-- Admins can view all audit logs
CREATE POLICY "Admins can view all audit logs"
    ON audit_logs FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- Backend services can insert audit logs (service_role)

-- ============================================================================
-- PIPELINE RUNS
-- ============================================================================

-- Only admins can view pipeline runs
CREATE POLICY "Admins can view pipeline runs"
    ON pipeline_runs FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- Backend services can insert/update pipeline runs (service_role)

-- ============================================================================
-- MODEL VERSIONS
-- ============================================================================

-- Only admins can view model versions
CREATE POLICY "Admins can view model versions"
    ON model_versions FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- Only admins can modify model versions
CREATE POLICY "Admins can modify model versions"
    ON model_versions FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = true
        )
    );

-- ============================================================================
-- FUNCTIONS FOR HELPER QUERIES
-- ============================================================================

-- Function to check if user has active subscription
CREATE OR REPLACE FUNCTION has_active_subscription(user_uuid UUID)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM profiles
        WHERE user_id = user_uuid
        AND is_active_subscriber = true
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check if AutoPilot is enabled for user
CREATE OR REPLACE FUNCTION is_autopilot_enabled(user_uuid UUID)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM profiles
        WHERE user_id = user_uuid
        AND autopilot_enabled = true
        AND is_active_subscriber = true
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get user's current exposure
CREATE OR REPLACE FUNCTION get_user_exposure(user_uuid UUID)
RETURNS DECIMAL AS $$
DECLARE
    total_exposure DECIMAL;
BEGIN
    SELECT COALESCE(SUM(quantity * current_price), 0)
    INTO total_exposure
    FROM positions
    WHERE user_id = user_uuid
    AND status = 'OPEN';

    RETURN total_exposure;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get user's daily P&L
CREATE OR REPLACE FUNCTION get_daily_pnl(user_uuid UUID, trade_date DATE)
RETURNS DECIMAL AS $$
DECLARE
    daily_pnl DECIMAL;
BEGIN
    SELECT COALESCE(net_pnl, 0)
    INTO daily_pnl
    FROM performance_daily
    WHERE user_id = user_uuid
    AND date = trade_date;

    RETURN daily_pnl;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================================================
-- GRANTS
-- ============================================================================

-- Grant usage on all sequences to authenticated users
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO service_role;

-- Grant execute on functions
GRANT EXECUTE ON FUNCTION has_active_subscription(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION is_autopilot_enabled(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_exposure(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION get_daily_pnl(UUID, DATE) TO authenticated;

-- ============================================================================
-- INITIAL DATA SEEDING
-- ============================================================================

-- Insert some common NSE instruments (you'll need to populate more)
INSERT INTO instruments (canonical_symbol, exchange, name, sector, lot_size) VALUES
    ('RELIANCE', 'NSE', 'Reliance Industries Ltd', 'Energy', 1),
    ('TCS', 'NSE', 'Tata Consultancy Services Ltd', 'IT', 1),
    ('INFY', 'NSE', 'Infosys Ltd', 'IT', 1),
    ('HDFCBANK', 'NSE', 'HDFC Bank Ltd', 'Finance', 1),
    ('ICICIBANK', 'NSE', 'ICICI Bank Ltd', 'Finance', 1),
    ('SBIN', 'NSE', 'State Bank of India', 'Finance', 1),
    ('ITC', 'NSE', 'ITC Ltd', 'FMCG', 1),
    ('BHARTIARTL', 'NSE', 'Bharti Airtel Ltd', 'Telecom', 1),
    ('HINDUNILVR', 'NSE', 'Hindustan Unilever Ltd', 'FMCG', 1),
    ('KOTAKBANK', 'NSE', 'Kotak Mahindra Bank Ltd', 'Finance', 1)
ON CONFLICT (canonical_symbol) DO NOTHING;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON POLICY "Users can view own profile" ON profiles IS 'Users can only see their own profile data';
COMMENT ON POLICY "Admins can view all profiles" ON profiles IS 'Admins have full visibility';
COMMENT ON POLICY "Admins can view trade intentions" ON trade_intentions IS 'Trade intentions are INTERNAL and not exposed to users';
COMMENT ON FUNCTION has_active_subscription(UUID) IS 'Check if user has an active subscription';
COMMENT ON FUNCTION is_autopilot_enabled(UUID) IS 'Check if AutoPilot is enabled for user';
COMMENT ON FUNCTION get_user_exposure(UUID) IS 'Calculate total current exposure for user';
COMMENT ON FUNCTION get_daily_pnl(UUID, DATE) IS 'Get daily P&L for user on specific date';
