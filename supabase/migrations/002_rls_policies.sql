-- Enable RLS on all tables
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE subscriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE signal_explanations ENABLE ROW LEVEL SECURITY;
ALTER TABLE scanner_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE scanner_candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE watchlists ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_limits ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;

-- Profiles Policies
CREATE POLICY "Users can view own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can update own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Admins can view all profiles"
    ON profiles FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = TRUE
        )
    );

-- Subscriptions Policies
CREATE POLICY "Users can view own subscription"
    ON subscriptions FOR SELECT
    USING (auth.uid() = user_id);

-- Signals Policies (Tier-gated)
CREATE POLICY "Users can view tier-allowed signals"
    ON signals FOR SELECT
    USING (
        -- Basic: top 3
        (
            EXISTS (
                SELECT 1 FROM profiles
                WHERE user_id = auth.uid() AND tier = 'basic'
            ) AND rank <= 3
        )
        OR
        -- Pro: top 10
        (
            EXISTS (
                SELECT 1 FROM profiles
                WHERE user_id = auth.uid() AND tier = 'pro'
            ) AND rank <= 10
        )
        OR
        -- Elite: unlimited
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND tier = 'elite'
        )
        OR
        -- Admins: all
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = TRUE
        )
    );

-- Signal Explanations Policies
CREATE POLICY "Users can view explanations for accessible signals"
    ON signal_explanations FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM signals
            WHERE signals.id = signal_explanations.signal_id
        )
    );

-- Scanner Runs Policies (Public read for authenticated users)
CREATE POLICY "Authenticated users can view scanner runs"
    ON scanner_runs FOR SELECT
    USING (auth.uid() IS NOT NULL);

-- Scanner Candidates Policies
CREATE POLICY "Authenticated users can view scanner candidates"
    ON scanner_candidates FOR SELECT
    USING (auth.uid() IS NOT NULL);

-- Watchlists Policies
CREATE POLICY "Users can view own watchlist"
    ON watchlists FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert to own watchlist"
    ON watchlists FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete from own watchlist"
    ON watchlists FOR DELETE
    USING (auth.uid() = user_id);

-- Positions Policies (Elite only)
CREATE POLICY "Elite users can view own positions"
    ON positions FOR SELECT
    USING (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND tier = 'elite'
        )
    );

CREATE POLICY "Elite users can insert own positions"
    ON positions FOR INSERT
    WITH CHECK (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND tier = 'elite'
        )
    );

CREATE POLICY "Elite users can update own positions"
    ON positions FOR UPDATE
    USING (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND tier = 'elite'
        )
    );

-- Orders Policies (Elite only)
CREATE POLICY "Elite users can view own orders"
    ON orders FOR SELECT
    USING (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND tier = 'elite'
        )
    );

CREATE POLICY "Elite users can insert own orders"
    ON orders FOR INSERT
    WITH CHECK (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND tier = 'elite'
        )
    );

-- Alerts Policies
CREATE POLICY "Users can view own alerts"
    ON alerts FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can update own alerts"
    ON alerts FOR UPDATE
    USING (auth.uid() = user_id);

-- Risk Limits Policies (Elite only)
CREATE POLICY "Elite users can view own risk limits"
    ON risk_limits FOR SELECT
    USING (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND tier = 'elite'
        )
    );

CREATE POLICY "Elite users can update own risk limits"
    ON risk_limits FOR UPDATE
    USING (
        auth.uid() = user_id
        AND EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND tier = 'elite'
        )
    );

-- Audit Logs Policies (Admin only)
CREATE POLICY "Admins can view all audit logs"
    ON audit_logs FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = TRUE
        )
    );

-- Pipeline Runs Policies (Admin only)
CREATE POLICY "Admins can view pipeline runs"
    ON pipeline_runs FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = TRUE
        )
    );

CREATE POLICY "Admins can insert pipeline runs"
    ON pipeline_runs FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = TRUE
        )
    );

CREATE POLICY "Admins can update pipeline runs"
    ON pipeline_runs FOR UPDATE
    USING (
        EXISTS (
            SELECT 1 FROM profiles
            WHERE user_id = auth.uid() AND is_admin = TRUE
        )
    );
