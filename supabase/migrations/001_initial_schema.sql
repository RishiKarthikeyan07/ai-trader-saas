-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- User Profiles Table
CREATE TABLE profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL UNIQUE,
    email TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'basic' CHECK (tier IN ('basic', 'pro', 'elite')),
    pro_mode_enabled BOOLEAN DEFAULT FALSE,
    cinematic_enabled BOOLEAN DEFAULT TRUE,
    auto_trading_enabled BOOLEAN DEFAULT FALSE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Subscriptions Table
CREATE TABLE subscriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(user_id) ON DELETE CASCADE,
    stripe_customer_id TEXT NOT NULL,
    stripe_subscription_id TEXT,
    tier TEXT NOT NULL CHECK (tier IN ('basic', 'pro', 'elite')),
    status TEXT NOT NULL CHECK (status IN ('active', 'cancelled', 'past_due')),
    current_period_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Signals Table
CREATE TABLE signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL CHECK (signal_type IN (
        'momentum_breakout',
        'squeeze_expansion',
        'pullback_continuation',
        'liquidity_sweep_reversal',
        'relative_strength'
    )),
    rank INTEGER NOT NULL,
    score DECIMAL(4,2) NOT NULL,
    confidence INTEGER NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
    risk_grade TEXT NOT NULL CHECK (risk_grade IN ('low', 'medium', 'high')),
    horizon TEXT NOT NULL CHECK (horizon IN ('intraday', 'swing', 'positional')),
    status TEXT NOT NULL DEFAULT 'wait' CHECK (status IN (
        'wait', 'ready', 'filled', 'tp1_hit', 'tp2_hit', 'sl_hit', 'exited', 'expired'
    )),
    entry_min DECIMAL(10,2) NOT NULL,
    entry_max DECIMAL(10,2) NOT NULL,
    stop_loss DECIMAL(10,2) NOT NULL,
    target_1 DECIMAL(10,2) NOT NULL,
    target_2 DECIMAL(10,2) NOT NULL,
    setup_tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confirmed_at TIMESTAMP WITH TIME ZONE
);

-- Signal Explanations Table (Black box - no model secrets)
CREATE TABLE signal_explanations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    why_now TEXT NOT NULL,
    key_factors TEXT[] NOT NULL,
    risk_notes TEXT[] NOT NULL,
    mtf_alignment JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Scanner Packs (defined in code, but runs stored here)
CREATE TABLE scanner_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pack_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    candidates_count INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Scanner Candidates
CREATE TABLE scanner_candidates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scanner_run_id UUID NOT NULL REFERENCES scanner_runs(id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    pack_score DECIMAL(4,2) NOT NULL,
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Watchlists
CREATE TABLE watchlists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(user_id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, symbol)
);

-- Positions
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(user_id) ON DELETE CASCADE,
    signal_id UUID NOT NULL REFERENCES signals(id),
    symbol TEXT NOT NULL,
    entry_price DECIMAL(10,2) NOT NULL,
    quantity INTEGER NOT NULL,
    stop_loss DECIMAL(10,2) NOT NULL,
    target_1 DECIMAL(10,2) NOT NULL,
    target_2 DECIMAL(10,2) NOT NULL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    pnl DECIMAL(10,2),
    is_paper BOOLEAN DEFAULT TRUE,
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE
);

-- Orders
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(user_id) ON DELETE CASCADE,
    signal_id UUID NOT NULL REFERENCES signals(id),
    position_id UUID REFERENCES positions(id),
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    type TEXT NOT NULL CHECK (type IN ('market', 'limit')),
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected')),
    is_paper BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    filled_at TIMESTAMP WITH TIME ZONE
);

-- Alerts
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES profiles(user_id) ON DELETE CASCADE,
    signal_id UUID REFERENCES signals(id),
    type TEXT NOT NULL CHECK (type IN (
        'signal_ready', 'tp1_hit', 'tp2_hit', 'sl_hit', 'daily_summary'
    )),
    message TEXT NOT NULL,
    read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk Limits
CREATE TABLE risk_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL UNIQUE REFERENCES profiles(user_id) ON DELETE CASCADE,
    max_positions INTEGER DEFAULT 5,
    max_position_size DECIMAL(10,2) DEFAULT 100000,
    max_daily_loss DECIMAL(10,2) DEFAULT 50000,
    max_total_exposure DECIMAL(10,2) DEFAULT 500000,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit Logs
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES profiles(user_id) ON DELETE SET NULL,
    event_type TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model Versions (for model registry)
CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    model_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    metrics JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(model_name, version)
);

-- Pipeline Runs
CREATE TABLE pipeline_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type TEXT NOT NULL CHECK (type IN ('daily', 'hourly')),
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
    signals_generated INTEGER,
    signals_updated INTEGER,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for performance
CREATE INDEX idx_signals_rank ON signals(rank);
CREATE INDEX idx_signals_status ON signals(status);
CREATE INDEX idx_signals_created_at ON signals(created_at DESC);
CREATE INDEX idx_positions_user_id ON positions(user_id);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_alerts_user_id ON alerts(user_id);
CREATE INDEX idx_alerts_read ON alerts(read);
CREATE INDEX idx_watchlists_user_id ON watchlists(user_id);

-- Updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_profiles_updated_at
    BEFORE UPDATE ON profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_subscriptions_updated_at
    BEFORE UPDATE ON subscriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_signals_updated_at
    BEFORE UPDATE ON signals
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();
