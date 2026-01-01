-- Performance Tracking Tables for Top-K Heatmap

-- Signal Outcomes Table (resolved signals)
CREATE TABLE signal_outcomes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL REFERENCES signals(id) ON DELETE CASCADE,
    horizon INTEGER NOT NULL CHECK (horizon IN (5, 10)), -- Days to resolution
    hit_tp BOOLEAN DEFAULT FALSE,
    hit_sl BOOLEAN DEFAULT FALSE,
    outcome TEXT NOT NULL CHECK (outcome IN ('TP1', 'TP2', 'SL', 'NONE', 'PENDING')),
    realized_r DECIMAL(6,2), -- Multiple of R (risk) realized
    entry_price DECIMAL(10,2),
    exit_price DECIMAL(10,2),
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(signal_id, horizon)
);

-- Performance Daily Aggregates
CREATE TABLE performance_daily (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    k_bucket TEXT NOT NULL CHECK (k_bucket IN ('K3', 'K10', 'KALL')),
    horizon_bucket TEXT NOT NULL CHECK (horizon_bucket IN ('H5', 'H10', 'BOTH')),
    win_rate DECIMAL(5,2) NOT NULL, -- Percentage 0-100
    avg_r DECIMAL(6,2) NOT NULL, -- Average R multiple
    total_signals INTEGER NOT NULL,
    winning_signals INTEGER NOT NULL,
    losing_signals INTEGER NOT NULL,
    profit_factor DECIMAL(6,2), -- Total wins / Total losses
    expectancy DECIMAL(8,2), -- Average $ per signal (proxy)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(date, k_bucket, horizon_bucket)
);

-- Performance Summary (rolling aggregates)
CREATE TABLE performance_summary (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    k_bucket TEXT NOT NULL CHECK (k_bucket IN ('K3', 'K10', 'KALL')),
    horizon_bucket TEXT NOT NULL CHECK (horizon_bucket IN ('H5', 'H10', 'BOTH')),
    period_days INTEGER NOT NULL, -- 7, 30, 90
    win_rate DECIMAL(5,2) NOT NULL,
    avg_r DECIMAL(6,2) NOT NULL,
    total_signals INTEGER NOT NULL,
    profit_factor DECIMAL(6,2),
    sharpe_ratio DECIMAL(6,2), -- If we compute it
    max_drawdown DECIMAL(5,2), -- Percentage
    computed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(k_bucket, horizon_bucket, period_days)
);

-- Create indexes for performance queries
CREATE INDEX idx_signal_outcomes_signal_id ON signal_outcomes(signal_id);
CREATE INDEX idx_signal_outcomes_outcome ON signal_outcomes(outcome);
CREATE INDEX idx_signal_outcomes_computed_at ON signal_outcomes(computed_at DESC);

CREATE INDEX idx_performance_daily_date ON performance_daily(date DESC);
CREATE INDEX idx_performance_daily_bucket ON performance_daily(k_bucket, horizon_bucket);

CREATE INDEX idx_performance_summary_bucket ON performance_summary(k_bucket, horizon_bucket);

-- Trigger for updated_at
CREATE TRIGGER update_performance_daily_updated_at
    BEFORE UPDATE ON performance_daily
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Function to compute signal outcome
CREATE OR REPLACE FUNCTION compute_signal_outcome(
    p_signal_id UUID,
    p_horizon INTEGER
) RETURNS void AS $$
DECLARE
    v_signal RECORD;
    v_entry_avg DECIMAL(10,2);
    v_tp1_hit BOOLEAN := FALSE;
    v_tp2_hit BOOLEAN := FALSE;
    v_sl_hit BOOLEAN := FALSE;
    v_outcome TEXT := 'PENDING';
    v_realized_r DECIMAL(6,2) := 0;
BEGIN
    -- Get signal details
    SELECT * INTO v_signal FROM signals WHERE id = p_signal_id;

    IF NOT FOUND THEN
        RETURN;
    END IF;

    v_entry_avg := (v_signal.entry_min + v_signal.entry_max) / 2;

    -- TODO: Fetch actual price data and check if TP/SL hit within horizon
    -- For now, use signal status as proxy

    CASE v_signal.status
        WHEN 'tp1_hit' THEN
            v_tp1_hit := TRUE;
            v_outcome := 'TP1';
            v_realized_r := ABS(v_signal.target_1 - v_entry_avg) / ABS(v_entry_avg - v_signal.stop_loss);
        WHEN 'tp2_hit' THEN
            v_tp2_hit := TRUE;
            v_outcome := 'TP2';
            v_realized_r := ABS(v_signal.target_2 - v_entry_avg) / ABS(v_entry_avg - v_signal.stop_loss);
        WHEN 'sl_hit' THEN
            v_sl_hit := TRUE;
            v_outcome := 'SL';
            v_realized_r := -1.0;
        ELSE
            v_outcome := 'NONE';
    END CASE;

    -- Insert or update outcome
    INSERT INTO signal_outcomes (
        signal_id,
        horizon,
        hit_tp,
        hit_sl,
        outcome,
        realized_r,
        entry_price,
        computed_at
    ) VALUES (
        p_signal_id,
        p_horizon,
        v_tp1_hit OR v_tp2_hit,
        v_sl_hit,
        v_outcome,
        v_realized_r,
        v_entry_avg,
        NOW()
    )
    ON CONFLICT (signal_id, horizon) DO UPDATE SET
        hit_tp = EXCLUDED.hit_tp,
        hit_sl = EXCLUDED.hit_sl,
        outcome = EXCLUDED.outcome,
        realized_r = EXCLUDED.realized_r,
        computed_at = EXCLUDED.computed_at;
END;
$$ LANGUAGE plpgsql;

-- Function to compute daily performance
CREATE OR REPLACE FUNCTION compute_daily_performance(
    p_date DATE,
    p_k_bucket TEXT,
    p_horizon_bucket TEXT
) RETURNS void AS $$
DECLARE
    v_rank_limit INTEGER;
    v_horizon_filter INTEGER[];
    v_total INTEGER := 0;
    v_wins INTEGER := 0;
    v_losses INTEGER := 0;
    v_avg_r DECIMAL(6,2) := 0;
    v_win_rate DECIMAL(5,2) := 0;
    v_profit_factor DECIMAL(6,2) := 0;
    v_total_wins DECIMAL(10,2) := 0;
    v_total_losses DECIMAL(10,2) := 0;
BEGIN
    -- Determine rank limit
    CASE p_k_bucket
        WHEN 'K3' THEN v_rank_limit := 3;
        WHEN 'K10' THEN v_rank_limit := 10;
        WHEN 'KALL' THEN v_rank_limit := 999;
    END CASE;

    -- Determine horizon filter
    CASE p_horizon_bucket
        WHEN 'H5' THEN v_horizon_filter := ARRAY[5];
        WHEN 'H10' THEN v_horizon_filter := ARRAY[10];
        WHEN 'BOTH' THEN v_horizon_filter := ARRAY[5, 10];
    END CASE;

    -- Compute metrics
    SELECT
        COUNT(*),
        COUNT(*) FILTER (WHERE outcome IN ('TP1', 'TP2')),
        COUNT(*) FILTER (WHERE outcome = 'SL'),
        COALESCE(AVG(realized_r), 0),
        COALESCE(SUM(realized_r) FILTER (WHERE realized_r > 0), 0),
        COALESCE(ABS(SUM(realized_r) FILTER (WHERE realized_r < 0)), 0)
    INTO
        v_total,
        v_wins,
        v_losses,
        v_avg_r,
        v_total_wins,
        v_total_losses
    FROM signal_outcomes so
    JOIN signals s ON so.signal_id = s.id
    WHERE s.created_at::date = p_date
        AND s.rank <= v_rank_limit
        AND so.horizon = ANY(v_horizon_filter)
        AND so.outcome IN ('TP1', 'TP2', 'SL');

    -- Calculate win rate
    IF v_total > 0 THEN
        v_win_rate := (v_wins::DECIMAL / v_total) * 100;
    END IF;

    -- Calculate profit factor
    IF v_total_losses > 0 THEN
        v_profit_factor := v_total_wins / v_total_losses;
    END IF;

    -- Insert or update
    INSERT INTO performance_daily (
        date,
        k_bucket,
        horizon_bucket,
        win_rate,
        avg_r,
        total_signals,
        winning_signals,
        losing_signals,
        profit_factor
    ) VALUES (
        p_date,
        p_k_bucket,
        p_horizon_bucket,
        v_win_rate,
        v_avg_r,
        v_total,
        v_wins,
        v_losses,
        v_profit_factor
    )
    ON CONFLICT (date, k_bucket, horizon_bucket) DO UPDATE SET
        win_rate = EXCLUDED.win_rate,
        avg_r = EXCLUDED.avg_r,
        total_signals = EXCLUDED.total_signals,
        winning_signals = EXCLUDED.winning_signals,
        losing_signals = EXCLUDED.losing_signals,
        profit_factor = EXCLUDED.profit_factor,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;
