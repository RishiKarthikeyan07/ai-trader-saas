// Core domain types

export type UserTier = 'basic' | 'pro' | 'elite';

export type SignalStatus = 'wait' | 'ready' | 'filled' | 'tp1_hit' | 'tp2_hit' | 'sl_hit' | 'exited' | 'expired';

export type SignalHorizon = 'intraday' | 'swing' | 'positional';

export type RiskGrade = 'low' | 'medium' | 'high';

export type SignalType =
  | 'momentum_breakout'
  | 'squeeze_expansion'
  | 'pullback_continuation'
  | 'liquidity_sweep_reversal'
  | 'relative_strength';

export interface User {
  id: string;
  email: string;
  tier: UserTier;
  pro_mode_enabled: boolean;
  cinematic_enabled: boolean;
  auto_trading_enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface Signal {
  id: string;
  symbol: string;
  signal_type: SignalType;
  rank: number;
  score: number;
  confidence: number;
  risk_grade: RiskGrade;
  horizon: SignalHorizon;
  status: SignalStatus;
  entry_min: number;
  entry_max: number;
  stop_loss: number;
  target_1: number;
  target_2: number;
  setup_tags: string[];
  created_at: string;
  updated_at: string;
  confirmed_at?: string;
}

export interface SignalExplanation {
  signal_id: string;
  why_now: string;
  key_factors: string[];
  risk_notes: string[];
  mtf_alignment?: {
    timeframe: string;
    trend: 'bullish' | 'bearish' | 'neutral';
    structure: string;
  }[];
}

export interface ScannerPack {
  id: string;
  name: string;
  description: string;
  icon: string;
  signal_type: SignalType;
}

export interface ScannerCandidate {
  id: string;
  scanner_run_id: string;
  symbol: string;
  pack_score: number;
  tags: string[];
  created_at: string;
}

export interface ScannerRun {
  id: string;
  pack_id: string;
  status: 'running' | 'completed' | 'failed';
  candidates_count: number;
  started_at: string;
  completed_at?: string;
}

export interface Watchlist {
  id: string;
  user_id: string;
  symbol: string;
  notes?: string;
  created_at: string;
}

export interface Position {
  id: string;
  user_id: string;
  signal_id: string;
  symbol: string;
  entry_price: number;
  quantity: number;
  stop_loss: number;
  target_1: number;
  target_2: number;
  status: 'open' | 'closed';
  pnl?: number;
  is_paper: boolean;
  opened_at: string;
  closed_at?: string;
}

export interface Order {
  id: string;
  user_id: string;
  signal_id: string;
  position_id?: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  quantity: number;
  price?: number;
  status: 'pending' | 'filled' | 'cancelled' | 'rejected';
  is_paper: boolean;
  created_at: string;
  filled_at?: string;
}

export interface Alert {
  id: string;
  user_id: string;
  signal_id?: string;
  type: 'signal_ready' | 'tp1_hit' | 'tp2_hit' | 'sl_hit' | 'daily_summary';
  message: string;
  read: boolean;
  created_at: string;
}

export interface PipelineRun {
  id: string;
  type: 'daily' | 'hourly';
  status: 'running' | 'completed' | 'failed';
  signals_generated?: number;
  signals_updated?: number;
  error_message?: string;
  started_at: string;
  completed_at?: string;
}

export interface RiskLimit {
  id: string;
  user_id: string;
  max_positions: number;
  max_position_size: number;
  max_daily_loss: number;
  max_total_exposure: number;
  updated_at: string;
}

// UI State types

export interface AppSettings {
  proMode: boolean;
  cinematicMode: boolean;
  theme: 'dark';
}

export interface FilterState {
  horizon?: SignalHorizon[];
  riskGrade?: RiskGrade[];
  signalType?: SignalType[];
  status?: SignalStatus[];
  search?: string;
}

// API Response types

export interface ApiResponse<T> {
  data: T;
  meta?: {
    total?: number;
    page?: number;
    limit?: number;
  };
}

export interface ApiError {
  error: string;
  detail?: string;
  code?: string;
}
