export type Signal = {
  id: string;
  symbol: string;
  signal_type: 'BUY' | 'SELL' | 'HOLD';
  entry_zone_low: number;
  entry_zone_high: number;
  stop_loss: number;
  target_1: number;
  target_2: number;
  confidence: number;
  expected_return?: number;
  expected_volatility?: number;
  tf_alignment?: Record<string, number>;
  smc_score?: number;
  smc_flags?: Record<string, number>;
  ready_state?: 'READY_TO_ENTER' | 'WAIT';
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export async function fetchSignals(tier: 'basic' | 'pro' | 'elite' = 'basic'): Promise<Signal[]> {
  const res = await fetch(`${API_BASE}/signals/latest?tier=${tier}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error('Failed to fetch signals');
  }
  return res.json();
}

export async function fetchSignalDetail(id: string): Promise<Signal> {
  const res = await fetch(`${API_BASE}/signals/${id}`, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error('Signal not found');
  }
  return res.json();
}
