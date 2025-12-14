'use client';

import { useEffect, useState } from 'react';
import SignalCard from '../components/SignalCard';
import { Signal, fetchSignals } from '../lib/api';

export default function Home() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState(true);
  const [tier, setTier] = useState<'basic' | 'pro' | 'elite'>('basic');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchSignals(tier)
      .then((data) => {
        setSignals(data);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [tier]);

  return (
    <div className="container">
      <section>
        <div className="flex-between">
          <div>
            <p className="pill" style={{ background: 'rgba(45, 212, 191, 0.18)', color: '#2dd4bf' }}>Daily institutional-grade swing signals</p>
            <h1 className="hero-title">AI swing trading desk for NSE (Monthly/Weekly/Daily + 4H/1H).</h1>
            <p className="hero-subtitle">Daily StockFormer/TFT/LightGBM fusion for direction + returns, Smart Money Concepts validation, hourly 1H/4H entry refinement, and FinRL PPO exit management for elite automation.</p>
          </div>
          <div className="card" style={{ minWidth: 260 }}>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Tier access</div>
            <div style={{ display: 'flex', gap: 10 }}>
              {(['basic', 'pro', 'elite'] as const).map((t) => (
                <button
                  key={t}
                  className="button"
                  style={{ opacity: tier === t ? 1 : 0.5, flex: 1, justifyContent: 'center' }}
                  onClick={() => setTier(t)}
                >
                  {t.toUpperCase()}
                </button>
              ))}
            </div>
            <p style={{ color: '#9fb1c5', marginTop: 10, fontSize: 13 }}>Signals are generated once centrally and served by tier. Hourly checks only refine entries.</p>
          </div>
        </div>
      </section>

      <section>
        <div className="card">
          <div className="flex-between">
            <h2 style={{ margin: 0 }}>Latest signals</h2>
            <div className="pill" style={{ background: 'rgba(255,255,255,0.08)', color: '#c7d3e0' }}>
              {loading ? 'Updating...' : `${signals.length} assets`}
            </div>
          </div>
          {error && <div style={{ color: '#f87171' }}>{error}</div>}
          {loading && <p>Loading signals...</p>}
          {!loading && signals.length === 0 && <p>No signals yet. Run daily pipeline.</p>}
          <div className="grid" style={{ marginTop: 12 }}>
            {signals.map((s) => (
              <SignalCard key={s.id} signal={s} />
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
