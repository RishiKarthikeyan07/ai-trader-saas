'use client';

import { Signal } from '../lib/api';

function signalColor(type: Signal['signal_type']) {
  if (type === 'BUY') return 'badge-success';
  if (type === 'SELL') return 'badge-danger';
  return 'badge-warn';
}

export default function SignalCard({ signal }: { signal: Signal }) {
  return (
    <div className="signal-card">
      <div className="flex-between">
        <div className="flex" style={{ gap: 12 }}>
          <div className={`pill ${signalColor(signal.signal_type)}`}>{signal.signal_type}</div>
          <h3>{signal.symbol}</h3>
        </div>
        <div className="pill" style={{ background: 'rgba(255,255,255,0.08)', color: '#e7edf4' }}>
          Confidence: {(signal.confidence * 100).toFixed(1)}%
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px,1fr))', gap: 8, marginTop: 10 }}>
        <Metric label="Entry" value={`${signal.entry_zone_low.toFixed(2)} - ${signal.entry_zone_high.toFixed(2)}`}/>
        <Metric label="Stop" value={signal.stop_loss.toFixed(2)} />
        <Metric label="Targets" value={`${signal.target_1.toFixed(2)} / ${signal.target_2.toFixed(2)}`} />
        <Metric label="SMC" value={(signal.smc_score ?? 0).toFixed(2)} />
        <Metric label="Ready" value={signal.ready_state || 'WAIT'} />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ color: '#c7d3e0', fontSize: 12 }}>
      <div style={{ opacity: 0.7 }}>{label}</div>
      <div style={{ fontSize: 16, color: '#fff', fontWeight: 600 }}>{value}</div>
    </div>
  );
}
