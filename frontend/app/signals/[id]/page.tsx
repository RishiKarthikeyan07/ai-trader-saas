import { fetchSignalDetail } from '../../../lib/api';
import Link from 'next/link';

export default async function SignalDetail({ params }: { params: { id: string } }) {
  const signal = await fetchSignalDetail(params.id);

  return (
    <div className="container">
      <Link href="/" style={{ color: '#8bc0ff' }}>&larr; Back to dashboard</Link>
      <div className="card" style={{ marginTop: 12 }}>
        <div className="flex-between">
          <h1 style={{ margin: 0 }}>{signal.symbol}</h1>
          <div className={`pill ${signal.signal_type === 'BUY' ? 'badge-success' : signal.signal_type === 'SELL' ? 'badge-danger' : 'badge-warn'}`}>{signal.signal_type}</div>
        </div>
        <p style={{ color: '#9fb1c5' }}>SMC score {signal.smc_score?.toFixed(2)} | Confidence {(signal.confidence * 100).toFixed(1)}%</p>
        <div className="metrics-grid">
          <Metric label="Entry zone" value={`${signal.entry_zone_low.toFixed(2)} - ${signal.entry_zone_high.toFixed(2)}`} />
          <Metric label="Stop loss" value={signal.stop_loss.toFixed(2)} />
          <Metric label="Targets" value={`${signal.target_1.toFixed(2)} / ${signal.target_2.toFixed(2)}`} />
          <Metric label="Expected return" value={`${(signal.expected_return ?? 0).toFixed(2)}%`} />
          <Metric label="Expected volatility" value={`${(signal.expected_volatility ?? 0).toFixed(3)}`} />
          <Metric label="Ready state" value={signal.ready_state ?? 'WAIT'} />
        </div>

        <h3 style={{ marginTop: 20 }}>Timeframe alignment</h3>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(140px,1fr))' }}>
          {signal.tf_alignment && Object.entries(signal.tf_alignment).map(([k, v]) => (
            <Metric key={k} label={k} value={v > 0 ? 'Bullish' : 'Bearish'} />
          ))}
        </div>

        <h3 style={{ marginTop: 20 }}>SMC flags</h3>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(140px,1fr))' }}>
          {signal.smc_flags && Object.entries(signal.smc_flags).map(([k, v]) => (
            <Metric key={k} label={k} value={String(v)} />
          ))}
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <div style={{ opacity: 0.7, fontSize: 12 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700 }}>{value}</div>
    </div>
  );
}
