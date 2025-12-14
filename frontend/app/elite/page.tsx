'use client';

import { useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export default function ElitePanel() {
  const [enabled, setEnabled] = useState(false);
  const [message, setMessage] = useState('');
  const [quantity, setQuantity] = useState(10);
  const [signalId, setSignalId] = useState('');

  async function toggle(value: boolean) {
    const res = await fetch(`${API_BASE}/elite/auto/enable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: value }),
    });
    const data = await res.json();
    setEnabled(data.elite_auto_enabled);
  }

  async function execute() {
    const res = await fetch(`${API_BASE}/elite/trade/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signal_id: signalId, quantity }),
    });
    const data = await res.json();
    setMessage(JSON.stringify(data, null, 2));
  }

  return (
    <div className="container">
      <h1>Elite automation</h1>
      <p style={{ color: '#9fb1c5' }}>BUY-only auto execution when hourly READY_TO_ENTER. PPO exit management runs on live positions (exit-only policy).</p>
      <div className="card">
        <div className="flex-between">
          <div>
            <p>Auto execution is {enabled ? 'enabled' : 'disabled'}</p>
            <button className="button" onClick={() => toggle(!enabled)}>{enabled ? 'Disable' : 'Enable'} auto</button>
          </div>
          <div style={{ maxWidth: 320 }}>
            <label style={{ display: 'block', marginBottom: 6 }}>Signal ID</label>
            <input value={signalId} onChange={(e) => setSignalId(e.target.value)} style={{ width: '100%', padding: 8, borderRadius: 10, border: '1px solid var(--border)', background: 'rgba(255,255,255,0.04)', color: '#fff' }} />
            <label style={{ display: 'block', margin: '10px 0 6px' }}>Quantity</label>
            <input type="number" value={quantity} onChange={(e) => setQuantity(Number(e.target.value))} style={{ width: '100%', padding: 8, borderRadius: 10, border: '1px solid var(--border)', background: 'rgba(255,255,255,0.04)', color: '#fff' }} />
            <button className="button" style={{ marginTop: 12 }} onClick={execute}>Execute BUY</button>
          </div>
        </div>
        {message && <pre style={{ marginTop: 14 }}>{message}</pre>}
      </div>
    </div>
  );
}
