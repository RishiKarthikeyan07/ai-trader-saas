'use client';

import { useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export default function AdminPanel() {
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);

  async function trigger(path: string) {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}${path}`, { method: 'POST' });
      const data = await res.json();
      setMessage(JSON.stringify(data));
    } catch (err) {
      setMessage('Failed: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1>Admin panel</h1>
      <p style={{ color: '#9fb1c5' }}>Run daily heavy models once/day. Hourly only refines entries for existing signals.</p>
      <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(240px,1fr))' }}>
        <button className="button" disabled={loading} onClick={() => trigger('/pipeline/run-daily')}>
          Run daily pipeline
        </button>
        <button className="button" disabled={loading} onClick={() => trigger('/pipeline/run-hourly')}>
          Run hourly refinement
        </button>
        <button className="button" style={{ background: 'linear-gradient(135deg,#f43f5e,#be123c)', color: '#fff' }} disabled={loading} onClick={() => trigger('/kill')}>
          Kill switch
        </button>
      </div>
      {message && <div className="card" style={{ marginTop: 16, whiteSpace: 'pre-wrap' }}>{message}</div>}
    </div>
  );
}
