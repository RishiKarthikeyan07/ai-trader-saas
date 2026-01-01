'use client';

import { GlassPanel, DataFreshnessBadge } from '@/components/design';
import { TrendingUp, Target, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

// Mock data - replace with real API call to admin-only endpoint
const mockIntentions = [
  {
    id: '1',
    date: '2025-01-26',
    canonical_symbol: 'RELIANCE',
    direction: 'BUY',
    entry_zone_low: 2401.0,
    entry_zone_high: 2449.0,
    sl: 2352.5,
    tp1: 2547.5,
    tp2: 2620.0,
    confidence: 0.8745,
    risk_grade: 'LOW',
    horizon: 'SWING',
    tags: ['trend aligned', 'volume surge'],
  },
  {
    id: '2',
    date: '2025-01-26',
    canonical_symbol: 'TCS',
    direction: 'BUY',
    entry_zone_low: 3577.0,
    entry_zone_high: 3649.0,
    sl: 3504.5,
    tp1: 3721.5,
    tp2: 3867.0,
    confidence: 0.8234,
    risk_grade: 'LOW',
    horizon: 'POSITIONAL',
    tags: ['trend aligned', 'normal volume'],
  },
  {
    id: '3',
    date: '2025-01-26',
    canonical_symbol: 'INFY',
    direction: 'BUY',
    entry_zone_low: 1391.6,
    entry_zone_high: 1420.4,
    sl: 1377.4,
    tp1: 1491.0,
    tp2: 1562.4,
    confidence: 0.7621,
    risk_grade: 'MEDIUM',
    horizon: 'SWING',
    tags: ['trend weak', 'volume surge'],
  },
];

export default function AdminIntentionsPage() {
  const getRiskColor = (grade: string) => {
    switch (grade) {
      case 'LOW':
        return 'text-accent-green bg-accent-green/10 border-accent-green/30';
      case 'MEDIUM':
        return 'text-accent-amber bg-accent-amber/10 border-accent-amber/30';
      case 'HIGH':
        return 'text-accent-red bg-accent-red/10 border-accent-red/30';
      default:
        return 'text-muted-foreground bg-muted/10 border-muted/30';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-accent-red">
            🔒 Trade Intentions (Internal Only)
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            NEVER expose this data to users - for admin eyes only
          </p>
        </div>
        <DataFreshnessBadge lastUpdated={new Date()} label="Generated" />
      </div>

      <div className="glass-panel p-4 border-accent-amber/50 bg-accent-amber/10">
        <div className="flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-accent-amber mt-0.5" />
          <div>
            <h3 className="font-semibold text-accent-amber mb-1">Admin-Only Data</h3>
            <p className="text-sm text-amber-200/80">
              This is the AI brain output. Users NEVER see this directly. They only see:
              <br />• Executed trades (what the bot did)
              <br />• Open positions (what's currently held)
              <br />• Performance (equity curve, stats)
            </p>
          </div>
        </div>
      </div>

      <GlassPanel variant="elevated" className="p-6">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Today's Intentions</h2>
          <div className="text-sm text-muted-foreground">
            {mockIntentions.length} opportunities identified
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-border/50">
                <th className="pb-3 font-medium text-sm text-muted-foreground">Symbol</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Confidence</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Risk</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Horizon</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Entry Zone</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">SL</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">TP1</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">TP2</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Tags</th>
              </tr>
            </thead>
            <tbody>
              {mockIntentions.map((intention) => (
                <tr
                  key={intention.id}
                  className="border-b border-border/30 hover:bg-glass-hover transition-colors"
                >
                  <td className="py-4 font-semibold">{intention.canonical_symbol}</td>
                  <td className="py-4">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-2 bg-graphite-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-accent-cyan rounded-full"
                          style={{ width: `${intention.confidence * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-sm">
                        {(intention.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="py-4">
                    <span
                      className={cn(
                        'px-2 py-1 rounded-full text-xs font-semibold border',
                        getRiskColor(intention.risk_grade)
                      )}
                    >
                      {intention.risk_grade}
                    </span>
                  </td>
                  <td className="py-4 text-sm">{intention.horizon}</td>
                  <td className="py-4 font-mono text-sm">
                    ₹{intention.entry_zone_low} - ₹{intention.entry_zone_high}
                  </td>
                  <td className="py-4 font-mono text-sm text-accent-red">
                    ₹{intention.sl}
                  </td>
                  <td className="py-4 font-mono text-sm text-accent-green">
                    ₹{intention.tp1}
                  </td>
                  <td className="py-4 font-mono text-sm text-accent-green">
                    ₹{intention.tp2 || '-'}
                  </td>
                  <td className="py-4">
                    <div className="flex flex-wrap gap-1">
                      {intention.tags.map((tag, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 rounded text-xs bg-muted text-muted-foreground"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassPanel>

      <GlassPanel variant="elevated" className="p-6">
        <h3 className="font-semibold mb-3">How It Works</h3>
        <ol className="space-y-2 text-sm text-muted-foreground list-decimal list-inside">
          <li>Daily brain runs at 7 AM IST</li>
          <li>PKScreener scans ~5,000 stocks for technical patterns</li>
          <li>AI model ranks candidates by confidence</li>
          <li>Top 20 become "trade intentions" (stored in this table)</li>
          <li>
            Executor checks these intentions every 15 min during market hours
          </li>
          <li>
            If price enters entry zone + user has capacity → creates order
          </li>
          <li>User sees ONLY the executed order, NOT the intention</li>
        </ol>
      </GlassPanel>
    </div>
  );
}
