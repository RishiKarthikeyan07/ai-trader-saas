'use client';

import { useMemo, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown } from 'lucide-react';

interface HeatmapData {
  date: string;
  [key: string]: number | string | null;
}

interface PerformanceHeatmapProps {
  data: HeatmapData[];
  metric: 'win_rate' | 'avg_r' | 'profit_factor';
}

const K_BUCKETS = [
  { key: 'K3', label: 'Top 3 (Basic)', tier: 'basic' },
  { key: 'K10', label: 'Top 10 (Pro)', tier: 'pro' },
  { key: 'KALL', label: 'All (Elite)', tier: 'elite' },
];

const HORIZON_BUCKETS = [
  { key: 'H5', label: '5-day' },
  { key: 'H10', label: '10-day' },
];

function getColorForValue(value: number, metric: string): string {
  if (metric === 'win_rate') {
    // Win rate: 0-100%
    if (value >= 70) return 'bg-emerald-500/80';
    if (value >= 60) return 'bg-emerald-500/60';
    if (value >= 50) return 'bg-emerald-500/40';
    if (value >= 40) return 'bg-amber-500/40';
    return 'bg-rose-500/40';
  } else if (metric === 'avg_r') {
    // Avg R: typically 0-5
    if (value >= 3) return 'bg-emerald-500/80';
    if (value >= 2) return 'bg-emerald-500/60';
    if (value >= 1) return 'bg-emerald-500/40';
    if (value >= 0.5) return 'bg-amber-500/40';
    return 'bg-rose-500/40';
  } else {
    // Profit factor: >1 is good
    if (value >= 2) return 'bg-emerald-500/80';
    if (value >= 1.5) return 'bg-emerald-500/60';
    if (value >= 1) return 'bg-emerald-500/40';
    if (value >= 0.5) return 'bg-amber-500/40';
    return 'bg-rose-500/40';
  }
}

function formatValue(value: number | null, metric: string): string {
  if (value === null) return 'N/A';

  if (metric === 'win_rate') {
    return `${value.toFixed(1)}%`;
  } else if (metric === 'avg_r') {
    return `${value.toFixed(2)}R`;
  } else {
    return value.toFixed(2);
  }
}

export default function PerformanceHeatmap({ data, metric }: PerformanceHeatmapProps) {
  const [selectedHorizon, setSelectedHorizon] = useState<'H5' | 'H10'>('H5');

  const metricLabels = {
    win_rate: 'Win Rate',
    avg_r: 'Avg R',
    profit_factor: 'Profit Factor',
  };

  // Calculate summary stats
  const summary = useMemo(() => {
    const result: Record<string, { value: number; trend: 'up' | 'down' | 'neutral' }> = {};

    K_BUCKETS.forEach(({ key }) => {
      const metricKey = `${key}_${selectedHorizon}_${metric}`;
      const values = data
        .map((d) => d[metricKey] as number)
        .filter((v) => v !== null && !isNaN(v));

      if (values.length === 0) {
        result[key] = { value: 0, trend: 'neutral' };
        return;
      }

      const avg = values.reduce((a, b) => a + b, 0) / values.length;

      // Calculate trend (last 7 days vs previous 7 days)
      const recent = values.slice(0, 7);
      const previous = values.slice(7, 14);

      let trend: 'up' | 'down' | 'neutral' = 'neutral';
      if (recent.length > 0 && previous.length > 0) {
        const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
        const previousAvg = previous.reduce((a, b) => a + b, 0) / previous.length;
        trend = recentAvg > previousAvg ? 'up' : recentAvg < previousAvg ? 'down' : 'neutral';
      }

      result[key] = { value: avg, trend };
    });

    return result;
  }, [data, metric, selectedHorizon]);

  return (
    <Card glass>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Top-K Performance: {metricLabels[metric]}</CardTitle>
          <div className="flex gap-2">
            {HORIZON_BUCKETS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setSelectedHorizon(key as 'H5' | 'H10')}
                className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                  selectedHorizon === key
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-3 gap-4">
          {K_BUCKETS.map(({ key, label, tier }) => (
            <div key={key} className="glass-card p-4 rounded-lg">
              <div className="text-xs text-muted-foreground mb-1">{label}</div>
              <div className="flex items-baseline gap-2">
                <div className="text-2xl font-bold">
                  {formatValue(summary[key]?.value, metric)}
                </div>
                {summary[key]?.trend === 'up' && (
                  <TrendingUp className="h-4 w-4 text-emerald-400" />
                )}
                {summary[key]?.trend === 'down' && (
                  <TrendingDown className="h-4 w-4 text-rose-400" />
                )}
              </div>
              <Badge variant={tier === 'elite' ? 'default' : 'secondary'} className="mt-2 text-xs">
                {tier.toUpperCase()}
              </Badge>
            </div>
          ))}
        </div>

        {/* Heatmap Grid */}
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground mb-2">
            Last {data.length} trading days
          </div>

          {/* Header */}
          <div className="grid grid-cols-[120px_repeat(3,1fr)] gap-2 text-xs font-medium text-muted-foreground">
            <div>Date</div>
            {K_BUCKETS.map(({ key, label }) => (
              <div key={key} className="text-center">
                {label}
              </div>
            ))}
          </div>

          {/* Rows */}
          <div className="space-y-1 max-h-[400px] overflow-y-auto custom-scrollbar">
            {data.map((row) => (
              <div key={row.date} className="grid grid-cols-[120px_repeat(3,1fr)] gap-2">
                <div className="text-xs text-muted-foreground flex items-center">
                  {new Date(row.date).toLocaleDateString('en-IN', {
                    month: 'short',
                    day: 'numeric',
                  })}
                </div>
                {K_BUCKETS.map(({ key }) => {
                  const metricKey = `${key}_${selectedHorizon}_${metric}`;
                  const value = row[metricKey] as number | null;
                  const totalKey = `${key}_${selectedHorizon}_total`;
                  const total = row[totalKey] as number | null;

                  return (
                    <div
                      key={key}
                      className={`p-2 rounded text-center text-xs font-semibold ${
                        value !== null ? getColorForValue(value, metric) : 'bg-muted/20'
                      }`}
                      title={total !== null ? `${total} signals` : 'No data'}
                    >
                      {formatValue(value, metric)}
                      {total !== null && total > 0 && (
                        <div className="text-[10px] text-white/60 mt-0.5">
                          n={total}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-emerald-500/80" />
            <span>Excellent</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-emerald-500/40" />
            <span>Good</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-amber-500/40" />
            <span>Fair</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-rose-500/40" />
            <span>Poor</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
