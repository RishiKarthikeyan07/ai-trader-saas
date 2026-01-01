'use client';

import { GlassPanel, MetricTile } from '@/components/design';
import { useUIStore } from '@/lib/stores/ui-store';
import { cn } from '@/lib/utils';
import { X, Activity, DollarSign, TrendingUp, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';

export function RightRail() {
  const { rightRailVisible, proMode, toggleRightRail } = useUIStore();

  if (!rightRailVisible || !proMode) return null;

  // Mock data - will be replaced with real data from API
  const quickMetrics = {
    openPositions: 3,
    dailyPnL: 2450.75,
    dailyPnLPercent: 1.23,
    exposure: 75000,
    maxExposure: 100000,
    alerts: 2,
  };

  return (
    <div className="w-64 border-l border-border/50 bg-background/95 backdrop-blur p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Quick Metrics</h3>
        <Button variant="ghost" size="icon" onClick={toggleRightRail}>
          <X className="w-4 h-4" />
        </Button>
      </div>

      {/* Metrics */}
      <div className="space-y-3">
        <GlassPanel className="p-3 space-y-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Activity className="w-3.5 h-3.5" />
            <span>Open Positions</span>
          </div>
          <div className="text-2xl font-semibold font-mono">
            {quickMetrics.openPositions}
          </div>
        </GlassPanel>

        <GlassPanel className="p-3 space-y-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <DollarSign className="w-3.5 h-3.5" />
            <span>Daily P&L</span>
          </div>
          <div
            className={cn(
              'text-2xl font-semibold font-mono',
              quickMetrics.dailyPnL >= 0 ? 'text-accent-green' : 'text-accent-red'
            )}
          >
            ₹{quickMetrics.dailyPnL.toLocaleString('en-IN')}
          </div>
          <div
            className={cn(
              'text-xs font-mono',
              quickMetrics.dailyPnLPercent >= 0
                ? 'text-accent-green'
                : 'text-accent-red'
            )}
          >
            {quickMetrics.dailyPnLPercent >= 0 ? '+' : ''}
            {quickMetrics.dailyPnLPercent}%
          </div>
        </GlassPanel>

        <GlassPanel className="p-3 space-y-2">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <TrendingUp className="w-3.5 h-3.5" />
            <span>Exposure</span>
          </div>
          <div className="text-lg font-semibold font-mono">
            ₹{(quickMetrics.exposure / 1000).toFixed(0)}K
          </div>
          <div className="w-full h-1.5 bg-graphite-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-accent-cyan rounded-full transition-all"
              style={{
                width: `${(quickMetrics.exposure / quickMetrics.maxExposure) * 100}%`,
              }}
            />
          </div>
          <div className="text-xs text-muted-foreground">
            of ₹{(quickMetrics.maxExposure / 1000).toFixed(0)}K max
          </div>
        </GlassPanel>

        {quickMetrics.alerts > 0 && (
          <GlassPanel className="p-3 space-y-2 border-accent-amber/30">
            <div className="flex items-center gap-2 text-xs text-accent-amber">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>{quickMetrics.alerts} Active Alerts</span>
            </div>
          </GlassPanel>
        )}
      </div>
    </div>
  );
}
