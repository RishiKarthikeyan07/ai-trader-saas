'use client';

import { GlassPanel, StatusChip, type AutoPilotStatus } from '@/components/design';
import { Activity, DollarSign, Shield, TrendingUp } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StatusStripProps {
  status: AutoPilotStatus;
  openPositions: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  exposure: number;
  maxExposure: number;
}

export function StatusStrip({
  status,
  openPositions,
  dailyPnL,
  dailyPnLPercent,
  exposure,
  maxExposure,
}: StatusStripProps) {
  const exposurePercent = (exposure / maxExposure) * 100;

  return (
    <GlassPanel className="p-4">
      <div className="flex items-center justify-between gap-6">
        {/* Status */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-accent-cyan/20 flex items-center justify-center">
            <Activity className="w-5 h-5 text-accent-cyan" />
          </div>
          <div>
            <div className="text-xs text-muted-foreground">AutoPilot Status</div>
            <StatusChip status={status} className="mt-1" />
          </div>
        </div>

        {/* Divider */}
        <div className="h-12 w-px bg-border" />

        {/* Open Positions */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-accent-cyan/10 flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-accent-cyan" />
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Open Positions</div>
            <div className="text-xl font-semibold font-mono">{openPositions}</div>
          </div>
        </div>

        {/* Divider */}
        <div className="h-12 w-px bg-border" />

        {/* Daily P&L */}
        <div className="flex items-center gap-3">
          <div
            className={cn(
              'w-10 h-10 rounded-lg flex items-center justify-center',
              dailyPnL >= 0 ? 'bg-accent-green/10' : 'bg-accent-red/10'
            )}
          >
            <DollarSign
              className={cn(
                'w-5 h-5',
                dailyPnL >= 0 ? 'text-accent-green' : 'text-accent-red'
              )}
            />
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Daily P&L</div>
            <div
              className={cn(
                'text-xl font-semibold font-mono',
                dailyPnL >= 0 ? 'text-accent-green' : 'text-accent-red'
              )}
            >
              ₹{dailyPnL.toLocaleString('en-IN')}
              <span className="text-sm ml-2">
                ({dailyPnLPercent >= 0 ? '+' : ''}
                {dailyPnLPercent}%)
              </span>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="h-12 w-px bg-border" />

        {/* Exposure */}
        <div className="flex items-center gap-3 flex-1">
          <div
            className={cn(
              'w-10 h-10 rounded-lg flex items-center justify-center',
              exposurePercent > 80
                ? 'bg-accent-red/10'
                : exposurePercent > 60
                  ? 'bg-accent-amber/10'
                  : 'bg-accent-green/10'
            )}
          >
            <Shield
              className={cn(
                'w-5 h-5',
                exposurePercent > 80
                  ? 'text-accent-red'
                  : exposurePercent > 60
                    ? 'text-accent-amber'
                    : 'text-accent-green'
              )}
            />
          </div>
          <div className="flex-1">
            <div className="text-xs text-muted-foreground">
              Exposure ({exposurePercent.toFixed(1)}%)
            </div>
            <div className="text-lg font-semibold font-mono">
              ₹{(exposure / 1000).toFixed(0)}K / ₹{(maxExposure / 1000).toFixed(0)}K
            </div>
            <div className="w-full h-1.5 bg-graphite-800 rounded-full overflow-hidden mt-1">
              <div
                className={cn(
                  'h-full rounded-full transition-all',
                  exposurePercent > 80
                    ? 'bg-accent-red'
                    : exposurePercent > 60
                      ? 'bg-accent-amber'
                      : 'bg-accent-green'
                )}
                style={{ width: `${exposurePercent}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </GlassPanel>
  );
}
