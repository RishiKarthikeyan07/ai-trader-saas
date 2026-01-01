'use client';

import { cn } from '@/lib/utils';
import { GlassPanel } from './GlassPanel';
import { ReactNode } from 'react';

export interface MetricTileProps {
  label: string;
  value: string | number;
  change?: {
    value: number;
    isPositive: boolean;
  };
  icon?: ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export function MetricTile({
  label,
  value,
  change,
  icon,
  trend,
  className,
}: MetricTileProps) {
  return (
    <GlassPanel className={cn('metric-tile', className)}>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="metric-tile-label">{label}</div>
          <div
            className={cn(
              'metric-tile-value mt-2',
              trend === 'up' && 'text-accent-green',
              trend === 'down' && 'text-accent-red'
            )}
          >
            {value}
          </div>
          {change && (
            <div
              className={cn(
                'mt-1 text-xs font-mono',
                change.isPositive ? 'text-accent-green' : 'text-accent-red'
              )}
            >
              {change.isPositive ? '+' : ''}
              {change.value}%
            </div>
          )}
        </div>
        {icon && <div className="text-graphite-500">{icon}</div>}
      </div>
    </GlassPanel>
  );
}
