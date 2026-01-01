'use client';

import { cn } from '@/lib/utils';

export type RiskLevel = 'low' | 'medium' | 'high';

export interface RiskMeterProps {
  label: string;
  current: number;
  max: number;
  riskLevel?: RiskLevel;
  className?: string;
}

export function RiskMeter({
  label,
  current,
  max,
  riskLevel,
  className,
}: RiskMeterProps) {
  const percentage = Math.min((current / max) * 100, 100);

  // Auto-determine risk level if not provided
  const level: RiskLevel =
    riskLevel ||
    (percentage < 50 ? 'low' : percentage < 80 ? 'medium' : 'high');

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between text-sm">
        <span className="text-graphite-500 font-medium">{label}</span>
        <span className="font-mono text-foreground">
          {current.toFixed(2)} / {max.toFixed(2)}
        </span>
      </div>
      <div className="risk-meter">
        <div
          className={cn('risk-meter-fill', level)}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
