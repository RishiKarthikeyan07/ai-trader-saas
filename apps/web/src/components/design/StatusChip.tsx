'use client';

import { cn } from '@/lib/utils';

export type AutoPilotStatus = 'ON' | 'PAUSED' | 'PROTECT';

export interface StatusChipProps {
  status: AutoPilotStatus;
  className?: string;
}

export function StatusChip({ status, className }: StatusChipProps) {
  const statusConfig = {
    ON: {
      label: 'Active',
      className: 'status-chip active',
      dot: true,
    },
    PAUSED: {
      label: 'Paused',
      className: 'status-chip paused',
      dot: false,
    },
    PROTECT: {
      label: 'Protect Mode',
      className: 'status-chip protect',
      dot: true,
    },
  };

  const config = statusConfig[status];

  return (
    <div className={cn(config.className, className)}>
      {config.dot && (
        <div className="w-2 h-2 rounded-full bg-current animate-pulse" />
      )}
      <span>{config.label}</span>
    </div>
  );
}
