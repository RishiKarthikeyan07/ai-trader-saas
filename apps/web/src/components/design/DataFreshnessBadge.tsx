'use client';

import { formatDistanceToNow } from 'date-fns';
import { useEffect, useState } from 'react';

export interface DataFreshnessBadgeProps {
  lastUpdated: Date | string;
  label?: string;
}

export function DataFreshnessBadge({
  lastUpdated,
  label = 'Updated',
}: DataFreshnessBadgeProps) {
  const [timeAgo, setTimeAgo] = useState('');

  useEffect(() => {
    const updateTimeAgo = () => {
      const date = typeof lastUpdated === 'string' ? new Date(lastUpdated) : lastUpdated;
      setTimeAgo(formatDistanceToNow(date, { addSuffix: true }));
    };

    updateTimeAgo();
    const interval = setInterval(updateTimeAgo, 30000); // Update every 30s

    return () => clearInterval(interval);
  }, [lastUpdated]);

  return (
    <div className="freshness-badge">
      <span>
        {label} {timeAgo}
      </span>
    </div>
  );
}
