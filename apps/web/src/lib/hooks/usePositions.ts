'use client';

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

interface Position {
  id: string;
  canonical_symbol: string;
  qty: number;
  avg_entry_price: number;
  current_price?: number;
  unrealized_pnl?: number;
  unrealized_pnl_percent?: number;
  sl: number;
  tp1: number;
  tp2?: number;
  status: 'OPEN' | 'CLOSED' | 'SL_HIT' | 'TP1_HIT' | 'TP2_HIT';
  entry_time: string;
  exit_time?: string;
  trailing_stop_data?: {
    initial_sl: number;
    current_sl: number;
    trail_points: number;
  };
}

interface PositionsResponse {
  positions: Position[];
  total_unrealized_pnl: number;
  total_unrealized_pnl_percent: number;
}

export function usePositions(status: 'OPEN' | 'CLOSED' | 'ALL' = 'OPEN') {
  return useQuery({
    queryKey: ['positions', status],
    queryFn: () => {
      const params = status !== 'ALL' ? `?status=${status}` : '';
      return apiClient.get<PositionsResponse>(`/portfolio/positions${params}`);
    },
    refetchInterval: 10000, // Refresh every 10 seconds for live positions
  });
}
