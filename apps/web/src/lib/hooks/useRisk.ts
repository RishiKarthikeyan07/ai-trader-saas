'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

interface RiskLimits {
  max_positions: number;
  max_daily_loss: number;
  max_daily_loss_percent: number;
  max_exposure: number;
  max_exposure_percent: number;
  per_trade_risk_percent: number;
  trailing_stop_enabled: boolean;
  trailing_stop_points: number;
}

interface RiskMetrics {
  current_positions: number;
  current_exposure: number;
  current_exposure_percent: number;
  daily_pnl: number;
  daily_pnl_percent: number;
  risk_limits: RiskLimits;
  violations: {
    max_positions_breached: boolean;
    max_daily_loss_breached: boolean;
    max_exposure_breached: boolean;
  };
}

export function useRiskLimits() {
  return useQuery({
    queryKey: ['risk', 'limits'],
    queryFn: () => apiClient.get<RiskLimits>('/risk/limits'),
  });
}

export function useRiskMetrics() {
  return useQuery({
    queryKey: ['risk', 'metrics'],
    queryFn: () => apiClient.get<RiskMetrics>('/risk/metrics'),
    refetchInterval: 10000, // Refresh every 10 seconds
  });
}

export function useUpdateRiskLimits() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (limits: Partial<RiskLimits>) =>
      apiClient.post('/risk/limits', limits),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['risk', 'limits'] });
      queryClient.invalidateQueries({ queryKey: ['risk', 'metrics'] });
    },
  });
}
