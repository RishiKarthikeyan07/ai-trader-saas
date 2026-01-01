'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import type { AutoPilotStatus } from '@/components/design';

interface AutoPilotStatusResponse {
  status: AutoPilotStatus;
  positions_count: number;
  open_orders_count: number;
  current_exposure: number;
  max_exposure: number;
  daily_pnl: number;
  daily_pnl_percent: number;
  can_take_new_positions: boolean;
  last_brain_run: string | null;
  broker_connected: boolean;
}

interface ToggleAutoPilotRequest {
  enable: boolean;
}

export function useAutoPilotStatus() {
  return useQuery({
    queryKey: ['autopilot', 'status'],
    queryFn: () => apiClient.get<AutoPilotStatusResponse>('/autopilot/status'),
    refetchInterval: 15000, // Refresh every 15 seconds
  });
}

export function useToggleAutoPilot() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: ToggleAutoPilotRequest) =>
      apiClient.post('/autopilot/toggle', data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['autopilot', 'status'] });
    },
  });
}

export function useKillSwitch() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => apiClient.post('/autopilot/kill-switch'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['autopilot', 'status'] });
      queryClient.invalidateQueries({ queryKey: ['positions'] });
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });
}
