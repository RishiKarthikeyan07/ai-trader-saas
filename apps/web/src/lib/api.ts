import { ApiResponse, Signal, ScannerRun, Position, Order, Alert } from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit
): Promise<ApiResponse<T>> {
  const response = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'API request failed');
  }

  return response.json();
}

// Signals API
export async function getLatestSignals(limit: number = 10): Promise<Signal[]> {
  const response = await fetchAPI<Signal[]>(`/signals/latest?limit=${limit}`);
  return response.data;
}

export async function getSignalById(id: string): Promise<Signal> {
  const response = await fetchAPI<Signal>(`/signals/${id}`);
  return response.data;
}

export async function getSignalExplanation(id: string) {
  const response = await fetchAPI(`/signals/${id}/explanation`);
  return response.data;
}

// Scanner API
export async function getScannerPacks() {
  const response = await fetchAPI('/scanner/packs');
  return response.data;
}

export async function getLatestScannerRun(packId: string): Promise<ScannerRun> {
  const response = await fetchAPI<ScannerRun>(`/scanner/run/latest?pack_id=${packId}`);
  return response.data;
}

export async function getScannerCandidates(runId: string) {
  const response = await fetchAPI(`/scanner/run/${runId}/candidates`);
  return response.data;
}

// User API
export async function getCurrentUserProfile() {
  const response = await fetchAPI('/me');
  return response.data;
}

// Alerts API
export async function subscribeToAlert(signalId: string) {
  const response = await fetchAPI('/alerts/subscribe', {
    method: 'POST',
    body: JSON.stringify({ signal_id: signalId }),
  });
  return response.data;
}

export async function getAlerts(): Promise<Alert[]> {
  const response = await fetchAPI<Alert[]>('/alerts');
  return response.data;
}

// Elite API
export async function enableAutoTrading(enabled: boolean) {
  const response = await fetchAPI('/elite/auto/enable', {
    method: 'POST',
    body: JSON.stringify({ enabled }),
  });
  return response.data;
}

export async function getPositions(): Promise<Position[]> {
  const response = await fetchAPI<Position[]>('/elite/positions');
  return response.data;
}

export async function getOrders(): Promise<Order[]> {
  const response = await fetchAPI<Order[]>('/elite/orders');
  return response.data;
}

export async function killSwitch() {
  const response = await fetchAPI('/kill', {
    method: 'POST',
  });
  return response.data;
}

// Admin API
export async function runDailyPipeline() {
  const response = await fetchAPI('/admin/pipeline/run-daily', {
    method: 'POST',
  });
  return response.data;
}

export async function runHourlyPipeline() {
  const response = await fetchAPI('/admin/pipeline/run-hourly', {
    method: 'POST',
  });
  return response.data;
}
