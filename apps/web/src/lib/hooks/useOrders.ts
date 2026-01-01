'use client';

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';

interface Order {
  id: string;
  canonical_symbol: string;
  broker: string;
  side: 'BUY' | 'SELL';
  qty: number;
  order_type: 'MARKET' | 'LIMIT';
  limit_price?: number;
  status: 'CREATED' | 'PENDING' | 'COMPLETED' | 'REJECTED' | 'CANCELLED';
  broker_order_id?: string;
  filled_qty?: number;
  avg_fill_price?: number;
  requested_at: string;
  filled_at?: string;
  rejection_reason?: string;
}

interface OrdersResponse {
  orders: Order[];
  total_count: number;
}

interface OrderFilters {
  status?: string;
  symbol?: string;
  side?: 'BUY' | 'SELL';
  date_from?: string;
  date_to?: string;
  limit?: number;
  offset?: number;
}

export function useOrders(filters: OrderFilters = {}) {
  const queryString = new URLSearchParams(
    Object.entries(filters).reduce(
      (acc, [key, value]) => {
        if (value !== undefined) acc[key] = String(value);
        return acc;
      },
      {} as Record<string, string>
    )
  ).toString();

  return useQuery({
    queryKey: ['orders', filters],
    queryFn: () => {
      const endpoint = queryString ? `/orders?${queryString}` : '/orders';
      return apiClient.get<OrdersResponse>(endpoint);
    },
    refetchInterval: 5000, // Refresh every 5 seconds for active orders
  });
}
