'use client';

import { useState } from 'react';
import { GlassPanel, DataFreshnessBadge } from '@/components/design';
import { useOrders } from '@/lib/hooks/useOrders';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { format } from 'date-fns';
import { Filter, Download } from 'lucide-react';

export default function OrdersPage() {
  const [filters, setFilters] = useState({});
  const { data, isLoading, error } = useOrders(filters);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-muted-foreground">Loading orders...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-accent-red">Error loading orders: {error.message}</div>
      </div>
    );
  }

  const orders = data?.orders || [];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'COMPLETED':
        return 'text-accent-green bg-accent-green/10 border-accent-green/30';
      case 'PENDING':
        return 'text-accent-cyan bg-accent-cyan/10 border-accent-cyan/30';
      case 'REJECTED':
      case 'CANCELLED':
        return 'text-accent-red bg-accent-red/10 border-accent-red/30';
      default:
        return 'text-muted-foreground bg-muted/10 border-muted/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Order Blotter</h1>
        <div className="flex items-center gap-3">
          <DataFreshnessBadge lastUpdated={new Date()} />
          <Button variant="outline" size="sm" className="gap-2">
            <Filter className="w-4 h-4" />
            Filters
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Download className="w-4 h-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Orders Table */}
      <GlassPanel variant="elevated" className="p-6">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">All Orders</h2>
          <div className="text-sm text-muted-foreground">
            {orders.length} orders
          </div>
        </div>

        {orders.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <p>No orders found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left border-b border-border/50">
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Time</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Symbol</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Side</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Type</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Qty</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Price</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Status</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Broker ID</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order) => (
                  <tr
                    key={order.id}
                    className="border-b border-border/30 hover:bg-glass-hover transition-colors"
                  >
                    <td className="py-4 font-mono text-sm">
                      {format(new Date(order.requested_at), 'HH:mm:ss')}
                    </td>
                    <td className="py-4 font-semibold">{order.canonical_symbol}</td>
                    <td className="py-4">
                      <span
                        className={cn(
                          'px-2 py-1 rounded text-xs font-semibold',
                          order.side === 'BUY'
                            ? 'bg-accent-green/20 text-accent-green'
                            : 'bg-accent-red/20 text-accent-red'
                        )}
                      >
                        {order.side}
                      </span>
                    </td>
                    <td className="py-4 text-sm">{order.order_type}</td>
                    <td className="py-4 font-mono">{order.qty}</td>
                    <td className="py-4 font-mono">
                      {order.avg_fill_price
                        ? `₹${order.avg_fill_price.toFixed(2)}`
                        : order.limit_price
                          ? `₹${order.limit_price.toFixed(2)}`
                          : 'Market'}
                    </td>
                    <td className="py-4">
                      <span
                        className={cn(
                          'px-2 py-1 rounded-full text-xs font-semibold border',
                          getStatusColor(order.status)
                        )}
                      >
                        {order.status}
                      </span>
                    </td>
                    <td className="py-4 font-mono text-xs text-muted-foreground">
                      {order.broker_order_id || '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </GlassPanel>
    </div>
  );
}
