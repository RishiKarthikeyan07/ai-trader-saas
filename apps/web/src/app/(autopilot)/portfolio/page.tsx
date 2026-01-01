'use client';

import { GlassPanel, MetricTile, DataFreshnessBadge } from '@/components/design';
import { usePositions } from '@/lib/hooks/usePositions';
import { TrendingUp, TrendingDown, Activity, DollarSign } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function PortfolioPage() {
  const { data, isLoading, error } = usePositions('OPEN');

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-muted-foreground">Loading positions...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-accent-red">Error loading positions: {error.message}</div>
      </div>
    );
  }

  const positions = data?.positions || [];
  const totalPnL = data?.total_unrealized_pnl || 0;
  const totalPnLPercent = data?.total_unrealized_pnl_percent || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Portfolio</h1>
        <DataFreshnessBadge lastUpdated={new Date()} />
      </div>

      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricTile
          label="Open Positions"
          value={positions.length}
          icon={<Activity className="w-5 h-5" />}
        />
        <MetricTile
          label="Total Unrealized P&L"
          value={`₹${totalPnL.toLocaleString('en-IN')}`}
          trend={totalPnL >= 0 ? 'up' : 'down'}
          icon={<DollarSign className="w-5 h-5" />}
        />
        <MetricTile
          label="Total Return"
          value={`${totalPnLPercent >= 0 ? '+' : ''}${totalPnLPercent.toFixed(2)}%`}
          trend={totalPnLPercent >= 0 ? 'up' : 'down'}
          icon={<TrendingUp className="w-5 h-5" />}
        />
      </div>

      {/* Positions Table */}
      <GlassPanel variant="elevated" className="p-6">
        <h2 className="text-lg font-semibold mb-4">Open Positions</h2>

        {positions.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <Activity className="w-16 h-16 mx-auto mb-4 opacity-30" />
            <p>No open positions</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left border-b border-border/50">
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Symbol</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Qty</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Entry</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Current</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">SL</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">TP1</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">P&L</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Return</th>
                  <th className="pb-3 font-medium text-sm text-muted-foreground">Status</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => {
                  const isProfitable = (position.unrealized_pnl || 0) >= 0;
                  return (
                    <tr
                      key={position.id}
                      className="border-b border-border/30 hover:bg-glass-hover transition-colors"
                    >
                      <td className="py-4 font-semibold">{position.canonical_symbol}</td>
                      <td className="py-4 font-mono">{position.qty}</td>
                      <td className="py-4 font-mono">₹{position.avg_entry_price.toFixed(2)}</td>
                      <td className="py-4 font-mono">
                        ₹{(position.current_price || 0).toFixed(2)}
                      </td>
                      <td className="py-4 font-mono text-accent-red">
                        ₹{position.sl.toFixed(2)}
                      </td>
                      <td className="py-4 font-mono text-accent-green">
                        ₹{position.tp1.toFixed(2)}
                      </td>
                      <td
                        className={cn(
                          'py-4 font-mono font-semibold',
                          isProfitable ? 'text-accent-green' : 'text-accent-red'
                        )}
                      >
                        {isProfitable ? (
                          <div className="flex items-center gap-1">
                            <TrendingUp className="w-4 h-4" />
                            +₹{Math.abs(position.unrealized_pnl || 0).toFixed(2)}
                          </div>
                        ) : (
                          <div className="flex items-center gap-1">
                            <TrendingDown className="w-4 h-4" />
                            -₹{Math.abs(position.unrealized_pnl || 0).toFixed(2)}
                          </div>
                        )}
                      </td>
                      <td
                        className={cn(
                          'py-4 font-mono',
                          isProfitable ? 'text-accent-green' : 'text-accent-red'
                        )}
                      >
                        {isProfitable ? '+' : ''}
                        {(position.unrealized_pnl_percent || 0).toFixed(2)}%
                      </td>
                      <td className="py-4">
                        <span className="status-chip active text-xs">
                          {position.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </GlassPanel>
    </div>
  );
}
