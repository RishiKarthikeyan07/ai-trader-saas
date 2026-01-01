'use client';

import { GlassPanel, MetricTile } from '@/components/design';
import { TrendingUp, Target, Award, Calendar } from 'lucide-react';

// Mock data - will be replaced with API calls
const mockPerformance = {
  totalPnL: 24750.5,
  totalReturn: 12.38,
  winRate: 68.5,
  avgWin: 3200.0,
  avgLoss: -1800.0,
  totalTrades: 47,
  winningTrades: 32,
  losingTrades: 15,
  sharpeRatio: 1.85,
  maxDrawdown: -5.2,
  heatmapData: [
    { date: '2025-01-01', pnl: 1200 },
    { date: '2025-01-02', pnl: -800 },
    { date: '2025-01-03', pnl: 2100 },
    { date: '2025-01-04', pnl: 1500 },
    { date: '2025-01-05', pnl: -300 },
    { date: '2025-01-08', pnl: 1800 },
    { date: '2025-01-09', pnl: 2400 },
    { date: '2025-01-10', pnl: -1200 },
    { date: '2025-01-11', pnl: 3100 },
    { date: '2025-01-12', pnl: 1900 },
  ],
};

export default function PerformancePage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <h1 className="text-3xl font-bold">Performance</h1>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile
          label="Total P&L"
          value={`₹${mockPerformance.totalPnL.toLocaleString('en-IN')}`}
          trend="up"
          icon={<TrendingUp className="w-5 h-5" />}
          change={{ value: mockPerformance.totalReturn, isPositive: true }}
        />
        <MetricTile
          label="Win Rate"
          value={`${mockPerformance.winRate}%`}
          trend="up"
          icon={<Target className="w-5 h-5" />}
        />
        <MetricTile
          label="Sharpe Ratio"
          value={mockPerformance.sharpeRatio.toFixed(2)}
          trend="up"
          icon={<Award className="w-5 h-5" />}
        />
        <MetricTile
          label="Total Trades"
          value={mockPerformance.totalTrades}
          icon={<Calendar className="w-5 h-5" />}
        />
      </div>

      {/* Performance Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trade Statistics */}
        <GlassPanel variant="elevated" className="p-6">
          <h2 className="text-lg font-semibold mb-4">Trade Statistics</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Winning Trades</span>
              <span className="text-accent-green font-semibold font-mono">
                {mockPerformance.winningTrades}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Losing Trades</span>
              <span className="text-accent-red font-semibold font-mono">
                {mockPerformance.losingTrades}
              </span>
            </div>
            <div className="flex items-center justify-between pt-4 border-t border-border/50">
              <span className="text-muted-foreground">Avg Win</span>
              <span className="text-accent-green font-semibold font-mono">
                ₹{mockPerformance.avgWin.toLocaleString('en-IN')}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Avg Loss</span>
              <span className="text-accent-red font-semibold font-mono">
                ₹{Math.abs(mockPerformance.avgLoss).toLocaleString('en-IN')}
              </span>
            </div>
            <div className="flex items-center justify-between pt-4 border-t border-border/50">
              <span className="text-muted-foreground">Max Drawdown</span>
              <span className="text-accent-red font-semibold font-mono">
                {mockPerformance.maxDrawdown}%
              </span>
            </div>
          </div>
        </GlassPanel>

        {/* Risk Metrics */}
        <GlassPanel variant="elevated" className="p-6">
          <h2 className="text-lg font-semibold mb-4">Risk Metrics</h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Sharpe Ratio</span>
              <span className="text-foreground font-semibold font-mono">
                {mockPerformance.sharpeRatio}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Win/Loss Ratio</span>
              <span className="text-foreground font-semibold font-mono">
                {(mockPerformance.avgWin / Math.abs(mockPerformance.avgLoss)).toFixed(2)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Profit Factor</span>
              <span className="text-accent-green font-semibold font-mono">
                {(
                  (mockPerformance.avgWin * mockPerformance.winningTrades) /
                  Math.abs(mockPerformance.avgLoss * mockPerformance.losingTrades)
                ).toFixed(2)}
              </span>
            </div>
          </div>
        </GlassPanel>
      </div>

      {/* Daily P&L Heatmap */}
      <GlassPanel variant="elevated" className="p-6">
        <h2 className="text-lg font-semibold mb-4">Daily P&L Heatmap</h2>
        <div className="grid grid-cols-10 gap-2">
          {mockPerformance.heatmapData.map((day) => {
            const isProfitable = day.pnl >= 0;
            const intensity = Math.min(Math.abs(day.pnl) / 3000, 1);

            return (
              <div
                key={day.date}
                className="aspect-square rounded flex items-center justify-center relative group cursor-pointer transition-transform hover:scale-110"
                style={{
                  backgroundColor: isProfitable
                    ? `rgba(16, 185, 129, ${intensity * 0.8})`
                    : `rgba(239, 68, 68, ${intensity * 0.8})`,
                  border: `1px solid ${isProfitable ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
                }}
              >
                {/* Tooltip */}
                <div className="absolute bottom-full mb-2 hidden group-hover:block glass-panel p-2 text-xs whitespace-nowrap z-10">
                  <div className="font-mono">{day.date}</div>
                  <div
                    className={`font-semibold ${isProfitable ? 'text-accent-green' : 'text-accent-red'}`}
                  >
                    {isProfitable ? '+' : ''}₹{day.pnl.toLocaleString('en-IN')}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
        <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-accent-red/80" />
              <span>Loss</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded bg-accent-green/80" />
              <span>Profit</span>
            </div>
          </div>
          <div>Last 10 trading days</div>
        </div>
      </GlassPanel>
    </div>
  );
}
