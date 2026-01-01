'use client';

import { MetricTile, GlassPanel, DataFreshnessBadge } from '@/components/design';
import { StatusStrip } from '@/components/features/StatusStrip';
import { FocusList } from '@/components/features/FocusList';
import { Activity, TrendingUp, Target, Zap } from 'lucide-react';
import { useUIStore } from '@/lib/stores/ui-store';

// Mock data - will be replaced with API calls
const mockData = {
  autopilotStatus: 'ON' as const,
  openPositions: 3,
  dailyPnL: 2450.75,
  dailyPnLPercent: 1.23,
  exposure: 75000,
  maxExposure: 100000,
  winRate: 68.5,
  avgWin: 3.2,
  opportunities: [
    {
      symbol: 'RELIANCE',
      confidence: 0.87,
      riskGrade: 'LOW' as const,
      expectedReturn: 5.2,
    },
    {
      symbol: 'TCS',
      confidence: 0.82,
      riskGrade: 'LOW' as const,
      expectedReturn: 4.1,
    },
    {
      symbol: 'INFY',
      confidence: 0.76,
      riskGrade: 'MEDIUM' as const,
      expectedReturn: 6.8,
    },
    {
      symbol: 'HDFCBANK',
      confidence: 0.71,
      riskGrade: 'MEDIUM' as const,
      expectedReturn: 3.9,
    },
    {
      symbol: 'ICICIBANK',
      confidence: 0.68,
      riskGrade: 'HIGH' as const,
      expectedReturn: 8.5,
    },
  ],
  positions: [
    {
      symbol: 'RELIANCE',
      qty: 50,
      entryPrice: 2450.0,
      currentPrice: 2512.5,
      unrealizedPnL: 3125.0,
      unrealizedPnLPercent: 2.55,
      sl: 2377.5,
      tp1: 2572.5,
      status: 'OPEN' as const,
    },
    {
      symbol: 'TCS',
      qty: 30,
      entryPrice: 3650.0,
      currentPrice: 3598.0,
      unrealizedPnL: -1560.0,
      unrealizedPnLPercent: -1.42,
      sl: 3540.5,
      tp1: 3832.5,
      status: 'OPEN' as const,
    },
    {
      symbol: 'INFY',
      qty: 100,
      entryPrice: 1420.0,
      currentPrice: 1462.8,
      unrealizedPnL: 4280.0,
      unrealizedPnLPercent: 3.01,
      sl: 1377.4,
      tp1: 1491.0,
      status: 'OPEN' as const,
    },
  ],
  lastUpdated: new Date(),
};

export default function HomePage() {
  const { proMode } = useUIStore();

  return (
    <div className="space-y-6">
      {/* Status Strip */}
      <StatusStrip
        status={mockData.autopilotStatus}
        openPositions={mockData.openPositions}
        dailyPnL={mockData.dailyPnL}
        dailyPnLPercent={mockData.dailyPnLPercent}
        exposure={mockData.exposure}
        maxExposure={mockData.maxExposure}
      />

      {/* Metric Tiles Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile
          label="Open Positions"
          value={mockData.openPositions}
          icon={<Activity className="w-5 h-5" />}
          trend="neutral"
        />
        <MetricTile
          label="Win Rate"
          value={`${mockData.winRate}%`}
          icon={<Target className="w-5 h-5" />}
          trend="up"
        />
        <MetricTile
          label="Avg Win"
          value={`${mockData.avgWin}%`}
          icon={<TrendingUp className="w-5 h-5" />}
          trend="up"
        />
        <MetricTile
          label="Opportunities"
          value={mockData.opportunities.length}
          icon={<Zap className="w-5 h-5" />}
          trend="neutral"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Opportunities List */}
        <GlassPanel variant="elevated" className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Top Opportunities</h2>
            <DataFreshnessBadge lastUpdated={mockData.lastUpdated} />
          </div>
          <div className="space-y-3">
            {mockData.opportunities.map((opp) => (
              <div
                key={opp.symbol}
                className="flex items-center justify-between p-3 rounded-lg bg-graphite-900/50 hover:bg-graphite-900/70 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      opp.riskGrade === 'LOW'
                        ? 'bg-accent-green'
                        : opp.riskGrade === 'MEDIUM'
                        ? 'bg-accent-amber'
                        : 'bg-accent-red'
                    }`}
                  />
                  <div>
                    <div className="font-semibold">{opp.symbol}</div>
                    <div className="text-xs text-muted-foreground">
                      {opp.riskGrade} Risk
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-accent-green">
                    +{opp.expectedReturn}%
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {(opp.confidence * 100).toFixed(0)}% confidence
                  </div>
                </div>
              </div>
            ))}
          </div>
        </GlassPanel>

        {/* Focus List - Live Positions */}
        <FocusList positions={mockData.positions} />
      </div>

      {/* Today's Action Timeline */}
      <GlassPanel variant="elevated" className="p-6">
        <h2 className="text-lg font-semibold mb-4">Today's Activity</h2>
        <div className="space-y-3">
          <div className="flex items-start gap-4">
            <div className="w-2 h-2 rounded-full bg-accent-green mt-2" />
            <div className="flex-1">
              <div className="text-sm font-medium">Position Entered: INFY</div>
              <div className="text-xs text-muted-foreground">
                09:45 AM - Entry confirmed at ₹1420.00
              </div>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="w-2 h-2 rounded-full bg-accent-cyan mt-2" />
            <div className="flex-1">
              <div className="text-sm font-medium">Daily Brain Run Completed</div>
              <div className="text-xs text-muted-foreground">
                07:00 AM - {mockData.opportunities.length} opportunities identified
              </div>
            </div>
          </div>
        </div>
      </GlassPanel>
    </div>
  );
}
