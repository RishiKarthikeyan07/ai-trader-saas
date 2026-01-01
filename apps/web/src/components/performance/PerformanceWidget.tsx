'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, ArrowRight, BarChart3 } from 'lucide-react';
import Link from 'next/link';

interface PerformanceSummary {
  K3?: { win_rate: number; avg_r: number; total_signals: number };
  K10?: { win_rate: number; avg_r: number; total_signals: number };
  KALL?: { win_rate: number; avg_r: number; total_signals: number };
}

interface PerformanceWidgetProps {
  summary: PerformanceSummary;
  userTier: 'basic' | 'pro' | 'elite';
}

export default function PerformanceWidget({ summary, userTier }: PerformanceWidgetProps) {
  const k3Data = summary.K3;
  const k10Data = summary.K10;
  const kallData = summary.KALL;

  const showUpsell = userTier !== 'elite';

  return (
    <Card glass className="border-primary/20">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Performance (Last 30d)</CardTitle>
          </div>
          <Link href="/performance">
            <Button variant="ghost" size="sm" className="gap-2">
              View Full
              <ArrowRight className="h-4 w-4" />
            </Button>
          </Link>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Current Tier Performance */}
        <div className="pro-panel">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium">Your Tier ({userTier.toUpperCase()})</span>
            <Badge variant={userTier === 'elite' ? 'default' : 'secondary'}>
              {userTier === 'basic' ? 'Top 3' : userTier === 'pro' ? 'Top 10' : 'All Signals'}
            </Badge>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div>
              <div className="text-xs text-muted-foreground">Win Rate</div>
              <div className="text-xl font-bold text-emerald-400">
                {userTier === 'basic' && k3Data
                  ? `${k3Data.win_rate.toFixed(1)}%`
                  : userTier === 'pro' && k10Data
                  ? `${k10Data.win_rate.toFixed(1)}%`
                  : kallData
                  ? `${kallData.win_rate.toFixed(1)}%`
                  : 'N/A'}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Avg R</div>
              <div className="text-xl font-bold text-cyan-400">
                {userTier === 'basic' && k3Data
                  ? `${k3Data.avg_r.toFixed(2)}R`
                  : userTier === 'pro' && k10Data
                  ? `${k10Data.avg_r.toFixed(2)}R`
                  : kallData
                  ? `${kallData.avg_r.toFixed(2)}R`
                  : 'N/A'}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Signals</div>
              <div className="text-xl font-bold">
                {userTier === 'basic' && k3Data
                  ? k3Data.total_signals
                  : userTier === 'pro' && k10Data
                  ? k10Data.total_signals
                  : kallData
                  ? kallData.total_signals
                  : 'N/A'}
              </div>
            </div>
          </div>
        </div>

        {/* Upsell Comparison */}
        {showUpsell && (
          <div className="glass-card p-3 rounded-lg border-2 border-primary/30">
            <div className="flex items-start gap-3">
              <TrendingUp className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
              <div className="flex-1 space-y-2">
                <p className="text-sm font-medium">
                  {userTier === 'basic'
                    ? 'See what you\'re missing with Pro'
                    : 'Elite unlocks full performance'}
                </p>

                {userTier === 'basic' && k10Data && (
                  <div className="text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Pro (Top 10):</span>
                      <span className="font-semibold">
                        {k10Data.win_rate.toFixed(1)}% WR • {k10Data.avg_r.toFixed(2)}R
                      </span>
                    </div>
                  </div>
                )}

                {kallData && (
                  <div className="text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Elite (All):</span>
                      <span className="font-semibold">
                        {kallData.win_rate.toFixed(1)}% WR • {kallData.avg_r.toFixed(2)}R
                      </span>
                    </div>
                  </div>
                )}

                <Link href="/pricing">
                  <Button size="sm" className="w-full mt-2">
                    Upgrade to {userTier === 'basic' ? 'Pro' : 'Elite'}
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        )}

        {/* Mini Heatmap Preview (optional) */}
        <div className="text-xs text-muted-foreground text-center pt-2 border-t border-border/50">
          <Link href="/performance" className="hover:text-primary transition-colors">
            View full heatmap, trends, and historical performance →
          </Link>
        </div>
      </CardContent>
    </Card>
  );
}
