'use client';

import { GlassPanel } from '@/components/design';
import { cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, AlertCircle } from 'lucide-react';

interface Position {
  symbol: string;
  qty: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  sl: number;
  tp1: number;
  status: 'OPEN' | 'SL_HIT' | 'TP1_HIT';
}

interface FocusListProps {
  positions: Position[];
}

export function FocusList({ positions }: FocusListProps) {
  return (
    <GlassPanel variant="elevated" className="p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Live Positions</h2>
        <div className="text-sm text-muted-foreground">
          {positions.length} active
        </div>
      </div>

      <div className="space-y-3">
        {positions.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No open positions</p>
          </div>
        ) : (
          positions.map((position) => {
            const isProfitable = position.unrealizedPnL >= 0;
            const distanceToSL = ((position.currentPrice - position.sl) / position.currentPrice) * 100;
            const distanceToTP = ((position.tp1 - position.currentPrice) / position.currentPrice) * 100;

            return (
              <GlassPanel key={position.symbol} className="p-4 hover:bg-glass-hover transition-all">
                <div className="flex items-center justify-between">
                  {/* Symbol & Qty */}
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold text-lg">{position.symbol}</h3>
                      <span className="text-xs text-muted-foreground">
                        {position.qty} qty
                      </span>
                    </div>
                    <div className="text-sm text-muted-foreground mt-1">
                      Entry: ₹{position.entryPrice.toFixed(2)}
                    </div>
                  </div>

                  {/* Current Price */}
                  <div className="flex-1 text-center">
                    <div className="text-xs text-muted-foreground mb-1">Current</div>
                    <div className="text-2xl font-semibold font-mono">
                      ₹{position.currentPrice.toFixed(2)}
                    </div>
                  </div>

                  {/* P&L */}
                  <div className="flex-1 text-right">
                    <div
                      className={cn(
                        'flex items-center justify-end gap-2 text-xl font-semibold font-mono',
                        isProfitable ? 'text-accent-green' : 'text-accent-red'
                      )}
                    >
                      {isProfitable ? (
                        <TrendingUp className="w-5 h-5" />
                      ) : (
                        <TrendingDown className="w-5 h-5" />
                      )}
                      <span>
                        {isProfitable ? '+' : ''}₹
                        {Math.abs(position.unrealizedPnL).toFixed(2)}
                      </span>
                    </div>
                    <div
                      className={cn(
                        'text-sm mt-1',
                        isProfitable ? 'text-accent-green' : 'text-accent-red'
                      )}
                    >
                      {isProfitable ? '+' : ''}
                      {position.unrealizedPnLPercent.toFixed(2)}%
                    </div>
                  </div>
                </div>

                {/* SL/TP Indicators */}
                <div className="mt-3 flex items-center gap-4 text-xs">
                  <div>
                    <span className="text-muted-foreground">SL: </span>
                    <span className="text-accent-red font-mono">
                      ₹{position.sl.toFixed(2)}
                    </span>
                    <span className="text-muted-foreground ml-1">
                      ({distanceToSL.toFixed(1)}%)
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">TP1: </span>
                    <span className="text-accent-green font-mono">
                      ₹{position.tp1.toFixed(2)}
                    </span>
                    <span className="text-muted-foreground ml-1">
                      (+{distanceToTP.toFixed(1)}%)
                    </span>
                  </div>
                </div>
              </GlassPanel>
            );
          })
        )}
      </div>
    </GlassPanel>
  );
}
