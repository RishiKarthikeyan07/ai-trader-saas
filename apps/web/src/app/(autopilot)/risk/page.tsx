'use client';

import { GlassPanel, RiskMeter, StatusChip } from '@/components/design';
import { useRiskMetrics, useRiskLimits, useUpdateRiskLimits } from '@/lib/hooks/useRisk';
import { useKillSwitch } from '@/lib/hooks/useAutoPilot';
import { Button } from '@/components/ui/button';
import { AlertTriangle, Shield, AlertOctagon } from 'lucide-react';
import { useState } from 'react';
import toast from 'react-hot-toast';

export default function RiskPage() {
  const { data: metrics, isLoading: metricsLoading } = useRiskMetrics();
  const { data: limits, isLoading: limitsLoading } = useRiskLimits();
  const updateLimits = useUpdateRiskLimits();
  const killSwitch = useKillSwitch();

  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState<any>(limits || {});

  const handleKillSwitch = async () => {
    if (
      !confirm(
        'Are you sure you want to activate the KILL SWITCH? This will:\n\n' +
          '• Cancel ALL pending orders\n' +
          '• Pause AutoPilot\n' +
          '• Send emergency notification\n\n' +
          'This action cannot be undone.'
      )
    ) {
      return;
    }

    try {
      await killSwitch.mutateAsync();
      toast.success('Kill Switch activated successfully');
    } catch (error: any) {
      toast.error(`Failed to activate Kill Switch: ${error.message}`);
    }
  };

  if (metricsLoading || limitsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-muted-foreground">Loading risk metrics...</div>
      </div>
    );
  }

  const hasViolations =
    metrics?.violations.max_positions_breached ||
    metrics?.violations.max_daily_loss_breached ||
    metrics?.violations.max_exposure_breached;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Risk Desk</h1>
        <Button
          variant="destructive"
          size="lg"
          onClick={handleKillSwitch}
          disabled={killSwitch.isPending}
          className="gap-2 bg-accent-red hover:bg-accent-red/90"
        >
          <AlertOctagon className="w-5 h-5" />
          {killSwitch.isPending ? 'Activating...' : 'KILL SWITCH'}
        </Button>
      </div>

      {/* Violations Alert */}
      {hasViolations && (
        <GlassPanel className="p-4 border-accent-red/50 bg-accent-red/10">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-6 h-6 text-accent-red mt-0.5" />
            <div className="flex-1">
              <h3 className="font-semibold text-accent-red mb-2">
                Risk Limit Violations Detected
              </h3>
              <ul className="space-y-1 text-sm">
                {metrics?.violations.max_positions_breached && (
                  <li>• Maximum positions limit exceeded</li>
                )}
                {metrics?.violations.max_daily_loss_breached && (
                  <li>• Daily loss limit breached</li>
                )}
                {metrics?.violations.max_exposure_breached && (
                  <li>• Maximum exposure limit exceeded</li>
                )}
              </ul>
              <StatusChip status="PROTECT" className="mt-3" />
            </div>
          </div>
        </GlassPanel>
      )}

      {/* Risk Meters */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Current Risk Metrics */}
        <GlassPanel variant="elevated" className="p-6">
          <div className="flex items-center gap-3 mb-6">
            <Shield className="w-6 h-6 text-accent-cyan" />
            <h2 className="text-lg font-semibold">Current Risk Metrics</h2>
          </div>

          <div className="space-y-6">
            <RiskMeter
              label="Positions"
              current={metrics?.current_positions || 0}
              max={metrics?.risk_limits.max_positions || 10}
            />

            <RiskMeter
              label="Daily P&L"
              current={Math.abs(metrics?.daily_pnl || 0)}
              max={metrics?.risk_limits.max_daily_loss || 5000}
              riskLevel={
                (metrics?.daily_pnl || 0) < 0
                  ? Math.abs(metrics?.daily_pnl || 0) >
                    (metrics?.risk_limits.max_daily_loss || 5000) * 0.8
                    ? 'high'
                    : 'medium'
                  : 'low'
              }
            />

            <RiskMeter
              label="Exposure"
              current={metrics?.current_exposure || 0}
              max={metrics?.risk_limits.max_exposure || 100000}
            />

            <div className="pt-4 border-t border-border/50">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground">Exposure %</div>
                  <div className="text-xl font-semibold font-mono mt-1">
                    {(metrics?.current_exposure_percent || 0).toFixed(1)}%
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Daily P&L %</div>
                  <div
                    className={`text-xl font-semibold font-mono mt-1 ${
                      (metrics?.daily_pnl_percent || 0) >= 0
                        ? 'text-accent-green'
                        : 'text-accent-red'
                    }`}
                  >
                    {(metrics?.daily_pnl_percent || 0) >= 0 ? '+' : ''}
                    {(metrics?.daily_pnl_percent || 0).toFixed(2)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        </GlassPanel>

        {/* Risk Limits Configuration */}
        <GlassPanel variant="elevated" className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold">Risk Limits</h2>
            {!editMode ? (
              <Button variant="outline" size="sm" onClick={() => setEditMode(true)}>
                Edit
              </Button>
            ) : (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setEditMode(false);
                    setFormData(limits || {});
                  }}
                >
                  Cancel
                </Button>
                <Button
                  size="sm"
                  onClick={async () => {
                    try {
                      await updateLimits.mutateAsync(formData);
                      toast.success('Risk limits updated');
                      setEditMode(false);
                    } catch (error: any) {
                      toast.error(`Failed to update: ${error.message}`);
                    }
                  }}
                  disabled={updateLimits.isPending}
                >
                  {updateLimits.isPending ? 'Saving...' : 'Save'}
                </Button>
              </div>
            )}
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-sm text-muted-foreground">Max Positions</label>
              <input
                type="number"
                value={editMode ? formData.max_positions : limits?.max_positions}
                onChange={(e) =>
                  setFormData({ ...formData, max_positions: parseInt(e.target.value) })
                }
                disabled={!editMode}
                className="w-full mt-1 px-3 py-2 glass-panel rounded-lg font-mono text-lg disabled:opacity-50"
              />
            </div>

            <div>
              <label className="text-sm text-muted-foreground">
                Max Daily Loss (₹)
              </label>
              <input
                type="number"
                value={editMode ? formData.max_daily_loss : limits?.max_daily_loss}
                onChange={(e) =>
                  setFormData({ ...formData, max_daily_loss: parseFloat(e.target.value) })
                }
                disabled={!editMode}
                className="w-full mt-1 px-3 py-2 glass-panel rounded-lg font-mono text-lg disabled:opacity-50"
              />
            </div>

            <div>
              <label className="text-sm text-muted-foreground">
                Max Exposure (₹)
              </label>
              <input
                type="number"
                value={editMode ? formData.max_exposure : limits?.max_exposure}
                onChange={(e) =>
                  setFormData({ ...formData, max_exposure: parseFloat(e.target.value) })
                }
                disabled={!editMode}
                className="w-full mt-1 px-3 py-2 glass-panel rounded-lg font-mono text-lg disabled:opacity-50"
              />
            </div>

            <div>
              <label className="text-sm text-muted-foreground">
                Per Trade Risk (%)
              </label>
              <input
                type="number"
                step="0.1"
                value={
                  editMode
                    ? formData.per_trade_risk_percent
                    : limits?.per_trade_risk_percent
                }
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    per_trade_risk_percent: parseFloat(e.target.value),
                  })
                }
                disabled={!editMode}
                className="w-full mt-1 px-3 py-2 glass-panel rounded-lg font-mono text-lg disabled:opacity-50"
              />
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-border/50">
              <label className="text-sm">Trailing Stop</label>
              <input
                type="checkbox"
                checked={
                  editMode
                    ? formData.trailing_stop_enabled
                    : limits?.trailing_stop_enabled
                }
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    trailing_stop_enabled: e.target.checked,
                  })
                }
                disabled={!editMode}
                className="w-5 h-5"
              />
            </div>
          </div>
        </GlassPanel>
      </div>
    </div>
  );
}
