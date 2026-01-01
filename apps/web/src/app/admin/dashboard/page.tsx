'use client';

import { GlassPanel, MetricTile } from '@/components/design';
import { Button } from '@/components/ui/button';
import { Users, Activity, TrendingUp, AlertTriangle, Play, StopCircle } from 'lucide-react';
import { useState } from 'react';
import toast from 'react-hot-toast';

export default function AdminDashboardPage() {
  const [loading, setLoading] = useState(false);

  const handleTriggerDailyBrain = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/admin/pipeline/run-daily', {
        method: 'POST',
      });
      const data = await res.json();
      toast.success(`Daily brain completed! Generated ${data.intentions_count} intentions`);
    } catch (error: any) {
      toast.error(`Failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleTriggerExecutor = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/admin/pipeline/run-executor', {
        method: 'POST',
      });
      const data = await res.json();
      toast.success(`Executor completed! Created ${data.orders_created} orders`);
    } catch (error: any) {
      toast.error(`Failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGlobalKillSwitch = async () => {
    if (!confirm('Are you sure you want to activate GLOBAL KILL SWITCH?\n\nThis will:\n• Pause ALL user AutoPilots\n• Cancel ALL pending orders\n• Send emergency notifications\n\nThis cannot be undone.')) {
      return;
    }

    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/admin/kill/global', {
        method: 'POST',
      });
      await res.json();
      toast.success('Global Kill Switch activated!');
    } catch (error: any) {
      toast.error(`Failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Mock data - replace with real API calls
  const stats = {
    totalUsers: 47,
    activeSubscribers: 32,
    activePilots: 28,
    totalExposure: 2450000,
    todayPnL: 125000,
    openPositions: 156,
  };

  return (
    <div className="space-y-6">
      {/* System Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricTile
          label="Total Users"
          value={stats.totalUsers}
          icon={<Users className="w-5 h-5" />}
        />
        <MetricTile
          label="Active Subscribers"
          value={stats.activeSubscribers}
          icon={<Activity className="w-5 h-5" />}
          trend="up"
        />
        <MetricTile
          label="Active AutoPilots"
          value={stats.activePilots}
          icon={<TrendingUp className="w-5 h-5" />}
          trend="up"
        />
      </div>

      {/* Pipeline Controls */}
      <GlassPanel variant="elevated" className="p-6">
        <h2 className="text-lg font-semibold mb-4">Pipeline Controls</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Button
            onClick={handleTriggerDailyBrain}
            disabled={loading}
            className="h-20 gap-3 bg-accent-cyan hover:bg-accent-cyan/90"
          >
            <Play className="w-5 h-5" />
            <div>
              <div className="font-semibold">Run Daily Brain</div>
              <div className="text-xs opacity-80">Generate trade intentions</div>
            </div>
          </Button>

          <Button
            onClick={handleTriggerExecutor}
            disabled={loading}
            className="h-20 gap-3 bg-accent-green hover:bg-accent-green/90"
          >
            <Play className="w-5 h-5" />
            <div>
              <div className="font-semibold">Run Executor</div>
              <div className="text-xs opacity-80">Execute pending trades</div>
            </div>
          </Button>

          <Button
            onClick={handleGlobalKillSwitch}
            disabled={loading}
            variant="destructive"
            className="h-20 gap-3 bg-accent-red hover:bg-accent-red/90"
          >
            <StopCircle className="w-5 h-5" />
            <div>
              <div className="font-semibold">Global Kill Switch</div>
              <div className="text-xs opacity-80">Emergency stop all</div>
            </div>
          </Button>
        </div>
      </GlassPanel>

      {/* System Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <GlassPanel variant="elevated" className="p-6">
          <h3 className="font-semibold mb-4">Trading Stats</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Open Positions</span>
              <span className="font-mono font-semibold">{stats.openPositions}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Total Exposure</span>
              <span className="font-mono font-semibold">
                ₹{(stats.totalExposure / 100000).toFixed(1)}L
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Today's P&L</span>
              <span className="font-mono font-semibold text-accent-green">
                +₹{(stats.todayPnL / 1000).toFixed(1)}K
              </span>
            </div>
          </div>
        </GlassPanel>

        <GlassPanel variant="elevated" className="p-6">
          <h3 className="font-semibold mb-4">System Health</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">API Status</span>
              <span className="px-2 py-1 rounded-full text-xs font-semibold bg-accent-green/20 text-accent-green">
                Online
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Database</span>
              <span className="px-2 py-1 rounded-full text-xs font-semibold bg-accent-green/20 text-accent-green">
                Healthy
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Workers</span>
              <span className="px-2 py-1 rounded-full text-xs font-semibold bg-accent-amber/20 text-accent-amber">
                Manual
              </span>
            </div>
          </div>
        </GlassPanel>
      </div>

      {/* Recent Pipeline Runs */}
      <GlassPanel variant="elevated" className="p-6">
        <h3 className="font-semibold mb-4">Recent Pipeline Runs</h3>
        <div className="text-sm text-muted-foreground">
          <p>Connect to API to see pipeline run history</p>
        </div>
      </GlassPanel>
    </div>
  );
}
