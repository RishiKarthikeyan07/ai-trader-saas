'use client';

import { StatusChip, type AutoPilotStatus } from '@/components/design';
import { useUIStore } from '@/lib/stores/ui-store';
import { Bell, Command, Maximize2, Minimize2, Monitor, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export function TopBar() {
  const {
    proMode,
    cinematicMode,
    toggleProMode,
    toggleCinematicMode,
    openCommandPalette,
  } = useUIStore();

  // Mock data - will be replaced with real data from API
  const autopilotStatus: AutoPilotStatus = 'ON';

  return (
    <div className="h-14 border-b border-border/50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex h-full items-center justify-between px-6">
        {/* Left: Logo + Status */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-accent-cyan" />
            <span className="font-semibold text-lg">AutoPilot</span>
          </div>
          <StatusChip status={autopilotStatus} />
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-3">
          {/* Command Palette */}
          <Button
            variant="ghost"
            size="sm"
            onClick={openCommandPalette}
            className="gap-2"
          >
            <Command className="w-4 h-4" />
            <span className="text-xs text-muted-foreground">⌘K</span>
          </Button>

          {/* Pro Mode Toggle */}
          <Button
            variant={proMode ? 'default' : 'ghost'}
            size="sm"
            onClick={toggleProMode}
            className={cn('gap-2', proMode && 'bg-accent-cyan/20 text-accent-cyan')}
          >
            <Monitor className="w-4 h-4" />
            <span className="text-xs">Pro</span>
          </Button>

          {/* Cinematic Mode Toggle */}
          <Button
            variant={cinematicMode ? 'default' : 'ghost'}
            size="sm"
            onClick={toggleCinematicMode}
            className={cn(
              'gap-2',
              cinematicMode && 'bg-accent-cyan/20 text-accent-cyan'
            )}
          >
            {cinematicMode ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
            <span className="text-xs">Cinematic</span>
          </Button>

          {/* Notifications */}
          <Button variant="ghost" size="icon" className="relative">
            <Bell className="w-4 h-4" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-accent-red rounded-full" />
          </Button>

          {/* User Avatar */}
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-accent-cyan to-purple-500 flex items-center justify-center text-xs font-semibold">
            U
          </div>
        </div>
      </div>
    </div>
  );
}
