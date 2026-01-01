'use client';

import { useAppStore } from '@/lib/store';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Zap,
  LayoutGrid,
  Sparkles,
  Command,
  Bell,
  User,
} from 'lucide-react';

export function Header() {
  const { user, settings, toggleProMode, toggleCinematicMode } = useAppStore();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 glass-card backdrop-blur-xl">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-cyan-500">
              <Zap className="h-5 w-5 text-white" />
            </div>
            <span className="text-xl font-bold gradient-text">AI Trader</span>
          </div>

          <nav className="hidden md:flex items-center gap-1">
            <Button variant="ghost" size="sm">
              AI Radar
            </Button>
            <Button variant="ghost" size="sm">
              Scanner
            </Button>
            <Button variant="ghost" size="sm">
              Signals
            </Button>
            <Button variant="ghost" size="sm">
              Watchlist
            </Button>
            {user?.tier === 'elite' && (
              <Button variant="ghost" size="sm">
                Console
              </Button>
            )}
          </nav>
        </div>

        <div className="flex items-center gap-3">
          {/* Pro Mode Toggle */}
          {user && ['pro', 'elite'].includes(user.tier) && (
            <Button
              variant={settings.proMode ? 'default' : 'ghost'}
              size="sm"
              onClick={toggleProMode}
              className="gap-2"
            >
              <LayoutGrid className="h-4 w-4" />
              Pro
            </Button>
          )}

          {/* Cinematic Mode Toggle */}
          <Button
            variant={settings.cinematicMode ? 'default' : 'ghost'}
            size="sm"
            onClick={toggleCinematicMode}
            className="gap-2"
          >
            <Sparkles className="h-4 w-4" />
          </Button>

          {/* Command Palette */}
          <Button variant="ghost" size="sm" className="gap-2">
            <Command className="h-4 w-4" />
            <span className="hidden sm:inline text-xs text-muted-foreground">
              ⌘K
            </span>
          </Button>

          {/* Alerts */}
          <Button variant="ghost" size="icon">
            <Bell className="h-4 w-4" />
          </Button>

          {/* User */}
          <div className="flex items-center gap-3 pl-3 border-l border-border">
            {user && (
              <Badge variant={user.tier === 'elite' ? 'default' : 'secondary'}>
                {user.tier.toUpperCase()}
              </Badge>
            )}
            <Button variant="ghost" size="icon">
              <User className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
}
