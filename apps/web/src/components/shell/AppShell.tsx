'use client';

import { cn } from '@/lib/utils';
import { useUIStore } from '@/lib/stores/ui-store';
import { TopBar } from './TopBar';
import { LeftNav } from './LeftNav';
import { RightRail } from './RightRail';
import { CommandPalette } from './CommandPalette';

export function AppShell({ children }: { children: React.ReactNode }) {
  const { proMode, cinematicMode } = useUIStore();

  return (
    <div
      className={cn(
        'h-screen flex flex-col overflow-hidden',
        proMode && 'pro-mode',
        cinematicMode && 'cinematic-mode'
      )}
    >
      {/* Top Bar */}
      <TopBar />

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Navigation */}
        <LeftNav />

        {/* Main Content */}
        <main className="flex-1 overflow-auto custom-scrollbar bg-graphite-950">
          <div className="container mx-auto p-6">{children}</div>
        </main>

        {/* Right Rail (Pro Mode Only) */}
        <RightRail />
      </div>

      {/* Command Palette */}
      <CommandPalette />
    </div>
  );
}
