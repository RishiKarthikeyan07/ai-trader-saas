'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useUIStore } from '@/lib/stores/ui-store';
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command';
import {
  Home,
  Link as LinkIcon,
  Briefcase,
  Receipt,
  Shield,
  TrendingUp,
  Settings,
  Power,
  AlertOctagon,
} from 'lucide-react';

interface CommandItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  action: () => void;
  keywords?: string[];
}

export function CommandPalette() {
  const router = useRouter();
  const { commandPaletteOpen, closeCommandPalette } = useUIStore();
  const [search, setSearch] = useState('');

  // Register keyboard shortcut
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        useUIStore.getState().openCommandPalette();
      }
    };

    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, []);

  const commands: CommandItem[] = [
    {
      icon: Home,
      label: 'Command Center',
      action: () => router.push('/home'),
      keywords: ['home', 'dashboard'],
    },
    {
      icon: LinkIcon,
      label: 'Broker Connection',
      action: () => router.push('/broker'),
      keywords: ['broker', 'connect', 'zerodha', 'upstox'],
    },
    {
      icon: Briefcase,
      label: 'Portfolio',
      action: () => router.push('/portfolio'),
      keywords: ['portfolio', 'positions', 'holdings'],
    },
    {
      icon: Receipt,
      label: 'Order Blotter',
      action: () => router.push('/orders'),
      keywords: ['orders', 'trades', 'executions'],
    },
    {
      icon: Shield,
      label: 'Risk Desk',
      action: () => router.push('/risk'),
      keywords: ['risk', 'limits', 'exposure'],
    },
    {
      icon: TrendingUp,
      label: 'Performance',
      action: () => router.push('/performance'),
      keywords: ['performance', 'pnl', 'returns'],
    },
    {
      icon: AlertOctagon,
      label: 'Activate Kill Switch',
      action: () => {
        // TODO: Implement kill switch activation
        console.log('Kill switch activated');
      },
      keywords: ['kill', 'emergency', 'stop'],
    },
    {
      icon: Power,
      label: 'Toggle AutoPilot',
      action: () => {
        // TODO: Implement autopilot toggle
        console.log('AutoPilot toggled');
      },
      keywords: ['autopilot', 'toggle', 'pause', 'resume'],
    },
  ];

  const handleSelect = (action: () => void) => {
    action();
    closeCommandPalette();
    setSearch('');
  };

  return (
    <CommandDialog open={commandPaletteOpen} onOpenChange={closeCommandPalette}>
      <CommandInput
        placeholder="Search commands..."
        value={search}
        onValueChange={setSearch}
      />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        <CommandGroup heading="Navigation">
          {commands
            .filter((cmd) => cmd.keywords?.includes('home') || cmd.keywords?.includes('broker') || cmd.keywords?.includes('portfolio') || cmd.keywords?.includes('orders') || cmd.keywords?.includes('risk') || cmd.keywords?.includes('performance'))
            .map((cmd) => {
              const Icon = cmd.icon;
              return (
                <CommandItem
                  key={cmd.label}
                  onSelect={() => handleSelect(cmd.action)}
                >
                  <Icon className="mr-2 h-4 w-4" />
                  <span>{cmd.label}</span>
                </CommandItem>
              );
            })}
        </CommandGroup>
        <CommandGroup heading="Actions">
          {commands
            .filter((cmd) => cmd.keywords?.includes('kill') || cmd.keywords?.includes('autopilot'))
            .map((cmd) => {
              const Icon = cmd.icon;
              return (
                <CommandItem
                  key={cmd.label}
                  onSelect={() => handleSelect(cmd.action)}
                >
                  <Icon className="mr-2 h-4 w-4" />
                  <span>{cmd.label}</span>
                </CommandItem>
              );
            })}
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
