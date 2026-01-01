'use client';

import { useUIStore } from '@/lib/stores/ui-store';
import { cn } from '@/lib/utils';
import {
  Home,
  Link as LinkIcon,
  Briefcase,
  Receipt,
  Shield,
  TrendingUp,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Button } from '@/components/ui/button';

interface NavItem {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  href: string;
}

const navItems: NavItem[] = [
  { icon: Home, label: 'Command Center', href: '/home' },
  { icon: LinkIcon, label: 'Broker', href: '/broker' },
  { icon: Briefcase, label: 'Portfolio', href: '/portfolio' },
  { icon: Receipt, label: 'Orders', href: '/orders' },
  { icon: Shield, label: 'Risk Desk', href: '/risk' },
  { icon: TrendingUp, label: 'Performance', href: '/performance' },
];

export function LeftNav() {
  const pathname = usePathname();
  const { leftNavCollapsed, toggleLeftNav } = useUIStore();

  return (
    <div
      className={cn(
        'border-r border-border/50 bg-background/95 backdrop-blur transition-all duration-300',
        leftNavCollapsed ? 'w-16' : 'w-56'
      )}
    >
      <div className="flex flex-col h-full py-4">
        {/* Nav Items */}
        <nav className="flex-1 px-2 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;

            return (
              <Link key={item.href} href={item.href}>
                <div
                  className={cn(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors',
                    isActive
                      ? 'bg-accent-cyan/20 text-accent-cyan'
                      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                  )}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  {!leftNavCollapsed && (
                    <span className="text-sm font-medium">{item.label}</span>
                  )}
                </div>
              </Link>
            );
          })}
        </nav>

        {/* Collapse Toggle */}
        <div className="px-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleLeftNav}
            className="w-full justify-start gap-3"
          >
            {leftNavCollapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <>
                <ChevronLeft className="w-4 h-4" />
                <span className="text-sm">Collapse</span>
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
