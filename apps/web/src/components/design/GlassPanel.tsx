import { cn } from '@/lib/utils';
import { HTMLAttributes, forwardRef } from 'react';

export interface GlassPanelProps extends HTMLAttributes<HTMLDivElement> {
  variant?: 'default' | 'elevated';
}

export const GlassPanel = forwardRef<HTMLDivElement, GlassPanelProps>(
  ({ className, variant = 'default', ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          variant === 'elevated' ? 'glass-card' : 'glass-panel',
          className
        )}
        {...props}
      />
    );
  }
);

GlassPanel.displayName = 'GlassPanel';
