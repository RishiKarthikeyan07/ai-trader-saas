'use client';

import { useEffect, useState } from 'react';
import { createClient } from '@/lib/supabase/client';
import { useQueryClient } from '@tanstack/react-query';

interface Position {
  id: string;
  canonical_symbol: string;
  qty: number;
  current_price?: number;
  unrealized_pnl?: number;
  status: string;
}

export function useRealtimePositions(userId?: string) {
  const queryClient = useQueryClient();
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    if (!userId) return;

    const supabase = createClient();

    // Subscribe to positions table changes
    const channel = supabase
      .channel('positions-changes')
      .on(
        'postgres_changes',
        {
          event: '*', // Listen to all events (INSERT, UPDATE, DELETE)
          schema: 'public',
          table: 'positions',
          filter: `user_id=eq.${userId}`,
        },
        (payload: any) => {
          console.log('Position change detected:', payload);

          // Invalidate positions query to trigger refetch
          queryClient.invalidateQueries({ queryKey: ['positions'] });
        }
      )
      .subscribe((status: string) => {
        if (status === 'SUBSCRIBED') {
          setIsConnected(true);
          console.log('✅ Realtime subscription active for positions');
        }
      });

    // Cleanup subscription on unmount
    return () => {
      channel.unsubscribe();
      setIsConnected(false);
    };
  }, [userId, queryClient]);

  return { isConnected };
}
