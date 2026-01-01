'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { useState } from 'react';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      <Toaster
        position="top-right"
        toastOptions={{
          className: 'glass-card',
          style: {
            background: 'rgba(20, 22, 28, 0.9)',
            color: '#fff',
            border: '1px solid rgba(255, 255, 255, 0.08)',
          },
        }}
      />
    </QueryClientProvider>
  );
}
