import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'AI Swing Trading SaaS',
  description: 'Hedge-fund style AI swing signals for NSE with institutional SMC.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
