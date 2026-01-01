'use client';

import { GlassPanel } from '@/components/design';
import { Button } from '@/components/ui/button';
import { Check, Link as LinkIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

const BROKERS = [
  { id: 'zerodha', name: 'Zerodha Kite', logo: '🔷', status: 'ready', popular: true },
  { id: 'upstox', name: 'Upstox', logo: '🟣', status: 'coming_soon', popular: true },
  { id: 'angel_one', name: 'Angel One', logo: '🔴', status: 'coming_soon', popular: false },
  { id: 'dhan', name: 'Dhan', logo: '🟢', status: 'coming_soon', popular: false },
  { id: 'fyers', name: 'FYERS', logo: '🟡', status: 'coming_soon', popular: false },
  { id: 'icici_breeze', name: 'ICICI Breeze', logo: '🔶', status: 'coming_soon', popular: false },
  { id: 'kotak_neo', name: 'Kotak Neo', logo: '🔵', status: 'coming_soon', popular: false },
  { id: 'fivepaisa', name: '5Paisa', logo: '⚫', status: 'coming_soon', popular: false },
];

export default function BrokerPage() {
  const connectedBroker = null; // Will be fetched from API

  const handleConnect = (brokerId: string) => {
    // TODO: Implement OAuth flow for broker connection
    console.log('Connecting to:', brokerId);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Broker Connection</h1>
        <p className="text-muted-foreground mt-2">
          Connect your trading account to enable AutoPilot execution
        </p>
      </div>

      {/* Current Connection Status */}
      {connectedBroker && (
        <GlassPanel variant="elevated" className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-lg bg-accent-green/20 flex items-center justify-center">
                <Check className="w-6 h-6 text-accent-green" />
              </div>
              <div>
                <h3 className="font-semibold text-lg">Connected to Zerodha Kite</h3>
                <p className="text-sm text-muted-foreground">Active since Jan 15, 2025</p>
              </div>
            </div>
            <Button variant="outline" size="sm">
              Disconnect
            </Button>
          </div>
        </GlassPanel>
      )}

      {/* Available Brokers */}
      <div>
        <h2 className="text-xl font-semibold mb-4">
          {connectedBroker ? 'Switch Broker' : 'Select Your Broker'}
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {BROKERS.map((broker) => (
            <GlassPanel
              key={broker.id}
              className={cn(
                'p-6 cursor-pointer hover:border-accent-cyan/50 transition-all',
                broker.status === 'coming_soon' && 'opacity-60'
              )}
              onClick={() =>
                broker.status === 'ready' && handleConnect(broker.id)
              }
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-4">
                  <div className="text-4xl">{broker.logo}</div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold text-lg">{broker.name}</h3>
                      {broker.popular && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30">
                          Popular
                        </span>
                      )}
                    </div>
                    {broker.status === 'ready' ? (
                      <p className="text-sm text-accent-green mt-1">
                        ✓ Available now
                      </p>
                    ) : (
                      <p className="text-sm text-muted-foreground mt-1">
                        Coming soon
                      </p>
                    )}
                  </div>
                </div>
                {broker.status === 'ready' && (
                  <Button size="sm" className="gap-2">
                    <LinkIcon className="w-4 h-4" />
                    Connect
                  </Button>
                )}
              </div>
            </GlassPanel>
          ))}
        </div>
      </div>

      {/* Help Section */}
      <GlassPanel variant="elevated" className="p-6">
        <h3 className="font-semibold mb-3">How Broker Connection Works</h3>
        <ol className="space-y-2 text-sm text-muted-foreground list-decimal list-inside">
          <li>Click "Connect" on your broker</li>
          <li>You'll be redirected to your broker's login page (OAuth flow)</li>
          <li>Grant AutoPilot permission to place orders on your behalf</li>
          <li>Your credentials are encrypted and stored securely</li>
          <li>AutoPilot can now execute trades automatically</li>
        </ol>
        <div className="mt-4 p-3 glass-panel rounded-lg text-xs">
          <strong>Security Note:</strong> We use bank-grade encryption (AES-256) to
          store your broker credentials. We never see your password.
        </div>
      </GlassPanel>
    </div>
  );
}
