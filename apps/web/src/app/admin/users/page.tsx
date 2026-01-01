'use client';

import { GlassPanel } from '@/components/design';
import { Search, UserCheck, UserX, Shield } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

// Mock data - replace with API
const mockUsers = [
  {
    id: '1',
    email: 'user1@example.com',
    created_at: '2025-01-15',
    is_active_subscriber: true,
    autopilot_enabled: true,
    is_admin: false,
    broker: 'zerodha',
    open_positions: 3,
  },
  {
    id: '2',
    email: 'user2@example.com',
    created_at: '2025-01-10',
    is_active_subscriber: true,
    autopilot_enabled: false,
    is_admin: false,
    broker: 'upstox',
    open_positions: 0,
  },
  {
    id: '3',
    email: 'admin@example.com',
    created_at: '2025-01-01',
    is_active_subscriber: true,
    autopilot_enabled: true,
    is_admin: true,
    broker: 'zerodha',
    open_positions: 5,
  },
];

export default function AdminUsersPage() {
  const [searchTerm, setSearchTerm] = useState('');

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">User Management</h1>
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search users..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 glass-panel rounded-lg w-64"
            />
          </div>
        </div>
      </div>

      <GlassPanel variant="elevated" className="p-6">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-left border-b border-border/50">
                <th className="pb-3 font-medium text-sm text-muted-foreground">Email</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Joined</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Subscription</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">AutoPilot</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Broker</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Positions</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Role</th>
                <th className="pb-3 font-medium text-sm text-muted-foreground">Actions</th>
              </tr>
            </thead>
            <tbody>
              {mockUsers
                .filter((user) =>
                  user.email.toLowerCase().includes(searchTerm.toLowerCase())
                )
                .map((user) => (
                  <tr
                    key={user.id}
                    className="border-b border-border/30 hover:bg-glass-hover transition-colors"
                  >
                    <td className="py-4 font-medium">{user.email}</td>
                    <td className="py-4 text-sm text-muted-foreground">{user.created_at}</td>
                    <td className="py-4">
                      {user.is_active_subscriber ? (
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold bg-accent-green/20 text-accent-green">
                          <UserCheck className="w-3 h-3" />
                          Active
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold bg-accent-red/20 text-accent-red">
                          <UserX className="w-3 h-3" />
                          Inactive
                        </span>
                      )}
                    </td>
                    <td className="py-4">
                      {user.autopilot_enabled ? (
                        <span className="text-accent-cyan text-sm font-semibold">ON</span>
                      ) : (
                        <span className="text-muted-foreground text-sm">OFF</span>
                      )}
                    </td>
                    <td className="py-4 text-sm capitalize">{user.broker || '-'}</td>
                    <td className="py-4 font-mono text-sm">{user.open_positions}</td>
                    <td className="py-4">
                      {user.is_admin && (
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-semibold bg-accent-red/20 text-accent-red">
                          <Shield className="w-3 h-3" />
                          Admin
                        </span>
                      )}
                    </td>
                    <td className="py-4">
                      <Button variant="ghost" size="sm">
                        View Details
                      </Button>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </GlassPanel>
    </div>
  );
}
