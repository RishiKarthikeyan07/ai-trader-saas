import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 2,
  }).format(value);
}

export function formatPercent(value: number): string {
  return new Intl.NumberFormat('en-IN', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value / 100);
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-IN').format(value);
}

export function formatDate(date: string | Date): string {
  return new Intl.DateTimeFormat('en-IN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  }).format(new Date(date));
}

export function formatDateTime(date: string | Date): string {
  return new Intl.DateTimeFormat('en-IN', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(date));
}

export function getRiskColor(risk: string): string {
  switch (risk) {
    case 'low':
      return 'text-emerald-400';
    case 'medium':
      return 'text-amber-400';
    case 'high':
      return 'text-rose-400';
    default:
      return 'text-slate-400';
  }
}

export function getStatusColor(status: string): string {
  switch (status) {
    case 'ready':
      return 'text-emerald-400';
    case 'wait':
      return 'text-amber-400';
    case 'filled':
      return 'text-blue-400';
    case 'tp1_hit':
    case 'tp2_hit':
      return 'text-emerald-400';
    case 'sl_hit':
      return 'text-rose-400';
    case 'exited':
    case 'expired':
      return 'text-slate-400';
    default:
      return 'text-slate-400';
  }
}

export function calculateRiskReward(
  entry: number,
  stopLoss: number,
  target: number
): number {
  const risk = Math.abs(entry - stopLoss);
  const reward = Math.abs(target - entry);
  return reward / risk;
}
