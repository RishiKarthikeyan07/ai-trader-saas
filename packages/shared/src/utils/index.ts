// Shared utility functions
export function calculatePositionSize(
  capital: number,
  riskPercent: number,
  entryPrice: number,
  stopLoss: number
): number {
  const riskAmount = capital * (riskPercent / 100);
  const riskPerShare = Math.abs(entryPrice - stopLoss);
  return Math.floor(riskAmount / riskPerShare);
}

export function calculatePnL(
  quantity: number,
  entryPrice: number,
  exitPrice: number,
  side: 'LONG' | 'SHORT'
): number {
  if (side === 'LONG') {
    return quantity * (exitPrice - entryPrice);
  } else {
    return quantity * (entryPrice - exitPrice);
  }
}

export function calculatePnLPercent(pnl: number, capital: number): number {
  return (pnl / capital) * 100;
}

export function isMarketOpen(): boolean {
  const now = new Date();
  const istNow = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }));
  const hours = istNow.getHours();
  const minutes = istNow.getMinutes();
  const day = istNow.getDay();

  // Market closed on weekends
  if (day === 0 || day === 6) return false;

  // Market hours: 9:15 AM - 3:30 PM IST
  const currentTime = hours * 60 + minutes;
  const marketOpen = 9 * 60 + 15; // 9:15 AM
  const marketClose = 15 * 60 + 30; // 3:30 PM

  return currentTime >= marketOpen && currentTime <= marketClose;
}

export function formatCurrency(amount: number, locale = 'en-IN'): string {
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 2,
  }).format(amount);
}

export function formatPercent(value: number, decimals = 2): string {
  return `${value.toFixed(decimals)}%`;
}
