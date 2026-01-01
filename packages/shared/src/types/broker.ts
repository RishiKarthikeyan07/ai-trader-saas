/**
 * Broker Types for Multi-Broker Support
 * Unified interface for all NSE brokers
 */

export type BrokerType =
  | 'zerodha'
  | 'upstox'
  | 'angel_one'
  | 'dhan'
  | 'fyers'
  | 'icici_breeze'
  | 'kotak_neo'
  | 'fivepaisa';

export type BrokerStatus =
  | 'connected'
  | 'disconnected'
  | 'error'
  | 'refreshing';

export type OrderSide = 'BUY' | 'SELL';
export type OrderType = 'MARKET' | 'LIMIT' | 'SL' | 'SL-M';
export type ProductType = 'MIS' | 'CNC' | 'NRML';

export type OrderStatus =
  | 'PENDING'
  | 'OPEN'
  | 'COMPLETE'
  | 'CANCELLED'
  | 'REJECTED'
  | 'TRIGGER_PENDING';

export interface BrokerConnection {
  user_id: string;
  broker_type: BrokerType;
  status: BrokerStatus;
  encrypted_tokens: string; // JSON encrypted
  last_refresh: string;
  created_at: string;
  updated_at?: string;
}

export interface BrokerAuthPayload {
  broker_type: BrokerType;
  // Zerodha
  request_token?: string;
  api_key?: string;
  api_secret?: string;
  // Upstox
  access_token?: string;
  // Angel One
  client_id?: string;
  password?: string;
  totp_secret?: string;
  // Generic API key based
  auth_token?: string;
}

export interface BrokerOrder {
  symbol: string; // Canonical NSE symbol
  exchange: 'NSE' | 'BSE';
  side: OrderSide;
  order_type: OrderType;
  product: ProductType;
  quantity: number;
  price?: number; // For LIMIT orders
  trigger_price?: number; // For SL orders
  disclosed_quantity?: number;
  validity?: 'DAY' | 'IOC';
  tag?: string; // For tracking AutoPilot orders
}

export interface BrokerOrderResponse {
  broker_order_id: string;
  status: OrderStatus;
  message?: string;
  timestamp: string;
}

export interface BrokerPosition {
  symbol: string;
  exchange: 'NSE' | 'BSE';
  product: ProductType;
  quantity: number;
  average_price: number;
  last_price: number;
  pnl: number;
  pnl_percent: number;
}

export interface BrokerHolding {
  symbol: string;
  exchange: 'NSE' | 'BSE';
  quantity: number;
  average_price: number;
  last_price: number;
  pnl: number;
}

export interface BrokerFunds {
  available_cash: number;
  used_margin: number;
  available_margin: number;
  total_collateral: number;
}

export interface BrokerOrderUpdate {
  broker_order_id: string;
  status: OrderStatus;
  filled_quantity: number;
  average_price?: number;
  timestamp: string;
  message?: string;
}

/**
 * Instrument Registry
 * Maps canonical symbols to broker-specific tokens
 */
export interface Instrument {
  canonical_symbol: string; // e.g., "RELIANCE"
  exchange: 'NSE' | 'BSE';
  name: string;
  isin?: string;
  lot_size: number;
  tick_size: number;
}

export interface BrokerInstrument {
  broker_type: BrokerType;
  canonical_symbol: string;
  broker_token: string; // Broker-specific instrument token
  broker_symbol: string; // Broker-specific symbol format
  extra?: Record<string, any>; // Broker-specific metadata
}

/**
 * BrokerHub Interface
 * All broker connectors must implement this
 */
export interface IBrokerConnector {
  // Connection Management
  connect(auth_payload: BrokerAuthPayload): Promise<BrokerConnection>;
  disconnect(user_id: string): Promise<void>;
  refresh(user_id: string): Promise<BrokerConnection>;
  healthCheck(user_id: string): Promise<boolean>;

  // Order Management
  placeOrder(user_id: string, order: BrokerOrder): Promise<BrokerOrderResponse>;
  modifyOrder(
    user_id: string,
    broker_order_id: string,
    changes: Partial<BrokerOrder>
  ): Promise<BrokerOrderResponse>;
  cancelOrder(user_id: string, broker_order_id: string): Promise<void>;
  getOrders(user_id: string): Promise<BrokerOrderUpdate[]>;

  // Portfolio Management
  getPositions(user_id: string): Promise<BrokerPosition[]>;
  getHoldings(user_id: string): Promise<BrokerHolding[]>;
  getFunds(user_id: string): Promise<BrokerFunds>;

  // Market Data (optional, for confirmation logic)
  getQuote?(symbol: string): Promise<{
    ltp: number;
    open: number;
    high: number;
    low: number;
    volume: number;
  }>;

  // Streaming (optional, for real-time updates)
  subscribeOrderUpdates?(
    user_id: string,
    callback: (update: BrokerOrderUpdate) => void
  ): void;
}
