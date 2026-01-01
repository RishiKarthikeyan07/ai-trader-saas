/**
 * User and Subscription Types
 */

export type PlanPeriod = 'monthly' | 'yearly';

export interface Profile {
  user_id: string; // Supabase auth.users.id
  email?: string;
  full_name?: string;
  is_active_subscriber: boolean;
  autopilot_enabled: boolean;
  pro_mode: boolean; // UI toggle for dense panels
  cinematic_mode: boolean; // UI toggle for 3D/animations
  is_admin: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface Subscription {
  id?: string;
  user_id: string;
  razorpay_subscription_id: string;
  razorpay_customer_id: string;
  plan_period: PlanPeriod;
  status: 'active' | 'cancelled' | 'paused' | 'expired';
  current_period_start: string;
  current_period_end: string;
  cancel_at_period_end: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface RazorpayWebhookPayload {
  event: string;
  payload: {
    subscription?: {
      entity: {
        id: string;
        customer_id: string;
        status: string;
        current_start: number;
        current_end: number;
        charge_at: number;
      };
    };
    payment?: {
      entity: {
        id: string;
        amount: number;
        status: string;
      };
    };
  };
}
