-- AI_TRADER Foundation Schema
-- Tier-gated signals platform with auth, profiles, orders, and positions
-- Run this in Supabase SQL Editor

-- Enable useful extensions
create extension if not exists pgcrypto;

-- 1) Tiers enum
do $$ begin
  create type public.user_tier as enum ('basic', 'pro', 'elite');
exception
  when duplicate_object then null;
end $$;

-- 2) Profiles table (one row per auth user)
create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email text,
  tier public.user_tier not null default 'basic',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  -- Elite / risk settings (safe defaults)
  auto_trade_enabled boolean not null default false,
  risk_per_trade numeric not null default 0.01, -- 1%
  max_open_positions int not null default 5,
  max_daily_loss numeric not null default 0.03  -- 3%
);

create index if not exists profiles_tier_idx on public.profiles(tier);

-- 3) Daily signals table (generated centrally)
-- rank_position is what enforces Basic/Pro limits
create table if not exists public.signals_daily (
  signal_id uuid primary key default gen_random_uuid(),
  trade_date date not null,
  symbol text not null,

  signal_type text not null check (signal_type in ('LONG','SHORT','NO_TRADE')),
  status text not null default 'WAIT'
    check (status in ('NEW','WAIT','READY','INVALIDATED','EXPIRED','ENTERED','CLOSED')),

  entry_zone_low numeric,
  entry_zone_high numeric,
  recommended_limit numeric,
  stop_loss numeric,
  target_1 numeric,
  target_2 numeric,

  confidence numeric not null default 0.0,
  rank_score numeric not null default 0.0,
  rank_position int not null,

  -- explanations / metadata
  smc_score numeric,
  tf_alignment jsonb,
  smc_flags jsonb,
  model_versions jsonb,

  created_at timestamptz not null default now(),

  -- Unique per day per symbol
  unique(trade_date, symbol)
);

create index if not exists signals_daily_date_rank_idx
  on public.signals_daily(trade_date, rank_position);

create index if not exists signals_daily_symbol_date_idx
  on public.signals_daily(symbol, trade_date);

-- 4) Orders & positions (for Elite auto + paper trading)
create table if not exists public.orders (
  order_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(user_id) on delete cascade,
  signal_id uuid references public.signals_daily(signal_id) on delete set null,

  symbol text not null,
  side text not null check (side in ('BUY','SELL')),
  order_type text not null check (order_type in ('LIMIT','MARKET')),
  tif text not null default 'DAY' check (tif in ('DAY','GTC')),
  limit_price numeric,
  qty int not null,
  status text not null default 'NEW'
    check (status in ('NEW','PLACED','FILLED','CANCELLED','REJECTED','EXPIRED')),

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists orders_user_time_idx on public.orders(user_id, created_at desc);

create table if not exists public.positions (
  position_id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(user_id) on delete cascade,
  symbol text not null,

  entry_price numeric not null,
  qty int not null,
  stop_loss numeric,
  target_1 numeric,
  target_2 numeric,

  status text not null default 'OPEN' check (status in ('OPEN','CLOSED')),
  opened_at timestamptz not null default now(),
  closed_at timestamptz
);

create index if not exists positions_user_status_idx on public.positions(user_id, status);

-- 5) Audit logs (important for safety)
create table if not exists public.audit_logs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid,
  action text not null,
  details jsonb,
  created_at timestamptz not null default now()
);

-- ─────────────────────────────────────────────────────────────
-- RLS + tier gating
-- ─────────────────────────────────────────────────────────────
alter table public.profiles enable row level security;
alter table public.signals_daily enable row level security;
alter table public.orders enable row level security;
alter table public.positions enable row level security;
alter table public.audit_logs enable row level security;

-- Helper: is_admin (optional for your admin panel later)
-- This assumes you will set JWT custom claims for admin users.
create or replace function public.is_admin()
returns boolean
language sql
stable
as $$
  select coalesce((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin', false);
$$;

-- Helper: tier limit (DB is the source of truth)
create or replace function public.current_tier_limit()
returns int
language plpgsql
stable
security definer
as $$
declare
  t public.user_tier;
begin
  select tier into t from public.profiles where user_id = auth.uid();

  if t = 'basic' then
    return 3;
  elsif t = 'pro' then
    return 10;
  elsif t = 'elite' then
    return 1000000;
  else
    return 0;
  end if;
end;
$$;

revoke all on function public.current_tier_limit() from public;
grant execute on function public.current_tier_limit() to authenticated;

-- Profiles policies
create policy "profiles_select_own"
on public.profiles
for select
to authenticated
using (user_id = auth.uid() or public.is_admin());

create policy "profiles_update_own"
on public.profiles
for update
to authenticated
using (user_id = auth.uid() or public.is_admin())
with check (user_id = auth.uid() or public.is_admin());

-- Signals policies: tier gating by rank_position
-- (Admin can see all.)
create policy "signals_select_by_tier"
on public.signals_daily
for select
to authenticated
using (
  public.is_admin()
  or rank_position <= public.current_tier_limit()
);

-- Orders policies: user can only see/modify their own
create policy "orders_select_own"
on public.orders
for select
to authenticated
using (user_id = auth.uid() or public.is_admin());

create policy "orders_insert_own"
on public.orders
for insert
to authenticated
with check (user_id = auth.uid() or public.is_admin());

create policy "orders_update_own"
on public.orders
for update
to authenticated
using (user_id = auth.uid() or public.is_admin())
with check (user_id = auth.uid() or public.is_admin());

-- Positions policies
create policy "positions_select_own"
on public.positions
for select
to authenticated
using (user_id = auth.uid() or public.is_admin());

create policy "positions_insert_own"
on public.positions
for insert
to authenticated
with check (user_id = auth.uid() or public.is_admin());

create policy "positions_update_own"
on public.positions
for update
to authenticated
using (user_id = auth.uid() or public.is_admin())
with check (user_id = auth.uid() or public.is_admin());

-- Audit logs policies (users can see their own; admin sees all)
create policy "audit_select_own"
on public.audit_logs
for select
to authenticated
using (public.is_admin() or user_id = auth.uid());

create policy "audit_insert_admin_or_system"
on public.audit_logs
for insert
to authenticated
with check (public.is_admin() or user_id = auth.uid());

-- Trigger: create profile row on signup
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
as $$
begin
  insert into public.profiles (user_id, email)
  values (new.id, new.email)
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;

create trigger on_auth_user_created
after insert on auth.users
for each row execute procedure public.handle_new_user();
