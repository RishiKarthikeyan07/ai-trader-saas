# AutoPilot AI Trading Bot - Institutional Grade SaaS for NSE India

> **Production-ready AutoPilot AI swing trading bot with multi-broker support**

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white)](https://supabase.com/)
[![Razorpay](https://img.shields.io/badge/Razorpay-02042B?style=flat&logo=razorpay&logoColor=white)](https://razorpay.com/)

---

## 🎯 Overview

**AutoPilot** is NOT a signals platform. It's a fully automated AI trading bot that:

- ✅ **Executes trades automatically** using AI-powered trade intentions
- ✅ **Supports 8 major NSE brokers** (Zerodha, Upstox, Angel One, Dhan, FYERS, ICICI Breeze, Kotak Neo, 5paisa)
- ✅ **Paper trading first** - Prove the bot works before risking capital
- ✅ **Institutional-grade risk management** - Max positions, daily loss limits, exposure controls
- ✅ **Black-box AI** - Model logic protected, high-level insights only
- ✅ **Single plan** - AutoPilot Monthly (₹4,999) or Yearly (₹47,990 - 20% off)

### Key Differentiator

**Users DON'T see "signals"**. They see:
- ✅ Executed trades (what the bot did)
- ✅ Open positions (what's currently held)
- ✅ Risk controls (limits & safety)
- ✅ Performance (equity curve, heatmap, stats)
- ✅ Logs (full transparency)

Internally, the system generates **trade_intentions** (from PKScreener + AI), but these are NEVER exposed to customers.

---

## 🚀 Quick Start

```bash
# 1. Clone & Install
git clone <repo-url>
cd AI_TRADER
npm install

# 2. Install shared package dependencies
cd packages/shared && npm install && cd ../..

# 3. Install web dependencies
cd apps/web && npm install && cd ../..

# 4. Install API dependencies
cd apps/api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ../..

# 5. Install worker dependencies
cd apps/worker
pip install -r requirements.txt
cd ../..

# 6. Configure environment (see .env.example in each app)
# 7. Run Supabase migrations (see Database Setup below)

# 8. Start development
npm run dev  # Runs web + api + worker
```

**Local URLs**:
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 📁 Project Structure

```
AI_TRADER/
├── apps/
│   ├── web/           # Next.js 15 (App Router) - AutoPilot UI
│   ├── api/           # FastAPI - REST API, BrokerHub
│   └── worker/        # Python - Daily brain pipeline, executor
├── packages/
│   └── shared/        # TypeScript types, schemas, utilities
├── supabase/
│   └── migrations/    # Database schema (003_autopilot_schema.sql, 004_autopilot_rls.sql)
├── notebooks/         # Jupyter notebooks for research/dataset building
└── docs/              # Documentation
```

---

## 🎨 Features

### Core Product Features

| Feature | Description |
|---------|-------------|
| **AutoPilot** | Fully automated BUY-only swing trading (SELL coming soon) |
| **Multi-Broker** | 8 broker connectors (Zerodha, Upstox, Angel One, etc.) |
| **Paper Trading** | Default mode; live trading requires explicit toggle |
| **Risk Controls** | Max positions, daily loss limits, exposure caps |
| **Kill Switch** | Per-user + global admin emergency stop |
| **Black-Box AI** | Model logic protected, high-level tags only |
| **Performance Tracking** | Daily heatmap, equity curve, win rate, avg R |

### UI WOW Features

**Opportunity Radar (3D)**
- 3D radar visualization on AutoPilot Home
- Shows Top-K "focus instruments" for today
- Blips represent bot's current watch list
- Click blip → High-level intent summary
- Toggleable via Cinematic Mode (disabled on mobile/low power)

**Performance Heatmap**
- Last 30 days performance grid
- Color-coded: green = win, red = loss
- Metrics: Win rate, Avg R, Drawdown
- Views: Monthly, weekly, by holding duration
- Trust builder - transparent performance

**Hybrid UI Modes**
1. **Clean Mode** (default): Simplified, consumer-friendly
2. **Pro Mode**: Dense panels, advanced metrics, institutional feel
3. **Cinematic Mode**: 3D graphics + heavy animations (auto-disabled on mobile)

---

## 📊 Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 15, React 19, TypeScript, Tailwind, shadcn/ui, Framer Motion, react-three-fiber |
| **Backend** | FastAPI, Python 3.11, Pydantic, asyncio |
| **Database** | Supabase (Postgres + Auth + Storage) |
| **Queue** | Upstash Redis + RQ (lightweight job queue) |
| **Payment** | Razorpay (India-only) |
| **Monitoring** | Sentry (web + API) |
| **Deploy** | Vercel (web), Render/Fly (API + worker) |

---

## 🗄️ Database Schema

### Core Tables (17 total)

| Table | Description |
|-------|-------------|
| `profiles` | User settings, AutoPilot state |
| `subscriptions` | Razorpay subscription data |
| `broker_connections` | Multi-broker auth tokens (encrypted) |
| `instruments` | Canonical NSE/BSE symbol registry |
| `broker_instruments` | Broker-specific token mappings |
| `trade_intentions` | **INTERNAL** - AI-generated trade ideas (NOT exposed) |
| `positions` | Active user trades with trailing stops |
| `orders` | Order execution history |
| `fills` | Order fill details |
| `risk_limits` | Per-user risk management settings |
| `performance_daily` | Daily performance aggregation for heatmap |
| `notifications` | System alerts |
| `audit_logs` | Security & compliance trail |
| `pipeline_runs` | Background job tracking |
| `model_versions` | AI model artifact registry |

### Row Level Security (RLS)

✅ All tables secured with RLS
✅ Users see only their own data (positions, orders, notifications)
✅ `trade_intentions` table is admin-only (NOT exposed to users)
✅ Admins can view all data
✅ Backend uses service_role key to bypass RLS for pipelines

---

## 🔌 BrokerHub (Multi-Broker Support)

### Supported Brokers

| Broker | Status | Auth Method |
|--------|--------|-------------|
| Zerodha Kite | ✅ Full impl | OAuth |
| Upstox | 🟡 Scaffold | OAuth |
| Angel One | 🟡 Scaffold | API Key + TOTP |
| Dhan | 🟡 Scaffold | API Key |
| FYERS | 🟡 Scaffold | OAuth |
| ICICI Breeze | 🟡 Scaffold | API Key |
| Kotak Neo | 🟡 Scaffold | OAuth |
| 5paisa | 🟡 Scaffold | API Key |

**Beta Requirement**: Zerodha Kite fully functional. Other connectors scaffolded with health checks.

### BrokerHub Interface

All brokers implement:

```python
connect(user_id, broker_type, auth_payload)
refresh(user_id)
place_order(user_id, order)
modify_order(user_id, order_id, changes)
cancel_order(user_id, order_id)
get_positions(user_id)
get_holdings(user_id)
get_funds(user_id)
get_orders(user_id)
```

### Instrument Registry

- **Canonical format**: NSE symbol (e.g., "RELIANCE")
- **Mapping**: canonical_symbol ↔ broker_token
- All broker responses normalized to canonical format

---

## 🤖 Pipelines

### Daily "Brain" Pipeline

**Frequency**: Once daily (7 AM IST)

**Steps**:
1. Tradability gate (NSE universe)
2. PKScreener packs (5 configured strategies)
3. Candidate shortlist (~100-200)
4. AI ranking engine (or stub scoring)
5. Generate **trade_intentions** (INTERNAL)
6. Store to database

**Output**: `trade_intentions` table (NOT exposed to users)

### Market-Hours Executor

**Frequency**: Every 15 minutes (9:15 AM - 3:30 PM IST)

**Steps**:
1. For each user with AutoPilot ON + active subscription:
   - Load today's trade_intentions
   - Apply user risk limits
   - Entry timing confirmation (1H/4H price action)
   - Calculate position size (risk % based)
   - Place BUY orders via BrokerHub
2. Manage open positions:
   - Rule-based trailing stops
   - Partial exits (TP1, TP2)
   - Optional PPO exit policy (feature-flagged)
3. Write orders/fills/positions
4. Audit log all actions
5. Pause AutoPilot if broker unhealthy or data stale

**Safety**:
- Idempotent (re-runs safe)
- Kill switch checks before every order
- Paper trading default

---

## 💳 Billing (Razorpay)

### Plans

| Plan | Price | Discount |
|------|-------|----------|
| AutoPilot Monthly | ₹4,999/month | - |
| AutoPilot Yearly | ₹47,990/year | 20% |

### Implementation

- Razorpay Subscriptions API
- Webhook signature validation (RAW body)
- Update `subscriptions` table on events
- Gate AutoPilot: requires `is_active_subscriber = true`

### Endpoints

- `POST /billing/razorpay/create-subscription`
- `POST /billing/razorpay/webhook`
- `GET /billing/status`

---

## 🎨 Pages (Next.js App Router)

| Page | Route | Description |
|------|-------|-------------|
| **Landing/Pricing** | `/` | One plan, Razorpay checkout, glass hero |
| **Auth** | `/auth/*` | Supabase auth (sign in/up) |
| **AutoPilot Home** | `/dashboard` | Command center, status, Opportunity Radar |
| **Broker Connect** | `/broker` | Multi-broker connection flow |
| **Portfolio** | `/portfolio` | Positions, P&L, exposure |
| **Order Blotter** | `/orders` | Institutional-grade order log |
| **Risk Desk** | `/risk` | Risk controls + kill switch |
| **Performance** | `/performance` | Equity curve + heatmap |
| **Settings** | `/settings` | Capital allocation, notifications |
| **Admin Panel** | `/admin` | Pipeline controls, user monitoring |

### Removed Pages (Old Signals SaaS)
❌ AI Radar (signals page)
❌ Scanner Packs
❌ Signal Detail
❌ Watchlist
❌ Alerts Center

---

## 🔐 Security & Safety

### Kill Switch

1. **Per-user kill switch**: User can pause AutoPilot anytime
2. **Global admin kill switch**: Admin can stop ALL trading immediately

### Risk Controls (Mandatory)

- Max open positions (default: 5)
- Max daily loss (default: ₹10,000 or 2% of capital)
- Max exposure (default: 80% of capital)
- Per-trade risk % (default: 1% of capital)

### Black-Box Product

- Model names NOT revealed
- Feature importance NOT shown
- Only high-level tags: "trend aligned", "volatility favorable"
- No confidence scores exposed

### Audit Logs

- All AutoPilot actions logged
- Order placement/modification/cancellation
- Risk limit breaches
- Kill switch activations
- Broker connection events

---

## 🚀 Development

### Prerequisites

- Node.js >= 20.0.0
- npm >= 10.0.0
- Python >= 3.11
- Supabase account
- Razorpay account

### Environment Setup

**apps/web/.env.local**:
```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**apps/api/.env**:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
RAZORPAY_KEY_ID=your_key_id
RAZORPAY_KEY_SECRET=your_key_secret
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret
```

**apps/worker/.env**:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
REDIS_URL=your_redis_url
```

### Database Setup

1. Create Supabase project
2. Run migrations:

```sql
-- In Supabase SQL Editor:
-- 1. Run supabase/migrations/003_autopilot_schema.sql
-- 2. Run supabase/migrations/004_autopilot_rls.sql
```

3. Create test user and set to admin:

```sql
-- After creating user via Supabase Auth UI:
UPDATE profiles
SET is_admin = true, is_active_subscriber = true
WHERE email = 'your@email.com';
```

### Run Development

```bash
# All services
npm run dev

# Individual services
npm run dev:web      # Frontend only
npm run dev:api      # Backend only
npm run dev:worker   # Worker only
```

### Trigger Pipelines Manually

```bash
# Daily brain pipeline
npm run pipeline:daily

# Executor pipeline
npm run pipeline:executor
```

---

## 📦 Deployment

### Frontend (Vercel)

```bash
cd apps/web
vercel --prod
```

### Backend (Render/Fly)

```bash
cd apps/api
docker build -t autopilot-api .
fly deploy  # or deploy to Render
```

### Worker (Render/Fly)

```bash
cd apps/worker
docker build -t autopilot-worker .
fly deploy
```

### Cron Jobs

Use Vercel Cron or GitHub Actions:

**Daily Brain Pipeline**: 7:00 AM IST daily
**Executor**: Every 15 min (9:15 AM - 3:30 PM IST) on market days

---

## 📊 Monitoring

- **Sentry**: Error tracking (web + API)
- **Supabase Logs**: Database queries
- **Uptime**: API health checks every 1 min
- **Custom Metrics**: Order success rate, AutoPilot uptime, broker health

---

## 📝 Documentation

| Document | Description |
|----------|-------------|
| [AUTOPILOT_TRANSFORMATION.md](AUTOPILOT_TRANSFORMATION.md) | Complete transformation plan |
| [GETTING_STARTED.md](GETTING_STARTED.md) | 10-minute setup guide |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment guide |
| [API.md](API.md) | API endpoints documentation |

---

## 🎯 Status

**Version**: 2.0.0-beta
**Phase**: Foundation Complete (Database + Types + Structure)

### Completed ✅
- [x] Transformation plan
- [x] Monorepo structure (web, api, worker, shared)
- [x] Complete database schema (17 tables)
- [x] RLS policies (all tables secured)
- [x] TypeScript shared types
- [x] Broker interface types
- [x] Trading types (positions, orders, risk)

### In Progress 🟡
- [ ] BrokerHub core + Zerodha connector
- [ ] Worker pipelines (daily brain, executor)
- [ ] UI pages (AutoPilot Home, Portfolio, etc.)
- [ ] Opportunity Radar 3D
- [ ] Performance Heatmap
- [ ] Razorpay integration

### Upcoming 🔲
- [ ] Remaining 7 broker connectors (scaffolds)
- [ ] End-to-end testing (paper trading)
- [ ] Production deployment
- [ ] Beta user testing
- [ ] Live trading (post-beta)

---

## 🤝 Contributing

This is a proprietary project. If you're on the team, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

Proprietary - All Rights Reserved

---

**Built by**: Founder-level Principal Engineer + Senior AI Engineer + SaaS Architect
**Built for**: Indian retail traders who demand institutional-grade automation
**Built to**: Scale from beta to 10,000+ users

🚀 **Let's ship this.**
