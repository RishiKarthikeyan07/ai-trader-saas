# AutoPilot Trading Bot - Route Map

## ✅ Live Application Routes

**Base URL**: http://localhost:3000

---

## 📱 User Routes (AutoPilot Features)

### 1. Home / Dashboard
**URL**: http://localhost:3000/home
**File**: `src/app/(autopilot)/home/page.tsx`
**Features**:
- AutoPilot ON/OFF toggle
- Live positions grid
- Daily P&L metrics
- 3D OpportunityRadar visualization
- Quick stats (open positions, win rate, exposure)

### 2. Broker Connections
**URL**: http://localhost:3000/broker
**File**: `src/app/(autopilot)/broker/page.tsx`
**Features**:
- Connect to 7 Indian brokers
- OAuth flow for each broker
- Connection status
- Token management

### 3. Portfolio
**URL**: http://localhost:3000/portfolio
**File**: `src/app/(autopilot)/portfolio/page.tsx`
**Features**:
- Live positions table
- Entry price, current price
- Unrealized P&L
- Real-time updates via Supabase

### 4. Orders
**URL**: http://localhost:3000/orders
**File**: `src/app/(autopilot)/orders/page.tsx`
**Features**:
- Order history
- Order status
- Fill details
- Timestamps

### 5. Risk Management
**URL**: http://localhost:3000/risk
**File**: `src/app/(autopilot)/risk/page.tsx`
**Features**:
- Max positions limit
- Max drawdown %
- Daily loss limit
- Portfolio heat settings

### 6. Performance Analytics
**URL**: http://localhost:3000/performance
**File**: `src/app/(autopilot)/performance/page.tsx`
**Features**:
- Daily P&L chart
- Win rate %
- Sharpe ratio
- Max drawdown
- Monthly returns

---

## 🔐 Admin Routes

### 1. Admin Dashboard
**URL**: http://localhost:3000/admin/dashboard
**File**: `src/app/admin/dashboard/page.tsx`
**Features**:
- **Trigger Daily Brain** button (runs PKScreener scan)
- **Trigger Executor** button (executes trades)
- **Global Kill Switch** (emergency stop with confirmation)
- System metrics
- Pipeline status

### 2. User Management
**URL**: http://localhost:3000/admin/users
**File**: `src/app/admin/users/page.tsx`
**Features**:
- All users table
- Subscription status (Basic/Pro/Elite)
- AutoPilot status (ON/OFF)
- Connected brokers
- Search/filter

### 3. Trade Intentions (ADMIN-ONLY)
**URL**: http://localhost:3000/admin/intentions
**File**: `src/app/admin/intentions/page.tsx`
**⚠️ SECURITY WARNING**: This page shows internal AI-generated signals
**Features**:
- AI brain output (raw intentions)
- Risk grades (A-F)
- Confidence scores
- Target/stop prices
- **NEVER exposed to users**

---

## 🌐 API Routes

### Backend API
**Base URL**: http://localhost:8000
**API Docs**: http://localhost:8000/docs

#### Health Check
```bash
GET http://localhost:8000/health
```

#### Root
```bash
GET http://localhost:8000/
```

---

## 📂 Route Group Structure

Next.js 15 uses route groups to organize routes without affecting the URL path:

```
src/app/
├── page.tsx                    → / (redirects to /home)
├── layout.tsx                  → Root layout
│
├── (autopilot)/                ← Route group (NOT in URL)
│   ├── layout.tsx              → Shared layout for user pages
│   ├── home/page.tsx           → /home
│   ├── broker/page.tsx         → /broker
│   ├── portfolio/page.tsx      → /portfolio
│   ├── orders/page.tsx         → /orders
│   ├── risk/page.tsx           → /risk
│   └── performance/page.tsx    → /performance
│
└── admin/                      ← Regular route (in URL)
    ├── layout.tsx              → Admin layout
    ├── dashboard/page.tsx      → /admin/dashboard
    ├── users/page.tsx          → /admin/users
    └── intentions/page.tsx     → /admin/intentions
```

**Key Point**: Route groups like `(autopilot)` are used for organization and shared layouts, but **do NOT appear in the URL path**.

---

## 🔑 Keyboard Shortcuts

- **Cmd+K** (Mac) / **Ctrl+K** (Windows): Open command palette
- **Esc**: Close command palette/modals

---

## 🧪 Testing Routes

### Quick Test (from terminal)
```bash
# Root (should redirect to /home)
curl -I http://localhost:3000/

# Home page
curl -I http://localhost:3000/home

# Admin dashboard
curl -I http://localhost:3000/admin/dashboard

# Backend health
curl http://localhost:8000/health
```

### Browser Test
1. Open http://localhost:3000 (should redirect to /home)
2. Navigate through sidebar links
3. Test command palette (Cmd+K)
4. Access admin pages

---

## 🚀 Navigation Flow

```
User lands on:
http://localhost:3000/
       ↓
Redirects to:
http://localhost:3000/home
       ↓
Shows:
- AutoPilot toggle
- Live positions
- 3D radar
- Metrics
```

---

## 📝 Notes

1. **Route Groups** `(autopilot)`:
   - Used for shared layouts
   - NOT part of URL path
   - All user pages share the AutoPilot layout

2. **Admin Routes**:
   - Separate `/admin/*` path
   - Different layout (no sidebar)
   - Admin-specific features

3. **API Proxy**:
   - Frontend API calls go to `/api/backend/*`
   - Proxied to backend at `http://localhost:8000/*`
   - Configured in `next.config.js`

---

**Last Updated**: December 27, 2025
**Next.js Version**: 15.5.9
**Status**: ✅ All routes working
