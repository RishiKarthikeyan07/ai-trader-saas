from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
# New AutoPilot routes
from app.api.routes import (
    autopilot,
    broker,
    portfolio,
    orders,
    risk,
    razorpay,
    admin_new as admin,
    user,
    performance,
)
from datetime import datetime

app = FastAPI(
    title="AutoPilot Trading Bot API",
    version="2.0.0-beta",
    description="Institutional-grade AutoPilot AI trading bot for NSE India",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://autopilot-trading.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "AutoPilot Trading Bot API",
        "version": "2.0.0-beta",
        "description": "AI-powered automated trading for NSE India",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "2.0.0-beta",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT,
    }


# Include AutoPilot routers
app.include_router(autopilot.router)  # /autopilot
app.include_router(broker.router)  # /broker
app.include_router(portfolio.router)  # /portfolio
app.include_router(orders.router)  # /orders
app.include_router(risk.router)  # /risk
app.include_router(razorpay.router)  # /billing/razorpay
app.include_router(admin.router)  # /admin
app.include_router(user.router)  # /me
app.include_router(performance.router)  # /performance


# Initialize Sentry if configured
if settings.SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        environment=settings.ENVIRONMENT,
        traces_sample_rate=1.0 if settings.ENVIRONMENT == "development" else 0.1,
    )
