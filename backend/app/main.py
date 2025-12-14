from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import health, pipeline, signals, elite, kill, metrics

settings = get_settings()

app = FastAPI(
    title="AI Swing Trading SaaS",
    version="0.1.0",
    description="Hedge-fund style AI swing trading platform for NSE.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="", tags=["health"])
app.include_router(pipeline.router, prefix="", tags=["pipeline"])
app.include_router(signals.router, prefix="", tags=["signals"])
app.include_router(elite.router, prefix="", tags=["elite"])
app.include_router(kill.router, prefix="", tags=["controls"])
app.include_router(metrics.router, prefix="", tags=["metrics"])


@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Swing Trading SaaS backend"}
