from fastapi import APIRouter, Depends, Response

from app.core.config import Settings, get_settings
from app.services.store import signal_counts

router = APIRouter()


@router.get("/metrics")
async def metrics(settings: Settings = Depends(get_settings)):
    counts = signal_counts(settings)
    body = """
# HELP signals_total Total signals generated
# TYPE signals_total gauge
signals_total {total}
signals_buy {buy}
signals_sell {sell}
signals_hold {hold}
""".format(**counts)
    return Response(content=body.strip(), media_type="text/plain")
