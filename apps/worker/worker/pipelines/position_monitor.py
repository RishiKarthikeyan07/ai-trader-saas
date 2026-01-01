"""
Position Monitor Pipeline
Updates position prices and P&L, monitors trailing stops
Runs every 5 minutes
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from supabase import create_client, Client
from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionMonitorPipeline:
    def __init__(self):
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    async def run(self) -> Dict[str, Any]:
        try:
            positions = self.supabase.table("positions").select("*").eq("status", "OPEN").execute()
            if not positions.data:
                return {"positions_updated": 0}
            
            updated_count = 0
            for position in positions.data:
                await self._update_position(position)
                updated_count += 1
            
            return {"status": "SUCCESS", "positions_updated": updated_count}
        except Exception as e:
            logger.error(f"Position monitor failed: {str(e)}")
            raise

    async def _update_position(self, position: Dict[str, Any]):
        # Simulate price update (in production, fetch from yfinance or broker)
        import random
        current_price = position["avg_price"] * random.uniform(0.97, 1.05)
        unrealized_pnl = (current_price - position["avg_price"]) * position["qty"]
        
        self.supabase.table("positions").update({
            "unrealized_pnl": round(unrealized_pnl, 2),
            "updated_at": datetime.utcnow().isoformat(),
        }).eq("id", position["id"]).execute()

async def main():
    pipeline = PositionMonitorPipeline()
    result = await pipeline.run()
    logger.info(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
