"""
Executor Pipeline
Executes trade entries/exits based on trade_intentions
Runs every 15 minutes during market hours
"""
import asyncio
import logging
from datetime import datetime, date
from typing import List, Dict, Any
from supabase import create_client, Client
from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutorPipeline:
    def __init__(self):
        self.supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)

    async def run(self) -> Dict[str, Any]:
        run_id = None
        try:
            run = self.supabase.table("pipeline_runs").insert({"type": "executor", "status": "RUNNING", "started_at": datetime.utcnow().isoformat()}).execute()
            run_id = run.data[0]["id"]
            
            users = await self._get_eligible_users()
            orders_created = 0
            
            for user in users:
                result = await self._process_user(user)
                orders_created += result["orders_created"]
            
            self.supabase.table("pipeline_runs").update({"status": "SUCCESS", "ended_at": datetime.utcnow().isoformat(), "metadata": {"users_processed": len(users), "orders_created": orders_created}}).eq("id", run_id).execute()
            
            return {"run_id": run_id, "status": "SUCCESS", "orders_created": orders_created}
        except Exception as e:
            logger.error(f"Executor failed: {str(e)}")
            if run_id:
                self.supabase.table("pipeline_runs").update({"status": "FAILED", "ended_at": datetime.utcnow().isoformat(), "error_message": str(e)}).eq("id", run_id).execute()
            raise

    async def _get_eligible_users(self) -> List[Dict[str, Any]]:
        result = self.supabase.table("profiles").select("user_id").eq("autopilot_enabled", True).eq("is_active_subscriber", True).execute()
        return result.data if result.data else []

    async def _process_user(self, user: Dict[str, Any]) -> Dict[str, Any]:
        user_id = user["user_id"]
        if not await self._check_risk_limits(user_id):
            return {"orders_created": 0}
        entry_orders = await self._process_entries(user_id)
        exit_orders = await self._process_exits(user_id)
        return {"orders_created": entry_orders + exit_orders}

    async def _check_risk_limits(self, user_id: str) -> bool:
        limits = self.supabase.table("risk_limits").select("*").eq("user_id", user_id).execute()
        if not limits.data:
            return False
        positions_count = self.supabase.table("positions").select("id", count="exact").eq("user_id", user_id).eq("status", "OPEN").execute().count or 0
        return positions_count < limits.data[0]["max_positions"]

    async def _process_entries(self, user_id: str) -> int:
        today = date.today().isoformat()
        intentions = self.supabase.table("trade_intentions").select("*").eq("date", today).execute()
        if not intentions.data:
            return 0
        # Entry logic stub - create paper orders
        return 0

    async def _process_exits(self, user_id: str) -> int:
        # Exit logic stub
        return 0

async def main():
    pipeline = ExecutorPipeline()
    result = await pipeline.run()
    logger.info(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
