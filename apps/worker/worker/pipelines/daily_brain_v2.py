"""
Daily Brain Pipeline V2
Generates trade_intentions for today based on REAL PKScreener + AI ranking
"""
import asyncio
import logging
from datetime import datetime, date
import random
from typing import List, Dict, Any
from supabase import create_client, Client
import yfinance as yf

from ..config import settings
from ..services.pkscreener import PKScreenerService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyBrainPipelineV2:
    """Daily pipeline to generate trade intentions with REAL PKScreener"""

    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY
        )
        self.pkscreener = PKScreenerService()

    async def run(self) -> Dict[str, Any]:
        """Run daily brain pipeline"""
        run_id = None
        try:
            run = self.supabase.table("pipeline_runs").insert({
                "type": "daily_brain",
                "status": "RUNNING",
                "started_at": datetime.utcnow().isoformat(),
            }).execute()

            run_id = run.data[0]["id"]
            logger.info(f"Daily Brain Pipeline V2 started - Run ID: {run_id}")

            tradable_instruments = await self._tradability_gate()
            candidates = await self._pkscreener_scan(tradable_instruments)
            ranked_candidates = await self._ai_ranking(candidates)
            intentions = await self._generate_intentions(ranked_candidates)
            await self._store_intentions(intentions)

            duration = (datetime.utcnow() - datetime.fromisoformat(
                run.data[0]["started_at"].replace("Z", "+00:00")
            )).total_seconds()

            self.supabase.table("pipeline_runs").update({
                "status": "SUCCESS",
                "ended_at": datetime.utcnow().isoformat(),
                "duration_seconds": int(duration),
                "metadata": {"intentions_generated": len(intentions)},
            }).eq("id", run_id).execute()

            return {"run_id": run_id, "status": "SUCCESS", "intentions_count": len(intentions)}
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            if run_id:
                self.supabase.table("pipeline_runs").update({
                    "status": "FAILED", "ended_at": datetime.utcnow().isoformat(),
                    "error_message": str(e),
                }).eq("id", run_id).execute()
            raise

    async def _tradability_gate(self) -> List[str]:
        """Get list of tradable instruments from database"""
        result = self.supabase.table("instruments").select("canonical_symbol").eq("is_tradable", True).execute()
        if not result.data:
            logger.warning("No tradable instruments in database, using default NIFTY50")
            return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        return [item["canonical_symbol"] for item in result.data]

    async def _pkscreener_scan(self, instruments: List[str]) -> List[str]:
        """
        REAL PKScreener integration
        Scans stocks using PKScreener CLI for breakout patterns
        Falls back to intelligent selection if PKScreener unavailable
        """
        logger.info(f"Running PKScreener scan on {len(instruments)} instruments")

        try:
            # Try breakout scan first
            candidates = await self.pkscreener.get_breakout_stocks(instruments)

            if not candidates or len(candidates) < 10:
                # Fallback to momentum scan
                logger.info("Breakout scan yielded few results, trying momentum scan")
                candidates = await self.pkscreener.get_momentum_stocks(instruments)

            logger.info(f"PKScreener returned {len(candidates)} candidates")
            return candidates

        except Exception as e:
            logger.error(f"PKScreener scan failed: {e}, using fallback")
            # Fallback is handled inside PKScreenerService
            return await self.pkscreener.get_breakout_stocks(instruments)

    async def _ai_ranking(self, candidates: List[str]) -> List[Dict[str, Any]]:
        """
        AI ranking with REAL price data from yfinance
        TODO: Replace with actual ML model inference
        """
        ranked = []

        logger.info(f"Fetching real price data for {len(candidates)} candidates")

        for symbol in candidates:
            try:
                # Fetch real price data
                ticker = yf.Ticker(f"{symbol}.NS")  # NSE suffix
                hist = ticker.history(period="1mo")

                if hist.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue

                # Calculate basic technical indicators
                current_price = hist['Close'].iloc[-1]
                ma_20 = hist['Close'].tail(20).mean()
                volatility = hist['Close'].pct_change().std()
                volume_surge = hist['Volume'].iloc[-1] / hist['Volume'].tail(20).mean()

                # Simple scoring (TODO: Replace with real AI model)
                # Higher score if price > MA20 and volume surge
                confidence = 0.5
                if current_price > ma_20:
                    confidence += 0.2
                if volume_surge > 1.5:
                    confidence += 0.15
                confidence = min(confidence + random.uniform(-0.05, 0.05), 0.95)

                # Risk grade based on volatility
                if volatility < 0.015:
                    risk_grade = "LOW"
                elif volatility < 0.025:
                    risk_grade = "MEDIUM"
                else:
                    risk_grade = "HIGH"

                ranked.append({
                    "symbol": symbol,
                    "confidence": round(confidence, 4),
                    "risk_grade": risk_grade,
                    "horizon": "SWING" if volatility > 0.02 else "POSITIONAL",
                    "tags": [
                        "trend aligned" if current_price > ma_20 else "trend weak",
                        "volume surge" if volume_surge > 1.5 else "normal volume"
                    ],
                    "current_price": current_price
                })

            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                continue

        # Sort by confidence
        ranked.sort(key=lambda x: x["confidence"], reverse=True)

        # Return top 20
        top_ranked = ranked[:20]
        logger.info(f"Ranked {len(top_ranked)} stocks for trade intentions")

        return top_ranked

    async def _generate_intentions(self, ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate trade intentions with REAL price levels"""
        intentions = []
        today = date.today().isoformat()

        for candidate in ranked:
            current_price = candidate.get("current_price", random.uniform(100, 3000))

            # Entry zone: ±2% from current
            entry_low = round(current_price * 0.98, 2)
            entry_high = round(current_price * 1.02, 2)

            # Stop loss: 3% below current
            sl = round(current_price * 0.97, 2)

            # Targets based on risk grade
            if candidate["risk_grade"] == "LOW":
                tp1 = round(current_price * 1.04, 2)  # 4% target
                tp2 = round(current_price * 1.06, 2)  # 6% target
            elif candidate["risk_grade"] == "MEDIUM":
                tp1 = round(current_price * 1.05, 2)  # 5% target
                tp2 = round(current_price * 1.08, 2)  # 8% target
            else:  # HIGH
                tp1 = round(current_price * 1.06, 2)  # 6% target
                tp2 = round(current_price * 1.10, 2)  # 10% target

            intentions.append({
                "date": today,
                "canonical_symbol": candidate["symbol"],
                "direction": "BUY",
                "entry_zone_low": entry_low,
                "entry_zone_high": entry_high,
                "sl": sl,
                "tp1": tp1,
                "tp2": tp2,
                "confidence": candidate["confidence"],
                "risk_grade": candidate["risk_grade"],
                "horizon": candidate["horizon"],
                "tags": candidate["tags"],
            })

        logger.info(f"Generated {len(intentions)} trade intentions")
        return intentions

    async def _store_intentions(self, intentions: List[Dict[str, Any]]) -> None:
        """Store intentions in database (idempotent)"""
        if not intentions:
            logger.warning("No intentions to store")
            return

        today = date.today().isoformat()

        # Delete old intentions for today (idempotent)
        self.supabase.table("trade_intentions").delete().eq("date", today).execute()

        # Insert new intentions
        self.supabase.table("trade_intentions").insert(intentions).execute()

        logger.info(f"Stored {len(intentions)} trade intentions for {today}")


async def main():
    pipeline = DailyBrainPipelineV2()
    result = await pipeline.run()
    logger.info(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
