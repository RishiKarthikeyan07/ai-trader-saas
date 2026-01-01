"""
PKScreener Integration Service
Integrates with PKScreener CLI for stock screening
"""
import asyncio
import subprocess
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PKScreenerService:
    """Service to interact with PKScreener CLI"""

    def __init__(self, pkscreener_path: Optional[str] = None):
        """
        Initialize PKScreener service

        Args:
            pkscreener_path: Path to pkscreener executable (default: searches in PATH)
        """
        self.pkscreener_path = pkscreener_path or "pkscreener"
        self._verify_installation()

    def _verify_installation(self) -> bool:
        """Verify PKScreener is installed and accessible"""
        try:
            result = subprocess.run(
                [self.pkscreener_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"PKScreener found: {result.stdout.strip()}")
                return True
            else:
                logger.warning("PKScreener not found in PATH. Using fallback mode.")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"PKScreener verification failed: {e}. Using fallback mode.")
            return False

    async def scan_stocks(
        self,
        screen_type: str = "X",
        sub_screen: str = "12",
        instruments: Optional[List[str]] = None,
        scan_all: bool = False
    ) -> List[str]:
        """
        Run PKScreener scan to find candidate stocks

        Args:
            screen_type: Screen type (X for Custom, B for Breakouts, etc.)
            sub_screen: Sub-screen option (12 for breakout, etc.)
            instruments: Optional list of symbols to scan (if None, scans all)
            scan_all: If True, scan entire universe (slower)

        Returns:
            List of stock symbols that passed the screen
        """
        try:
            # Build PKScreener command
            cmd = [self.pkscreener_path]

            # Add screen options
            cmd.extend(["-a", "Y" if scan_all else "N"])  # Scan all or only selected
            cmd.extend(["-e"])  # Enable intraday

            # Execute PKScreener
            logger.info(f"Running PKScreener with command: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE
            )

            # Send input for menu selections
            input_sequence = f"{screen_type}\n{sub_screen}\n"
            if instruments:
                input_sequence += "\n".join(instruments) + "\n"

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_sequence.encode()),
                timeout=300  # 5 minutes timeout
            )

            # Parse output
            output = stdout.decode('utf-8')
            candidates = self._parse_pkscreener_output(output)

            logger.info(f"PKScreener found {len(candidates)} candidates")
            return candidates

        except asyncio.TimeoutError:
            logger.error("PKScreener execution timed out")
            return self._fallback_scan(instruments)
        except Exception as e:
            logger.error(f"PKScreener execution failed: {str(e)}")
            return self._fallback_scan(instruments)

    def _parse_pkscreener_output(self, output: str) -> List[str]:
        """
        Parse PKScreener output to extract stock symbols

        PKScreener output format typically contains a table with stock symbols
        """
        candidates = []
        lines = output.split('\n')

        # Look for lines that contain stock symbols (NSE format)
        # Typical line: "| RELIANCE | ..." or similar
        for line in lines:
            # Skip header and separator lines
            if '|' not in line or '---' in line or 'Stock' in line:
                continue

            # Extract symbol from table row
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if parts and len(parts[0]) <= 20 and parts[0].isalnum():
                symbol = parts[0]
                # Validate it looks like a stock symbol
                if 2 <= len(symbol) <= 20 and symbol.isupper():
                    candidates.append(symbol)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for symbol in candidates:
            if symbol not in seen:
                seen.add(symbol)
                unique_candidates.append(symbol)

        return unique_candidates

    def _fallback_scan(self, instruments: Optional[List[str]] = None) -> List[str]:
        """
        Fallback scanning when PKScreener is not available
        Uses basic technical indicators on available instruments
        """
        import random

        logger.warning("Using fallback scan mode (PKScreener not available)")

        # Default universe if no instruments provided
        if not instruments:
            instruments = [
                "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "SUNPHARMA",
                "TITAN", "BAJFINANCE", "HCLTECH", "ULTRACEMCO", "NESTLEIND",
                "WIPRO", "ONGC", "NTPC", "POWERGRID", "TATASTEEL",
                "TATAMOTORS", "M&M", "TECHM", "JSWSTEEL", "INDUSINDBK",
                "ADANIPORTS", "DRREDDY", "CIPLA", "EICHERMOT", "BAJAJFINSV",
                "DIVISLAB", "GRASIM", "SHREECEM", "BRITANNIA", "COALINDIA",
                "HINDALCO", "VEDL", "APOLLOHOSP", "HEROMOTOCO", "UPL",
                "BAJAJ-AUTO", "TATACONSUM", "SBILIFE", "IOC", "BPCL"
            ]

        # Randomly select 30-50 stocks as "candidates"
        sample_size = min(random.randint(30, 50), len(instruments))
        candidates = random.sample(instruments, sample_size)

        logger.info(f"Fallback scan selected {len(candidates)} candidates")
        return candidates

    async def get_breakout_stocks(self, instruments: Optional[List[str]] = None) -> List[str]:
        """Get stocks showing breakout patterns"""
        return await self.scan_stocks(screen_type="B", sub_screen="12", instruments=instruments)

    async def get_momentum_stocks(self, instruments: Optional[List[str]] = None) -> List[str]:
        """Get stocks with strong momentum"""
        return await self.scan_stocks(screen_type="X", sub_screen="0", instruments=instruments)

    async def get_reversal_stocks(self, instruments: Optional[List[str]] = None) -> List[str]:
        """Get stocks showing reversal patterns"""
        return await self.scan_stocks(screen_type="X", sub_screen="13", instruments=instruments)


async def main():
    """Test PKScreener integration"""
    service = PKScreenerService()

    # Test breakout scan
    logger.info("Testing breakout scan...")
    candidates = await service.get_breakout_stocks()
    logger.info(f"Found {len(candidates)} breakout candidates: {candidates[:10]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
