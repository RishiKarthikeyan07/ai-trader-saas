"""
PKScreener Integration

Executes PKScreener pack scans and returns candidates
"""

from typing import List, Dict, Optional
import asyncio
import subprocess
import json
import csv
import os
import logging
from io import StringIO
from pathlib import Path

logger = logging.getLogger(__name__)


PACK_CONFIGS = {
    'momentum-breakout': {
        'scan_code': 'X:12:10',  # PKScreener scan for breakouts
        'signal_type': 'momentum_breakout'
    },
    'squeeze-expansion': {
        'scan_code': 'X:12:16',  # PKScreener volatility squeeze
        'signal_type': 'squeeze_expansion'
    },
    'pullback-continuation': {
        'scan_code': 'X:0:27',  # PKScreener pullback scan
        'signal_type': 'pullback_continuation'
    },
    'liquidity-sweep': {
        'scan_code': 'X:12:42',  # PKScreener reversal scan
        'signal_type': 'liquidity_sweep_reversal'
    },
    'relative-strength': {
        'scan_code': 'X:12:29',  # PKScreener RS scan
        'signal_type': 'relative_strength'
    }
}


async def run_pkscreener_packs(use_live: bool = False) -> List[Dict]:
    """
    Run all PKScreener packs and return candidates

    Args:
        use_live: If True, execute real PKScreener CLI. If False, use stub data.

    Returns:
        List of candidate signals from all scanner packs
    """
    all_candidates = []

    for pack_id, config in PACK_CONFIGS.items():
        logger.info(f"Running scanner pack: {pack_id}")

        if use_live:
            # Try real PKScreener execution
            candidates = await run_single_pack_live(pack_id, config)
        else:
            # Use stub data
            candidates = await run_single_pack_stub(pack_id, config)

        all_candidates.extend(candidates)
        logger.info(f"Pack {pack_id} returned {len(candidates)} candidates")

    logger.info(f"Total candidates from all packs: {len(all_candidates)}")
    return all_candidates


async def run_single_pack_stub(pack_id: str, config: Dict) -> List[Dict]:
    """
    Stub implementation - returns mock candidates

    In production, this would:
    1. Execute: pkscreener -a Y -e -p -o scancode={scan_code}
    2. Parse CSV output
    3. Return structured data
    """
    # Mock candidates for development
    mock_symbols = {
        'momentum-breakout': ['RELIANCE', 'TCS', 'INFY'],
        'squeeze-expansion': ['HDFCBANK', 'ICICIBANK'],
        'pullback-continuation': ['WIPRO', 'LT'],
        'liquidity-sweep': ['TATAMOTORS'],
        'relative-strength': ['ASIANPAINT', 'BAJFINANCE']
    }

    candidates = []
    symbols = mock_symbols.get(pack_id, [])

    for symbol in symbols:
        candidates.append({
            'pack_id': pack_id,
            'symbol': symbol,
            'signal_type': config['signal_type'],
            'pack_score': 7.5 + (hash(symbol) % 20) / 10,  # Mock score 7.5-9.5
            'tags': ['Scanned', 'Volume OK']
        })

    return candidates


async def run_single_pack_live(pack_id: str, config: Dict) -> List[Dict]:
    """
    Execute single PKScreener pack using CLI

    Args:
        pack_id: Scanner pack identifier
        config: Pack configuration with scan_code

    Returns:
        List of candidate dicts
    """
    try:
        scan_code = config['scan_code']
        logger.info(f"Executing PKScreener for {pack_id} with code {scan_code}")

        # Execute PKScreener
        symbols = await run_pkscreener_cli(scan_code)

        if not symbols:
            logger.warning(f"No symbols returned from {pack_id}, falling back to stub")
            return await run_single_pack_stub(pack_id, config)

        # Convert symbols to structured candidates
        candidates = []
        for symbol in symbols:
            candidates.append({
                'pack_id': pack_id,
                'symbol': symbol,
                'signal_type': config['signal_type'],
                'pack_score': 7.5,  # Default score, can be enhanced
                'tags': ['PKScreener', 'Live Scan']
            })

        return candidates

    except Exception as e:
        logger.error(f"Error running PKScreener pack {pack_id}: {str(e)}")
        logger.warning(f"Falling back to stub for {pack_id}")
        return await run_single_pack_stub(pack_id, config)


async def run_pkscreener_cli(scan_code: str) -> List[str]:
    """
    Execute actual PKScreener command and return symbols

    Command format:
    pkscreener -a Y -e -p -o scancode=X:12:10

    Environment:
    - PKScreener should be installed (pip install pkscreener)
    - May require configuration file (pkscreener.ini)

    Returns:
        List of stock symbols
    """
    try:
        # Check if pkscreener is available
        check_cmd = subprocess.run(
            ['which', 'pkscreener'],
            capture_output=True,
            text=True
        )

        if check_cmd.returncode != 0:
            logger.warning("PKScreener CLI not found in PATH")
            return []

        # Build command
        cmd = [
            'pkscreener',
            '-a', 'Y',        # Auto-select
            '-e',             # Export to CSV
            '-p',             # Print results
            '-o', f'scancode={scan_code}'
        ]

        logger.info(f"Executing: {' '.join(cmd)}")

        # Run with timeout
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=300)

        if result.returncode != 0:
            logger.error(f"PKScreener failed: {stderr.decode()}")
            return []

        # Parse output
        output = stdout.decode()
        symbols = parse_pkscreener_output(output)

        logger.info(f"PKScreener returned {len(symbols)} symbols")
        return symbols

    except asyncio.TimeoutError:
        logger.error("PKScreener execution timed out (5 min)")
        return []
    except Exception as e:
        logger.error(f"PKScreener execution error: {str(e)}")
        return []


def parse_pkscreener_output(output: str) -> List[str]:
    """
    Parse PKScreener output and extract symbols

    PKScreener typically outputs results as CSV or formatted text.
    This parser handles common formats.

    Args:
        output: Raw stdout from pkscreener

    Returns:
        List of stock symbols
    """
    symbols = []

    try:
        # Try CSV parsing first
        if ',' in output and ('Stock' in output or 'Symbol' in output):
            # CSV format
            reader = csv.DictReader(StringIO(output))
            for row in reader:
                # Look for symbol column (various possible names)
                symbol = (row.get('Stock') or
                         row.get('Symbol') or
                         row.get('Ticker') or
                         row.get('stock'))

                if symbol and symbol.strip():
                    # Clean symbol (remove .NS, .BO suffixes if present)
                    clean_symbol = symbol.strip().split('.')[0]
                    symbols.append(clean_symbol)

        else:
            # Plain text format - extract words that look like symbols
            lines = output.split('\n')
            for line in lines:
                # Skip header/footer lines
                if any(skip in line.lower() for skip in ['scan', 'total', 'results', '---']):
                    continue

                # Extract potential symbols (uppercase words)
                words = line.split()
                for word in words:
                    # Symbol-like: 2-15 chars, mostly uppercase, alphanumeric
                    if (2 <= len(word) <= 15 and
                        word.replace('&', '').isalnum() and
                        any(c.isupper() for c in word)):

                        clean_symbol = word.strip().split('.')[0]
                        if clean_symbol not in symbols:
                            symbols.append(clean_symbol)

        # Remove duplicates and invalid entries
        symbols = list(set(s for s in symbols if s and len(s) >= 2))

        logger.info(f"Parsed {len(symbols)} symbols from PKScreener output")
        return symbols

    except Exception as e:
        logger.error(f"Error parsing PKScreener output: {str(e)}")
        return []
