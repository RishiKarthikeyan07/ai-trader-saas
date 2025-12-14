#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

from app.core.config import get_settings
from app.services.kronos_loader import download_kronos_model, DEFAULT_KRONOS_REPO


def main():
    parser = argparse.ArgumentParser(description="Download Kronos model from Hugging Face")
    parser.add_argument("--dest", type=Path, help="Destination directory for Kronos model")
    parser.add_argument("--repo", type=str, default=DEFAULT_KRONOS_REPO, help="Hugging Face repo id")
    args = parser.parse_args()

    settings = get_settings()
    target = download_kronos_model(settings=settings, dest=args.dest, repo_id=args.repo)
    print(f"Kronos model downloaded to: {target}")


if __name__ == "__main__":
    # Allow running as `PYTHONPATH=.. python scripts/download_kronos.py`
    sys.exit(main())
