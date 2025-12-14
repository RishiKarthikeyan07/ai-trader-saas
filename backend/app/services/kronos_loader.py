from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from app.core.config import Settings


DEFAULT_KRONOS_REPO = "NeoQuasar/Kronos-base"
DEFAULT_KRONOS_MODEL_ID = "NeoQuasar/Kronos-base"
DEFAULT_KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
DEFAULT_KRONOS_GIT = "https://github.com/shiyu-coder/Kronos.git"


def download_kronos_model(
    settings: Optional[Settings] = None,
    dest: Optional[Path] = None,
    repo_id: str = DEFAULT_KRONOS_REPO,
) -> Path:
    """Download Kronos model snapshot from Hugging Face into dest directory.

    If settings provided and dest is None, uses settings.kronos_artifact_path or artifacts/kronos.
    """
    target = dest or (settings.kronos_artifact_path if settings else None) or Path("artifacts/kronos")
    target = Path(target).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    # Hugging Face token can be provided via HF_TOKEN if needed for gated models.
    token = os.getenv("HF_TOKEN")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        local_dir_use_symlinks=False,
        token=token,
        revision=None,
        resume_download=True,
    )
    return target


def load_kronos_hf(
    model_id: str = DEFAULT_KRONOS_MODEL_ID,
    tokenizer_id: str = DEFAULT_KRONOS_TOKENIZER_ID,
    device: str = "cpu",
    max_context: int = 512,
):
    """Load Kronos model + tokenizer from Hugging Face Hub using the upstream API.

    Example:
        predictor = load_kronos_hf(device="cuda:0", max_context=512)
        emb = predictor.encode(inputs)
    """
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
    except ImportError:
        ensure_kronos_repo()
        try:
            from model import Kronos, KronosTokenizer, KronosPredictor  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Kronos classes not found. Ensure the Kronos repo/package is installed.") from exc

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
    model = Kronos.from_pretrained(model_id)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    return predictor


def ensure_kronos_repo(target_dir: Path | str = Path(".cache/kronos_repo")) -> Path:
    target = Path(target_dir)
    if target.exists() and (target / "model.py").exists():
        if str(target) not in sys.path:
            sys.path.append(str(target))
        return target
    target.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["git", "clone", DEFAULT_KRONOS_GIT, str(target)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # pragma: no cover
        # If directory already exists, continue; otherwise raise
        if not (target / "model").exists():
            raise RuntimeError(f"Failed to clone Kronos repo: {exc}")
    if str(target) not in sys.path:
        sys.path.append(str(target))
    return target
