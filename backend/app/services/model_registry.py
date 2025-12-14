from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

from app.core.config import Settings
from app.services.kronos_loader import download_kronos_model, load_kronos_hf

try:  # lazy imports to keep optional deps
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

from app.ml.stockformer.model import StockFormer
from app.ml.tft.model import TFT
from app.ml.preprocess.normalize import build_veto_vec, build_tf_align_vec, build_smc_vec, build_ta_vec


@dataclass
class ModelInput:
    symbol: str
    asof: pd.Timestamp
    ohlcv_120: np.ndarray  # (120,5) float32 normalized
    tf_align: np.ndarray
    smc_vec: np.ndarray
    ta_vec: np.ndarray
    raw_features: Dict[str, Any]


@dataclass
class ModelOutput:
    direction_prob: float
    expected_return: float
    expected_volatility: float
    reliability: float
    regime: float
    veto_score: float
    upper_band: Optional[float]
    lower_band: Optional[float]
    model_versions: Dict[str, str]


class KronosEncoder:
    def __init__(self, artifact_path: Path | None):
        if torch is None:
            raise ImportError("torch is required for Kronos encoder")
        self.predictor = None
        self.model = None
        self._init_model(artifact_path)

    def _init_model(self, artifact_path: Path | None):
        if artifact_path and Path(artifact_path).exists():
            self.model = self._load_torch_model(Path(artifact_path))
            self.model.eval()
            return
        # fallback to HF loader
        self.predictor = load_kronos_hf()

    def _load_torch_model(self, artifact_path: Path):
        artifact_path = Path(artifact_path)
        if artifact_path.is_dir():
            candidates = list(artifact_path.glob("*.pt")) + list(artifact_path.glob("*.bin"))
            if not candidates:
                raise FileNotFoundError(f"No torch model file found in {artifact_path}")
            model_path = candidates[0]
        else:
            model_path = artifact_path
        return torch.jit.load(model_path) if hasattr(torch, "jit") else torch.load(model_path, map_location="cpu")

    def encode(self, ohlcv_tensor: "torch.Tensor") -> np.ndarray:
        with torch.no_grad():
            if self.predictor is not None:
                # Use tokenizer embed as pooled embedding
                device = getattr(self.predictor, "device", "cpu")
                x = ohlcv_tensor.to(device)
                tok = self.predictor.tokenizer
                if hasattr(tok, "embed"):
                    # Kronos tokenizer expects 6 features (price + volume + amount); pad amount if missing.
                    if x.shape[-1] == 5:
                        amt = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
                        x = torch.cat([x, amt], dim=-1)
                    z = tok.embed(x)
                    if isinstance(z, tuple):
                        z = z[0]
                    emb = z.mean(dim=1)
                    emb_np = emb.detach().cpu().numpy()
                    if emb_np.shape[1] < 512:
                        pad = np.zeros((emb_np.shape[0], 512 - emb_np.shape[1]), dtype=emb_np.dtype)
                        emb_np = np.concatenate([emb_np, pad], axis=1)
                    elif emb_np.shape[1] > 512:
                        emb_np = emb_np[:, :512]
                    return emb_np
                raise RuntimeError("Kronos tokenizer does not expose embed method for embeddings.")
            emb = self.model(ohlcv_tensor)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            return emb.detach().cpu().numpy()


class StockFormerPredictor:
    def __init__(self, ckpt_path: Path, context_dim: int = 29):
        if torch is None:
            raise ImportError("torch is required for StockFormer")
        self.model = StockFormer(context_dim=context_dim)
        state = torch.load(str(ckpt_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, model_input: ModelInput, kronos_emb: np.ndarray) -> Dict[str, Any]:
        x_price = torch.tensor(model_input.ohlcv_120, dtype=torch.float32).unsqueeze(0)
        x_kronos = torch.tensor(kronos_emb, dtype=torch.float32).unsqueeze(0)
        x_context = torch.tensor(np.concatenate([model_input.tf_align, model_input.smc_vec, model_input.ta_vec]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            ret_pred, up_prob = self.model(x_price, x_kronos, x_context)
        return {"ret": ret_pred.detach().cpu().numpy(), "prob": up_prob.detach().cpu().numpy()}


class TFTPredictor:
    def __init__(self, ckpt_path: Path, context_dim: int = 29):
        if torch is None:
            raise ImportError("torch is required for TFT")
        self.model = TFT(context_dim=context_dim)
        state = torch.load(str(ckpt_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, model_input: ModelInput, kronos_emb: np.ndarray) -> Dict[str, Any]:
        x_price = torch.tensor(model_input.ohlcv_120, dtype=torch.float32).unsqueeze(0)
        x_kronos = torch.tensor(kronos_emb, dtype=torch.float32).unsqueeze(0)
        x_context = torch.tensor(np.concatenate([model_input.tf_align, model_input.smc_vec, model_input.ta_vec]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x_price, x_kronos, x_context)
        return out


class VetoModel:
    def __init__(self, model_path: Path):
        if lgb is None:
            raise ImportError("lightgbm is required for veto model")
        self.model = lgb.Booster(model_file=str(model_path))

    def veto(self, features: np.ndarray) -> float:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        prob = self.model.predict(features, raw_score=False)
        return float(prob[0])


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mode = (settings.mode or "stub").lower()
        self.artifacts = {
            "kronos": settings.kronos_artifact_path,
            "stockformer": settings.stockformer_artifact_path,
            "tft": settings.tft_artifact_path,
            "lightgbm": settings.lightgbm_artifact_path,
        }
        self.kronos: Optional[KronosEncoder] = None
        self.stockformer: Optional[StockFormerPredictor] = None
        self.tft: Optional[TFTPredictor] = None
        self.veto_model: Optional[VetoModel] = None
        if self.mode == "prod":
            self._load_models()

    def _load_models(self) -> None:
        self.ensure_ready()
        self.kronos = KronosEncoder(self.artifacts["kronos"])
        # The following loaders assume user supplies correct checkpoints/modules.
        self.stockformer = StockFormerPredictor(self.artifacts["stockformer"], context_dim=29)
        self.tft = TFTPredictor(self.artifacts["tft"], context_dim=29)
        self.veto_model = VetoModel(self.artifacts["lightgbm"])

    def ensure_ready(self) -> None:
        if self.mode != "prod":
            return
        missing = [name for name, path in self.artifacts.items() if path is None or not Path(path).exists()]
        if missing:
            raise RuntimeError(
                f"Prod mode requires model artifacts for: {', '.join(missing)}. Set paths in settings or switch to stub mode."
            )

    def infer(self, model_input: ModelInput) -> ModelOutput:
        self.ensure_ready()
        if self.mode == "stub":
            return self._stub_infer(model_input)
        if not all([self.kronos, self.stockformer, self.tft, self.veto_model]):
            raise RuntimeError("Prod mode models are not loaded")
        # Build feature payloads for models; user must ensure proper preprocessing upstream.
        if torch is None:
            raise ImportError("torch required for prod inference")
        ohlcv_tensor = torch.tensor(model_input.ohlcv_120, dtype=torch.float32).unsqueeze(0)
        kronos_emb = self.kronos.encode(ohlcv_tensor)
        sf_out = self.stockformer.predict(model_input, kronos_emb)
        tft_out = self.tft.predict(model_input, kronos_emb)
        veto_feats = self._build_veto_features(model_input, sf_out, tft_out)
        veto_score = self.veto_model.veto(veto_feats)

        direction_prob = float(np.clip(np.mean(sf_out["up_prob"]), 0, 1))
        expected_return = float(np.mean(sf_out["ret"]))
        expected_vol = float(tft_out.get("vol_10d", [[model_input.raw_features.get("volatility", 0.0)]])[0][0]) if isinstance(tft_out, dict) else float(model_input.raw_features.get("volatility", 0.0))
        upper_band = tft_out.get("upper_10d", [[None]])[0][0] if isinstance(tft_out, dict) else None
        lower_band = tft_out.get("lower_10d", [[None]])[0][0] if isinstance(tft_out, dict) else None
        regime = float(model_input.raw_features.get("ema_trend", 0))
        versions = {
            "kronos": str(self.artifacts["kronos"]),
            "stockformer": str(self.artifacts["stockformer"]),
            "tft": str(self.artifacts["tft"]),
            "lightgbm": str(self.artifacts["lightgbm"]),
        }
        return ModelOutput(
            direction_prob=direction_prob,
            expected_return=expected_return,
            expected_volatility=expected_vol,
            reliability=1 - veto_score,
            regime=regime,
            veto_score=veto_score,
            upper_band=upper_band if isinstance(upper_band, (int, float)) else None,
            lower_band=lower_band if isinstance(lower_band, (int, float)) else None,
            model_versions=versions,
        )

    def _stub_infer(self, model_input: ModelInput) -> ModelOutput:
        features = model_input.raw_features
        ema_signal = 1 if features.get("ema_fast", 0) > features.get("ema_slow", 0) else -1
        rsi = features.get("rsi", 50)
        smc_score = features.get("smc_score", 0)
        base_prob = 0.5 + 0.2 * ema_signal + 0.001 * (rsi - 50) + 0.2 * smc_score
        direction_prob = float(np.clip(base_prob, 0, 1))
        expected_return = float(features.get("returns", 0) * 5)
        expected_volatility = float(features.get("volatility", 0))
        reliability = float(np.clip(0.6 + 0.2 * smc_score, 0, 1))
        regime = float(features.get("ema_trend", 0))
        versions = {
            "kronos": "v0.1-stub",
            "stockformer": "v0.1-stub",
            "tft": "v0.1-stub",
            "lightgbm": "v0.1-stub",
        }
        return ModelOutput(
            direction_prob=direction_prob,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            reliability=reliability,
            regime=regime,
            veto_score=0.0,
            upper_band=None,
            lower_band=None,
            model_versions=versions,
        )

    def _build_veto_features(self, model_input: ModelInput, sf_out: Dict[str, Any], tft_out: Dict[str, Any]) -> np.ndarray:
        return build_veto_vec(sf_out, tft_out, model_input.smc_vec, model_input.tf_align, model_input.ta_vec, model_input.raw_features)
