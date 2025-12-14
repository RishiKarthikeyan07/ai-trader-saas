import pytest

from app.core.config import Settings
from app.services.model_registry import ModelRegistry, ModelOutput


def test_stub_infer_returns_model_output():
    settings = Settings(mode="stub")
    registry = ModelRegistry(settings)
    from app.services.model_registry import ModelInput
    import numpy as np
    import pandas as pd

    mi = ModelInput(
        symbol="TEST",
        asof=pd.Timestamp.utcnow(),
        ohlcv_120=np.ones((120, 5), dtype=np.float32),
        tf_align=np.zeros(5, dtype=np.float32),
        smc_vec=np.zeros(8, dtype=np.float32),
        ta_vec=np.zeros(4, dtype=np.float32),
        raw_features={"ema_fast": 2, "ema_slow": 1, "rsi": 55, "smc_score": 0.3},
    )
    out = registry.infer(mi)
    assert isinstance(out, ModelOutput)
    assert 0 <= out.direction_prob <= 1
    assert out.model_versions


def test_prod_mode_without_artifacts_raises():
    settings = Settings(mode="prod")
    with pytest.raises(RuntimeError):
        ModelRegistry(settings)
