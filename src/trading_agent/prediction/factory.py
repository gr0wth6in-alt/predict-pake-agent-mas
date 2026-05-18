"""Predictor factory.

Centralizes how the rest of the codebase chooses a predictor by name. Keeps the
TradingAgent / API / CLI free of hard-coded conditionals about which predictor to use.
"""

from __future__ import annotations

from pathlib import Path

from trading_agent.config import Settings, load_settings
from trading_agent.prediction.baseline import MovingAverageMomentumPredictor
from trading_agent.prediction.llm import ClaudePredictor
from trading_agent.prediction.ml import TrainedModelPredictor
from trading_agent.prediction.multi_indicator import MultiIndicatorPredictor
from trading_agent.prediction.protocols import Predictor


PREDICTOR_AUTO = "auto"
PREDICTOR_BASELINE = "baseline"
PREDICTOR_MULTI = "multi"
PREDICTOR_ML = "ml"
PREDICTOR_LLM = "llm"

PREDICTOR_NAMES = (
    PREDICTOR_AUTO,
    PREDICTOR_BASELINE,
    PREDICTOR_MULTI,
    PREDICTOR_ML,
    PREDICTOR_LLM,
)


def build_predictor(
    name: str = PREDICTOR_AUTO,
    *,
    model_path: str | Path | None = None,
    symbol: str | None = None,
    settings: Settings | None = None,
) -> Predictor:
    """Return a predictor by name. ``auto`` picks the best available given the env."""

    resolved_settings = settings or load_settings()
    chosen = (name or PREDICTOR_AUTO).strip().lower()

    if chosen == PREDICTOR_BASELINE:
        return _build_baseline(resolved_settings)
    if chosen == PREDICTOR_MULTI:
        return MultiIndicatorPredictor()
    if chosen == PREDICTOR_ML:
        return _load_ml_or_raise(model_path, symbol)
    if chosen == PREDICTOR_LLM:
        return ClaudePredictor()

    if chosen != PREDICTOR_AUTO:
        raise ValueError(
            f"unknown predictor '{name}'. Valid options: {', '.join(PREDICTOR_NAMES)}"
        )

    # Auto selection: ML if a trained model exists for this symbol, otherwise multi.
    ml = _try_load_ml(model_path, symbol)
    if ml is not None:
        return ml
    return MultiIndicatorPredictor()


def _build_baseline(settings: Settings) -> MovingAverageMomentumPredictor:
    return MovingAverageMomentumPredictor(
        short_window=settings.prediction_short_window,
        long_window=settings.prediction_long_window,
    )


def _try_load_ml(
    model_path: str | Path | None,
    symbol: str | None,
) -> TrainedModelPredictor | None:
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.exists():
        return None
    predictor = TrainedModelPredictor.load(path)
    if symbol is not None and predictor.model.symbol.upper() != symbol.upper():
        return None
    return predictor


def _load_ml_or_raise(
    model_path: str | Path | None,
    symbol: str | None,
) -> TrainedModelPredictor:
    if model_path is None:
        raise ValueError("ml predictor requires a model_path")
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"model file does not exist: {path}")
    predictor = TrainedModelPredictor.load(path)
    if symbol is not None and predictor.model.symbol.upper() != symbol.upper():
        raise ValueError(
            f"trained model symbol {predictor.model.symbol!r} does not match request {symbol!r}"
        )
    return predictor
