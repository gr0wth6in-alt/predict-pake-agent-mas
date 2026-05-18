"""Anthropic Claude based predictor.

Sends a compact view of recent OHLC plus computed indicators to Claude and parses a
strict JSON response back into a :class:`Prediction`. The Anthropic SDK is an optional
dependency installed via the ``llm`` extra. The predictor never executes a live trade
on its own. It only produces a direction score consumed by the rest of the pipeline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from trading_agent.features.indicators import (
    clamp,
    compute_indicator_snapshot,
)
from trading_agent.models import Candle, Prediction


DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 600
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RECENT_CANDLES = 30
DEFAULT_TIMEOUT_SECONDS = 30.0

SYSTEM_PROMPT = (
    "You are a careful crypto market analyst. You answer ONLY with strict JSON of the shape "
    '{"direction": "up" | "down" | "flat", "confidence": <float 0..1>, "rationale": "<short>"}. '
    "Never include markdown, code fences or any text outside the JSON object. Be conservative: "
    "if signals conflict, lean toward 'flat' and lower confidence."
)


@dataclass(frozen=True)
class LLMResponse:
    direction: str
    confidence: float
    rationale: str


class LLMConfigurationError(RuntimeError):
    """Raised when the predictor is asked to run without a usable Anthropic configuration."""


class ClaudePredictor:
    """Predictor that delegates the call to Anthropic Claude.

    The Anthropic SDK is imported lazily so the rest of the project keeps working even
    when the ``llm`` extra is not installed. If no API key is configured, ``predict``
    raises :class:`LLMConfigurationError` instead of silently producing a fake score.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        recent_candles: int = DEFAULT_RECENT_CANDLES,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        horizon_candles: int = 1,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        if recent_candles <= 0:
            raise ValueError("recent_candles must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0 and 1")

        self.model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.recent_candles = recent_candles
        self.timeout_seconds = timeout_seconds
        self.horizon_candles = horizon_candles
        self.system_prompt = system_prompt

    @property
    def min_history(self) -> int:
        # Match the multi-indicator predictor so MACD and Bollinger fields are populated.
        return 35

    def predict(self, candles: list[Candle]) -> Prediction:
        if len(candles) < self.min_history:
            raise ValueError(f"need at least {self.min_history} candles")
        if not self._api_key:
            raise LLMConfigurationError(
                "ANTHROPIC_API_KEY is not set; install the llm extra and configure the key."
            )

        client = self._build_client()
        prompt = self._build_user_prompt(candles)

        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout_seconds,
        )

        response_text = self._extract_text(message)
        parsed = self._parse_response(response_text)

        direction_score = _direction_to_score(parsed.direction, parsed.confidence)
        return Prediction(
            symbol=candles[-1].symbol,
            direction_score=direction_score,
            confidence=clamp(parsed.confidence, 0.0, 1.0),
            horizon_candles=self.horizon_candles,
            rationale=f"claude:{self.model}: {parsed.rationale}".strip(),
        )

    def _build_client(self) -> object:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise LLMConfigurationError(
                "The anthropic SDK is not installed. Run: pip install -e \".[llm]\""
            ) from exc

        return anthropic.Anthropic(api_key=self._api_key)

    def _build_user_prompt(self, candles: list[Candle]) -> str:
        recent = candles[-self.recent_candles :]
        closes = [candle.close for candle in candles]
        highs = [candle.high for candle in candles]
        lows = [candle.low for candle in candles]
        snapshot = compute_indicator_snapshot(closes, highs, lows)

        candle_rows = [
            {
                "t": candle.timestamp.isoformat(),
                "o": candle.open,
                "h": candle.high,
                "l": candle.low,
                "c": candle.close,
                "v": candle.volume,
            }
            for candle in recent
        ]

        payload = {
            "symbol": candles[-1].symbol,
            "horizon_candles": self.horizon_candles,
            "indicators": snapshot.to_dict(),
            "recent_candles": candle_rows,
            "instructions": (
                "Predict the direction of the next "
                f"{self.horizon_candles} candle(s). Respond with strict JSON only: "
                '{"direction": "up"|"down"|"flat", "confidence": 0..1, "rationale": "..."}.'
            ),
        }
        return json.dumps(payload, default=str)

    @staticmethod
    def _extract_text(message: object) -> str:
        content = getattr(message, "content", None)
        if not content:
            raise LLMConfigurationError("Claude returned an empty response")

        chunks: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                chunks.append(block["text"])
        if not chunks:
            raise LLMConfigurationError("Claude response did not contain any text content")
        return "\n".join(chunks)

    @staticmethod
    def _parse_response(text: str) -> LLMResponse:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise LLMConfigurationError(f"Could not find JSON in Claude response: {text[:200]}")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMConfigurationError(
                f"Claude response was not valid JSON: {match.group(0)[:200]}"
            ) from exc

        direction = str(data.get("direction", "flat")).strip().lower()
        if direction not in {"up", "down", "flat"}:
            direction = "flat"
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        rationale = str(data.get("rationale", "")).strip()
        return LLMResponse(direction=direction, confidence=confidence, rationale=rationale)


def _direction_to_score(direction: str, confidence: float) -> float:
    bounded = clamp(confidence, 0.0, 1.0)
    if direction == "up":
        return bounded
    if direction == "down":
        return -bounded
    return 0.0
