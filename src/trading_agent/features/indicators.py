from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev


def simple_moving_average(values: list[float], window: int) -> list[float | None]:
    if window <= 0:
        raise ValueError("window must be positive")

    averages: list[float | None] = []
    running_sum = 0.0

    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]

        if index + 1 < window:
            averages.append(None)
        else:
            averages.append(running_sum / window)

    return averages


def exponential_moving_average(values: list[float], window: int) -> list[float | None]:
    """Standard EMA with alpha = 2 / (window + 1). Seeds with the SMA of the first ``window`` values."""

    if window <= 0:
        raise ValueError("window must be positive")
    if len(values) < window:
        return [None] * len(values)

    alpha = 2.0 / (window + 1)
    ema_values: list[float | None] = [None] * (window - 1)
    seed = sum(values[:window]) / window
    ema_values.append(seed)
    previous = seed

    for value in values[window:]:
        previous = alpha * value + (1 - alpha) * previous
        ema_values.append(previous)

    return ema_values


def relative_strength_index(values: list[float], window: int = 14) -> list[float | None]:
    """RSI using Wilder smoothing. Returns ``None`` until enough data is available."""

    if window <= 0:
        raise ValueError("window must be positive")
    if len(values) <= window:
        return [None] * len(values)

    gains: list[float] = []
    losses: list[float] = []
    for index in range(1, len(values)):
        change = values[index] - values[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window

    rsi: list[float | None] = [None] * (window + 1)
    rsi[window] = _rsi_from(avg_gain, avg_loss)

    for index in range(window, len(gains)):
        avg_gain = (avg_gain * (window - 1) + gains[index]) / window
        avg_loss = (avg_loss * (window - 1) + losses[index]) / window
        rsi.append(_rsi_from(avg_gain, avg_loss))

    return rsi[: len(values)]


def macd(
    values: list[float],
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Returns (macd_line, signal_line, histogram)."""

    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("MACD windows must be positive")
    if fast >= slow:
        raise ValueError("fast window must be smaller than slow window")

    fast_ema = exponential_moving_average(values, fast)
    slow_ema = exponential_moving_average(values, slow)
    macd_line: list[float | None] = []
    for fast_value, slow_value in zip(fast_ema, slow_ema):
        if fast_value is None or slow_value is None:
            macd_line.append(None)
        else:
            macd_line.append(fast_value - slow_value)

    macd_only = [value for value in macd_line if value is not None]
    if len(macd_only) < signal:
        return macd_line, [None] * len(values), [None] * len(values)

    signal_tail = exponential_moving_average(macd_only, signal)
    pad = len(macd_line) - len(signal_tail)
    signal_line: list[float | None] = [None] * pad + list(signal_tail)

    histogram: list[float | None] = []
    for line_value, sig_value in zip(macd_line, signal_line):
        if line_value is None or sig_value is None:
            histogram.append(None)
        else:
            histogram.append(line_value - sig_value)

    return macd_line, signal_line, histogram


def average_true_range(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    window: int = 14,
) -> list[float | None]:
    """Wilder ATR. ``highs``, ``lows`` and ``closes`` must be the same length."""

    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows and closes must have the same length")
    if window <= 0:
        raise ValueError("window must be positive")
    if len(closes) <= window:
        return [None] * len(closes)

    true_ranges: list[float] = [highs[0] - lows[0]]
    for index in range(1, len(closes)):
        prev_close = closes[index - 1]
        true_ranges.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - prev_close),
                abs(lows[index] - prev_close),
            )
        )

    atr: list[float | None] = [None] * window
    seed = sum(true_ranges[1 : window + 1]) / window
    atr.append(seed)
    previous = seed
    for index in range(window + 1, len(true_ranges)):
        previous = (previous * (window - 1) + true_ranges[index]) / window
        atr.append(previous)

    return atr[: len(closes)]


def bollinger_bands(
    values: list[float],
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Returns (lower_band, middle_band, upper_band)."""

    if window <= 0:
        raise ValueError("window must be positive")
    if num_std <= 0:
        raise ValueError("num_std must be positive")

    middle = simple_moving_average(values, window)
    lower: list[float | None] = []
    upper: list[float | None] = []
    for index, mid in enumerate(middle):
        if mid is None:
            lower.append(None)
            upper.append(None)
            continue
        slice_values = values[index - window + 1 : index + 1]
        std = pstdev(slice_values) if len(slice_values) > 1 else 0.0
        lower.append(mid - num_std * std)
        upper.append(mid + num_std * std)

    return lower, middle, upper


def percentage_change(previous: float, current: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class IndicatorSnapshot:
    """Latest reading of a curated set of indicators. Plain floats so it serializes cleanly."""

    close: float
    sma_20: float | None
    ema_12: float | None
    ema_26: float | None
    ema_50: float | None
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_histogram: float | None
    atr_14: float | None
    bollinger_lower: float | None
    bollinger_middle: float | None
    bollinger_upper: float | None
    bollinger_percent: float | None
    return_1: float | None
    return_5: float | None
    return_20: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "close": self.close,
            "sma_20": self.sma_20,
            "ema_12": self.ema_12,
            "ema_26": self.ema_26,
            "ema_50": self.ema_50,
            "rsi_14": self.rsi_14,
            "macd": self.macd,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "atr_14": self.atr_14,
            "bollinger_lower": self.bollinger_lower,
            "bollinger_middle": self.bollinger_middle,
            "bollinger_upper": self.bollinger_upper,
            "bollinger_percent": self.bollinger_percent,
            "return_1": self.return_1,
            "return_5": self.return_5,
            "return_20": self.return_20,
        }


def compute_indicator_snapshot(
    closes: list[float],
    highs: list[float],
    lows: list[float],
) -> IndicatorSnapshot:
    """Compute indicator readings at the last index. Missing indicators come back as ``None``."""

    if not closes:
        raise ValueError("closes must not be empty")
    if len(closes) != len(highs) or len(closes) != len(lows):
        raise ValueError("closes, highs and lows must have the same length")

    sma_20 = simple_moving_average(closes, 20)[-1] if len(closes) >= 20 else None
    ema_12 = exponential_moving_average(closes, 12)[-1] if len(closes) >= 12 else None
    ema_26 = exponential_moving_average(closes, 26)[-1] if len(closes) >= 26 else None
    ema_50 = exponential_moving_average(closes, 50)[-1] if len(closes) >= 50 else None
    rsi_14 = relative_strength_index(closes, 14)[-1] if len(closes) > 14 else None

    macd_value: float | None = None
    macd_signal_value: float | None = None
    macd_histogram_value: float | None = None
    if len(closes) >= 26 + 9:
        macd_line, signal_line, histogram = macd(closes)
        macd_value = macd_line[-1]
        macd_signal_value = signal_line[-1]
        macd_histogram_value = histogram[-1]

    atr_14 = average_true_range(highs, lows, closes, 14)[-1] if len(closes) > 14 else None

    lower_band: float | None = None
    middle_band: float | None = None
    upper_band: float | None = None
    bollinger_percent: float | None = None
    if len(closes) >= 20:
        lower, middle, upper = bollinger_bands(closes, 20, 2.0)
        lower_band = lower[-1]
        middle_band = middle[-1]
        upper_band = upper[-1]
        if lower_band is not None and upper_band is not None and upper_band != lower_band:
            bollinger_percent = (closes[-1] - lower_band) / (upper_band - lower_band)

    return_1 = percentage_change(closes[-2], closes[-1]) if len(closes) >= 2 else None
    return_5 = percentage_change(closes[-6], closes[-1]) if len(closes) >= 6 else None
    return_20 = percentage_change(closes[-21], closes[-1]) if len(closes) >= 21 else None

    return IndicatorSnapshot(
        close=closes[-1],
        sma_20=sma_20,
        ema_12=ema_12,
        ema_26=ema_26,
        ema_50=ema_50,
        rsi_14=rsi_14,
        macd=macd_value,
        macd_signal=macd_signal_value,
        macd_histogram=macd_histogram_value,
        atr_14=atr_14,
        bollinger_lower=lower_band,
        bollinger_middle=middle_band,
        bollinger_upper=upper_band,
        bollinger_percent=bollinger_percent,
        return_1=return_1,
        return_5=return_5,
        return_20=return_20,
    )


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
