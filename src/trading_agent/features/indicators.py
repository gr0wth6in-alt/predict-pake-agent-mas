from __future__ import annotations


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


def percentage_change(previous: float, current: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
