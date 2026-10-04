"""Finite absolute and reference-relative metrics for Python QA records."""

from collections.abc import Iterable
import math
from numbers import Real


def _as_finite_real(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} is not a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} is not finite")
    return converted


def require_finite_numbers(values: Iterable[object], label: str) -> None:
    """Reject nonnumeric, NaN, or infinite values before a golden comparison."""
    for index, value in enumerate(values):
        _as_finite_real(value, f"{label}[{index}]")


def maximum_error_metrics(pairs: Iterable[tuple[object, object]]) -> tuple[float, float]:
    """Return maximum absolute and relative errors for finite actual/reference pairs.

    Relative error uses `abs(reference)` as its denominator. At an exact zero
    reference it uses 1.0, matching the zero-coefficient convention in H02;
    the reported value then remains the actual absolute difference in native
    units. This avoids inventing a zero metric or emitting infinity.
    """
    maximum_absolute = 0.0
    maximum_relative = 0.0
    count = 0
    for index, (actual_value, reference_value) in enumerate(pairs):
        actual = _as_finite_real(actual_value, f"actual[{index}]")
        reference = _as_finite_real(reference_value, f"reference[{index}]")
        absolute_error = abs(actual - reference)
        denominator = abs(reference) if reference != 0.0 else 1.0
        relative_error = absolute_error / denominator
        if not math.isfinite(absolute_error) or not math.isfinite(relative_error):
            raise ValueError(f"comparison[{index}] produced a non-finite error")
        maximum_absolute = max(maximum_absolute, absolute_error)
        maximum_relative = max(maximum_relative, relative_error)
        count += 1
    if count == 0:
        raise ValueError("at least one finite comparison pair is required")
    return maximum_absolute, maximum_relative
