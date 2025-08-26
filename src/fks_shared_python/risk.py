"""Namespace wrapper exposing root risk utilities under fks_shared_python.risk."""
from __future__ import annotations

try:
    from risk import (  # type: ignore
        kelly_fraction,
        fractional_position,
        volatility_target_position,
        correlation_sizing,
        dynamic_confidence_scale,
        composite_position,
        average_true_range,
        atr_trailing_stop,
        min_variance_weights,
        risk_parity_weights,
        regime_uncertainty_score,
        hedge_position_size,
        composite_with_hedge,
    )
except Exception:  # pragma: no cover
    # Fallback to relative import inside installed package context
    from ..risk import (  # type: ignore
        kelly_fraction,
        fractional_position,
        volatility_target_position,
        correlation_sizing,
        dynamic_confidence_scale,
        composite_position,
        average_true_range,
        atr_trailing_stop,
        min_variance_weights,
        risk_parity_weights,
    )

__all__ = [
    "kelly_fraction",
    "fractional_position",
    "volatility_target_position",
    "correlation_sizing",
    "dynamic_confidence_scale",
    "composite_position",
    "average_true_range",
    "atr_trailing_stop",
    "min_variance_weights",
    "risk_parity_weights",
    "regime_uncertainty_score",
    "hedge_position_size",
    "composite_with_hedge",
]
