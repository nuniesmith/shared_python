"""Risk management utilities: position sizing, stop-loss helpers, portfolio stubs."""
from __future__ import annotations

from typing import Sequence, Dict, Any, Tuple
import math
from unicodedata import name

try:  # prefer fully-qualified exported wrapper
    from fks_shared_python.types import RiskParams, PositionSizingResult  # type: ignore
except Exception:  # pragma: no cover
    try:
        from .types import RiskParams, PositionSizingResult  # type: ignore
    except Exception:  # final fallback (during local execution without package install)
        pass  # will raise later if referenced
try:
    from .exceptions import DataValidationError  # type: ignore
except Exception:  # pragma: no cover
    from exceptions import DataValidationError  # type: ignore
try:  # prefer custom structured logger
    from .logging import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging as _py_logging

    def get_logger(name: str):  # type: ignore
        return _py_logging.getLogger(name)

_log = get_logger(__name__)

# ------------------------------
# Position sizing
# ------------------------------
def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Classic Kelly fraction: f* = (bp - q)/b where b = win_loss_ratio, p = win_prob, q = 1-p.
    Returns 0 if invalid inputs produce negative recomm. No leverage floor/ceiling here.
    """
    if not (0 <= win_prob <= 1) or win_loss_ratio <= 0:
        raise DataValidationError("Invalid inputs for Kelly", {"win_prob": win_prob, "wl": win_loss_ratio})
    p = win_prob
    q = 1 - p
    b = win_loss_ratio
    f = (b * p - q) / b
    return max(0.0, f)

def fractional_position(equity: float, fraction: float, price: float, side: int) -> float:
    if equity <= 0 or price <= 0:
        raise DataValidationError("Equity/price must be positive", {"equity": equity, "price": price})
    units = (equity * fraction) / price
    return side * units

def volatility_target_position(equity: float, recent_vol: float, params: RiskParams, price: float, side: int) -> float:
    if params.target_vol is None or recent_vol <= 0:
        return fractional_position(equity, params.max_risk_per_trade, price, side)
    # scale risk fraction by ratio of target to realized vol
    scale = params.target_vol / recent_vol
    eff_fraction = min(params.max_risk_per_trade * scale, params.max_risk_per_trade * params.max_leverage)
    return fractional_position(equity, eff_fraction, price, side)

def correlation_sizing(base_size: float, avg_correlation: float, params: RiskParams) -> float:
    if avg_correlation < 0:
        return base_size
    if avg_correlation > params.correlation_cap:
        # linear scale down to cap -> 1.0 correlation gives half size (heuristic)
        excess = min(1.0, avg_correlation)
        scale = 1 - 0.5 * (excess - params.correlation_cap) / (1 - params.correlation_cap)
        return base_size * max(0.1, scale)
    return base_size

def dynamic_confidence_scale(confidence: float, params: RiskParams) -> float:
    # confidence expected in [0,1]; clamp
    c = min(1.0, max(0.0, confidence))
    return params.confidence_floor + (params.confidence_ceiling - params.confidence_floor) * c

def composite_position(
    equity: float,
    price: float,
    side: int,
    win_prob: float,
    win_loss_ratio: float,
    recent_vol: float,
    avg_correlation: float,
    confidence: float,
    params: RiskParams,
) -> PositionSizingResult:
    kelly = kelly_fraction(win_prob, win_loss_ratio)
    kelly_fraction_capped = min(kelly, params.max_risk_per_trade * params.max_leverage)
    base_units = fractional_position(equity, kelly_fraction_capped, price, side)
    vol_units = volatility_target_position(equity, recent_vol, params, price, side)
    # blend simple average for now
    blended = 0.5 * base_units + 0.5 * vol_units
    corr_adjusted = correlation_sizing(blended, avg_correlation, params)
    confidence_scale = dynamic_confidence_scale(confidence, params)
    final_units = corr_adjusted * confidence_scale
    leverage = abs(final_units * price) / max(equity, 1e-9)
    result = PositionSizingResult(
        position_size=final_units,
        leverage=leverage,
        method="composite",
        meta={
            "kelly_fraction": kelly,
            "kelly_capped": kelly_fraction_capped,
            "vol_units": vol_units,
            "corr_adjusted": corr_adjusted,
            "confidence_scale": confidence_scale,
        },
    )
    # Structured log for risk event
    try:  # pragma: no cover (logging side effects) 
        _log.info(
            "risk.sizing", 
            extra={
                "method": "composite",
                "position_size": result.position_size,
                "leverage": result.leverage,
                "kelly": kelly,
                "corr": avg_correlation,
                "confidence": confidence,
            },
        )
    except Exception:  # pragma: no cover
        pass
    return result

# ------------------------------
# Stop-loss / ATR helpers
# ------------------------------
def average_true_range(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> float:
    if not (len(high) == len(low) == len(close)) or len(close) < period + 1:
        raise DataValidationError("ATR input mismatch/insufficient length", {"len": len(close)})
    trs = []
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        trs.append(tr)
    recent = trs[-period:]
    return sum(recent) / len(recent)

def atr_trailing_stop(entry_price: float, atr: float, atr_mult: float, side: int) -> float:
    if atr <= 0 or atr_mult <= 0:
        raise DataValidationError("ATR/mult must be positive", {"atr": atr, "mult": atr_mult})
    if side >= 0:  # long
        return entry_price - atr_mult * atr
    return entry_price + atr_mult * atr

# ------------------------------
# Portfolio optimization stubs
# ------------------------------
def min_variance_weights(cov: Sequence[Sequence[float]]) -> Tuple[Sequence[float], float]:
    n = len(cov)
    if n == 0:
        return [], 0.0
    # naive inverse variance approximation (no external deps)
    inv_var = []
    for i in range(n):
        v = cov[i][i]
        inv_var.append(0.0 if v <= 0 else 1.0 / v)
    s = sum(inv_var) or 1.0
    w = [x / s for x in inv_var]
    # portfolio variance approx
    port_var = 0.0
    for i in range(n):
        for j in range(n):
            port_var += w[i] * w[j] * cov[i][j]
    return w, port_var

def risk_parity_weights(cov: Sequence[Sequence[float]], max_iter: int = 100, tol: float = 1e-6) -> Tuple[Sequence[float], int]:
    n = len(cov)
    if n == 0:
        return [], 0
    w = [1.0 / n] * n
    for it in range(max_iter):
        # compute marginal contributions ~ cov * w
        mc = [sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
        target = sum(w[i] * mc[i] for i in range(n)) / n or 1.0
        updated = []
        for i in range(n):
            if mc[i] <= 0:
                updated.append(w[i])
            else:
                updated.append(min(10.0, max(0.0, target / mc[i])))
        s = sum(updated) or 1.0
        new_w = [x / s for x in updated]
        diff = sum(abs(new_w[i] - w[i]) for i in range(n))
        w = new_w
        if diff < tol:
            return w, it + 1
    return w, max_iter

# ------------------------------
# Uncertainty / Hedging helpers
# ------------------------------
def regime_uncertainty_score(vol: float, ood_score: float, vol_threshold: float = 0.8, ood_threshold: float = 0.6) -> float:
    """Combine normalized realized volatility and out-of-distribution model score into an uncertainty score.

    Both vol and ood_score are expected already scaled to [0,1]. Simple average after threshold accentuation.
    """
    v = max(0.0, min(1.0, vol))
    o = max(0.0, min(1.0, ood_score))
    # Emphasize tail zones beyond thresholds
    if v > vol_threshold:
        v = min(1.0, v + 0.25 * (v - vol_threshold))
    if o > ood_threshold:
        o = min(1.0, o + 0.25 * (o - ood_threshold))
    return 0.5 * (v + o)

def hedge_position_size(base_size: float, uncertainty: float, max_hedge_fraction: float = 0.5) -> float:
    """Return hedge overlay (opposite sign) sized as fraction of base position scaled by uncertainty.

    If uncertainty=0 -> no hedge. If uncertainty=1 -> max_hedge_fraction of base size.
    """
    u = max(0.0, min(1.0, uncertainty))
    hedge = -base_size * max_hedge_fraction * u
    return hedge

def composite_with_hedge(*, equity: float, price: float, side: int, win_prob: float, win_loss_ratio: float,
                          recent_vol: float, avg_correlation: float, confidence: float, params: RiskParams,
                          normalized_vol: float, ood_score: float) -> PositionSizingResult:
    """Extended composite sizing that overlays a hedge when regime uncertainty is elevated.

    Adds meta fields: uncertainty_score, hedge_size.
    """
    base = composite_position(equity, price, side, win_prob, win_loss_ratio, recent_vol, avg_correlation, confidence, params)
    uncertainty = regime_uncertainty_score(normalized_vol, ood_score)
    hedge = hedge_position_size(base.position_size, uncertainty)
    final_units = base.position_size + hedge
    leverage = abs(final_units * price) / max(equity, 1e-9)
    enriched = PositionSizingResult(
        position_size=final_units,
        leverage=leverage,
        method="composite",
        meta={**(base.meta or {}), "uncertainty_score": uncertainty, "hedge_size": hedge},
    )
    try:  # pragma: no cover
        _log.info("risk.sizing.hedged", extra={
            "position_size": enriched.position_size,
            "hedge_size": hedge,
            "uncertainty": uncertainty,
        })
    except Exception:
        pass
    return enriched

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
"""Risk management utilities: position sizing, stop-loss helpers, portfolio stubs."""
from __future__ import annotations

from typing import Sequence, Dict, Any, Tuple
import math

try:  # prefer fully-qualified exported wrapper
    from fks_shared_python.types import RiskParams, PositionSizingResult  # type: ignore
except Exception:  # pragma: no cover
    try:
        from .types import RiskParams, PositionSizingResult  # type: ignore
    except Exception:  # final fallback (during local execution without package install)
        pass  # will raise later if referenced
try:
    from .exceptions import DataValidationError  # type: ignore
except Exception:  # pragma: no cover
    from exceptions import DataValidationError  # type: ignore

try:  # prefer custom structured logger
    from .logging import get_logger  # type: ignore
except Exception:  # pragma: no cover
    import logging as _py_logging

    def get_logger(name: str):  # type: ignore
        return _py_logging.getLogger(name)

_log = get_logger(__name__)


# ------------------------------
# Position sizing
# ------------------------------
def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Classic Kelly fraction: f* = (bp - q)/b where b = win_loss_ratio, p = win_prob, q = 1-p.
    Returns 0 if invalid inputs produce negative recomm. No leverage floor/ceiling here.
    """
    if not (0 <= win_prob <= 1) or win_loss_ratio <= 0:
        raise DataValidationError("Invalid inputs for Kelly", {"win_prob": win_prob, "wl": win_loss_ratio})
    p = win_prob
    q = 1 - p
    b = win_loss_ratio
    f = (b * p - q) / b
    return max(0.0, f)


def fractional_position(equity: float, fraction: float, price: float, side: int) -> float:
    if equity <= 0 or price <= 0:
        raise DataValidationError("Equity/price must be positive", {"equity": equity, "price": price})
    units = (equity * fraction) / price
    return side * units


def volatility_target_position(equity: float, recent_vol: float, params: RiskParams, price: float, side: int) -> float:
    if params.target_vol is None or recent_vol <= 0:
        return fractional_position(equity, params.max_risk_per_trade, price, side)
    # scale risk fraction by ratio of target to realized vol
    scale = params.target_vol / recent_vol
    eff_fraction = min(params.max_risk_per_trade * scale, params.max_risk_per_trade * params.max_leverage)
    return fractional_position(equity, eff_fraction, price, side)


def correlation_sizing(base_size: float, avg_correlation: float, params: RiskParams) -> float:
    if avg_correlation < 0:
        return base_size
    if avg_correlation > params.correlation_cap:
        # linear scale down to cap -> 1.0 correlation gives half size (heuristic)
        excess = min(1.0, avg_correlation)
        scale = 1 - 0.5 * (excess - params.correlation_cap) / (1 - params.correlation_cap)
        return base_size * max(0.1, scale)
    return base_size


def dynamic_confidence_scale(confidence: float, params: RiskParams) -> float:
    # confidence expected in [0,1]; clamp
    c = min(1.0, max(0.0, confidence))
    return params.confidence_floor + (params.confidence_ceiling - params.confidence_floor) * c


def composite_position(
    equity: float,
    price: float,
    side: int,
    win_prob: float,
    win_loss_ratio: float,
    recent_vol: float,
    avg_correlation: float,
    confidence: float,
    params: RiskParams,
) -> PositionSizingResult:
    kelly = kelly_fraction(win_prob, win_loss_ratio)
    kelly_fraction_capped = min(kelly, params.max_risk_per_trade * params.max_leverage)
    base_units = fractional_position(equity, kelly_fraction_capped, price, side)
    vol_units = volatility_target_position(equity, recent_vol, params, price, side)
    # blend simple average for now
    blended = 0.5 * base_units + 0.5 * vol_units
    corr_adjusted = correlation_sizing(blended, avg_correlation, params)
    confidence_scale = dynamic_confidence_scale(confidence, params)
    final_units = corr_adjusted * confidence_scale
    leverage = abs(final_units * price) / max(equity, 1e-9)
    result = PositionSizingResult(
        position_size=final_units,
        leverage=leverage,
        method="composite",
        meta={
            "kelly_fraction": kelly,
            "kelly_capped": kelly_fraction_capped,
            "vol_units": vol_units,
            "corr_adjusted": corr_adjusted,
            "confidence_scale": confidence_scale,
        },
    )
    # Structured log for risk event
    try:  # pragma: no cover (logging side effects) 
        _log.info(
            "risk.sizing", 
            extra={
                "method": "composite",
                "position_size": result.position_size,
                "leverage": result.leverage,
                "kelly": kelly,
                "corr": avg_correlation,
                "confidence": confidence,
            },
        )
    except Exception:  # pragma: no cover
        pass
    return result


# ------------------------------
# Stop-loss / ATR helpers
# ------------------------------
def average_true_range(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> float:
    if not (len(high) == len(low) == len(close)) or len(close) < period + 1:
        raise DataValidationError("ATR input mismatch/insufficient length", {"len": len(close)})
    trs = []
    for i in range(1, len(close)):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        trs.append(tr)
    recent = trs[-period:]
    return sum(recent) / len(recent)


def atr_trailing_stop(entry_price: float, atr: float, atr_mult: float, side: int) -> float:
    if atr <= 0 or atr_mult <= 0:
        raise DataValidationError("ATR/mult must be positive", {"atr": atr, "mult": atr_mult})
    if side >= 0:  # long
        return entry_price - atr_mult * atr
    return entry_price + atr_mult * atr


# ------------------------------
# Portfolio optimization stubs
# ------------------------------
def min_variance_weights(cov: Sequence[Sequence[float]]) -> Tuple[Sequence[float], float]:
    n = len(cov)
    if n == 0:
        return [], 0.0
    # naive inverse variance approximation (no external deps)
    inv_var = []
    for i in range(n):
        v = cov[i][i]
        inv_var.append(0.0 if v <= 0 else 1.0 / v)
    s = sum(inv_var) or 1.0
    w = [x / s for x in inv_var]
    # portfolio variance approx
    port_var = 0.0
    for i in range(n):
        for j in range(n):
            port_var += w[i] * w[j] * cov[i][j]
    return w, port_var


def risk_parity_weights(cov: Sequence[Sequence[float]], max_iter: int = 100, tol: float = 1e-6) -> Tuple[Sequence[float], int]:
    n = len(cov)
    if n == 0:
        return [], 0
    w = [1.0 / n] * n
    for it in range(max_iter):
        # compute marginal contributions ~ cov * w
        mc = [sum(cov[i][j] * w[j] for j in range(n)) for i in range(n)]
        target = sum(w[i] * mc[i] for i in range(n)) / n or 1.0
        updated = []
        for i in range(n):
            if mc[i] <= 0:
                updated.append(w[i])
            else:
                updated.append(min(10.0, max(0.0, target / mc[i])))
        s = sum(updated) or 1.0
        new_w = [x / s for x in updated]
        diff = sum(abs(new_w[i] - w[i]) for i in range(n))
        w = new_w
        if diff < tol:
            return w, it + 1
    return w, max_iter


# ------------------------------
# Uncertainty / Hedging helpers
# ------------------------------
def regime_uncertainty_score(vol: float, ood_score: float, vol_threshold: float = 0.8, ood_threshold: float = 0.6) -> float:
    """Combine normalized realized volatility and out-of-distribution model score into an uncertainty score.

    Both vol and ood_score are expected already scaled to [0,1]. Simple average after threshold accentuation.
    """
    v = max(0.0, min(1.0, vol))
    o = max(0.0, min(1.0, ood_score))
    # Emphasize tail zones beyond thresholds
    if v > vol_threshold:
        v = min(1.0, v + 0.25 * (v - vol_threshold))
    if o > ood_threshold:
        o = min(1.0, o + 0.25 * (o - ood_threshold))
    return 0.5 * (v + o)


def hedge_position_size(base_size: float, uncertainty: float, max_hedge_fraction: float = 0.5) -> float:
    """Return hedge overlay (opposite sign) sized as fraction of base position scaled by uncertainty.

    If uncertainty=0 -> no hedge. If uncertainty=1 -> max_hedge_fraction of base size.
    """
    u = max(0.0, min(1.0, uncertainty))
    hedge = -base_size * max_hedge_fraction * u
    return hedge


def composite_with_hedge(*, equity: float, price: float, side: int, win_prob: float, win_loss_ratio: float,
                          recent_vol: float, avg_correlation: float, confidence: float, params: RiskParams,
                          normalized_vol: float, ood_score: float) -> PositionSizingResult:
    """Extended composite sizing that overlays a hedge when regime uncertainty is elevated.

    Adds meta fields: uncertainty_score, hedge_size.
    """
    base = composite_position(equity, price, side, win_prob, win_loss_ratio, recent_vol, avg_correlation, confidence, params)
    uncertainty = regime_uncertainty_score(normalized_vol, ood_score)
    hedge = hedge_position_size(base.position_size, uncertainty)
    final_units = base.position_size + hedge
    leverage = abs(final_units * price) / max(equity, 1e-9)
    enriched = PositionSizingResult(
        position_size=final_units,
        leverage=leverage,
        method="composite",
        meta={**(base.meta or {}), "uncertainty_score": uncertainty, "hedge_size": hedge},
    )
    try:  # pragma: no cover
        _log.info("risk.sizing.hedged", extra={
            "position_size": enriched.position_size,
            "hedge_size": hedge,
            "uncertainty": uncertainty,
        })
    except Exception:
        pass
    return enriched


__all__ = [
    # sizing
    "kelly_fraction",
    "fractional_position",
    "volatility_target_position",
    "correlation_sizing",
    "dynamic_confidence_scale",
    "composite_position",
    # ATR / stops
    "average_true_range",
    "atr_trailing_stop",
    # portfolio
    "min_variance_weights",
    "risk_parity_weights",
    # hedging
    "regime_uncertainty_score",
    "hedge_position_size",
    "composite_with_hedge",
]
