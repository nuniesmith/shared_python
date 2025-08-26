from __future__ import annotations

try:  # allow import whether loaded as package (relative) or top-level module
    from .config import get_settings  # type: ignore
    from .exceptions import RiskLimitExceeded, DataFetchError, DataValidationError  # type: ignore
except ImportError:  # pragma: no cover
    from config import get_settings  # type: ignore
    from exceptions import RiskLimitExceeded, DataFetchError, DataValidationError  # type: ignore
from typing import Iterable, Sequence, List, Tuple
import math

try:  # Optional heavy deps
    import numpy as _np  # type: ignore
    import numpy.typing as _npt  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _npt = None  # type: ignore


def get_risk_threshold(equity: float | int) -> float:
    s = get_settings()
    return float(equity) * float(s.RISK_MAX_PER_TRADE)


def enforce_risk(requested: float, equity: float | int) -> float:
    max_allowed = get_risk_threshold(equity)
    if requested > max_allowed:
        raise RiskLimitExceeded(requested=requested, max_allowed=max_allowed)
    return requested


from __future__ import annotations

try:  # allow import whether loaded as package (relative) or top-level module
    from .config import get_settings  # type: ignore
    from .exceptions import RiskLimitExceeded, DataFetchError, DataValidationError  # type: ignore
except ImportError:  # pragma: no cover
    from config import get_settings  # type: ignore
    from exceptions import RiskLimitExceeded, DataFetchError, DataValidationError  # type: ignore
from typing import Sequence, List, Tuple
import math

try:  # Optional heavy deps
    import numpy as _np  # type: ignore
    import numpy.typing as _npt  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore
    _npt = None  # type: ignore


def get_risk_threshold(equity: float | int) -> float:
    s = get_settings()
    return float(equity) * float(s.RISK_MAX_PER_TRADE)


def enforce_risk(requested: float, equity: float | int) -> float:
    max_allowed = get_risk_threshold(equity)
    if requested > max_allowed:
        raise RiskLimitExceeded(requested=requested, max_allowed=max_allowed)
    return requested


# ---------------------------
# Data preprocessing helpers
# ---------------------------
def zscore_outliers(series: Sequence[float], threshold: float = 3.0) -> List[int]:
    """Return indices of points whose z-score exceeds threshold.
    Falls back to simple mean/std if numpy unavailable.
    """
    if not series:
        return []
    if _np is not None:  # use numpy for speed
        arr = _np.asarray(series, dtype=float)
        m = float(arr.mean())
        std = float(arr.std(ddof=0)) or 1e-9
        z = _np.abs((arr - m) / std)
        idx = [int(i) for i, v in enumerate(z) if v > threshold]
        if not idx:
            # Robust fallback: if no outliers but distribution extremely concentrated except one point
            try:
                median = float(_np.median(arr))
                abs_dev = _np.abs(arr - median)
                mad = float(_np.median(abs_dev))
                if mad == 0:  # all identical except potential outlier
                    alt_idx = [int(i) for i, v in enumerate(arr) if abs(v - median) > threshold]
                    if alt_idx:
                        return alt_idx
            except Exception:  # pragma: no cover
                pass
        return idx
    # Python fallback
    n = len(series)
    m = sum(series) / n
    var = sum((x - m) ** 2 for x in series) / n
    std = math.sqrt(var) or 1e-9
    out: List[int] = []
    for i, v in enumerate(series):
        if abs((v - m) / std) > threshold:
            out.append(i)
    return out


def kmeans_regimes(series: Sequence[float], k: int = 3, max_iter: int = 50) -> List[int]:
    """Very small dependency-free k-means for quick regime clustering (1D). Not for production scale.
    Returns list of cluster labels (0..k-1).
    """
    if not series:
        return []
    vals = [float(v) for v in series]
    uniq = sorted(set(vals))
    if len(uniq) <= k:
        mapping = {v: i for i, v in enumerate(uniq)}
        return [mapping[v] for v in vals]
    # init centroids evenly through sorted unique values
    step = max(1, len(uniq) // k)
    centroids = [uniq[min(i * step, len(uniq) - 1)] for i in range(k)]
    for _ in range(max_iter):
        # assign
        labels = [min(range(k), key=lambda ci: abs(vals_i - centroids[ci])) for vals_i in vals]
        new_centroids: List[float] = []
        changed = False
        for ci in range(k):
            cluster_points = [v for v, lab in zip(vals, labels) if lab == ci]
            if cluster_points:
                nc = sum(cluster_points) / len(cluster_points)
            else:
                nc = centroids[ci]
            if abs(nc - centroids[ci]) > 1e-12:
                changed = True
            new_centroids.append(nc)
        centroids = new_centroids
        if not changed:
            break
    return labels


# ---------------------------
# Feature engineering helpers
# ---------------------------
def bid_ask_spread(bid: float, ask: float) -> float:
    if ask <= 0 or bid <= 0:
        raise DataValidationError("Bid/Ask must be > 0", {"bid": bid, "ask": ask})
    if ask < bid:
        raise DataValidationError("Ask must be >= Bid", {"bid": bid, "ask": ask})
    return (ask - bid) / ((ask + bid) / 2.0)


def price_divergence(series_a: Sequence[float], series_b: Sequence[float]) -> float:
    if len(series_a) != len(series_b) or not series_a:
        raise DataValidationError("Series length mismatch or empty", {"len_a": len(series_a), "len_b": len(series_b)})
    if _np is not None:
        a = _np.asarray(series_a, dtype=float)
        b = _np.asarray(series_b, dtype=float)
        return float(_np.mean(_np.abs(a - b) / (_np.abs(b) + 1e-9)))
    # fallback
    acc = 0.0
    for x, y in zip(series_a, series_b):
        acc += abs(x - y) / (abs(y) + 1e-9)
    return acc / len(series_a)


def cyclical_encode(value: float, period: float) -> Tuple[float, float]:
    if period <= 0:
        raise DataValidationError("Period must be positive", {"period": period})
    angle = (value % period) / period * 2 * math.pi
    return math.sin(angle), math.cos(angle)


__all__ = [
    "get_risk_threshold",
    "enforce_risk",
    "zscore_outliers",
    "kmeans_regimes",
    "bid_ask_spread",
    "price_divergence",
    "cyclical_encode",
]
def cyclical_encode(value: float, period: float) -> Tuple[float, float]:
    if period <= 0:
        raise DataValidationError("Period must be positive", {"period": period})
    angle = (value % period) / period * 2 * math.pi
    return math.sin(angle), math.cos(angle)


__all__ = [
    # risk
    "get_risk_threshold",
    "enforce_risk",
    # preprocessing
    "zscore_outliers",
    "kmeans_regimes",
    # features
    "bid_ask_spread",
    "price_divergence",
    "cyclical_encode",
]
