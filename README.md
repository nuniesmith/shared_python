# shared_python

Shared Python utilities for FKS services (config/env loading, logging helpers, risk calculations, hedging, shared domain types).

## Features

- Centralized environment + .env loader (`config.load_config()`)
- Lazy global settings object (`get_settings()`)
- Structured logging setup (`init_logging()` / JSON via `FKS_JSON_LOGS=1`)
- Shared exceptions (e.g. `RiskLimitExceeded`)
- Risk + numeric helpers (`utils.py`)
- Typed domain models via Pydantic (`types.py`)
- Advanced position sizing & portfolio helpers (`risk.py`): Kelly, volatility targeting, correlation scaling, risk parity, hedged composite sizing.
- Hedging utilities: regime uncertainty scoring + dynamic hedge overlay.

## Install (as submodule path dependency)

```bash
pip install -e repo/shared/python  # from service repo root
```

Or if split into its own repository and added as git submodule under `shared/python`:

```bash
git submodule add https://github.com/your-org/shared_python repo/shared/python
pip install -e repo/shared/python
```

## Usage

```python
from fks_shared_python.config import get_settings  # new canonical namespace
from fks_shared_python.risk import composite_with_hedge, RiskParams

# Legacy alias still works:
# from shared_python.config import get_settings

settings = get_settings()
params = RiskParams()
res = composite_with_hedge(
	equity=100_000,
	price=25_000,
	side=1,
	win_prob=0.55,
	win_loss_ratio=1.2,
	recent_vol=0.4,
	avg_correlation=0.5,
	confidence=0.7,
	params=params,
	normalized_vol=0.9,
	ood_score=0.6,
)
print(res.position_size, res.meta["uncertainty_score"])
```

## Environment Variables

| Var | Description | Default |
|-----|-------------|---------|
| APP_ENV | Environment name | dev |
| LOG_LEVEL | Logging level | INFO |
| RISK_MAX_PER_TRADE | Fraction of equity per trade | 0.01 |
| DEBUG_MODE | Extra debug toggles | false |

Loads values from process env then optional `.env` (root search upward) using `python-dotenv`.

## Tests

```bash
pytest -q  # from project root after editable install
```

 
## Versioning

Tag releases (e.g. `v0.1.0`) in the standalone repo; services pin submodule commit hashes. Legacy import path `shared_python` retained for backward compatibility; plan migration to `fks_shared_python` only in next minor version.
