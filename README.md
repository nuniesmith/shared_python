# fks-shared-python

Shared Python utilities for FKS services (config/env loading, logging helpers, risk calculations, shared domain types).

## Features

- Centralized environment + .env loader (`config.load_config()`)
- Lazy global settings object (`get_settings()`)
- Structured logging setup (`init_logging()`)
- Shared exceptions (e.g. `RiskLimitExceeded`)
- Risk + numeric helpers (`utils.py`)
- Typed domain models via Pydantic (`types.py`)

## Install (as submodule path dependency)

```bash
pip install -e repo/shared/python  # from service repo root
```

Or if split into its own repository and added as git submodule under `shared/python`:

```bash
git submodule add https://github.com/your-org/fks-shared-python repo/shared/python
pip install -e repo/shared/python
```

## Usage

```python
from fks_shared_python.config import get_settings
from fks_shared_python.utils import get_risk_threshold

settings = get_settings()
env = settings.APP_ENV
risk = get_risk_threshold(equity=10_000)
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
pytest -q repo/shared/python
```

## Versioning
Tag releases (e.g. `v0.1.0`) in the standalone repo; services pin submodule commit hashes.
