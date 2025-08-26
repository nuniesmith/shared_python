# Implementation Timeline Progress

Status codes: ✅ done | 🟡 in progress / partial | ⏳ not started | 🔄 planned automation

## Month 1: Infrastructure & Data

| Week | Objective | Key Tasks | Status | Notes |
|------|-----------|----------|--------|-------|
| 1 | Docker centralization & shared submodules | Generate unified Docker templates; add `fks_shared_python` submodule/symlink to each service | ✅ | Symlinks + templates applied across all services (scripts executed) |
| 2 | API connections (`fks_data`) | Integrate config/env loading, placeholder data fetch adapters | ✅ | Unified adapter layer (base + Binance + Polygon), retries/backoff, JSON logging, manager integration + tests |
| 3 | DB schema (`fks_data`) | Define schema + validate with JSON schemas | ⏳ | Requires inventory of existing DB models |
| 4 | Feature pipeline (`fks_transformer`) | Wire preprocessing & feature utils (already in shared) | ⏳ | Shared utilities ready (zscore, regimes, cyclical) |
| 5 | Viz tools (`fks_web`) | Add dashboards + shared deploy smoke test | ⏳ | Monitoring stub present (`monitor.sh`) |

## Month 2: Model Development

| Week | Objective | Status | Notes |
|------|-----------|--------|-------|
| 6 | Baselines training integration | 🟡 | Baseline param schemas + risk infra ready; wrappers for metrics present |
| 7 | Time-series CV | 🟡 | CV scaffolding present (earlier phase) – needs service wiring |
| 8 | Hyperopt / ensemble | ⏳ | Placeholder only |
| 9 | Initial ensemble cross-service tests | ⏳ | Await prior weeks completion |

## Month 3: Refinement & Validation

| Week | Objective | Status | Notes |
|------|-----------|--------|-------|
| 10 | Feature selection & SHAP | ⏳ | Shared infra can host SHAP utilities later |
| 11 | Walk-forward & robustness | ⏳ | Risk & simulation modules prepared |
| 12 | Risk module finalization | ✅ | Implemented sizing, stops, hedging, portfolio stubs |
| 13 | Integration & health checks | 🟡 | Monitor script stub added |

## Month 4: Paper Trading & Live Testing

| Week | Objective | Status | Notes |
|------|-----------|--------|-------|
| 14 | Paper env + monitoring | ⏳ | Monitoring stub exists; needs service hooks |
| 15 | Dashboards & evaluation | ⏳ | Pending web integration |
| 16 | Paper evaluation & drift | ⏳ | OOD + uncertainty scoring available |
| 17 | Live cutover plan | ⏳ | Will aggregate prior artifacts |

## Automation / Scripts Added

| Script | Path | Purpose |
|--------|------|---------|
| add_shared_submodule.sh | shared_scripts/tools/ | Add or update `fks_shared_python` submodule/symlink across services |
| centralize_docker.sh | shared_scripts/docker/ | Stub to sync shared Docker templates into each service |
| monitor.sh | shared_scripts/utils/ | Basic monitoring stub (already present) |

## Next Immediate Actions

1. Execute `shared_scripts/tools/add_shared_submodule.sh` in monorepo root to ensure all services reference the shared package.
2. Flesh out `centralize_docker.sh` with concrete template copy logic (pull from `shared/shared_docker`).
3. Begin Week 2 API connection scaffolding in `fks_data` (adapters + config usage).
4. Draft DB schema validation harness (JSON schema + pydantic) for Week 3.
