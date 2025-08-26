#!/usr/bin/env bash
set -euo pipefail

# Stub script to propagate shared Docker templates into service directories.
# Enhancements: template variable substitution, checksum comparison, dry-run mode.

TEMPLATE_ROOT=${TEMPLATE_ROOT:-"shared/shared_docker"}
FILES=("Dockerfile.template" "docker-compose.template.yml")

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <service_dir> [service_dir ...]" >&2
  exit 1
fi

for svc in "$@"; do
  if [[ ! -d "$svc" ]]; then
    echo "[skip] $svc missing" >&2
    continue
  fi
  for f in "${FILES[@]}"; do
    if [[ -f "$TEMPLATE_ROOT/$f" ]]; then
      target="$svc/${f/.template/}"
      cp "$TEMPLATE_ROOT/$f" "$target"
      echo "[sync] $f -> $target"
    fi
  done
done

echo "Docker template sync complete (stub)."
