#!/usr/bin/env bash
set -euo pipefail

# Adds or updates the fks_shared_python submodule (or creates a symlink fallback) in each listed service directory.
# Usage:
#   ./shared_scripts/tools/add_shared_submodule.sh ../fks/fks_api ../fks/fks_worker
# Environment variables:
#   SHARED_REPO_URL  (optional) remote git URL of shared repo
#   SHARED_PATH      (optional) relative path inside service (default: shared/shared_python)

SHARED_REPO_URL=${SHARED_REPO_URL:-"git@github.com:your-org/shared-python.git"}
TARGET_PATH=${SHARED_PATH:-"shared/shared_python"}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <service_dir> [service_dir ...]" >&2
  exit 1
fi

for svc in "$@"; do
  if [[ ! -d "$svc" ]]; then
    echo "[skip] $svc not a directory" >&2
    continue
  fi
  pushd "$svc" >/dev/null
  if [[ -d .git ]]; then
    if git submodule status "$TARGET_PATH" &>/dev/null; then
      echo "[update] submodule exists in $svc ($TARGET_PATH)"
      git submodule update --init --remote "$TARGET_PATH" || true
    else
      echo "[add] adding submodule $TARGET_PATH to $svc"
      mkdir -p "$(dirname "$TARGET_PATH")"
      git submodule add "$SHARED_REPO_URL" "$TARGET_PATH" || true
    fi
  else
    # Fallback: create symlink if not a git repo (e.g. during local dev sandbox)
    echo "[link] creating symlink fallback in $svc"
    mkdir -p "$(dirname "$TARGET_PATH")"
    ln -sfn "$(realpath ../../shared/shared_python)" "$TARGET_PATH"
  fi
  popd >/dev/null
done

echo "Done. Remember to commit submodule changes in each service repo."
