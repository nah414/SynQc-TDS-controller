#!/usr/bin/env sh
set -eu

# POSIX smoke test for SynQc backend + UI
# Usage: ./scripts/smoke_test.sh [BASE_URL] [UI_URL]
# Defaults: BASE_URL=http://127.0.0.1:8000/api/v1/synqc, UI_URL=http://127.0.0.1:8000/ui

BASE=${1:-${SYNQC_API_BASE:-http://127.0.0.1:8000/api/v1/synqc}}
UI=${2:-${SYNQC_UI_URL:-http://127.0.0.1:8000/ui}}

check() {
  url="$1"
  code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
  if [ "$code" -ge 200 ] && [ "$code" -lt 400 ]; then
    echo "OK: $url -> $code"
    return 0
  else
    echo "FAIL: $url -> $code"
    return 1
  fi
}

echo "[smoke] Checking backend health..."
check "$BASE/health"

echo "[smoke] Checking sessions endpoint..."
check "$BASE/sessions"

echo "[smoke] Fetching UI page..."
check "$UI"

echo "[smoke] All checks passed."
