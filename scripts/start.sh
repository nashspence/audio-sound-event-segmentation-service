#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-/app:/opt/atst-sed}"

exec python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${SERVICE_PORT:-8000}" \
  --workers 1 \
  --timeout-keep-alive 5 \
  --log-level "${LOG_LEVEL:-info}"
