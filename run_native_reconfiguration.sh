#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec ./scripts/python_mjbatch.sh -m scripts.run_native_reconfiguration "$@"
