#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec ./scripts/python_mjbatch.sh -m scripts.view_native_connector_geometry "$@"
