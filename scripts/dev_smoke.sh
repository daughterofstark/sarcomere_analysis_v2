#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

"${PYTHON_BIN}" -m pytest
"${PYTHON_BIN}" scripts/build_manifest.py --config configs/default.yaml --dry-run
"${PYTHON_BIN}" scripts/run_image_metrics.py --config configs/default.yaml --image-id 2.007-1 --write-all
"${PYTHON_BIN}" scripts/run_batch_metrics.py --config configs/default.yaml --limit 3 --write-tables --write-provenance --continue-on-error
