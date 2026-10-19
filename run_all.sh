#!/usr/bin/env bash
# Runs analyses 01 to 06 in order, then the tests. Needs AIREADI_DATA_DIR (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
: "${AIREADI_DATA_DIR:?Set AIREADI_DATA_DIR to the folder that holds the AI-READI parquet and split file}"
mkdir -p restricted
for script in analyses/0*.py; do
  start=$(date +%s)
  python "$script"
  echo "$(basename "$script"): $(( $(date +%s) - start )) s" | tee -a restricted/runtimes.txt
done
python -m pytest -q
