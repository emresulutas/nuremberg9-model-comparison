#!/usr/bin/env bash
# Refresh the Nuremberg-9 paper report from saved results.
set -euo pipefail
cd "$(dirname "$0")/../.."
boxx_python=/HDD2/esulutas/envs/churro-comparison/bin/python
if [[ ! -v PYTHON_BIN && -x $boxx_python ]]; then
    PYTHON_BIN=$boxx_python
fi
"${PYTHON_BIN:-python}" nuremberg-9/scripts/build_report9.py
