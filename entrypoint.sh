#!/usr/bin/env bash
# entrypoint.sh – Container entry point for the AMD Caption Agent.
#
# Behaviour:
#   /input/tasks.json present  →  headless batch mode  (python run_headless.py)
#   /input/tasks.json absent   →  interactive UI mode  (streamlit run app.py)

set -euo pipefail

if [ -f "/input/tasks.json" ]; then
    echo "[entrypoint] /input/tasks.json detected – starting headless batch mode."
    exec python run_headless.py
else
    echo "[entrypoint] No /input/tasks.json found – starting Streamlit dashboard."
    exec streamlit run app.py \
        --server.address 0.0.0.0 \
        --server.port 8501
fi
