#!/usr/bin/env bash
set -euo pipefail
python -m dsta_rag.stage1.build_traces "$@"
# if both outputs are provided together, this script can also call build_sft automatically.
