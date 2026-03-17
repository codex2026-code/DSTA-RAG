#!/usr/bin/env bash
set -euo pipefail
python -m dsta_rag.stage2.replay_eval "$@"
