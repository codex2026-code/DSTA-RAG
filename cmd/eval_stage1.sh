#!/usr/bin/env bash
set -euo pipefail
python -m dsta_rag.stage1.eval_sft "$@"
