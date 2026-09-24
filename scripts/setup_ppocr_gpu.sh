#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
python -m pip install paddlepaddle-gpu==3.3.1 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
