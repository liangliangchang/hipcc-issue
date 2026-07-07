#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${ROOT}/compare_hot.py" "${ROOT}/build/asm" "${ROOT}/build/hot"
