#!/usr/bin/env bash
# Optional: refresh vendored kernel/ support headers from a local llama.cpp checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA="${LLAMA:-/proj/gdba/lichang/llama.cpp}"
K="${ROOT}/kernel"

if [[ ! -d "${LLAMA}/ggml" ]]; then
    echo "LLAMA not found: ${LLAMA}" >&2
    exit 1
fi

mkdir -p "${K}/ggml/include" "${K}/ggml/src/ggml-cuda/vendors" \
         "${K}/variants/stripped" "${K}/variants/split" "${K}/variants/fix"

cp -a "${LLAMA}/ggml/include/." "${K}/ggml/include/"
cp "${LLAMA}/ggml/src/ggml-common.h" "${K}/ggml/src/"
cp "${LLAMA}/ggml/src/ggml-impl.h" "${K}/ggml/src/"
cp "${LLAMA}/ggml/src/ggml-cuda/common.cuh" "${K}/ggml/src/ggml-cuda/"
cp "${LLAMA}/ggml/src/ggml-cuda/mma.cuh" "${K}/ggml/src/ggml-cuda/"
cp "${LLAMA}/ggml/src/ggml-cuda/vecdotq.cuh" "${K}/ggml/src/ggml-cuda/"
cp "${LLAMA}/ggml/src/ggml-cuda/vendors/hip.h" "${K}/ggml/src/ggml-cuda/vendors/"

git -C "${LLAMA}" show 55945ef57:ggml/src/ggml-cuda/mmq.cuh > "${K}/variants/stripped/mmq.cuh"
git -C "${LLAMA}" show 1c862d51f:ggml/src/ggml-cuda/mmq.cuh > "${K}/variants/split/mmq.cuh"
cp "${LLAMA}/ggml/src/ggml-cuda/mmq.cuh" "${K}/variants/fix/mmq.cuh"

echo "Refreshed ${K}/ from ${LLAMA}"
