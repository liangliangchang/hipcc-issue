#!/usr/bin/env bash
# Compile three mmq.cuh variants to AMDGPU ASM with stock ROCm hipcc/clang.
# Self-contained: uses kernel/ vendored sources (no llama.cpp checkout required).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL="${ROOT}/kernel"
ROCM="${ROCM:-/proj/gdba/lichang/rocm}"
ARCH="${ARCH:-gfx1151}"
CLANG="${ROCM}/llvm/bin/clang++"

CUDA_INC="${KERNEL}/ggml/src/ggml-cuda"
INST="${CUDA_INC}/template-instances/mmq-instance-q4_k.cu"
ASM="${ROOT}/build/asm"

VARIANTS=(stripped split fix)
VARIANT_COMMITS=(
    "55945ef57 single hot loop, regression baseline"
    "1c862d51f cold/hot duplicate via threadIdx.z > 0"
    "fix MMQ_HIP_TILE_BARRIER (vendored mmq.cuh)"
)

mkdir -p "${ASM}"

COMMON_FLAGS=(
    -x hip
    -std=c++17
    -O3
    "--offload-arch=${ARCH}"
    -DGGML_USE_HIP
    -DAMD_WMMA_AVAILABLE
    -DRDNA3_5
    -I"${CUDA_INC}"
    -I"${KERNEL}/ggml/src"
    -I"${KERNEL}/ggml/include"
)

compile_variant() {
    local name="$1"
    local variant_mmq="${KERNEL}/variants/${name}/mmq.cuh"
    if [[ ! -f "${variant_mmq}" ]]; then
        echo "missing ${variant_mmq}" >&2
        exit 1
    fi
    cp "${variant_mmq}" "${CUDA_INC}/mmq.cuh"
    echo "==> ${name}"
    "${CLANG}" "${COMMON_FLAGS[@]}" -S -o "${ASM}/${name}.s" "${INST}"
}

for name in "${VARIANTS[@]}"; do
    compile_variant "${name}"
done

echo
echo "ASM outputs (kernel/: self-contained, no llama.cpp):"
for i in "${!VARIANTS[@]}"; do
    echo "  ${ASM}/${VARIANTS[$i]}.s  (${VARIANT_COMMITS[$i]})"
done
