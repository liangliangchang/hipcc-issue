# HIP MMQ codegen reproducer (RDNA3.5 / gfx1151)

Minimal repro for an AMDGPU **post-RA scheduling** regression in llama.cpp `mul_mat_q` when a duplicate cold K-loop was removed.

**Self-contained:** everything needed to compile lives under `kernel/`. You only need ROCm (`hipcc`/`clang++`) — **no llama.cpp checkout**.

## Layout

```
hipcc-issue/
├── kernel/                          # vendored compile inputs (~1.3 MB)
│   ├── ggml/include/                # ggml.h, ggml-cuda.h, …
│   ├── ggml/src/ggml-common.h
│   ├── ggml/src/ggml-impl.h
│   ├── ggml/src/ggml-cuda/
│   │   ├── common.cuh, mma.cuh, vecdotq.cuh
│   │   ├── vendors/hip.h
│   │   └── template-instances/mmq-instance-q4_k.cu
│   └── variants/
│       ├── stripped/mmq.cuh         # @ 55945ef57
│       ├── split/mmq.cuh            # @ 1c862d51f (cold branch)
│       └── fix/mmq.cuh              # MMQ_HIP_TILE_BARRIER
├── build.sh
├── compare.sh
├── compare_hot.py
└── vendor_from_llama.sh             # optional: refresh kernel/ from llama.cpp
```

## Three source variants

| Variant | `mmq.cuh` | What it does |
|---------|-----------|--------------|
| **stripped** | `55945ef57` | Single hot K-loop — **bad** schedule (`lgk_max≈11`) |
| **split** | `1c862d51f` | Cold + hot duplicate via `if (threadIdx.z > 0)` — **good** (`lgk_max≈3`) |
| **fix** | vendored fix | Single loop + `MMQ_HIP_TILE_BARRIER()` — **good** (`lgk_max≈4`) |

All compile `Q4_K` / `mmq_x=128` via `mmq-instance-q4_k.cu`.

## Quick start

```bash
cd /proj/gdba/lichang/hipcc-issue
export ROCM=/proj/gdba/lichang/rocm   # only external dependency

./build.sh    # → build/asm/{stripped,split,fix}.s
./compare.sh  # hot-path metrics + PASS/FAIL
```

Outputs:

- `build/asm/*.s` — full AMDGPU assembly from hipcc
- `build/hot/*_hot.s` — extracted hot-path regions

## Do we need llama.cpp?

**No**, for building and comparing ASM. The repro vendors:

- `mmq.cuh` (three pinned variants)
- Supporting CUDA/HIP headers: `common.cuh`, `mma.cuh`, `vecdotq.cuh`, `vendors/hip.h`
- ggml type/API headers: `ggml.h`, `ggml-common.h`, etc.

These are header-only for this compile — no llama binary, no CMake, no model weights.

To refresh vendored files after upstream changes:

```bash
LLAMA=/path/to/llama.cpp ./vendor_from_llama.sh
```

## What hipcc generates (expected on gfx1151)

| | stripped | split | fix |
|--|----------|-------|-----|
| WMMA | 32 | 32 | 32 |
| `lgk_max` | **11** | **3** | **4** |
| `s_barrier` | 6 | 4 | 9 |

Same math; the regression is **instruction order** (DS / WMMA / FMA_MIX), not missing WMMA.

Opcode preview (`d`=LDS `W`=WMMA `F`=fma_mix `b`=barrier):

- **stripped:** `bdFWF` — FMA before WMMA cluster
- **split / fix:** `bdWF…` — LDS adjacent to WMMA

## Why the workarounds work

**Cold split:** duplicate K-loop + extra `__syncthreads()` in dead branch → extra CFG/scheduling regions → post-RA scheduler emits DS→WMMA→FMA on hot path.

**Tile barriers:** `MMQ_HIP_TILE_BARRIER()` adds `s_barrier` scheduling boundaries on HIP+RDNA3.5 only — same effect without dead code.

## Root cause

First diverging LLVM pass: `postmisched`. Stripped hot MIR has prefix FMA before WMMA; split/fix cluster DS before WMMA. `SIInsertWaitcnts` then emits deep `lgkmcnt` on stripped.

## Environment

| Variable | Default | Required |
|----------|---------|----------|
| `ROCM` | `/proj/gdba/lichang/rocm` | yes |
| `ARCH` | `gfx1151` | no |
| `LLAMA` | — | only for `vendor_from_llama.sh` |
