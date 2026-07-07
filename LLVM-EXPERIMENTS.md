# LLVM investigation summary

Notes from bisecting the RDNA3.5 (`gfx1151`) MMQ codegen regression. The forked LLVM work described here was **explored and removed**; the shipped fix is the source-level `MMQ_HIP_TILE_BARRIER()` workaround in `kernel/variants/fix/mmq.cuh`.

Fork tree (no longer in use): `/proj/gdba/lichang/llvm-project` @ ROCm base `5c9bfa94`.

---

## Problem statement

Removing a duplicate “cold” K-loop from `mul_mat_q_process_tile` (commit `55945ef57`, **stripped**) caused ~12% `pp128` throughput loss on gfx1151, even though only the hot loop runs at launch (`blockDim.z == 1`, cold predicate always false).

The hot path still executes the same WMMA count; the regression is **instruction schedule**, not missing math.

| Hot-path metric | stripped | split (cold branch) | fix (tile barriers) |
|-----------------|----------|---------------------|---------------------|
| WMMA | 32 | 32 | 32 |
| `lgk_max` | **11** | **3** | **4** |
| pp128 tok/s (Qwen2.5-0.5B Q4_K_M) | ~9.4k | ~10.7k | ~10.8k |

(`lgk_max` = max `N` in `s_waitcnt lgkmcnt(N)` on the hot path — see `compare_hot.py`.)

---

## Why cold split and stripped differ

### Observable ASM difference

On the **first** DS→WMMA cluster in the hot K-loop:

- **stripped:** `… d F W …` — `v_fma_mix` between `ds_load_b128` and WMMA
- **split / fix:** `… d W F …` — LDS cluster adjacent to WMMA

Stripped forces `SIInsertWaitcnts` to keep many LDS operations in flight while VALU (FMA) runs → deep `lgkmcnt(11)`. Split/fix get shallow `lgkmcnt(3–4)`.

### Cause chain (confirmed by MIR bisect)

```
source: remove cold/hot duplicate (stripped)
   ↓
IR: vec_dot inlined once (4 llvm.amdgcn.wmma in kernel) vs twice in split (8)
   ↓
pre-RA machine scheduler: hot MIR still matches split (similarity ≈ 1.0)
   ↓
post-RA machine scheduler (pass: postmisched): hot MIR diverges (similarity ≈ 0.59)
   ↓   stripped: prefix FMA_MIX before WMMA in scheduling region
   ↓   split:    WMMA before interleaved prefix FMA_MIX
   ↓
SIInsertWaitcnts, si-pre-emit-peephole: inherit bad order → lgkmcnt(11)
```

**First diverging pass:** `postmisched` (`GCNPostScheduleDAGMILive` + `PostGenericScheduler`).

- `stop-before postmisched`: split-hot vs stripped-hot MIR similarity **1.000**
- `stop-after postmisched`: similarity **≈ 0.587**

Passes **before** `postmisched` (first `machine-scheduler`, `si-load-store-opt`, `si-form-memory-clauses`, `si-fold-operands`) do **not** diverge the hot region. Passes **after** (`si-insert-waitcnts`, etc.) amplify the damage but are not the root scheduler bug.

### MIR smoking gun (hot loop, pre-postmisched)

Instruction order in the post-RA scheduling region:

| Build | `first_wmma` | `first_fma_mix` | Order |
|-------|--------------|-----------------|-------|
| stripped | 50 | 48 | **F before W** |
| split | 48 | 56 | **W before F** |

The two prefix `FMA_MIX` ops before WMMA on stripped use `$vgpr207/208` (scale temps), not WMMA outputs — they are reorderable in principle, but **`postmisched` does not repair** `first_fma < first_wmma` once that order is established.

### Why the cold branch fixes it (without running cold code)

The cold duplicate is not a runtime fix; it changes **compiler context**:

1. **CFG:** extra basic blocks from `if (threadIdx.z > 0)` split the function’s scheduling regions.
2. **IR duplication:** two full K-loop bodies → 8 WMMA intrinsics in the kernel instead of 4.
3. **Extra `__syncthreads()`** in the cold path → additional post-RA scheduling boundaries (`s_barrier` in ASM).

Together these isolate the DS-load region from the WMMA+FMA region on the **hot** path, even though only hot executes.

---

## Stock ROCm flags / source tricks tried (no fork)

| Experiment | Result |
|------------|--------|
| `-mllvm -amdgpu-unroll-threshold-local=N` sweep (400–2000) on stripped | No improvement; N=2000 over-unrolls |
| `-mllvm -amdgpu-sched-strategy=max-memory-clause` | No improvement |
| `-mllvm -enable-post-misched=false` on stripped | Still `lgk_max=11` on hot path |
| `__attribute__((always_inline))`, `flatten`, compiler fences | Bad schedule unchanged |
| Separate TU / minimal cold sync-only branch variants | Bad schedule unchanged |
| `__builtin_amdgcn_sched_barrier` / `sched_group_barrier` patches in stripped `mmq.cuh` | Did not recover split-class `lgk_max` |
| Synthetic mini-kernel repro (scaled WMMA+LDS loop) | Did **not** reproduce regression; real `mmq-instance-q4_k.cu` required |

**Conclusion:** frontend flags and light source intrinsics are insufficient; the bug needs either CFG/barrier region boundaries (cold split, tile barriers) or a backend scheduler fix.

---

## Fork LLVM experiments

We patched ROCm LLVM (`llvm/lib/Target/AMDGPU/`) with MMQ-specific hooks. All of this was later **reverted** from the fork; summary below is for upstream reporting.

### Infrastructure added

| Component | Purpose |
|-----------|---------|
| `AMDGPUMMQSchedLog` | `-amdgpu-mmq-sched-log` — region stats, opcode previews, pre/post `postmisched` deltas |
| `AMDGPUMMQSchedHack` | Test-only scheduler DAG mutations behind `-amdgpu-mmq-sched-hack-*` flags |
| `AMDGPUMMQRegionSplit` | Extra post-RA scheduling boundaries in `SIInstrInfo::isSchedulingBoundary()` |
| `AMDGPUMMQReorder` | Pre-`postmisched` MachineFunction pass — move WMMA cluster before prefix FMA when deps allow |

Integrated in `AMDGPUTargetMachine.cpp` (pre-RA mutations + `addPreSched2` pass) and `GCNSchedStrategy.cpp` (logging around `GCNPostScheduleDAGMILive::schedule()`).

### Scheduler hacks (`-amdgpu-mmq-sched-hack-*`)

| Flag | Intent | Hot-path `lgk_max` (stripped MIR→ASM) |
|------|--------|--------------------------------------|
| `defer-fma` | Push `FMA_MIX` later in the DAG | 10–11 (still bad) |
| `order-dswmmafma` | Enforce DS→WMMA→FMA dependency edges | 10–11 |
| `force-iglp` | Force IGLP-only post-RA mutations | 10–11 |
| `split-s-barrier` | Treat each `s_barrier` as scheduling boundary | marginal |

Hacks could flip `wmma_before_first_fma` in **logs** but did not reliably produce split-class ASM in isolation.

### Region split (`-amdgpu-mmq-sched-split-wmma-after-ds`)

Split post-RA scheduling regions **before the first WMMA that follows DS reads** in a block (mimics cold/hot CFG isolation).

| Variant | `lgk_max` | Notes |
|---------|-----------|-------|
| Split at **WMMA-after-FMA** (first attempt) | **11** | **Harmful** — cemented FMA prefix region |
| Split at **WMMA-after-DS** (second attempt) | improved in MIR harness | Better region boundaries |

### Reorder pass (`-amdgpu-mmq-reorder-dswmmafma`)

Conservative pre-`postmisched` MIR pass: if prefix `FMA_MIX` ops are dependency-safe, physically move WMMA cluster before them.

- Aggressive full-BB topological reorder → **MIR verifier failure** (353 errors); abandoned.
- Conservative WMMA-before-prefix-FMA only → valid MIR.

### MIR round-trip harness (what *did* work)

Pipeline used for LLVM-side validation (scripts since removed from this repo):

1. Stock ROCm `clang++ -mllvm -stop-before=postmisched` → stripped MIR
2. Fork `llc -start-before=postmisched` with fork flags → ASM
3. `compare_hot.py` on output

| Build | `lgk_max` | Notes |
|-------|-----------|-------|
| stripped (stock clang ASM) | 11 | baseline |
| tile barriers only (stock clang) | **4** | source fix, no fork |
| stripped + fork reorder + region-split (MIR→`llc`) | **2** | best ASM in harness |
| split cold branch (stock clang) | 3 | reference |

The fork could beat split **in the MIR harness** when reorder + WMMA-after-DS region split were both enabled.

---

## Why LLVM fixes did not ship end-to-end

| Blocker | Detail |
|---------|--------|
| **Full compile via fork `clang++` failed** | `HIP_CLANG_PATH` → fork `clang++` hit ROCm include/stdlib path issues; never got a reliable hipcc + fork production build |
| **MIR harness ≠ production pipeline** | Fixing stripped MIR at `postmisched` with `llc` skips earlier fork integration; validating every kernel instance that way is fragile |
| **Hacks incomplete alone** | Pre-RA mutations fixed log metrics (`wmma_before_first_fma`) but hot ASM often stayed at `lgk_max` 10–11 without region split + reorder together |
| **Region split placement sensitive** | WMMA-after-**FMA** split actively hurt; only WMMA-after-**DS** split matched cold CFG intent |
| **Source fix simpler and faster** | `MMQ_HIP_TILE_BARRIER()` on HIP+RDNA3.5 alone → `lgk_max=4`, ~10.8k tok/s, no dead branch, stock ROCm clang |
| **Fork maintenance cost** | MMQ-specific passes in private LLVM fork are hard to upstream as-is; source workaround is mergeable to llama.cpp immediately |

**End-to-end winner:** source-level tile barriers (`kernel/variants/fix/mmq.cuh`), not the LLVM patch bundle.

---

## Upstream ask (AMD/ROCm LLVM)

**Title:** Post-RA GCN scheduler emits worse DS→WMMA order when duplicate cold loop removed (`gfx1151`)

**Minimal repro:** this repo — `./build.sh && ./compare.sh`

**Ask:** `GCNPostScheduleDAGMILive` / `postmisched` should emit the same DS→WMMA cluster order for the hot `vec_dot` loop whether or not a provably-dead duplicate loop (or equivalent `s_barrier` boundaries) exists in the same kernel. Alternatively, `SIInsertWaitcnts` should compute minimal `lgkmcnt` for the stripped ordering without penalizing throughput.

**Key files in upstream LLVM:**

| File | Role |
|------|------|
| `llvm/lib/CodeGen/MachineScheduler.cpp` | `postmisched` pass entry |
| `llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp` | `GCNMaxOccupancySchedStrategy` post-RA heuristics |
| `llvm/lib/Target/AMDGPU/SIInstrInfo.cpp` | `isSchedulingBoundary()` — models `s_barrier` / region splits |
| `llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp` | Inserts `s_waitcnt lgkmcnt(N)` from final order |

---

## Timeline / outcome

1. Identified regression after removing `MMQ_CODEGEN_SPLIT_COLD` (`threadIdx.z > 0`).
2. Bisected to `postmisched` as first diverging pass; characterized F-before-W vs W-before-F MIR order.
3. Built fork LLVM with logging, hacks, region split, reorder pass; validated via MIR→`llc` harness.
4. Shipped **source fix** (`MMQ_HIP_TILE_BARRIER`) — beats split in llama-bench (~10.8k vs ~10.7k pp128).
5. **Removed all fork LLVM changes** — not needed for production; this doc preserves the investigation record.
