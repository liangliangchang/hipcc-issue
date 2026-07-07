#!/usr/bin/env python3
"""Compare hot-path AMDGPU schedule for stripped / split / fix MMQ builds."""

from __future__ import annotations

import re
import sys
from pathlib import Path

KERNEL_TAG = "ELi128ELb0"

VARIANTS = ("stripped", "split", "fix")


def asm_kernel(path: Path) -> list[str]:
    lines = path.read_text(errors="ignore").splitlines()
    start = next(i for i, l in enumerate(lines) if "globl" in l and KERNEL_TAG in l)
    end = next(
        i
        for i in range(start + 1, len(lines))
        if "globl" in lines[i] and KERNEL_TAG not in lines[i] and lines[i].strip().startswith(".globl")
    )
    return lines[start:end]


def asm_schedule(body: list[str]) -> dict:
    text = "\n".join(body)
    waits = [int(x) for x in re.findall(r"s_waitcnt lgkmcnt\((\d+)\)", text)]
    return {
        "wmma": text.count("v_wmma_i32_16x16x16_iu8"),
        "b128": text.count("ds_load_b128"),
        "addr2": text.count("ds_load_2addr"),
        "fma_mix": text.count("v_fma_mix_f32"),
        "barrier": text.count("s_barrier"),
        "lgk_max": max(waits) if waits else 0,
        "lgk_avg": round(sum(waits) / len(waits), 1) if waits else 0.0,
    }


def asm_split_hot(body: list[str]) -> list[str]:
    """Executed hot path in split build (duplicate loop with good schedule)."""
    best = None
    for i, line in enumerate(body):
        if not (line.strip().endswith(":") and ".LBB" in line):
            continue
        m = asm_schedule(body[i:])
        if 28 <= m["wmma"] <= 36 and m["b128"] >= 20:
            key = (m["lgk_max"], m["barrier"], -m["b128"])
            if best is None or key < best[0]:
                best = (key, i)
    if best is not None:
        return body[best[1] :]
    wmma = [i for i, l in enumerate(body) if "v_wmma_i32_16x16x16_iu8" in l]
    return body[wmma[len(wmma) // 2] :] if wmma else body


def asm_hot(body: list[str], variant: str) -> list[str]:
    if variant == "split":
        return asm_split_hot(body)
    return body


def opcode_preview(body: list[str], limit: int = 96) -> str:
    """Compact schedule: d=ds_load_b128 W=WMMA F=fma_mix b=s_barrier."""
    out: list[str] = []
    for line in body:
        if "ds_load_b128" in line:
            out.append("d")
        elif "v_wmma_i32_16x16x16_iu8" in line:
            out.append("W")
        elif "v_fma_mix_f32" in line:
            out.append("F")
        elif "s_barrier" in line:
            out.append("b")
        if len(out) >= limit:
            break
    # collapse runs for readability
    if not out:
        return ""
    collapsed = [out[0]]
    for ch in out[1:]:
        if ch != collapsed[-1]:
            collapsed.append(ch)
    return "".join(collapsed)


def gap_before_wmma(body: list[str]) -> int:
    """Instructions between last ds_load_b128 and first WMMA in hot path."""
    last_d = None
    first_w = None
    for i, line in enumerate(body):
        if "ds_load_b128" in line:
            last_d = i
        if first_w is None and "v_wmma_i32_16x16x16_iu8" in line:
            first_w = i
    if last_d is None or first_w is None or first_w <= last_d:
        return -1
    gap = body[last_d + 1 : first_w]
    return sum(
        1
        for l in gap
        if l.strip()
        and not l.strip().endswith(":")
        and not l.strip().startswith(".")
        and not l.strip().startswith("//")
    )


def fmt_asm(m: dict) -> str:
    return (
        f"wmma={m['wmma']:>2}  b128={m['b128']:>2}  2addr={m['addr2']:>2}  "
        f"fma_mix={m['fma_mix']:>3}  bar={m['barrier']:>2}  "
        f"lgk_max={m['lgk_max']:>2}  lgk_avg={m['lgk_avg']}"
    )


def compare(asm_dir: Path, out_dir: Path | None) -> int:
    metrics: dict[str, dict] = {}
    hots: dict[str, list[str]] = {}

    for name in VARIANTS:
        path = asm_dir / f"{name}.s"
        if not path.is_file():
            print(f"missing {path} — run ./build.sh first", file=sys.stderr)
            return 2
        body = asm_kernel(path)
        hot = asm_hot(body, name)
        hots[name] = hot
        metrics[name] = asm_schedule(hot)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, hot in hots.items():
            (out_dir / f"{name}_hot.s").write_text("\n".join(hot) + "\n")

    print("Hot-path AMDGPU schedule  (Q4_K mmq_x=128, gfx1151, -O3)\n")
    for name in VARIANTS:
        m = metrics[name]
        gap = gap_before_wmma(hots[name])
        gap_s = str(gap) if gap >= 0 else "n/a"
        print(f"  {name:8s}  {fmt_asm(m)}  gap_d→W={gap_s}")

    print("\nOpcode preview (d=ds_load_b128  W=WMMA  F=fma_mix  b=s_barrier, runs collapsed):")
    for name in VARIANTS:
        prev = opcode_preview(hots[name])
        print(f"  {name:8s}  {prev[:120]}{'…' if len(prev) > 120 else ''}")

    stripped, split, fix = metrics["stripped"], metrics["split"], metrics["fix"]

    checks = [
        ("same WMMA count on hot path", split["wmma"] == stripped["wmma"] == fix["wmma"]),
        ("stripped has worse lgkmcnt than split", stripped["lgk_max"] > split["lgk_max"]),
        ("fix matches or beats split lgkmcnt", fix["lgk_max"] <= split["lgk_max"] + 1),
        ("stripped interleaves FMA before WMMA cluster", "F" in opcode_preview(hots["stripped"])[:20]),
        ("split clusters DS before WMMA", opcode_preview(hots["split"]).find("W") > opcode_preview(hots["split"]).find("d")),
        ("fix clusters DS before WMMA", opcode_preview(hots["fix"]).find("W") > opcode_preview(hots["fix"]).find("d")),
    ]
    print("\nChecks:")
    ok = True
    for msg, cond in checks:
        if not cond:
            ok = False
        print(f"  [{'OK' if cond else 'FAIL'}] {msg}")

    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    root = Path(__file__).resolve().parent
    asm_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "build" / "asm"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "build" / "hot"
    return compare(asm_dir, out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
