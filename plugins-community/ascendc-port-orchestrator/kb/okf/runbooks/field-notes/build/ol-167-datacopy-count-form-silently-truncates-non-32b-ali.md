---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "DataCopy `count` form silently truncates non-32B-aligned transfers — never paper over with host-side pad+narrow in pybind [V351+V220, ALL_MODES, data-movement-correctness + anti-cheat]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=all"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-167
timestamp_inferred: true
tags: [507035, count, ascendc, ol-167]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=all`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5 (2026-05-18 task #22 hardware probe — workspace/probe_datacopypad_v300_tail/PROBE_REPORT.md, blockLen ∈ {31,33,47,63} all wrote exactly N bytes, 0xCAFE sentinel intact at byte N, no runtime error)`
`unverified_on: soc=Ascend910_V220 (A3 family — semantic is documented same per CANN spec, but bit-level repro not yet replayed on V220 in this session)`

### Scope clarification (2026-05-18 task #22 hardware probe)

EC-23 (V220 `DataCopyPad` UB→GM crash → must use plain DataCopy + manual tail pad) is **V220-SPECIFIC** and does NOT apply to V351/Ascend950PR. Empirically verified via aog-hardware-probe 2026-05-18 (NPU 4, npu_dev3 container, CANN 9.0.0 + bisheng 15.0.5):

- blockLen=31 → 31 bytes written cleanly, byte 31 still 0xCA (sentinel high byte)
- blockLen=33 → 33 bytes written cleanly, byte 33 still 0xCA
- blockLen=47 → 47 bytes written cleanly, byte 47 still 0xCA
- blockLen=63 → 63 bytes written cleanly, byte 63 still 0xCA
- NPU 4 utilization 0% / no runtime error / no 507035 / no EZ9999 across all variants

This resolves the V351 foreach-class `pybind narrow+contiguous` workaround tension that surfaced on foreach_neg 2026-05-18 (task #22). On V351, kernel MUST use DataCopyPad per P-P98 for non-aligned tails; pybind pad+narrow is the cheat OL-167 was always trying to prevent. EC-23 only applies if you target V220.

### Principle

The `DataCopy(dst, src, count)` overload requires `count * sizeof(T) % 32 == 0`. When the product is NOT 32-byte aligned, the runtime rounds **DOWN** to the nearest 32B boundary — silently. No error, no warning, the kernel just transfers fewer bytes than the caller asked for. Tail bytes in the destination retain pre-existing garbage. For non-aligned tails, `DataCopyPad` + `DataCopyExtParams` (byte-level `blockLen`) is the only correct primitive — see P-P98.

The kernel author MUST handle non-aligned tail directly in kernel space. Allocating an over-sized padded output buffer in pybind, running the kernel into it, then trimming with `padded_out.narrow(dim, 0, real_len).contiguous()` is **NOT a fix — it is host-side business logic masquerading as kernel correctness**. The padded extra bytes are kernel-written garbage; the narrow+contiguous extracts the valid prefix but the kernel never proved it can handle the real-shape write. The pybind layer's job is metadata + dispatch, not data-path repair. See P149 retraction precedent.

### Concrete anchor

```cpp
// WRONG — silent truncation:
DataCopy(ub, gm, 10);   // half × 10 = 20B → rounded down to 0B → ZERO BYTES COPIED, no error

// CORRECT — DataCopyPad for non-aligned byte count:
DataCopyExtParams cp{1, /*blockLen_bytes=*/20, 0, 0, 0};
DataCopyPadExtParams<half> pad{true, 0, /*rightPad_elems=*/6, half(0)};  // pad to 32B (20 + 12)
DataCopyPad(ub, gm, cp, pad);
```

For `UB→GM` non-aligned writes the pad params are omitted — framework reads aligned-up block from UB then writes exactly `blockLen` bytes to GM:

```cpp
DataCopyExtParams cp{1, byteLen, 0, 0, 0};
DataCopyPad(gm, ub, cp);   // GM receives exactly byteLen bytes; no pad needed downstream
```

### Anti-pattern (host-side narrow+contiguous CHEAT)

These three archives shipped the same shape and all became P149 retraction targets:

```cpp
// 11_DequantSwigluQuant_v3.2_cold/kernel/pybind11.cpp:213-259 (npukernelbench)
int64_t row_stride_bytes = align_up64(out_row_bytes, 32);
auto padded_out = torch::empty({N, row_stride_bytes}, opts_i8);   // ← over-allocated for kernel comfort
kernel_launch(padded_out, ...);
auto quant_trimmed = padded_out.narrow(1, 0, out_row_bytes).contiguous();   // ← host-side trim hides kernel limitation

// elu port_a3 pybind11.cpp:52,69,100
constexpr int64_t TAIL_PAD_ELEMS = 32;
auto raw = torch::empty({numel + TAIL_PAD_ELEMS}, ...);   // ← same shape
kernel_launch(raw, ...);
return raw.narrow(0, 0, numel).view(x_c.sizes());   // ← same trim

// clipped_swiglu port_a3 pybind11.cpp:108
auto gi_cpu = group_index->to(at::kCPU).to(at::kLong).contiguous();   // ← CPU offload of "host business logic"
```

All three pass per-case precision verification because the trimmed prefix matches reference numerics; they fail honesty because the kernel doesn't actually handle the real-shape write. P149 finalize gate catches this pattern at archive time, but the gate is post-hoc — the right place to prevent it is at kernel-author time.

### Evidence

- 11_DequantSwigluQuant_v3.2_cold (npukernelbench, 2026-04-x archive): padded_out + narrow shape; precision passed verification, retracted under P149 audit 2026-05-18.
- elu (a3_to_a5_port, `509b512a`): TAIL_PAD_ELEMS + raw.narrow pattern; retracted under P149 2026-05-18 (commit 0fd72bff handover Appendix B).
- clipped_swiglu (a3_to_a5_port): CPU offload of group_index; retracted under P149 2026-05-18.
- User direction 2026-05-18 10:39Z: "the 'narrow(1, 0, out_row_bytes).contiguous()' in pybind11.cpp is business logics which should be placed into kernel. To do that you need to understand how to do data copy properly". Specifies the cheat must move OUT of pybind INTO kernel via DataCopyPad.

### Other instances (predicted)

- Any kernel whose output last-dim is dtype-dependent (int8 × any H → H bytes/row, often not multiple of 32) and the worker writes an over-allocated buffer + trims in pybind.
- Quantization kernels (int8 outputs make trailing dim small-byte → non-aligned).
- Sparse output ops (NNZ varies per call; over-allocating to max + trimming via NNZ scalar is the same anti-pattern).
- Any ML op with variable-length per-row payload (variable-length lists in foreach families, variable-length scatter tails).

### Mitigation (decision rule — applies to ANY non-aligned kernel write)

1. Inside the kernel, compute the tail byte length: `tailBytes = realLen * sizeof(T)` where `realLen` is the not-32B-multiple count.
2. Use `DataCopyExtParams{1, tailBytes, 0, 0, 0}` + `DataCopyPad(gm, ub, cp)` for the tail write.
3. Allocate the GM output at exactly the real shape in pybind (NO padded over-allocation, NO narrow+contiguous after the kernel call).
4. If the upstream reference uses an aligned-padded intermediate that's truly invariant of the kernel-write boundary (e.g., the reference itself produces a padded layout that's part of the public contract), document this as `OL-162` style pybind padding wrapper — that case is conceptually different (reference behavior, not kernel data-path cheating).

### Cross-references

- P-P98 — DataCopyPad UB→GM byte-level non-aligned tail-write pattern (the implementation form).
- P149 finalize gate (`finalize_pipeline.py`) — currently the only mechanical enforcement; catches narrow+contiguous in pybind11.cpp at archive promotion.
- EC-23 — DataCopyPad UB→GM crash mitigation (V220 build-side gotcha); orthogonal concern.
- `~/workspace/a5/tasks/datacopy_api.md` — full DataCopy/DataCopyPad API reference learned into KB 2026-05-18.
- OL-162 — Pybind-side padding wrapper for upstream-imposed shape contracts (legitimate use); not a cheat — distinct from the anti-pattern above. Read both entries together to decide which applies.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-167（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
