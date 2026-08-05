---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "AscendC TilingData: struct-def / host-getter / kernel-register must all use the SAME registration style"
description: "AscendC has two mutually-exclusive TilingData registration styles; mixing the struct def, host getter, and kernel register across them makes host GetTilingData<T>() return null at aclnn tiling."
phenomenon: build_failure
signal:
  - "Host context->GetTilingData<T>() returns nullptr → \"<Op> do tiling failed, ret is -1\" (EZ9999) at aclnn tiling; opc warns \"do not registe tiling struct\". Build itself SUCCEEDS — failure is tiling-registration, not compile."
confidence: single_run
original_id: OL-268
classified_by: llm-assisted
timestamp_inferred: true
tags: [tiling, build, ol-268, tilingdata, registration-style, aclnn]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

AscendC host-side tiling registers TilingData via one of **two mutually-exclusive styles**. Three pieces must all be the same style: (1) the struct definition (location + form), (2) the host `TilingFunc` getter, (3) the kernel-side register/unpack. Mixing them makes the framework fail to recognize the struct → host `context->GetTilingData<T>()` returns `nullptr` at aclnn tiling → `<Op> do tiling failed, ret is -1` (EZ9999) and opc warns `do not registe tiling struct`. The build itself succeeds (the `.o` links fine) — this is a **tiling-registration** failure, not a compile failure.

Note: `GetTilingData<T>()` (host, `gert::TilingContext`) ≠ the `GET_TILING_DATA` macro (kernel-side, unpacks the raw tiling buffer). The host getter returning null is a HOST-side registration failure, not a kernel one.

## 根因 / 教训

Pick one style and keep all three pieces consistent.

**Style A (macro DSL)** — in `op_host/<op>_tiling.h`, `namespace optiling`:
- Struct: `BEGIN_TILING_DATA_DEF(Struct) TILING_DATA_FIELD_DEF(type,name)... END_TILING_DATA_DEF; REGISTER_TILING_DATA_CLASS(OpType, Struct)`.
- Host TilingFunc: local var + `set_field(v)` + `SaveToBuffer(context->GetRawTilingData()->GetData(), cap)` + `SetDataSize(tiling.GetDataSize())`.
- Kernel: `GET_TILING_DATA(tilingData, tiling)` (**no** `REGISTER_TILING_DEFAULT`).

**Style B (standard C++ struct)** — in `op_kernel/<op>_tiling.h`, a plain POD `struct Struct {...};`:
- Host TilingFunc: `Struct *tiling = context->GetTilingData<Struct>(); tiling->field = v;`.
- Kernel entry: `REGISTER_TILING_DEFAULT(Struct); GET_TILING_DATA(tilingData, tiling)`.

**Key correction (kw-2 真根因):** Style B **still requires** `REGISTER_TILING_DEFAULT` + the `GET_TILING_DATA` macro pairing — the macro is NOT macro-DSL-exclusive. If the kernel drops `GET_TILING_DATA` and hand-reads the buffer via a raw `(__gm__ Struct*)tiling` cast, the tiling is never unpacked and the host `GetTilingData<T>()` returns null. The correct kernel form uses `GET_TILING_DATA(tiling_data, tiling)` and then `tiling_data.<field>` directly (the `tiling_data` type is the struct registered by `REGISTER_TILING_DEFAULT` — do not cast to a `__gm__` pointer).

### Evidence

- A fused matmul-reduce scaffold on A2/910B4 with CANN 8.5.1 changed its style-B `tiling.h` into style A but left the host `TilingFunc` on `GetTilingData<>()`. The build succeeded, yet runtime tiling returned null because the three registration pieces no longer matched.
- Follow-up diagnosis found that the scaffold had treated `GET_TILING_DATA` as macro-DSL-only and replaced it with a raw `(__gm__ Struct*)tiling` read. Style B also requires `REGISTER_TILING_DEFAULT` + `GET_TILING_DATA`; restoring that pair fixed the registration contract.

### Other instances (predicted)

Any AscendC op where the `tiling.h` struct style is changed (scaffold patch, worker rewrite, cross-op copy) without updating both the host getter and the kernel register/unpack to match.
