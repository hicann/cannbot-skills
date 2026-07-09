# API Mapping Schema

`api-mapping/device_api.yaml` and `api-mapping/runtime_api.yaml` are the source of truth for CUDA-to-Ascend API mapping data.

The Markdown files remain as human-readable browse views. Update YAML first. If the Markdown view is kept, regenerate or sync it afterward.

## Record Structure

Each API mapping record uses this shape:

```yaml
- cuda_api: cudaMalloc
  category: runtime/memory-management
  status: mapped
  mapping_type: direct
  ascend_api: aclrtMalloc
  ascend_signature: "aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy)"
  action: direct_replace
  note: ""
  fallback: ""
  source:
    - reference/api-mapping/runtime_api.md
  reviewed_by: human
  reviewed_at: "2026-04-06"
```

## Field Definitions

- `cuda_api`
  - Required.
  - CUDA API name.
  - Unique within the file.

- `category`
  - Required.
  - For `api-mapping/device_api.yaml`, use `device/<subcategory-kebab-case>`.
  - The `<subcategory-kebab-case>` part should match the corresponding section name in `api-mapping/device_api.md`.
  - Example: section `math` maps to category `device/math`.
  - Example values: `device/math`, `device/warp`, `device/type-conversion`.
  - For `api-mapping/runtime_api.yaml`, use `runtime/<subcategory-kebab-case>`.
  - The subcategory should match the corresponding section in `api-mapping/runtime_api.md`.
  - Example values: `runtime/device-management`, `runtime/stream-management`, `runtime/memory-management`.

- `status`
  - Required.
  - One of:
    - `mapped`: confirmed usable mapping exists
    - `partial`: partially mapped, semantics or coverage still need care
    - `unsupported`: confirmed unsupported in the current migration model
    - `unknown`: not investigated yet

- `mapping_type`
  - Required.
  - One of:
    - `direct`: direct API replacement is the main path
    - `semantic`: similar purpose, but not a strict one-to-one replacement
    - `manual-rewrite`: migrate by rewriting logic instead of replacing the API directly
    - `none`: no mapping path currently recorded

- `ascend_api`
  - Required, may be empty.
  - Ascend-side API name when known.

- `ascend_signature`
  - Required, may be empty.
  - Function signature or key callable form.

- `action`
  - Required.
  - One of:
    - `direct_replace`
    - `check_docs`
    - `manual_implementation`
    - `stop_and_report`

- `note`
  - Required, may be empty.
  - Short explanation for ambiguity, caveats, or current investigation state.

- `fallback`
  - Required, may be empty.
  - What to do if there is no direct mapping.

- `source`
  - Required.
  - List of source references used to justify the entry.

- `reviewed_by`
  - Required, may be empty.
  - `human`, `agent`, or a more specific reviewer label.

- `reviewed_at`
  - Required, may be empty.
  - ISO date string.

## Status and Mapping Rules

- If `status` is `mapped`, `mapping_type` should normally be `direct`, `semantic`, or `manual-rewrite`.
- If `status` is `partial`, `mapping_type` should normally be `semantic` or `manual-rewrite`.
- If `status` is `unsupported`, `mapping_type` should be `none` and `action` should be `stop_and_report`.
- If `status` is `unknown`, `mapping_type` should be `none` and `action` should usually be `check_docs`.

## Maintenance Rules

- Do not use blank rows to mean “not investigated”. Use `status: unknown`.
- Do not change `unknown` to `mapped` without adding at least one concrete source and a reviewed date.
- When only part of the CUDA API can be represented on Ascend, use `status: partial`, not `mapped`.
- When there is no supported migration path in the current skill scope, use `status: unsupported`.
- If the mapping is uncertain, keep `status: partial` or `unknown`; do not guess.

## Agent Update Workflow

When an agent improves the mapping references:

1. Select one API or one small category at a time.
2. Read official local references.
3. Update the YAML record first.
4. Add or revise `note`, `action`, and `source`.
5. Leave the record as `unknown` if the evidence is insufficient.

## Bootstrap Note

The current YAML files were bootstrapped from the existing Markdown tables.

That means:

- rows with no mapping were converted to `status: unknown`
- rows with a mapped Ascend API were converted conservatively
- some entries may still need human or agent refinement
