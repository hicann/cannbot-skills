---
title: DataCopy Best Practices
purpose: Record DataCopy and DataCopyPad usage as compatibility and fallback material, not as the default regbase data-movement model.
read_when:
  - You need to compare a regbase path with a queued compatibility path.
  - A fallback or contrast design needs DataCopy/DataCopyPad rules spelled out.
not_for:
  - Default VF load/store semantics
  - Main regbase kernel mental model
keywords:
  - datacopy
  - datacopypad
  - compatibility
  - fallback
next_reads:
  - pipeline_and_buffer.md
  - regbase_api_whitelist.md
  - ../pitfalls/api_misuse.md
depth: intermediate
topic_type: api
---

# DataCopy Best Practices

This card records GM and UB movement rules as compatibility and contrast material for regbase direct-invoke work. Use it when you need to compare against membase-style buffer handling or document a non-regbase fallback path.

## Selection Rules

When you are documenting or comparing a queued compatibility path, prefer `DataCopyPad` when the alignment is uncertain, tile sizes are irregular, or the design may need copy-in and copy-out padding symmetry.

Use plain `DataCopy` only when the data layout is already known to be safe and aligned for the exact transfer you are making in that compatibility path.

`SetValue` and `GetValue` are debugging aids, not production data movement APIs. They are too slow for normal kernel code and should not be treated as the regbase default path.

## Alignment Rules

- `DataCopy` expects the transfer to respect the platform alignment requirements.
- In compatibility or fallback notes, choose `DataCopyPad` when the layout is not obviously safe and make the padding explicit.
- Do not mix a padded copy-in with an unpadded copy-out unless you have proved the downstream shape is still aligned and correct.

## Padding Behavior

`DataCopyPad` can either let the framework manage padding or use an explicit pad value in a compatibility path:

- `isPad = false`: framework-controlled padding behavior
- `isPad = true`: use the provided `paddingValue`

Treat the copy-in and copy-out sides as a pair. If the input side needs padding, the output side usually needs the same discipline.

## Stride And Layout

Keep the stride unit straight:

- GM side stride is measured in bytes.
- UB side stride is measured in 32-byte blocks.

That difference is a common source of silent offset bugs, especially when copy loops are generated from tiling data.

## Tiling And Limits

- Keep `blockCount` within the documented limit.
- Prefer batch copy when multiple rows share the same row length.
- Size the tile so the buffer shape and the copy shape agree.
- Handle the tail tile explicitly instead of assuming the last batch is regular.

## Debug Sequence

When copy output looks wrong:

1. Verify copy-in alone.
2. Verify copy-out alone.
3. Check whether the requested size is actually aligned.
4. Confirm the same padding rule is used on both ends.
5. Re-check the stride unit on the GM and UB sides.

## Practical Rule Of Thumb

- Unknown alignment in a compatibility path: use `DataCopyPad`.
- Regular and aligned in a compatibility path: `DataCopy` is acceptable.
- Debug print or single-element inspection: `GetValue` only.

## Related Documents

- [[regbase_api_reference]]
- [[regbase_api_sync]]
- [[pipeline_and_buffer]]
- [[precision_and_runtime]]
