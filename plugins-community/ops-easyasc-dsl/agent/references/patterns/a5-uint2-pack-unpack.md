# A5 Exact uint2 Carrier Pack and Unpack

## Applies when

Use this pattern for an exact unsigned 2-bit ABI whose logical BF16 values are
guaranteed to be exactly `{0,1,2,3}`. Four consecutive values are packed into
one byte in low-index-first bit order.

## Logical dataflow

Encode:

```text
BF16 -> half -> sparse uint8 lanes -> dense bytes
     -> two shift/or packing stages -> packed uint8 carrier
```

Decode:

```text
packed uint8 carrier -> one byte per uint32 lane -> two bit-spread stages
-> dense-byte UB scratch -> unpack reload -> half -> BF16
```

For values `a,b,c,d`, the carrier is
`a | (b << 2) | (c << 4) | (d << 6)`.

## Physical invariants

- the encode contract has no clamp or range check; adding one changes the
  algorithm and instruction sequence;
- BF16 -> half -> uint8 uses `RegLayout.ZERO`, leaving one value in the low
  byte of each uint16 lane;
- native `pack(..., HighLowPart.LOWEST)` compacts those sparse bytes;
- `DIST_PACK4_B32` writes the low byte from each active uint32 lane;
- decode `DIST_UNPACK4_B8` places one carrier byte in the low byte of a uint32
  lane;
- a uint8 -> half P0 cast consumes even bytes of uint16 lanes, not the first
  contiguous 128 uint8 lanes;
- therefore decode must store dense bytes to UB, issue a STORE -> LOAD local
  barrier, and reload with `DIST_UNPACK_B8` before casting;
- `DIST_NORM_B8` has a full 256-byte register write footprint even under a
  prefix mask, so decode scratch must be at least 256 bytes.

## Minimal skeleton

Encode four dense bytes per uint32 lane:

```python
pack(dense_u8, sparse_u16, HighLowPart.LOWEST)
shiftrs(shifted6, dense_u32, 6, mask=mask_u32)
vor(pairs, dense_u32, shifted6, mask=mask_u32)
shiftrs(shifted12, pairs, 12, mask=mask_u32)
vor(quads, pairs, shifted12, mask=mask_u32)
dst[0] <<= pack_mask * quads.reinterpret(DT.uint8).pack4()
```

Decode one carrier byte into four dense bytes:

```python
ub_to_reg_unpack4(carrier_u8, src[0])
spread = (carrier_u32 | (carrier_u32 << 12)) & 0x000F000F
dense_u32 = (spread | (spread << 6)) & 0x03030303
reg_to_ub_normal(scratch[0], dense_u32.reinterpret(DT.uint8), mask_u8)
vf_barrier(VfPipe.STORE, VfPipe.LOAD)
ub_to_reg_unpack(cast_src_u8, scratch[0])
cast(dense_half, cast_src_u8, u8_to_half_cfg, mask_bf16)
cast(dense_bf16, dense_half, half_to_bf16_cfg, mask_bf16)
```

The expression form above abbreviates the actual `shiftls`/`vor`/`vand`
register operations; use the runnable reference for exact DSL calls.

## Failure signatures

- only the first/even decoded values are correct: dense bytes were cast without
  the UB store/barrier/unpack reload;
- scratch footprint warning or adjacent corruption: the decode scratch was
  allocated for 128 logical bytes instead of the 256-byte register store;
- byte fields appear reversed: the public bit order was not fixed before
  writing shifts;
- encode passes simple values but not all 256 quadruples: the shift/or chain or
  active-lane mask is incomplete;
- fractional/negative inputs produce arbitrary carriers: the exact-value
  precondition was violated rather than checked by the kernel.

## Runnable references

- `agent/example/kernels/a5/vec_only/bf16_to_uint2.py`: exhaustive encode over all 256 input
  quadruples.
- `agent/example/kernels/a5/vec_only/uint2_to_bf16.py`: exhaustive inverse over all 256
  carrier bytes and the required 256-byte scratch/reload path.
- `agent/references/constraints/a5.md`: register layout, full-footprint, and
  micro memory constraints.
- `agent/references/constraints/sync.md`: VF local-memory barrier semantics.

## Do not use when

- input values are not exactly unsigned integers in `[0,3]`;
- the ABI uses signed 2-bit fields or a different field order;
- the last dimension cannot be grouped into consecutive fours without an
  explicit tail contract;
- a standard dtype cast already provides the required public representation.

## Source escape

For a new bit width, signed interpretation, field order, or tail, follow
`agent/references/evidence-escalation.md`. Prove the register lane placement and
full store footprint with a minimal exhaustive carrier probe before composing a
new pack/unpack Pattern.
