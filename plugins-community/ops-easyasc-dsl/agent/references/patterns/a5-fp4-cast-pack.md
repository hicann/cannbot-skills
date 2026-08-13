# A5 Native FP4 Cast and Carrier Pack

## Applies when

Use this pattern when an A5 VF converts aligned BF16 values directly to native
unscaled FP4 E1M2/E2M1 payloads and exposes them through a packed `uint8` GM
carrier. This is a dtype conversion pattern, not MXFP4 scale selection or cube
matmul.

## Logical dataflow

```text
BF16 GM -> BF16 UB/register -> native FP4 register cast
        -> live uint8 carrier alias -> pack4 -> packed uint8 UB/GM
```

Two consecutive logical FP4 payloads share one byte; the lower-index value is
stored in the low nibble.

## Physical invariants

- use `MaskReg(DT.bfloat16)` because the cast mask follows the wider BF16
  source width;
- use `MaskMergeMode.ZEROING` and a concrete `RegLayout.ZERO/ONE/TWO/THREE`;
- `RegLayout.UNKNOWN` is illegal for the C310 BF16-to-FP4 `vcvt`;
- `RegLayout.ZERO` places carrier bytes in every fourth uint8 register slot;
- reinterpret the written FP4 Reg as a live uint8 Reg alias; a pre-cast value
  copy observes stale data;
- compact the sparse carrier with `pack4()` before UB/GM writeback;
- preserve signed zero: a negative value rounded to zero emits nibble `0x8`.

Supported BF16 -> FP4 round modes are `TO_EVEN`, `AWAY_FROM_ZERO`, `FLOOR`,
`CEIL`, and `TRUNC`. FP4 -> BF16 uses `RoundMode.NONE`.

## Minimal skeleton

```python
@vf()
def cast_bf16_to_fp4_vf(src: Tensor, dst: Tensor):
    src_reg = Reg(DT.bfloat16)
    fp4_reg = Reg(DT.fp4_e1m2)
    carrier_reg = fp4_reg.reinterpret(DT.uint8)
    mask = MaskReg(DT.bfloat16)
    cfg = CastConfig(
        round_mode=RoundMode.TO_EVEN,
        reg_layout=RegLayout.ZERO,
    )

    src_reg <<= src[0]
    cast(fp4_reg, src_reg, cfg, mask)
    dst[0] <<= carrier_reg.pack4()
```

Change the FP4 dtype and round mode only when the public contract requires it.
The public output remains `DT.uint8`; FP4 register dtype is an internal compute
view.

## Failure signatures

- all-zero or stale output carrier: reinterpret lowered as a value copy rather
  than a live reference alias;
- every fourth byte is correct but GM output is sparse: `pack4()` is missing;
- compiler rejects `vcvt` layout: `RegLayout.UNKNOWN` or an unsupported layout
  reached native codegen;
- negative values near zero differ by nibble `0x8`: signed-zero semantics were
  omitted from the reference;
- decoded values look plausible but carrier bytes mismatch: validation checked
  only dequantized numerics rather than the packed ABI.

## Runnable references

- `agent/example/kernels/a5/vec_only/bf16_to_fp4_e1m2.py`: fixed `[64,128]` BF16 to packed
  E1M2 carrier, exhaustive carrier comparison, simulator and CANN validation.
- `easyasc/dtypehelper/fp4_fp32.py`: host carrier codec and round modes.
- `agent/references/constraints/a5.md`: native cast mask/layout constraints.
- `agent/references/constraints/precision.md`: cast-order and numeric-boundary
  ownership.

## Do not use when

- the kernel needs MXFP4 e8m0 scaling or `matmul_mx`;
- the required output is dequantized BF16/FP32 rather than a packed carrier;
- the target is A2, which does not expose the A5 micro register path;
- input/tail layout is not covered by the aligned row load/store contract.

## Source escape

For another FP4 dtype, round mode, register layout, or tail shape, follow
`agent/references/evidence-escalation.md`. Inspect the cast rule, micro
simulator, generated C++ alias, and one carrier-exact probe before extending
this Pattern.
