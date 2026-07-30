# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Low-precision input materialization (owner direction 2026-06-09: build the capability for
fp8 / mxfp8 / fp4 / mxfp4 — not just fp8, not GAP).

This is the case_gen-side MATERIALIZATION primitive: given a float32 tensor, produce the
low-precision representation an op consumes. Two families:

  * Per-tensor scaled fp8 (FA's low-precision per flash_attention_score_proto.h: q/k/v in
    FP8_E4M3/E5M2 + d_scale_q/k/v FP32 dequant). `materialize_fp8` → (fp8_tensor, d_scale).
  * OCP Microscaling MX (matmul-class low-precision; CANN 量化介绍.md: "FLOAT8_E8M0 + group
    size 32"). A block of 32 consecutive elements shares one power-of-2 (E8M0) scale; elements
    are FP8 (mxfp8) or FP4-E2M1 (mxfp4). `materialize_mx` → (elements, e8m0_exponents).

Grounded facts (local probe + CANN docs, 2026-06-09):
  - torch has float8_e4m3fn / float8_e5m2 (+uz) and float4_e2m1fn_x2 (packed, 2/byte).
  - MX group size = 32; scale dtype = FLOAT8_E8M0 (8-bit exponent-only, value 2^(e-127)).
  - Element-format maxima: e4m3=448, e5m2=57344, e2m1(FP4)=6.0.

Correctness is asserted by round-trip tests (materialize → dequantize → within format error),
NOT by inspection. Run: `python3 lowprec_materialize.py` (self-test).
"""
from __future__ import annotations

import torch

# Element-format descriptors: (max_normal, max_exponent e_max, torch_dtype-or-None).
# e_max = floor(log2(max_normal)) — the exponent of the largest representable magnitude.
_FP8_E4M3 = {"name": "e4m3", "max": 448.0, "emax": 8, "torch": torch.float8_e4m3fn}
_FP8_E5M2 = {"name": "e5m2", "max": 57344.0, "emax": 15, "torch": torch.float8_e5m2}
_FP4_E2M1 = {"name": "e2m1", "max": 6.0, "emax": 2, "torch": None}  # no scalar torch fp4; emulate

_FP8 = {"e4m3": _FP8_E4M3, "e5m2": _FP8_E5M2}

# FP4 E2M1 representable magnitudes (1 sign + 2 exp + 1 mantissa): {0,.5,1,1.5,2,3,4,6}.
_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

MX_BLOCK = 32  # OCP Microscaling group size (CANN: group size 32)


def _round_to_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest onto the FP4 E2M1 level set (sign-symmetric)."""
    sign = torch.sign(x)
    mag = x.abs()
    # nearest level by absolute difference
    diffs = (mag.unsqueeze(-1) - _FP4_LEVELS.to(x.device)).abs()
    idx = diffs.argmin(dim=-1)
    return sign * _FP4_LEVELS.to(x.device)[idx]


def materialize_fp8(t: torch.Tensor, fmt: str = "e4m3", scope: str = "per_tensor"):
    """Per-tensor (or per-row) scaled fp8 — FA's low-precision path.
    Returns (fp8_tensor, d_scale) where dequant = fp8_tensor.float() * d_scale.
    d_scale = amax / fmt_max so the largest magnitude maps near the format max (no overflow)."""
    spec = _FP8[fmt]
    if scope == "per_tensor":
        amax = t.abs().amax().clamp_min(1e-12)
        d_scale = (amax / spec["max"]).to(torch.float32)
        q = (t / d_scale).clamp(-spec["max"], spec["max"]).to(spec["torch"])
        return q, d_scale
    if scope == "per_row":  # scale per last-dim row (FA per-head channel option)
        amax = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        d_scale = (amax / spec["max"]).to(torch.float32)
        q = (t / d_scale).clamp(-spec["max"], spec["max"]).to(spec["torch"])
        return q, d_scale
    raise ValueError(f"scope {scope!r} not in (per_tensor, per_row)")


def materialize_fp8_blockwise(t: torch.Tensor, fmt: str = "e4m3", blocksize: int = 128):
    """Block-wise scaled fp8 along the SEQUENCE dim — the vendor FA-quant scheme.

    Per aclnnQuantFlashAttentionScore: dScaleQ shape [B,N1,Ceil(Sq/128),1] (blocksize 128),
    dScaleK [...,Ceil(Skv/256),1] (256), dScaleV (512); all FP32. Each block of `blocksize`
    consecutive sequence positions (shared across the head_dim D too — the trailing scale dim
    is 1) gets one FP32 scale = block_amax / fmt_max.

    Input layout: [B, N, S, D] (BNSD canonical; S = dim -2, D = dim -1). Returns
    (fp8_tensor [B,N,S,D], d_scale [B,N,Ceil(S/blocksize),1]) where
    dequant = fp8_tensor.float() * d_scale broadcast per block. Handles S not a multiple of
    blocksize by zero-padding the tail block for the amax (padding does not affect amax)."""
    spec = _FP8[fmt]
    *lead, S, D = t.shape
    nblk = (S + blocksize - 1) // blocksize
    padS = nblk * blocksize
    tp = t
    if padS != S:
        pad = torch.zeros(*lead, padS - S, D, dtype=t.dtype, device=t.device)
        tp = torch.cat([t, pad], dim=-2)
    blk = tp.reshape(*lead, nblk, blocksize, D)               # [...,nblk,bs,D]
    amax = blk.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12)  # [...,nblk,1,1]
    d_scale = (amax / spec["max"]).to(torch.float32)          # [...,nblk,1,1]
    q_blk = (blk / d_scale).clamp(-spec["max"], spec["max"]).to(spec["torch"])
    q = q_blk.reshape(*lead, padS, D)[..., :S, :].contiguous()  # unpad → [...,S,D]
    d_scale_out = d_scale.reshape(*lead, nblk, 1)             # [B,N,Ceil(S/bs),1] vendor shape
    return q, d_scale_out


def _mx_shared_exp(block_amax: torch.Tensor, elem_emax: int) -> torch.Tensor:
    """OCP MX shared exponent: e = floor(log2(amax)) - elem_emax, clamped to E8M0 range
    [-127, 127]. The shared scale is 2^e; dividing a block by it maps its amax near the
    element format's max magnitude."""
    safe = block_amax.clamp_min(1e-30)
    e = torch.floor(torch.log2(safe)).to(torch.int32) - elem_emax
    return e.clamp(-127, 127)


def materialize_mx(t: torch.Tensor, elem: str = "e4m3"):
    """OCP Microscaling: group of MX_BLOCK(32) along the last dim shares one E8M0 power-of-2
    scale; elements are fp8 (mxfp8) or fp4-e2m1 (mxfp4). Returns (elements_float, e8m0_exp,
    dequant) where dequant = elements_float * 2^e8m0_exp (broadcast per block). elements_float
    is the quantized element magnitude in float (kept as float for portability; the on-device
    rep would pack to fp8/fp4). Last dim must be a multiple of 32."""
    if t.shape[-1] % MX_BLOCK != 0:
        raise ValueError(f"last dim {t.shape[-1]} not a multiple of MX_BLOCK={MX_BLOCK}")
    is_fp4 = elem == "e2m1"
    spec = _FP4_E2M1 if is_fp4 else _FP8[elem]
    orig_shape = t.shape
    blocks = t.reshape(*orig_shape[:-1], orig_shape[-1] // MX_BLOCK, MX_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)            # [..., nblk, 1]
    exp = _mx_shared_exp(amax, spec["emax"])                  # [..., nblk, 1] int
    scale = torch.pow(2.0, exp.to(torch.float32))
    scaled = blocks / scale
    if is_fp4:
        q = _round_to_fp4(scaled.clamp(-spec["max"], spec["max"]))
    else:
        q = scaled.clamp(-spec["max"], spec["max"]).to(spec["torch"]).float()
    dequant = (q * scale).reshape(orig_shape)
    return q.reshape(orig_shape), exp.squeeze(-1), dequant


# ---------------------------------------------------------------------------------------------
# Self-test: round-trip error must be within each format's expected bound (proves correctness).
def _selftest():
    torch.manual_seed(0)
    t = torch.randn(4, 64) * 3.0
    fails = []

    # fp8 per-tensor: rel error should be small (e4m3 ~2^-3 worst-case relative step).
    for fmt, bound in (("e4m3", 0.10), ("e5m2", 0.30)):
        q, ds = materialize_fp8(t, fmt)
        deq = q.float() * ds
        rel = ((deq - t).abs() / t.abs().clamp_min(1e-3)).mean().item()
        ok = rel < bound
        print(f"  fp8 {fmt} per_tensor: mean_rel_err={rel:.4f} (<{bound}) {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"fp8 {fmt}")

    # mxfp8 / mxfp4: block-scaled, last dim 64 = 2 blocks of 32.
    for elem, bound in (("e4m3", 0.10), ("e5m2", 0.30), ("e2m1", 0.40)):
        q, exp, deq = materialize_mx(t, elem)
        rel = ((deq - t).abs() / t.abs().clamp_min(1e-3)).mean().item()
        ok = rel < bound
        kind = "mxfp4" if elem == "e2m1" else "mxfp8"
        print(f"  {kind} ({elem}) group32: mean_rel_err={rel:.4f} (<{bound}) exp_shape={list(exp.shape)} "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"mx {elem}")

    # block-wise fp8 (vendor FA-quant): [B,N,S,D]; verify dScale shape + round-trip.
    qkv = torch.randn(1, 2, 384, 128) * 3.0   # S=384 → blocks: q(128)=3, k(256)=2, v(512)=1
    for fmt, bs, want_nblk, bound in (("e4m3", 128, 3, 0.10), ("e4m3", 256, 2, 0.10),
                                      ("e4m3", 512, 1, 0.10)):
        q, ds = materialize_fp8_blockwise(qkv, fmt, blocksize=bs)
        deq = q.float() * ds.repeat_interleave(bs, dim=-2)[..., :qkv.shape[-2], :]
        rel = ((deq - qkv).abs() / qkv.abs().clamp_min(1e-3)).mean().item()
        shape_ok = list(ds.shape) == [1, 2, want_nblk, 1]
        ok = rel < bound and shape_ok
        print(f"  fp8-blockwise {fmt} bs={bs}: dScale={list(ds.shape)} (want nblk={want_nblk}) "
              f"mean_rel_err={rel:.4f} (<{bound}) {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"fp8-blockwise bs={bs}")

    # shape guard
    try:
        materialize_mx(torch.randn(4, 30), "e4m3")
        fails.append("shape-guard not raised")
    except ValueError:
        print("  mx shape-guard (non-multiple-of-32): correctly raised OK")

    if fails:
        print(f"SELFTEST FAILED: {fails}")
        raise SystemExit(1)
    print("SELFTEST OK — fp8 + fp4 + mxfp8 + mxfp4 materialization round-trips within format error.")


if __name__ == "__main__":
    _selftest()
