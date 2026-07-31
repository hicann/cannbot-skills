#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Roofline calculator POC for AscendC operators on Ascend950PR (A5).

Pure-python analytical calculator — NO NPU required. Given an op's
(FLOPs, HBM bytes moved, cross-core comm volume) + the A5 hardware spec,
it computes:

  1. compute-bound ceiling   = FLOPs / peak_FLOPS
  2. memory-bound  ceiling   = bytes / HBM_BW
  3. regime (arithmetic intensity OI vs ridge point)
  4. perfect-overlap best-case = sum over tiles of max(MTE2, CUBE, VEC) per
     tile — the comm/load-hidden lower bound (overlappable pipes run
     concurrently, so wall time is gated by the slowest pipe per tile).

Two demos: (a) plain GEMM 4096^3 fp16 sanity check, (b) FlashAttention
dense B1/H8/S2048/D128 fp16 causal with a measured vendor-relative point.

EVERY hardware number below is sourced from
  src/skills/references/hardware/target/ascend950pr.md      (spec)
  src/skills/references/target/ascendc/ROOFLINE_MODEL.md    (VEC roofline)
  src/scripts/orchestrator/roofline_eval.py                 (A3 empirical calib)
Numbers NOT in the spec are marked ASSUMPTION with the derivation shown.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# A5 (Ascend950PR) hardware constants — sourced, with assumptions flagged
# ---------------------------------------------------------------------------
# All "SPEC" lines are verbatim from ascend950pr.md. We use the PG-binned
# variant (28 AIC / 56 AIV) that the actual A5 server reports via npu-smi,
# NOT the full-die 32/64, so ceilings match what the measured kernel sees.

CLOCK_GHZ = 1.65     # SPEC ascend950pr.md L34 "Clock frequency 1.65 GHz"
N_AIC = 28       # SPEC L28 "AIC total 28 (PG-28)"
N_AIV = 56       # SPEC L27 "AIV total 56 (PG-28)"
HBM_BW_PEAK_GBs = 1600.0   # SPEC L60 "HBM bandwidth (peak) 1.6 TB/s"
HBM_BW_MEAS_GBs = 1100.0   # SPEC L61 "HBM bandwidth (measured) ~1.1 TB/s"

# --- CUBE (matmul) peak FLOPS ---------------------------------------------
# The spec does NOT state a CUBE TFLOPS figure directly. It gives the MMAD
# tile shape fp16 [m,k,n]=[16,16,16] (L323) and clock+core count, but NOT
# the MMAD issue rate (cycles per [16,16,16] MMAD). So CUBE peak must be
# ASSUMED. We anchor it three ways and take the convergent value:
#
#   (A1) A3 EMPIRICAL: roofline_eval.py L63 measured A3 (32 AIC, prev gen)
#        fp16 cube = 292 TFLOPS → 9.1 TFLOPS/AIC.
#   (A2) A5 WHITEPAPER: ascend950pr.md L708 "per-core FP16/FP32 TFLOPS +100%
#        vs prev gen" → ~18.2 TFLOPS/AIC.
#   (A3) Apply A2 uplift to A1 per-core, scale by A5 PG core count (28 AIC):
#        18.2 * 28 ≈ 510 TFLOPS fp16.
# >>> ASSUMPTION CUBE-1: CUBE fp16 peak = 510 TFLOPS (PG-28). Full-die(32)
#     would be ~583. This is the single largest lever on the FA result.
PEAK_CUBE_FP16_TFLOPS = 510.0   # ASSUMPTION (derivation above)

# --- VEC (vector) peak FLOPS ----------------------------------------------
# SPEC-adjacent: ROOFLINE_MODEL.md L11-12 gives VEC fp16 ~56 TFLOPS
# (56 cores * 512 FLOP/cyc * 1 GHz). That model used 1 GHz; the spec clock
# is 1.65 GHz, and whitepaper says vector +100% vs prev gen. We keep the
# KB's published 56 TFLOPS for VEC to stay consistent with the harness's
# existing roofline_eval.py SOC_A5 (peak_fp16_tflops=56), and flag it.
# >>> ASSUMPTION VEC-1: VEC fp16 peak = 56 TFLOPS (from ROOFLINE_MODEL.md).
PEAK_VEC_FP16_TFLOPS = 56.0     # ROOFLINE_MODEL.md L12 (KB-published)

# Which HBM BW to use for the memory ceiling. Peak is the true roofline
# ceiling; measured (~1.1 TB/s, ~69% of peak) is what kernels realistically
# attain. We report ceilings against PEAK and annotate measured.
HBM_BW_GBs = HBM_BW_PEAK_GBs


# ---------------------------------------------------------------------------
# Core roofline primitives
# ---------------------------------------------------------------------------

def fmt_ms(x: float) -> str:
    return f"{x*1e3:.4f}ms"


def tflops_to_flops(tflops: float) -> float:
    return tflops * 1e12


def gbs_to_bytes_per_s(gbs: float) -> float:
    return gbs * 1e9


def compute_ceiling_s(flops: float, peak_tflops: float) -> float:
    """Lower bound on time if perfectly compute-bound (no stalls)."""
    return flops / tflops_to_flops(peak_tflops)


def memory_ceiling_s(bytes_moved: float, bw_gbs: float = HBM_BW_GBs) -> float:
    """Lower bound on time if perfectly memory-bound (full BW)."""
    return bytes_moved / gbs_to_bytes_per_s(bw_gbs)


def operational_intensity(flops: float, bytes_moved: float) -> float:
    return flops / bytes_moved


def ridge_point(peak_tflops: float, bw_gbs: float = HBM_BW_GBs) -> float:
    """OI (FLOP/byte) at which compute and memory ceilings cross."""
    return tflops_to_flops(peak_tflops) / gbs_to_bytes_per_s(bw_gbs)


@dataclass
class TileProfile:
    """Per-tile work decomposed onto the overlappable A5 pipes.

    On A5 the 6 pipelines (MTE2 load / CUBE matmul / VEC compute / MTE3
    store / MTE1 / FIXP) run concurrently. For a steady-state tiled kernel
    the per-tile wall time is gated by the SLOWEST pipe (the others hide
    under it). Perfect-overlap best-case = n_tiles * max(pipe times).
    """
    mte2_load_s: float = 0.0   # HBM -> L1/UB
    cube_s: float = 0.0        # CUBE matmul
    vec_s: float = 0.0         # VEC softmax/elementwise
    mte3_store_s: float = 0.0  # UB -> HBM
    label: str = ""

    def tile_time_s(self) -> float:
        return max(self.mte2_load_s, self.cube_s, self.vec_s, self.mte3_store_s)


@dataclass
class RooflineResult:
    name: str
    flops: float
    bytes_moved: float
    comm_bytes: float
    compute_ceiling_s: float
    memory_ceiling_s: float
    oi: float
    ridge: float
    regime: str
    overlap_best_s: float
    notes: list = field(default_factory=list)

    def fmt(self) -> str:
        def ms(x):
            return f"{x*1e3:.4f} ms"

        L = [
            f"=== {self.name} ===",
            f"  FLOPs            : {self.flops:.3e}",
            f"  HBM bytes        : {self.bytes_moved:.3e} ({self.bytes_moved/1e6:.1f} MB)",
            f"  cross-core comm  : {self.comm_bytes:.3e} ({self.comm_bytes/1e6:.1f} MB)",
            f"  OI (FLOP/byte)   : {self.oi:.2f}",
            f"  ridge point      : {self.ridge:.2f} FLOP/byte  "
            f"(peak={PEAK_CUBE_FP16_TFLOPS}TF / {HBM_BW_GBs/1000}TB/s)",
            f"  regime           : {self.regime}",
            f"  compute ceiling  : {ms(self.compute_ceiling_s)}",
            f"  memory  ceiling  : {ms(self.memory_ceiling_s)}",
            f"  roofline ceiling : {ms(max(self.compute_ceiling_s, self.memory_ceiling_s))}  (max of the two)",
            f"  overlap best-case: {ms(self.overlap_best_s)}  (sum tiles max(pipe))",
        ]
        for n in self.notes:
            L.append(f"  note: {n}")
        return "\n".join(L)


def build_result(name, flops, bytes_moved, comm_bytes, tiles,
                 peak_tflops=PEAK_CUBE_FP16_TFLOPS, notes=None):
    cc = compute_ceiling_s(flops, peak_tflops)
    mc = memory_ceiling_s(bytes_moved)
    oi = operational_intensity(flops, bytes_moved)
    rp = ridge_point(peak_tflops)
    regime = "COMPUTE-bound" if oi > rp else "MEMORY-bound"
    overlap = sum(t.tile_time_s() for t in tiles)
    return RooflineResult(
        name=name, flops=flops, bytes_moved=bytes_moved, comm_bytes=comm_bytes,
        compute_ceiling_s=cc, memory_ceiling_s=mc, oi=oi, ridge=rp,
        regime=regime, overlap_best_s=overlap, notes=notes or [],
    )


# ---------------------------------------------------------------------------
# Demo (a): plain GEMM M=N=K=4096, fp16 — sanity check vs spec peak
# ---------------------------------------------------------------------------

def demo_gemm():
    M = N = K = 4096
    dt = 2  # fp16 bytes/elem
    # FLOPs for C[MxN] = A[MxK] @ B[KxN]: 2*M*N*K (one mul + one add per MAC)
    flops = 2.0 * M * N * K
    # HBM bytes (best case, each operand read once, C written once):
    #   read A + read B + write C
    bytes_moved = (M * K + K * N + M * N) * dt
    # No cross-core comm for a single fused GEMM (CUBE-local accumulate).
    comm = 0.0
    # Tiling: single big matmul, CUBE-dominated. The overlap best-case for a
    # compute-bound GEMM collapses to the CUBE ceiling itself (load of
    # operands hides under matmul once OI >> ridge). One "tile" = whole op.
    cube_t = compute_ceiling_s(flops, PEAK_CUBE_FP16_TFLOPS)
    load_t = memory_ceiling_s(bytes_moved)
    tiles = [TileProfile(mte2_load_s=load_t, cube_s=cube_t, label="gemm-whole")]
    notes = [
        "ASSUMPTION GEMM tiling: treated as one CUBE-bound op; operand "
        "reload across K-tiles ignored (single read each) — optimistic floor.",
        f"At peak {PEAK_CUBE_FP16_TFLOPS} TF the 4096^3 fp16 GEMM floor is "
        f"{cube_t*1e3:.4f} ms; cross-check A3 measured 8192^3 @292TF=3.76ms "
        f"=> 4096^3 A3 floor ~0.47ms, A5@510TF scales to ~0.27ms (consistent).",
    ]
    return build_result("GEMM 4096^3 fp16", flops, bytes_moved, comm, tiles, notes=notes)


# ---------------------------------------------------------------------------
# Demo (b): FlashAttention DENSE, B=1 H=8 S=2048 D=128 fp16 causal
# ---------------------------------------------------------------------------

def demo_flash_attention():
    B, H, S, D = 1, 8, 2048, 128
    dt = 2  # fp16

    # --- FLOPs --------------------------------------------------------------
    # Per (batch, head): two matmuls dominate, each S*S*D MACs:
    #   QK^T : [S,D]@[D,S] -> [S,S]   = 2*S*S*D
    #   PV   : [S,S]@[S,D] -> [S,D]   = 2*S*S*D
    # Causal mask => only lower triangle of the S*S score grid is computed
    # => ~0.5 factor on BOTH matmuls (an idealized vendor that skips the
    # upper triangle). Softmax VEC FLOPs (exp/max/sum/div over S*S) are
    # ~O(S*S) and small vs the 2*matmul; counted separately for the VEC pipe.
    causal = 0.5
    matmul_flops = 2 * (2.0 * S * S * D) * B * H * causal   # QK^T + PV, triangular
    # softmax elementwise: ~5 flops/score (sub-max, exp, add-to-sum, div, scale)
    softmax_flops = 5.0 * (S * S) * B * H * causal
    flops = matmul_flops + softmax_flops

    # --- HBM bytes (FlashAttention: O(S*D) IO, NOT O(S*S)) ------------------
    # FA keeps the S*S scores on-chip (tiled), so HBM traffic is just the
    # Q,K,V reads + O write, each B*H*S*D elements:
    bytes_moved = (4 * B * H * S * D) * dt   # Q + K + V read, O write
    # cross-core comm: FA on A5 uses the CV-fusion channel (Cube L1 <-> Vec
    # UB) on-chip; cross-CORE comm only for the head/row-block split. With
    # B*H=8 row-tiles distributed across cores, each core owns whole heads =>
    # negligible inter-core score exchange. Model as 0 (on-chip CV-fusion).
    comm = 0.0

    # --- per-tile pipe decomposition (perfect-overlap best-case) ------------
    # Tile the S (query) dim into blocks of Bq=128 rows; inner loop over K/V
    # blocks of Bk=128. For each (head, q-block) the steady state runs:
    #   MTE2 : load K,V blocks  | CUBE: QK^T + PV | VEC: softmax | MTE3: O
    # The CV-fusion channel hides K/V load + softmax under CUBE for a
    # compute-bound FA. So per (head,q-block) tile time = CUBE time of its
    # two triangular matmuls. Sum over all tiles == total CUBE ceiling.
    Bq = 128
    n_q_blocks = S // Bq
    n_tiles = B * H * n_q_blocks
    # CUBE flops per tile (already includes causal avg over the block):
    cube_flops_per_tile = matmul_flops / n_tiles
    vec_flops_per_tile = softmax_flops / n_tiles
    cube_t = compute_ceiling_s(cube_flops_per_tile, PEAK_CUBE_FP16_TFLOPS)
    vec_t = compute_ceiling_s(vec_flops_per_tile, PEAK_VEC_FP16_TFLOPS)
    # MTE2 load per tile: K,V for this q-block. Causal => avg half the K/V.
    load_bytes_per_tile = (2 * S * D * dt) * causal
    mte2_t = memory_ceiling_s(load_bytes_per_tile)
    tiles = [TileProfile(mte2_load_s=mte2_t, cube_s=cube_t, vec_s=vec_t,
                         label="fa-tile") for _ in range(n_tiles)]

    gate = "MTE2-gated" if mte2_t > cube_t else "CUBE-gated"
    notes = [
        "FA causal modeled with 0.5 triangular factor on matmul+softmax+load.",
        f"tiling Bq={Bq}, n_tiles={n_tiles}; per-tile pipes: "
        f"MTE2={mte2_t*1e6:.2f}us CUBE={cube_t*1e6:.2f}us VEC={vec_t*1e6:.2f}us "
        f"=> {gate}.",
        f"NOTE the overlap best-case ({fmt_ms(sum(t.tile_time_s() for t in tiles))}) "
        f"EXCEEDS the compute ceiling at Bq={Bq}: per q-block this naive tiling "
        f"RELOADS K,V from HBM (no K/V reuse across q-blocks), so MTE2 (load) "
        f"gates each tile. Keeping K/V resident (larger Bq or K/V-stationary "
        f"loop order) collapses overlap-best toward the "
        f"{compute_ceiling_s(matmul_flops, PEAK_CUBE_FP16_TFLOPS)*1e3:.4f}ms "
        f"CUBE floor. "
        f"This gap IS part of the recoverable vendor headroom.",
        "ASSUMPTION FA-IO: FlashAttention keeps S*S scores on-chip => global "
        "HBM traffic is O(S*D) (Q,K,V,O once each = 16.8MB). The per-tile "
        "MTE2 above models a K/V-reloading tiling => its sum is an upper "
        "bound; a K/V-stationary kernel hits the 16.8MB / O(S*D) floor.",
    ]
    return build_result("FlashAttention dense B1H8S2048D128 fp16 causal",
                        flops, bytes_moved, comm, tiles, notes=notes)


# ---------------------------------------------------------------------------
# Vendor / our-kernel positioning vs roofline (the actual question)
# ---------------------------------------------------------------------------

def fa_vs_roofline(fa: RooflineResult):
    # Measured data point: OUR generated FA dense runs at ~0.40x of vendor
    # npu_fusion_attention (i.e. we are 2.5x slower than vendor).
    OUR_RATIO_OF_VENDOR = 0.40   # measured (given)

    ceiling_s = max(fa.compute_ceiling_s, fa.memory_ceiling_s)  # roofline
    best_s = fa.overlap_best_s                               # overlap floor

    # We don't have vendor's absolute ms here; we have OUR/vendor=0.40 and we
    # can bound vendor against the roofline IF we assume vendor sits at some
    # efficiency. Instead we report the structural relationship and let the
    # orchestrator plug in the measured vendor ms when available. We give the
    # roofline ceiling so downstream code can divide it by the measured vendor
    # time to obtain vendor efficiency. Dividing it by our inferred time—the
    # vendor time divided by 0.40—makes our efficiency exactly 0.40 times the
    # vendor efficiency.
    # So WHATEVER vendor's efficiency is, OURS is exactly 0.40x of it.
    #
    # To produce concrete %s we adopt one anchor: a well-tuned vendor FA on
    # A5 (CV-fusion HW path, whitepaper §3 "1.5-2x single-core for FA")
    # typically reaches ~55-70% of cube roofline on dense. We take 60% as the
    # ANCHOR for vendor efficiency (flagged), then our% = 0.40 * vendor%.
    VENDOR_EFF_ANCHOR = 0.60   # ASSUMPTION (typical tuned dense-FA cube util)
    vendor_pct = VENDOR_EFF_ANCHOR
    our_pct = vendor_pct * OUR_RATIO_OF_VENDOR

    vendor_ms = ceiling_s / vendor_pct * 1e3
    our_ms = vendor_ms / OUR_RATIO_OF_VENDOR

    return {
        "roofline_ceiling_ms": ceiling_s * 1e3,
        "overlap_best_ms": best_s * 1e3,
        "regime": fa.regime,
        "our_ratio_of_vendor": OUR_RATIO_OF_VENDOR,
        "vendor_eff_anchor": VENDOR_EFF_ANCHOR,
        "vendor_pct_of_roofline": vendor_pct * 100,
        "our_pct_of_roofline": our_pct * 100,
        "implied_vendor_ms": vendor_ms,
        "implied_our_ms": our_ms,
        "interpretation": (
            "vendor ~60% of roofline (anchor) => recoverable headroom is the "
            "gap from our 24% to vendor's 60%: the 2.5x dense gap is OVERHEAD "
            "(load/softmax not overlapping, tiling, sync), NOT a roofline wall. "
            "If vendor were ~90% of roofline, our ~36% would still leave most "
            "of the 2.5x as recoverable overhead."
        ),
    }


# ---------------------------------------------------------------------------
def main():
    print("Ascend950PR (A5) Roofline POC — analytical, no NPU")
    print("=" * 64)
    print(f"hw: {N_AIC} AIC / {N_AIV} AIV @ {CLOCK_GHZ} GHz, "
          f"HBM peak {HBM_BW_GBs/1000} TB/s (meas {HBM_BW_MEAS_GBs/1000})")
    print(f"    CUBE fp16 peak {PEAK_CUBE_FP16_TFLOPS} TF [ASSUMPTION], "
          f"VEC fp16 peak {PEAK_VEC_FP16_TFLOPS} TF")
    print(f"    ridge (cube) = {ridge_point(PEAK_CUBE_FP16_TFLOPS):.2f} FLOP/byte\n")

    gemm = demo_gemm()
    print(gemm.fmt(), "\n")

    fa = demo_flash_attention()
    print(fa.fmt(), "\n")

    pos = fa_vs_roofline(fa)
    print("=== FA dense: vendor & our position vs roofline ===")
    for k, v in pos.items():
        if isinstance(v, float):
            print(f"  {k:24s}: {v:.4f}")
        else:
            print(f"  {k:24s}: {v}")


if __name__ == "__main__":
    main()
