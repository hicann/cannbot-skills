# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import tile.language as tl

# =============================================================================
# Transpose / Permute — generic N-D "contiguous-row strided-gather" baseline
# =============================================================================
# Semantics: y is torch.permute(x, perm) made contiguous.
#   output dim j takes its length from input dim perm[j]; pure layout move, no arithmetic.
#   the input coord along axis perm[j] equals the output coord along axis j.
#   the input offset is the sum over j of output_coord j times in_stride at perm[j].
#
# Anti-regression rules from failed runs:
#   * Do NOT build an output-dim stride table from permInv[j]. For output dim j,
#     the source stride is in_stride[perm[j]]. permInv is only for reconstructing
#     reconstructing the input coord at perm[j] from the output coord at j.
#   * Do NOT decompose row index from j=0 upward. rowIdx is row-major over the
#     first ndim-1 output dims, so split it from j=ndim-2 down to 0.
#   * Do NOT switch to input-contiguous read + strided output scatter. Keep
#     output rows contiguous and gather from input.
#   * Do NOT use a manual slotGap = 32 - sizeof(T) store after strided gather;
#     mirror the template's multi-block layout instead.
#
# Core idea (correctness-first, covers rank 2..8 / any perm / any dtype):
#   * Process the OUTPUT one "row" at a time. A row = the last output dim,
#     length lastDim = out_shape[-1]. The output is contiguous, so a row is
#     written as one contiguous block y[r*lastDim : (r+1)*lastDim].
#   * Advancing +1 along the last OUTPUT axis advances the INPUT linear offset
#     by a CONSTANT stride  S = in_stride[perm[-1]]  (unit: elements). So the
#     lastDim source elements of a row sit at base, base+S, base+2S, ...
#     -> a single strided gather per row.
#   * numRows = numel / lastDim, distributed across cores via pivot.
#   * Fast path: when S == 1 the source row is contiguous -> plain load.
#     Additionally merge adjacent output rows whose input bases are contiguous
#     into larger contiguous copies; this avoids one tiny DMA per row on
#     million-row tensors.
#
# Host precomputes and passes down:
#   ndim, numRows, lastDim, S, base, pivot,
#   out_shape[0..ndim-2] and in_stride_perm (each entry is in_stride at perm[j], int64).
#
# -----------------------------------------------------------------------------
# ⚠️ TWO HARDWARE CONSTRAINTS THAT LOWERING MUST HONOR (learned the hard way):
#
#   (H1) Strided gather lowers to a multi-block DataCopyPad (blockCount=chunk,
#        blockLen of 1 element, srcStride in BYTES of (stride_last-1)*sizeof(dtype). Each
#        1-element block lands in a 32B-aligned UB slot, i.e. the gathered data
#        is NOT tightly packed in UB. The contiguous store MUST use the SAME
#        multi-block (symmetric) layout, and the UB buffer MUST be sized
#        chunk * 32B (not chunk * sizeof(dtype)). A tight contiguous read-back
#        reads padding garbage -> nan/inf/wrong values.
#
#   (H2) DataCopyPad blockCount is a 12-bit field: max 4095. blockCount >= 4096
#        wraps to 0 and silently copies nothing. Therefore chunk the row so
#        CHUNK <= 2048 (safe margin under 4095). Rows longer than CHUNK are
#        split into ceil(lastDim / CHUNK) gathers.
#
# 🚀 PERFORMANCE NOTE:
#   This element-level strided gather is correctness-first and bandwidth-bound.
#   It is fine for small/medium tensors (2-3x speedup) but slow on LARGE 2D
#   transposes (huge numel, big S -> poor DMA locality). The baseline still
#   MUST merge contiguous row runs for S == 1, because that is a simple DMA
#   batching fix and prevents timeout on million-row contiguous layouts. For
#   large S != 1 2D cases, prefer an on-chip 2D-block transpose via vnchwconv /
#   TransDataTo5HD (NLast / GatherTranspose branches). Do the block-transpose
#   rewrite in a later optimization pass (cake-evo), not in this baseline.
# =============================================================================

CHUNK = 2048  # <= 4095 (H2); also bounds UB usage (chunk * 32B, H1)


@ascend_kernel
def transpose_kernel(
    x_ptr,                 # input,  row-major contiguous, numel elements
    y_ptr,                 # output, row-major contiguous, numel elements
    ndim,                  # tensor rank (2..8)
    num_rows,              # numel / lastDim
    last_dim,              # out_shape[-1]
    stride_last,           # in_stride[perm[-1]] (elements); ==1 -> contiguous
    base,                  # rows per core (pivot distribution)
    pivot,                 # first `pivot` cores get base+1 rows
    out_shape_head,        # out_shape[0 .. ndim-2], len ndim-1
    in_stride_perm,        # in_stride[perm[j]] for j in 0..ndim-2 (int64)
):
    pid = tl.program_id(0)

    # ---- pivot distribution: which rows does this core own ----
    my_count = base + (1 if pid < pivot else 0)
    my_start = pid * base + min(pid, pivot)

    # UB sized for one CHUNK. Per (H1) each element occupies a 32B slot after a
    # strided gather, so the lowering allocates chunk * 32B for this buffer.
    ub = tl.alloc_ub(CHUNK, dtype=x_ptr.dtype)

    i = 0
    while i < my_count:
        r = my_start + i

        # ---- decompose row index r -> leading (ndim-1) output coords, and
        #      accumulate the input base offset (int64 to avoid overflow on
        #      tensors with >2^31 elements) ----
        rem = r
        in_base = 0
        for j in range(ndim - 2, -1, -1):
            c = rem % out_shape_head[j]
            rem = rem // out_shape_head[j]
            in_base += c * in_stride_perm[j]      # int64 accumulation

        # ---- S==1 anti-timeout path: merge consecutive rows when their input
        #      bases are contiguous, then copy runRows*lastDim as a contiguous
        #      stream. This preserves semantics and avoids one tiny DMA per row.
        if stride_last == 1:
            run_rows = 1
            next_in_base = in_base + last_dim
            while i + run_rows < my_count:
                nr = r + run_rows
                rem2 = nr
                nb = 0
                for j in range(ndim - 2, -1, -1):
                    c2 = rem2 % out_shape_head[j]
                    rem2 = rem2 // out_shape_head[j]
                    nb += c2 * in_stride_perm[j]
                if nb != next_in_base:
                    break
                next_in_base += last_dim
                run_rows += 1

            if run_rows > 1:
                total = run_rows * last_dim
                done = 0
                while done < total:
                    cur = min(CHUNK, total - done)
                    with tl.copyin():
                        tl.load(x_ptr + in_base + done + tl.arange(0, cur), ub)
                    with tl.copyout():
                        tl.store(y_ptr + r * last_dim + done + tl.arange(0, cur), ub)
                    done += cur
                i += run_rows
                continue

        # ---- process one row in chunks of at most CHUNK (H2) ----
        done = 0
        while done < last_dim:
            cur = min(CHUNK, last_dim - done)
            out_off = r * last_dim + done

            with tl.copyin():
                if stride_last == 1:
                    # fast path: contiguous source row
                    tl.load(x_ptr + (in_base + done) + tl.arange(0, cur), ub)
                else:
                    # strided gather: element k of this chunk is at
                    # in_base + (done + k) * S  (constant stride S).
                    # Lowers to multi-block DataCopyPad; see (H1)/(H2).
                    src = (in_base + (done * stride_last)) + tl.arange(0, cur) * stride_last
                    tl.load(x_ptr + src, ub)

            with tl.copyout():
                # contiguous write-back. Store MUST mirror the multi-block
                # layout used on load (H1) so packing stays consistent.
                tl.store(y_ptr + out_off + tl.arange(0, cur), ub)

            done += cur
        i += 1
