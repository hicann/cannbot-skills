/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include <type_traits>

// DTYPE_X1 is injected by the build system per-dtype compilation:
//   float       for float32
//   half        for float16
//   bfloat16_t  for bfloat16
//
// On Ascend910b (dav_c220):
//   - Muls<half>      ✅  supported
//   - Muls<bfloat16_t> ❌  NOT supported
//   - Mul<bfloat16_t>  ❌  NOT supported
//   - Add<bfloat16_t>  ❌  NOT supported
//
// Fix: for bfloat16_t, upcast to float32 → compute → downcast back.
//      Uses if constexpr so only the relevant branch is compiled per variant.
#ifndef DTYPE_X1
#define DTYPE_X1 float   // fallback for IDE / linters
#endif

class KernelAdd {
private:
    AscendC::TPipe pipe;

    // I/O queues in native dtype DTYPE_X1
    AscendC::TQue<AscendC::TPosition::VECIN, 1>  x1Queue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1>  x2Queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yQueue;

    // Float32 intermediate compute buffers — only allocated/used on the bfloat16_t path.
    // For float / half, these TBufs are declared but pipe.InitBuffer is never called,
    // so they consume zero UB space in those compilation variants.
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1FloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x2FloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFloatBuf;

    // Global memory tensors in DTYPE_X1
    AscendC::GlobalTensor<DTYPE_X1> x1Gm;
    AscendC::GlobalTensor<DTYPE_X1> x2Gm;
    AscendC::GlobalTensor<DTYPE_X1> yGm;

    // Tiling parameters
    uint32_t baseElems;
    uint32_t pivot;
    uint32_t myElems;    // elements assigned to this core
    uint32_t myStart;    // global start index for this core
    uint32_t tileSize;
    uint32_t innerLoops;
    float    alpha;      // kept as float; cast to DTYPE_X1 (or used as-is for float32 path)

public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t baseElems, uint32_t pivot,
                                uint32_t tileSize, uint32_t innerLoops,
                                float alpha)
    {
        this->baseElems  = baseElems;
        this->pivot      = pivot;
        this->tileSize   = tileSize;
        this->innerLoops = innerLoops;
        this->alpha      = alpha;

        // Pivot distribution: first 'pivot' cores get one extra element
        uint32_t pid  = AscendC::GetBlockIdx();
        this->myElems = baseElems + (pid < pivot ? 1 : 0);
        this->myStart = pid * baseElems + (pid < pivot ? pid : pivot);

        // Each core only sees its own slice in global memory
        x1Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x1 + this->myStart, this->myElems);
        x2Gm.SetGlobalBuffer((__gm__ DTYPE_X1*)x2 + this->myStart, this->myElems);
        yGm.SetGlobalBuffer((__gm__ DTYPE_X1*)y  + this->myStart, this->myElems);

        // ---- UB budget ----
        // float / half path:  3 TQueues × tileSize × sizeof(T)
        //                     = 3 × 2048 × 4 = 24KB  (float32)
        //                     = 3 × 2048 × 2 = 12KB  (float16)
        // bfloat16_t path:    3 TQueues × 2048 × 2  +  3 TBufs × 2048 × 4
        //                     = 12KB + 24KB = 36KB  << 192KB UB limit
        uint32_t qBytes = tileSize * sizeof(DTYPE_X1);
        pipe.InitBuffer(x1Queue, 1, qBytes);
        pipe.InitBuffer(x2Queue, 1, qBytes);
        pipe.InitBuffer(yQueue,  1, qBytes);

        // Allocate float32 compute buffers only for the bfloat16_t compilation variant
        if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value) {
            uint32_t fBytes = tileSize * sizeof(float);
            pipe.InitBuffer(x1FloatBuf, fBytes);
            pipe.InitBuffer(x2FloatBuf, fBytes);
            pipe.InitBuffer(yFloatBuf,  fBytes);
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < innerLoops; i++) {
            uint32_t tileOffset = i * tileSize;
            // Guard: non-pivot cores run innerLoops (= ceil(maxElems/tileSize)) iterations,
            // but their myElems = baseElems may equal exactly k*tileSize.
            // Without this break, the last iteration would compute cnt=0 and access
            // x1Gm[myElems] which is one past the registered GM region (undefined behaviour).
            if (tileOffset >= myElems) break;
            uint32_t cnt = (tileOffset + tileSize <= myElems)
                         ? tileSize
                         : (myElems - tileOffset);
            CopyIn(i, cnt);
            Compute(i, cnt);
            CopyOut(i, cnt);
        }
    }

private:
    // ----------------------------------------------------------------
    // CopyIn: load x1 and x2 tiles from global memory into UB
    //
    // FIX (ADD-04): use isPad=true so the DMA reads exactly blockLen bytes
    // from GM and pads the remainder of the 32-byte block with zeros in UB.
    // With isPad=false the DMA reads ceil(blockLen/32)*32 bytes from GM,
    // which can exceed the registered GM region when cnt is not 32-byte-
    // aligned (e.g. bfloat16_t with cnt=1228 → blockLen=2456, DMA needs
    // 2464 bytes → 8 bytes past end of registered region → DDR out-of-range).
    //
    // We use the simple DataCopyPadParams API (not DataCopyPadExtParams<T>)
    // because the extended bfloat16_t specialisation produces incorrect reads.
    // DataCopyPadParams.rightPadding is uint8_t; max excess is 31 bytes
    // → at most 15 elements for 2-byte types, well within uint8_t range.
    // ----------------------------------------------------------------
    __aicore__ inline void CopyIn(uint32_t idx, uint32_t cnt)
    {
        uint32_t blockLen  = cnt * sizeof(DTYPE_X1);
        uint32_t paddedLen = ((blockLen + 31) / 32) * 32;
        uint8_t  rightPad  = static_cast<uint8_t>((paddedLen - blockLen) / sizeof(DTYPE_X1));

        AscendC::LocalTensor<DTYPE_X1> x1Local = x1Queue.AllocTensor<DTYPE_X1>();
        AscendC::DataCopyPad(x1Local, x1Gm[idx * tileSize],
                             {1, static_cast<uint16_t>(blockLen), 0, 0},
                             {true, static_cast<uint8_t>(0), rightPad, static_cast<uint64_t>(0)});
        x1Queue.EnQue(x1Local);

        AscendC::LocalTensor<DTYPE_X1> x2Local = x2Queue.AllocTensor<DTYPE_X1>();
        AscendC::DataCopyPad(x2Local, x2Gm[idx * tileSize],
                             {1, static_cast<uint16_t>(blockLen), 0, 0},
                             {true, static_cast<uint8_t>(0), rightPad, static_cast<uint64_t>(0)});
        x2Queue.EnQue(x2Local);
    }

    // ----------------------------------------------------------------
    // Compute: y = x1 + alpha * x2
    //
    // float / half path (else branch):
    //   Muls is supported → scale x2 in-place, then Add to x1
    //
    // bfloat16_t path (if constexpr branch):
    //   Muls/Add NOT supported for bfloat16_t on Ascend910b.
    //   → Cast bfloat16_t → float32, compute in float32, Cast back.
    // ----------------------------------------------------------------
    __aicore__ inline void Compute(uint32_t idx, uint32_t cnt)
    {
        AscendC::LocalTensor<DTYPE_X1> x1Local = x1Queue.DeQue<DTYPE_X1>();
        AscendC::LocalTensor<DTYPE_X1> x2Local = x2Queue.DeQue<DTYPE_X1>();
        AscendC::LocalTensor<DTYPE_X1> yLocal  = yQueue.AllocTensor<DTYPE_X1>();

        if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value) {
            // ---- bfloat16_t path: upcast → float32 compute → downcast ----
            AscendC::LocalTensor<float> x1F = x1FloatBuf.Get<float>();
            AscendC::LocalTensor<float> x2F = x2FloatBuf.Get<float>();
            AscendC::LocalTensor<float> yF  = yFloatBuf.Get<float>();

            // Step 1: bfloat16_t → float32 (exact conversion, no rounding needed)
            AscendC::Cast(x1F, x1Local, AscendC::RoundMode::CAST_NONE, cnt);
            AscendC::Cast(x2F, x2Local, AscendC::RoundMode::CAST_NONE, cnt);

            // Step 2: y_f32 = x1_f32 + alpha * x2_f32 (Muls/Add both supported for float)
            AscendC::Muls(x2F, x2F, this->alpha, cnt);
            AscendC::Add(yF, x1F, x2F, cnt);

            // Step 3: float32 → bfloat16_t
            // On Ascend910b (dav_c220), CAST_NONE for float→bfloat16_t triggers
            // ASCENDC_ASSERT(false) — the only legal modes are CAST_RINT/ROUND/FLOOR/CEIL/TRUNC.
            // Use CAST_RINT (round-to-nearest-even) to match PyTorch's bfloat16 rounding.
            AscendC::Cast(yLocal, yF, AscendC::RoundMode::CAST_RINT, cnt);
        } else {
            // ---- float / half path: direct computation ----
            // Muls supports: float, half — scale x2 by alpha in-place
            AscendC::Muls(x2Local, x2Local, static_cast<DTYPE_X1>(this->alpha), cnt);
            // Add: y = x1 + (alpha * x2)
            AscendC::Add(yLocal, x1Local, x2Local, cnt);
        }

        yQueue.EnQue(yLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    // ----------------------------------------------------------------
    // CopyOut: write output tile from UB back to global memory
    // DataCopyPad supports both aligned and non-aligned sizes
    // ----------------------------------------------------------------
    __aicore__ inline void CopyOut(uint32_t idx, uint32_t cnt)
    {
        AscendC::LocalTensor<DTYPE_X1> yLocal = yQueue.DeQue<DTYPE_X1>();
        AscendC::DataCopyPad(yGm[idx * tileSize], yLocal,
                             {1, static_cast<uint16_t>(cnt * sizeof(DTYPE_X1)), 0, 0});
        yQueue.FreeTensor(yLocal);
    }
};


extern "C" __global__ __aicore__ void add_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdd op;
    op.Init(x1, x2, y,
            tiling_data.baseElems, tiling_data.pivot,
            tiling_data.tileSize, tiling_data.innerLoops,
            tiling_data.alpha);
    op.Process();
}
