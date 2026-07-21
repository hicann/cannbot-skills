### Example input dsl
```python
# bf16 variant
import tile.language as tl

@ascend_kernel
def leaky_relu_kernel(input_ptr, output_ptr,
                      base_elems, pivot, tile_size, inner_loops, negative_slope):

    pid = tl.program_id(0)
    # Pivot distribution: first 'pivot' cores get one extra element
    my_elems = base_elems + (1 if pid < pivot else 0)
    start = pid * base_elems + min(pid, pivot)

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    x_ub          = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 UB compute for bf16 IO
    pos_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 UB compute for bf16 IO
    neg_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 UB compute for bf16 IO
    out_ub        = tl.alloc_ub(tile_size, dtype=tl.float32)  # float32 UB compute for bf16 IO

    # ------------------------------------------------------------
    # Tile loop
    # ------------------------------------------------------------
    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)

        # --------------------------------------------------------
        # COPYIN
        # --------------------------------------------------------
        with tl.copyin():
            tl.load(input_ptr + offsets, x_ub)

        # --------------------------------------------------------
        # COMPUTE
        # --------------------------------------------------------
        with tl.compute():
            # pos = max(x, 0)
            tl.vmax(pos_ub, x_ub, 0.0)

            # neg = min(x, 0)
            tl.vmin(neg_ub, x_ub, 0.0)

            # neg_scaled = neg * negative_slope
            tl.vmul_scalar(neg_ub, neg_ub, negative_slope)

            # out = pos + neg_scaled
            tl.vadd(out_ub, pos_ub, neg_ub)

        # --------------------------------------------------------
        # COPYOUT
        # --------------------------------------------------------
        with tl.copyout():
            tl.store(output_ptr + offsets, out_ub)


def leaky_relu_host(x: torch.Tensor, output: torch.Tensor, negative_slope: float):
    total_elems = x.numel()

    # ------------------------------------------------------------
    # Core Partitioning: dynamically query Vector core count
    # ------------------------------------------------------------
    n_cores    = tl.num_vec_cores()
    n_used     = min(n_cores, total_elems)
    base_elems = total_elems // n_used
    pivot      = total_elems % n_used

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size   = 2048
    max_elems   = base_elems + (1 if pivot > 0 else 0)
    inner_loops = (max_elems + tile_size - 1) // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    leaky_relu_kernel[n_used](
        x, output,
        base_elems, pivot,
        tile_size,
        inner_loops,
        negative_slope
    )
```
### Example input AscendC
```
#include "kernel_operator.h"

class KernelLeakyRelu {
private:
    AscendC::TPipe pipe;
    // bf16 IO queues (for DataCopy with GM)
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBf16;
    // fp32 compute queues (for vector operations)
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueFp32;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueFp32;
    // bf16 output queue (for DataCopy back to GM)
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueBf16;
    // fp32 computation buffers
    AscendC::TBuf<AscendC::TPosition::VECCALC> posBuf, negBuf;
    // GM tensors use bfloat16_t for bf16 data
    AscendC::GlobalTensor<bfloat16_t> inputGm;
    AscendC::GlobalTensor<bfloat16_t> outputGm;
    uint32_t baseElems;
    uint32_t pivot;
    uint32_t myElems;
    uint32_t myStart;
    uint32_t tileSize;
    uint32_t innerLoops;
    float alpha;

public:
    __aicore__ inline KernelLeakyRelu() {}
    __aicore__ inline void Init(GM_ADDR input_ptr, GM_ADDR output_ptr,
                                uint32_t baseElems, uint32_t pivot,
                                uint32_t tileSize, uint32_t innerLoops, float alpha)
    {
        this->baseElems  = baseElems;
        this->pivot      = pivot;
        this->tileSize   = tileSize;
        this->innerLoops = innerLoops;
        this->alpha      = alpha;

        // Pivot distribution: first 'pivot' cores get one extra element
        uint32_t pid = AscendC::GetBlockIdx();
        this->myElems = baseElems + (pid < pivot ? 1 : 0);
        this->myStart = pid * baseElems + (pid < pivot ? pid : pivot);

        inputGm.SetGlobalBuffer((__gm__ bfloat16_t *)input_ptr + this->myStart, this->myElems);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t *)output_ptr + this->myStart, this->myElems);

        // bf16 buffers: sizeof(bfloat16_t) = 2 bytes
        pipe.InitBuffer(inQueueBf16, 1, this->tileSize * sizeof(bfloat16_t));
        // fp32 buffers: sizeof(float) = 4 bytes
        pipe.InitBuffer(inQueueFp32, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outQueueFp32, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outQueueBf16, 1, this->tileSize * sizeof(bfloat16_t));
        // fp32 computation buffers
        pipe.InitBuffer(posBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(negBuf, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // TODO implemented
    }
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, tiling_data.baseElems, tiling_data.pivot, tiling_data.tileSize, tiling_data.innerLoops, tiling_data.alpha);
    op.Process();
}
```

### Example output AscendC
```
#include "kernel_operator.h"

class KernelLeakyRelu {
private:
    AscendC::TPipe pipe;
    // bf16 IO queues (for DataCopy with GM)
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBf16;
    // fp32 compute queues (for vector operations)
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueFp32;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueFp32;
    // bf16 output queue (for DataCopy back to GM)
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueBf16;
    // fp32 computation buffers
    AscendC::TBuf<AscendC::TPosition::VECCALC> posBuf, negBuf;
    // GM tensors use bfloat16_t for bf16 data
    AscendC::GlobalTensor<bfloat16_t> inputGm;
    AscendC::GlobalTensor<bfloat16_t> outputGm;
    uint32_t baseElems;
    uint32_t pivot;
    uint32_t myElems;
    uint32_t myStart;
    uint32_t tileSize;
    uint32_t innerLoops;
    float alpha;

public:
    __aicore__ inline KernelLeakyRelu() {}
    __aicore__ inline void Init(GM_ADDR input_ptr, GM_ADDR output_ptr,
                                uint32_t baseElems, uint32_t pivot,
                                uint32_t tileSize, uint32_t innerLoops, float alpha)
    {
        this->baseElems  = baseElems;
        this->pivot      = pivot;
        this->tileSize   = tileSize;
        this->innerLoops = innerLoops;
        this->alpha      = alpha;

        // Pivot distribution: first 'pivot' cores get one extra element
        uint32_t pid = AscendC::GetBlockIdx();
        this->myElems = baseElems + (pid < pivot ? 1 : 0);
        this->myStart = pid * baseElems + (pid < pivot ? pid : pivot);

        inputGm.SetGlobalBuffer((__gm__ bfloat16_t *)input_ptr + this->myStart, this->myElems);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t *)output_ptr + this->myStart, this->myElems);

        // bf16 buffers: sizeof(bfloat16_t) = 2 bytes
        pipe.InitBuffer(inQueueBf16, 1, this->tileSize * sizeof(bfloat16_t));
        // fp32 buffers: sizeof(float) = 4 bytes
        pipe.InitBuffer(inQueueFp32, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outQueueFp32, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outQueueBf16, 1, this->tileSize * sizeof(bfloat16_t));
        // fp32 computation buffers
        pipe.InitBuffer(posBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(negBuf, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->innerLoops; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx)
    {
        // Step 1: DataCopy bfloat16 from GM to bf16 UB
        AscendC::LocalTensor<bfloat16_t> inputBf16 = inQueueBf16.AllocTensor<bfloat16_t>();
        AscendC::DataCopy(inputBf16, inputGm[idx * this->tileSize], this->tileSize);
        inQueueBf16.EnQue(inputBf16);

        // Step 2: Cast bf16 -> fp32 for computation
        AscendC::LocalTensor<bfloat16_t> bf16Local = inQueueBf16.DeQue<bfloat16_t>();
        AscendC::LocalTensor<float> fp32Local = inQueueFp32.AllocTensor<float>();
        AscendC::Cast(fp32Local, bf16Local, AscendC::RoundMode::CAST_NONE, this->tileSize);
        inQueueBf16.FreeTensor(bf16Local);
        inQueueFp32.EnQue(fp32Local);
    }

    __aicore__ inline void Compute(uint32_t idx)
    {
        AscendC::LocalTensor<float> inputLocal = inQueueFp32.DeQue<float>();
        AscendC::LocalTensor<float> outputLocal = outQueueFp32.AllocTensor<float>();

        AscendC::LocalTensor<float> posLocal = posBuf.Get<float>();
        AscendC::LocalTensor<float> negLocal = negBuf.Get<float>();

        // pos = max(x, 0)
        AscendC::Maxs(posLocal, inputLocal, 0.0f, this->tileSize);

        // neg = min(x, 0)
        AscendC::Mins(negLocal, inputLocal, 0.0f, this->tileSize);

        // neg_scaled = neg * alpha
        AscendC::Muls(negLocal, negLocal, this->alpha, this->tileSize);

        // out = pos + neg_scaled
        AscendC::Add(outputLocal, posLocal, negLocal, this->tileSize);

        outQueueFp32.EnQue<float>(outputLocal);
        inQueueFp32.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t idx)
    {
        // Step 1: Get fp32 result
        AscendC::LocalTensor<float> fp32Out = outQueueFp32.DeQue<float>();

        // Step 2: Cast fp32 -> bf16
        // CAST_RINT required for f32→bf16 on dav_c220; CAST_NONE triggers
        // ASCENDC_ASSERT(false) and leaves the output buffer uninitialized.
        AscendC::LocalTensor<bfloat16_t> bf16Out = outQueueBf16.AllocTensor<bfloat16_t>();
        AscendC::Cast(bf16Out, fp32Out, AscendC::RoundMode::CAST_RINT, this->tileSize);
        outQueueFp32.FreeTensor(fp32Out);
        outQueueBf16.EnQue(bf16Out);

        // Step 3: DataCopy bf16 to GM
        AscendC::LocalTensor<bfloat16_t> bf16Final = outQueueBf16.DeQue<bfloat16_t>();
        AscendC::DataCopy(outputGm[idx * this->tileSize], bf16Final, this->tileSize);
        outQueueBf16.FreeTensor(bf16Final);
    }
};

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelLeakyRelu op;
    op.Init(x, y, tiling_data.baseElems, tiling_data.pivot, tiling_data.tileSize, tiling_data.innerLoops, tiling_data.alpha);
    op.Process();
}
```
