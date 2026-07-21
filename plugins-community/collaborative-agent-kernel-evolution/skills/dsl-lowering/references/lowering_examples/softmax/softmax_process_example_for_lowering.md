### Example input dsl
```python
# bf16 variant
import torch
import tile.language as tl

@ascend_kernel
def softmax_kernel(input_ptr, output_ptr, base_rows, pivot, tile_length):
    pid = tl.program_id(0)
    # Pivot distribution: first 'pivot' cores get one extra row
    my_rows = base_rows + (1 if pid < pivot else 0)
    row_start_idx = pid * base_rows + min(pid, pivot)
    row_end_idx = row_start_idx + my_rows

    # Allocate UB Buffers
    row_ub      = tl.alloc_ub(tile_length, dtype=tl.float32)  # float32 UB compute for bf16 IO
    exp_ub      = tl.alloc_ub(tile_length, dtype=tl.float32)  # float32 UB compute for bf16 IO
    shared_ub   = tl.alloc_ub(tile_length, dtype=tl.float32)  # float32 UB compute for bf16 IO
    out_ub      = tl.alloc_ub(tile_length, dtype=tl.float32)  # float32 UB compute for bf16 IO

    # Computation Logic
    for row_idx in range(row_start_idx, row_end_idx):
        offsets = row_idx * tile_length + tl.arange(0, tile_length)

        with tl.copyin():
            tl.load(input_ptr + offsets, row_ub)

        with tl.compute():
            tl.reduce_max(shared_ub, row_ub, shared_ub)
            row_max = extract_scalar(shared_ub, 0)
            tl.vsub_scalar(exp_ub, row_ub, row_max)
            tl.vexp(exp_ub, exp_ub)
            tl.reduce_sum(shared_ub, exp_ub, shared_ub)
            row_sum = extract_scalar(shared_ub, 0)
            tl.vdiv_scalar(out_ub, exp_ub, row_sum)

        with tl.copyout():
            tl.store(output_ptr + offsets, out_ub)

def softmax_host(x: torch.Tensor, output: torch.Tensor):
    rows = x.shape[0]
    cols = x.shape[1]

    # Core Partitioning: dynamically query Vector core count
    n_cores   = tl.num_vec_cores()
    n_used    = min(n_cores, rows)
    base_rows = rows // n_used
    pivot     = rows % n_used

    # Tiling Strategy: entire row fits into UB
    tile_length = cols

    softmax_kernel[n_used](x, output, base_rows, pivot, tile_length)

```

### Example input AscendC
```
// State after init_pass: class members and Init() are fully implemented.
// Only Process() (and its helper methods CopyIn/Compute/CopyOut) need to be added.
// Full class skeleton is in softmax_init_example_for_lowering.md.
//
// Add the following method to class KernelSoftmax (shown without class wrapper
// to avoid duplicate-class redefinition when concatenated with the output block):
    __aicore__ inline void Process()
    {
        // TODO implemented
    }
```

### Example output AscendC
```
#include "kernel_operator.h"

class KernelSoftmax {
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf, sharedBuf;
    // GM tensors use bfloat16_t for bf16 data
    AscendC::GlobalTensor<bfloat16_t> inputGm;
    AscendC::GlobalTensor<bfloat16_t> outputGm;
    uint32_t tileLength;
    uint32_t baseRows;
    uint32_t pivot;
    uint32_t myRows;
    uint32_t rowStart;

public:
    __aicore__ inline KernelSoftmax() {}
    __aicore__ inline void Init(GM_ADDR input_ptr, GM_ADDR output_ptr,
                                uint32_t tileLength, uint32_t baseRows, uint32_t pivot)
    {
        this->tileLength = tileLength;
        this->baseRows   = baseRows;
        this->pivot      = pivot;

        // Pivot distribution: first 'pivot' cores get one extra row
        uint32_t pid = AscendC::GetBlockIdx();
        this->myRows  = baseRows + (pid < pivot ? 1 : 0);
        this->rowStart = pid * baseRows + (pid < pivot ? pid : pivot);

        uint32_t totalElements = this->myRows * tileLength;

        inputGm.SetGlobalBuffer((__gm__ bfloat16_t *)input_ptr + this->rowStart * tileLength, totalElements);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t *)output_ptr + this->rowStart * tileLength, totalElements);

        // bf16 buffers: sizeof(bfloat16_t) = 2 bytes
        pipe.InitBuffer(inQueueBf16, 1, this->tileLength * sizeof(bfloat16_t));
        // fp32 buffers: sizeof(float) = 4 bytes
        pipe.InitBuffer(inQueueFp32, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueFp32, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueBf16, 1, this->tileLength * sizeof(bfloat16_t));
        // fp32 computation buffers
        pipe.InitBuffer(expBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(sharedBuf, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->myRows; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t rowIdx)
    {
        // Step 1: DataCopy bfloat16 from GM to bf16 UB
        AscendC::LocalTensor<bfloat16_t> inputBf16 = inQueueBf16.AllocTensor<bfloat16_t>();
        AscendC::DataCopy(inputBf16, inputGm[rowIdx * this->tileLength], this->tileLength);
        inQueueBf16.EnQue(inputBf16);

        // Step 2: Cast bf16 -> fp32 for computation
        AscendC::LocalTensor<bfloat16_t> bf16Local = inQueueBf16.DeQue<bfloat16_t>();
        AscendC::LocalTensor<float> fp32Local = inQueueFp32.AllocTensor<float>();
        AscendC::Cast(fp32Local, bf16Local, AscendC::RoundMode::CAST_NONE, this->tileLength);
        inQueueBf16.FreeTensor(bf16Local);
        inQueueFp32.EnQue(fp32Local);
    }

    __aicore__ inline void Compute(uint32_t rowIdx)
    {
        AscendC::LocalTensor<float> inputLocal = inQueueFp32.DeQue<float>();
        AscendC::LocalTensor<float> outputLocal = outQueueFp32.AllocTensor<float>();

        AscendC::LocalTensor<float> expLocalTensor = expBuf.Get<float>();
        AscendC::LocalTensor<float> sharedLocalTensor = sharedBuf.Get<float>();

        // Find max value in the row
        AscendC::ReduceMax(sharedLocalTensor, inputLocal, sharedLocalTensor, this->tileLength);
        float maxVal = sharedLocalTensor.GetValue(0);

        // Subtract max from all elements
        AscendC::Adds(expLocalTensor, inputLocal, -maxVal, this->tileLength);

        // Compute exponential
        AscendC::Exp(expLocalTensor, expLocalTensor, this->tileLength);

        // Compute sum of exponentials
        AscendC::ReduceSum(sharedLocalTensor, expLocalTensor, sharedLocalTensor, this->tileLength);

        // Divide by sum (scalar division avoids Reciprocal precision issues)
        float rowSum = sharedLocalTensor.GetValue(0);
        AscendC::Muls(outputLocal, expLocalTensor, 1.0f / rowSum, this->tileLength);

        outQueueFp32.EnQue<float>(outputLocal);
        inQueueFp32.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t rowIdx)
    {
        // Step 1: Get fp32 result
        AscendC::LocalTensor<float> fp32Out = outQueueFp32.DeQue<float>();

        // Step 2: Cast fp32 -> bf16
        AscendC::LocalTensor<bfloat16_t> bf16Out = outQueueBf16.AllocTensor<bfloat16_t>();
        AscendC::Cast(bf16Out, fp32Out, AscendC::RoundMode::CAST_RINT, this->tileLength);
        outQueueFp32.FreeTensor(fp32Out);
        outQueueBf16.EnQue(bf16Out);

        // Step 3: DataCopy bf16 to GM
        AscendC::LocalTensor<bfloat16_t> bf16Final = outQueueBf16.DeQue<bfloat16_t>();
        AscendC::DataCopy(outputGm[rowIdx * this->tileLength], bf16Final, this->tileLength);
        outQueueBf16.FreeTensor(bf16Final);
    }
};

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSoftmax op;
    op.Init(x, y, tiling_data.tileLength, tiling_data.baseRows, tiling_data.pivot);
    op.Process();
}
```
