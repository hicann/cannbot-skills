### Example input dsl
```python
import tile.language as tl
from vecpipe import ascend_kernel

@ascend_kernel
def mse_loss_kernel(pred_ptr, target_ptr, out_ptr,
                    n_elements, tile_length, n_tiles):

    pid = tl.program_id(0)  # single core

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    pred_ub     = tl.alloc_ub(tile_length, dtype=tl.float32)
    target_ub   = tl.alloc_ub(tile_length, dtype=tl.float32)
    diff_ub     = tl.alloc_ub(tile_length, dtype=tl.float32)
    sq_ub       = tl.alloc_ub(tile_length, dtype=tl.float32)

    acc_ub      = tl.alloc_ub(tile_length, dtype=tl.float32)
    shared_ub   = tl.alloc_ub(tile_length, dtype=tl.float32)

    out_ub      = tl.alloc_ub(1, dtype=tl.float32)   # ✅ explicit output transfer UB

    # ------------------------------------------------------------
    # Initialize accumulator
    # ------------------------------------------------------------
    with tl.compute():
        tl.duplicate(acc_ub, 0.0)

    # ------------------------------------------------------------
    # Tiled reduction over all elements
    # ------------------------------------------------------------
    for tile_id in range(n_tiles):

        tile_start = tile_id * tile_length
        offsets = tile_start + tl.arange(0, tile_length)

        # ---- Load tiles ----
        with tl.copyin():
            tl.load(pred_ptr + offsets, pred_ub)
            tl.load(target_ptr + offsets, target_ub)

        # ---- Compute squared error ----
        with tl.compute():
            tl.vsub(diff_ub, pred_ub, target_ub)
            tl.vmul(sq_ub, diff_ub, diff_ub)
            tl.vadd(acc_ub, acc_ub, sq_ub)

    # ------------------------------------------------------------
    # Final reduction and mean
    # ------------------------------------------------------------
    with tl.compute():
        tl.reduce_sum(shared_ub, acc_ub, shared_ub)
        total_sum = tl.extract_scalar(shared_ub, 0)
        mean_val  = total_sum / n_elements

        # write scalar into UB buffer
        tl.duplicate(out_ub, mean_val, count=1)

    # ------------------------------------------------------------
    # Copy-out
    # ------------------------------------------------------------
    with tl.copyout():
        tl.store(out_ptr, out_ub)

def mse_loss_host(pred: torch.Tensor, target: torch.Tensor, output: torch.Tensor):

    # ------------------------------------------------------------
    # Core Partitioning
    # ------------------------------------------------------------
    # Use one core for simplicity
    n_cores = 1

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    # UB can comfortably hold a few KB; we tile the 1D vector
    n_elements = pred.numel()

    tile_length = 1024          # matches Triton TILE_SIZE
    n_tiles = (n_elements + tile_length - 1) // tile_length

    # ------------------------------------------------------------
    # Kernel Launch
    # ------------------------------------------------------------
    mse_loss_kernel[n_cores](
        pred,
        target,
        output,
        n_elements,
        tile_length,
        n_tiles
    )

```

### Example input AscendC
```cpp
#include "kernel_operator.h"

class KernelMseLoss {
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> predQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> targetQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diffBuf, sqBuf, accBuf, sharedBuf;
    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<float> targetGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t nElements;
    uint32_t tileLength;
    uint32_t nTiles;

public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR pred_ptr, GM_ADDR target_ptr, GM_ADDR out_ptr, uint32_t nElements, uint32_t tileLength, uint32_t nTiles)
    {
        this->nElements = nElements;
        this->tileLength = tileLength;
        this->nTiles = nTiles;

        // Set global memory buffer. Single core processes all elements
        predGm.SetGlobalBuffer((__gm__ float *)pred_ptr, nElements);
        targetGm.SetGlobalBuffer((__gm__ float *)target_ptr, nElements);
        outputGm.SetGlobalBuffer((__gm__ float *)out_ptr, 1);

        // Initialize pipe buffer queues with one slot, each holding tileLength floats
        pipe.InitBuffer(predQueue, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(targetQueue, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, 1, 1 * sizeof(float));
        pipe.InitBuffer(diffBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(sqBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(accBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(sharedBuf, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // TODO implemented
    }
};

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLoss op;
    op.Init(predictions, targets, y, tiling_data.nElements, tiling_data.tileLength, tiling_data.nTiles);
    op.Process();
}
```

### Example output AscendC
```cpp
#include "kernel_operator.h"

class KernelMseLoss {
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> predQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> targetQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diffBuf, sqBuf, accBuf, sharedBuf;
    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<float> targetGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t nElements;
    uint32_t tileLength;
    uint32_t nTiles;

public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR pred_ptr, GM_ADDR target_ptr, GM_ADDR out_ptr, uint32_t nElements, uint32_t tileLength, uint32_t nTiles)
    {
        this->nElements = nElements;
        this->tileLength = tileLength;
        this->nTiles = nTiles;

        // Set global memory buffer. Single core processes all elements
        predGm.SetGlobalBuffer((__gm__ float *)pred_ptr, nElements);
        targetGm.SetGlobalBuffer((__gm__ float *)target_ptr, nElements);
        outputGm.SetGlobalBuffer((__gm__ float *)out_ptr, 1);

        // Initialize pipe buffer queues with one slot, each holding tileLength floats
        pipe.InitBuffer(predQueue, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(targetQueue, 1, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, 1, 1 * sizeof(float));
        pipe.InitBuffer(diffBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(sqBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(accBuf, this->tileLength * sizeof(float));
        pipe.InitBuffer(sharedBuf, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // Initialize accumulator
        ComputeInit();
        
        // Tiled reduction over all elements
        for (uint32_t tileId = 0; tileId < this->nTiles; tileId++) {
            CopyIn(tileId);
            Compute(tileId);
        }
        
        // Final reduction and mean
        ComputeFinal();
        
        // Copy-out
        CopyOut();
    }

private:
    __aicore__ inline void ComputeInit()
    {
        AscendC::LocalTensor<float> accLocal = accBuf.Get<float>();
        AscendC::Duplicate(accLocal, 0.0f, this->tileLength);
    }
    
    __aicore__ inline void CopyIn(uint32_t tileId)
    {
        AscendC::LocalTensor<float> predLocal = predQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> targetLocal = targetQueue.AllocTensor<float>();
        
        uint32_t tileStart = tileId * this->tileLength;
        AscendC::DataCopy(predLocal, predGm[tileStart], this->tileLength);
        AscendC::DataCopy(targetLocal, targetGm[tileStart], this->tileLength);
        
        predQueue.EnQue(predLocal);
        targetQueue.EnQue(targetLocal);
    }
    
    __aicore__ inline void Compute(uint32_t tileId)
    {
        AscendC::LocalTensor<float> predLocal = predQueue.DeQue<float>();
        AscendC::LocalTensor<float> targetLocal = targetQueue.DeQue<float>();
        
        AscendC::LocalTensor<float> diffLocal = diffBuf.Get<float>();
        AscendC::LocalTensor<float> sqLocal = sqBuf.Get<float>();
        AscendC::LocalTensor<float> accLocal = accBuf.Get<float>();
        
        // Compute squared error
        AscendC::Sub(diffLocal, predLocal, targetLocal, this->tileLength);
        AscendC::Mul(sqLocal, diffLocal, diffLocal, this->tileLength);
        AscendC::Add(accLocal, accLocal, sqLocal, this->tileLength);
        
        predQueue.FreeTensor(predLocal);
        targetQueue.FreeTensor(targetLocal);
    }
    
    __aicore__ inline void ComputeFinal()
    {
        AscendC::LocalTensor<float> accLocal = accBuf.Get<float>();
        AscendC::LocalTensor<float> sharedLocal = sharedBuf.Get<float>();
        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        
        // Reduce sum
        AscendC::ReduceSum(sharedLocal, accLocal, sharedLocal, this->tileLength);
        float totalSum = sharedLocal.GetValue(0);
        
        // Compute mean
        float meanVal = totalSum / this->nElements;
        
        // Write scalar into UB buffer
        AscendC::Duplicate(outLocal, meanVal, 1);
        
        outQueue.EnQue(outLocal);
    }
    
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> outLocal = outQueue.DeQue<float>();
        AscendC::DataCopy(outputGm[0], outLocal, 1);
        outQueue.FreeTensor(outLocal);
    }
};

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLoss op;
    op.Init(predictions, targets, y, tiling_data.nElements, tiling_data.tileLength, tiling_data.nTiles);
    op.Process();
}
```