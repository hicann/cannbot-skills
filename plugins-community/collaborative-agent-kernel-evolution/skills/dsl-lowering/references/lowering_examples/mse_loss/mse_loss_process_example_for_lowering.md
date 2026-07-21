### Example input dsl
```python
import tile.language as tl

@ascend_kernel
def mse_loss_kernel(
    pred_ptr,
    target_ptr,
    workspace,
    output_ptr,
    base_elems,
    pivot,
    tile_size,
    inner_loops,
    total_elems,
    n_used
):
    pid = tl.program_id(0)
    my_elems = base_elems + (1 if pid < pivot else 0)
    start = pid * base_elems + min(pid, pivot)

    pred_ub    = tl.alloc_ub(tile_size, dtype=tl.float32)
    target_ub  = tl.alloc_ub(tile_size, dtype=tl.float32)
    diff_ub    = tl.alloc_ub(tile_size, dtype=tl.float32)
    sq_ub      = tl.alloc_ub(tile_size, dtype=tl.float32)
    shared_ub  = tl.alloc_ub(tile_size, dtype=tl.float32)
    workspace_out_ub = tl.alloc_ub(tile_size, dtype=tl.float32)
    workspace_in_ub   = tl.alloc_ub(tile_size, dtype=tl.float32)
    output_ub   = tl.alloc_ub(tile_size, dtype=tl.float32)

    partial_sum = 0.0

    for i in range(inner_loops):
        tile_start = start + i * tile_size
        offsets = tile_start + tl.arange(0, tile_size)

        with tl.copyin():
            tl.load(pred_ptr   + offsets, pred_ub)
            tl.load(target_ptr + offsets, target_ub)

        with tl.compute():
            tl.vsub(diff_ub, pred_ub, target_ub)
            tl.vmul(sq_ub, diff_ub, diff_ub)
            tl.reduce_sum(sq_ub, sq_ub, shared_ub)
            tile_sum = extract_scalar(sq_ub, 0)
            partial_sum = partial_sum + tile_sum

    with tl.copyout():
        tl.set_scalar(workspace_out_ub, 0, partial_sum)
        tl.store(workspace+tl.arange(pid,pid+1), workspace_out_ub)

    if pid == 0:

        with tl.copyin():
            tl.load(workspace + tl.arange(0, tl.num_programs(0)), workspace_in_ub)

        with tl.compute():
            tl.reduce_sum(shared_ub, workspace_in_ub, shared_ub)
            sum_sq = extract_scalar(shared_ub, 0)
            mse = sum_sq / float(total_elems)
            tl.set_scalar(output_ub, 0, mse)

        with tl.copyout():
           tl.store(output_ptr+tl.arange(0, 1), output_ub)

def mse_loss_host(pred: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    total_elems = pred.numel()

    # Core Partitioning: dynamically query Vector core count
    n_cores = tl.num_vec_cores()
    n_used  = min(n_cores, total_elems)

    # Pivot-based distribution: each core processes either base_elems or base_elems + 1 elements.
    base_elems = total_elems // n_used
    remainder = total_elems % n_used
    pivot = remainder

    workspace = torch.empty(n_used, dtype=torch.float32, device=pred.device)

    tile_size = 2048
    max_elems = base_elems + (1 if pivot > 0 else 0)
    inner_loops = (max_elems + tile_size - 1) // tile_size

    mse_loss_kernel[n_used](
        pred, target, workspace, output,
        base_elems, pivot, tile_size, inner_loops, total_elems, n_used
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
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workspaceInQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> workspaceOutQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diffBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sqBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sharedBuf;
    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<float> targetGm;
    AscendC::GlobalTensor<float> outputGm;
    AscendC::GlobalTensor<float> workspaceGm;
    uint32_t totalElems;
    uint32_t baseElems;
    uint32_t pivot;
    uint32_t tileSize;
    uint32_t innerLoops;
    uint32_t nUsed;
    uint32_t programId;

public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR pred_ptr, GM_ADDR target_ptr, GM_ADDR output_ptr, GM_ADDR workspace_ptr,
                                 uint32_t totalElems, uint32_t baseElems, uint32_t pivot,
                                 uint32_t tileSize, uint32_t innerLoops, uint32_t nUsed)
    {
        this->totalElems = totalElems;
        this->baseElems  = baseElems;
        this->pivot      = pivot;
        this->tileSize   = tileSize;
        this->innerLoops = innerLoops;
        this->nUsed      = nUsed;
        this->programId  = AscendC::GetBlockIdx();

        uint32_t myElems = baseElems + (programId < pivot ? 1 : 0);
        uint32_t start   = programId * baseElems + (programId < pivot ? programId : pivot);

        predGm.SetGlobalBuffer((__gm__ float *)pred_ptr + start, myElems);
        targetGm.SetGlobalBuffer((__gm__ float *)target_ptr + start, myElems);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace_ptr, nUsed);
        outputGm.SetGlobalBuffer((__gm__ float *)output_ptr, 1);

        pipe.InitBuffer(predQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(targetQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(workspaceInQueue, 1, nUsed * sizeof(float));

        pipe.InitBuffer(workspaceOutQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outputQueue, 1, this->tileSize * sizeof(float));

        pipe.InitBuffer(diffBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(sqBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(sharedBuf, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // TODO
    }

};

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLoss op;
    op.Init(predictions, targets, y, workspace, tiling_data.totalElems, tiling_data.baseElems, tiling_data.pivot, tiling_data.tileSize, tiling_data.innerLoops, tiling_data.nUsed);
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
    AscendC::TQue<AscendC::TPosition::VECIN, 1> workspaceInQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> workspaceOutQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diffBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sqBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sharedBuf;
    AscendC::GlobalTensor<float> predGm;
    AscendC::GlobalTensor<float> targetGm;
    AscendC::GlobalTensor<float> outputGm;
    AscendC::GlobalTensor<float> workspaceGm;
    uint32_t totalElems;
    uint32_t baseElems;
    uint32_t pivot;
    uint32_t tileSize;
    uint32_t innerLoops;
    uint32_t nUsed;
    uint32_t programId;

public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR pred_ptr, GM_ADDR target_ptr, GM_ADDR output_ptr, GM_ADDR workspace_ptr,
                                 uint32_t totalElems, uint32_t baseElems, uint32_t pivot,
                                 uint32_t tileSize, uint32_t innerLoops, uint32_t nUsed)
    {
        this->totalElems = totalElems;
        this->baseElems  = baseElems;
        this->pivot      = pivot;
        this->tileSize   = tileSize;
        this->innerLoops = innerLoops;
        this->nUsed      = nUsed;
        this->programId  = AscendC::GetBlockIdx();

        uint32_t myElems = baseElems + (programId < pivot ? 1 : 0);
        uint32_t start   = programId * baseElems + (programId < pivot ? programId : pivot);

        predGm.SetGlobalBuffer((__gm__ float *)pred_ptr + start, myElems);
        targetGm.SetGlobalBuffer((__gm__ float *)target_ptr + start, myElems);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace_ptr, nUsed);
        outputGm.SetGlobalBuffer((__gm__ float *)output_ptr, 1);

        pipe.InitBuffer(predQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(targetQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(workspaceInQueue, 1, nUsed * sizeof(float));

        pipe.InitBuffer(workspaceOutQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outputQueue, 1, this->tileSize * sizeof(float));

        pipe.InitBuffer(diffBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(sqBuf, this->tileSize * sizeof(float));
        pipe.InitBuffer(sharedBuf, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // Phase 1: per-core partial reduction of squared errors
        float partialSum = 0.0f;

        for (uint32_t i = 0; i < this->innerLoops; i++) {
            CopyIn1(i);
            Compute1(i, partialSum);
        }

        CopyOut1(partialSum);
        AscendC::SyncAll();

        // Phase 2: Core 0 performs final reduce + mean
        if (programId == 0) {
            CopyIn2();
            Compute2();
            CopyOut2();
        }
    }

private:
    __aicore__ inline void CopyIn1(uint32_t i)
    {
        AscendC::LocalTensor<float> predLocal = predQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> targetLocal = targetQueue.AllocTensor<float>();

        uint32_t tileStart = i * this->tileSize;
        AscendC::DataCopy(predLocal, predGm[tileStart], this->tileSize);
        AscendC::DataCopy(targetLocal, targetGm[tileStart], this->tileSize);

        predQueue.EnQue(predLocal);
        targetQueue.EnQue(targetLocal);
    }

    __aicore__ inline void Compute1(uint32_t i, float& partialSum)
    {
        AscendC::LocalTensor<float> predLocal = predQueue.DeQue<float>();
        AscendC::LocalTensor<float> targetLocal = targetQueue.DeQue<float>();
        AscendC::LocalTensor<float> diffLocal = diffBuf.Get<float>();
        AscendC::LocalTensor<float> sqLocal = sqBuf.Get<float>();
        AscendC::LocalTensor<float> sharedLocal = sharedBuf.Get<float>();

        AscendC::Sub(diffLocal, predLocal, targetLocal, this->tileSize);
        AscendC::Mul(sqLocal, diffLocal, diffLocal, this->tileSize);
        AscendC::ReduceSum(sqLocal, sqLocal, sharedLocal, this->tileSize);
        float tileSum = sqLocal.GetValue(0);
        partialSum = partialSum + tileSum;

        predQueue.FreeTensor(predLocal);
        targetQueue.FreeTensor(targetLocal);
    }

    __aicore__ inline void CopyOut1(float partialSum)
    {
        AscendC::LocalTensor<float> uploadWorkspaceLocal = workspaceOutQueue.AllocTensor<float>();
        uploadWorkspaceLocal.SetValue(0, partialSum);
        AscendC::DataCopy(workspaceGm[programId], uploadWorkspaceLocal, 1);
        workspaceOutQueue.FreeTensor(uploadWorkspaceLocal);
    }

    __aicore__ inline void CopyIn2()
    {
        AscendC::LocalTensor<float> workspaceLocal = workspaceInQueue.AllocTensor<float>();
        // Load all partial sums: nUsed elements (dynamic, not hardcoded)
        AscendC::DataCopy(workspaceLocal, workspaceGm[0], this->nUsed);
        workspaceInQueue.EnQue(workspaceLocal);
    }

    __aicore__ inline void Compute2()
    {
        AscendC::LocalTensor<float> workspaceLocal = workspaceInQueue.DeQue<float>();
        AscendC::LocalTensor<float> sharedLocal = sharedBuf.Get<float>();
        AscendC::LocalTensor<float> outputLocal = outputQueue.AllocTensor<float>();

        // Reduce sum across all cores (nUsed, not hardcoded)
        AscendC::ReduceSum(sharedLocal, workspaceLocal, sharedLocal, this->nUsed);
        float sumSq = sharedLocal.GetValue(0);
        float mse = sumSq / this->totalElems;

        outputLocal.SetValue(0, mse);

        workspaceInQueue.FreeTensor(workspaceLocal);
        outputQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut2()
    {
        AscendC::LocalTensor<float> outputLocal = outputQueue.DeQue<float>();
        AscendC::DataCopy(outputGm[0], outputLocal, 1);
        outputQueue.FreeTensor(outputLocal);
    }
};

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLoss op;
    op.Init(predictions, targets, y, workspace, tiling_data.totalElems, tiling_data.baseElems, tiling_data.pivot, tiling_data.tileSize, tiling_data.innerLoops, tiling_data.nUsed);
    op.Process();
}
```
