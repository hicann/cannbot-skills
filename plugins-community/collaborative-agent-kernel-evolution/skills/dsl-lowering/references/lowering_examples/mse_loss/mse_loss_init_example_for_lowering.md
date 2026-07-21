### Example input dsl
```python
import tile.language as tl

@ascend_kernel
def mse_loss_kernel(
    pred_ptr,                 # [N]
    target_ptr,               # [N]
    workspace,               # [n_used]
    output_ptr,               # final scalar
    elems_per_core,
    tile_size,
    inner_loops,
    total_elems,              # for computing mean
    n_used                    # actual launched core count
):
    pid = tl.program_id(0)
    start = pid * elems_per_core

    # ------------------------------------------------------------
    # UB Buffers
    # ------------------------------------------------------------
    pred_ub    = tl.alloc_ub(tile_size, dtype=tl.float32)
    target_ub  = tl.alloc_ub(tile_size, dtype=tl.float32)
    diff_ub    = tl.alloc_ub(tile_size, dtype=tl.float32)
    sq_ub      = tl.alloc_ub(tile_size, dtype=tl.float32)
    shared_ub  = tl.alloc_ub(tile_size, dtype=tl.float32)
    workspace_out_ub = tl.alloc_ub(tile_size, dtype=tl.float32)
    workspace_in_ub   = tl.alloc_ub(tile_size, dtype=tl.float32)
    output_ub   = tl.alloc_ub(tile_size, dtype=tl.float32)

    # ------------------------------------------------------------
    # Phase 1: per-core partial reduction of squared errors
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Write per-core partial sum
    # ------------------------------------------------------------
    with tl.copyout():
        tl.set_scalar(workspace_out_ub, 0, partial_sum)
        tl.store(workspace+tl.arange(pid,pid+1), workspace_out_ub)

    # ------------------------------------------------------------
    # Phase 2: Core 0 performs final reduce + mean
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Core Partitioning: dynamically query Vector core count
    # ------------------------------------------------------------
    n_cores = tl.num_vec_cores()
    n_used  = min(n_cores, total_elems)
    elems_per_core = total_elems // n_used

    # GM buffer for per-core partial results
    workspace = torch.empty(n_used, dtype=torch.float32, device=pred.device)

    # ------------------------------------------------------------
    # Tiling Strategy
    # ------------------------------------------------------------
    tile_size = 2048
    inner_loops = elems_per_core // tile_size

    # ------------------------------------------------------------
    # Launch kernel
    # ------------------------------------------------------------
    mse_loss_kernel[n_used](
        pred,
        target,
        workspace,
        output,
        elems_per_core,
        tile_size,
        inner_loops,
        total_elems,
        n_used
    )


```

### Example input AscendC
```
host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalElems);
  TILING_DATA_FIELD_DEF(uint32_t, elemsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerLoops);
  TILING_DATA_FIELD_DEF(uint32_t, nUsed);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossCustom, MseLossCustomTilingData)
}
"""

host_operator_src="""
#include "mse_loss_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t TILE_SIZE = 2048;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  MseLossCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);

  // Calculate total number of elements
  uint32_t totalElems = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
    totalElems *= x1_shape->GetStorageShape().GetDim(i);
  }

  // Dynamic core count: query Vector (AIV) core count at runtime
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  if (coreNum == 0) { coreNum = 1; }
  uint32_t usedCoreNum = (totalElems < coreNum) ? totalElems : coreNum;
  context->SetBlockDim(usedCoreNum);

  // Core partitioning
  uint32_t elemsPerCore = totalElems / usedCoreNum;

  // Tiling strategy
  uint32_t innerLoops = elemsPerCore / TILE_SIZE;

  tiling.set_totalElems(totalElems);
  tiling.set_elemsPerCore(elemsPerCore);
  tiling.set_tileSize(TILE_SIZE);
  tiling.set_innerLoops(innerLoops);
  tiling.set_nUsed(usedCoreNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  // Set workspace size for per-core partial results (dynamic: usedCoreNum floats)
  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t requiredWorkspaceBytes = usedCoreNum * sizeof(float);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = requiredWorkspaceBytes + sysWorkspaceSize;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class MseLossCustom : public OpDef {
public:
    explicit MseLossCustom(const char* name) : OpDef(name)
    {
        this->Input("predictions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("targets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(MseLossCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
"""
```

### Example output AscendC
```
host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalElems);
  TILING_DATA_FIELD_DEF(uint32_t, elemsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerLoops);
  TILING_DATA_FIELD_DEF(uint32_t, nUsed);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossCustom, MseLossCustomTilingData)
}
"""

host_operator_src="""
#include "mse_loss_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t TILE_SIZE = 2048;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  MseLossCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);

  // Calculate total number of elements
  uint32_t totalElems = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
    totalElems *= x1_shape->GetStorageShape().GetDim(i);
  }

  // Dynamic core count: query Vector (AIV) core count at runtime
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  if (coreNum == 0) { coreNum = 1; }
  uint32_t usedCoreNum = (totalElems < coreNum) ? totalElems : coreNum;
  context->SetBlockDim(usedCoreNum);

  // Core partitioning
  uint32_t elemsPerCore = totalElems / usedCoreNum;

  // Tiling strategy
  uint32_t innerLoops = elemsPerCore / TILE_SIZE;

  tiling.set_totalElems(totalElems);
  tiling.set_elemsPerCore(elemsPerCore);
  tiling.set_tileSize(TILE_SIZE);
  tiling.set_innerLoops(innerLoops);
  tiling.set_nUsed(usedCoreNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  // Set workspace size for per-core partial results (dynamic: usedCoreNum floats)
  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t requiredWorkspaceBytes = usedCoreNum * sizeof(float);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = requiredWorkspaceBytes + sysWorkspaceSize;

  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class MseLossCustom : public OpDef {
public:
    explicit MseLossCustom(const char* name) : OpDef(name)
    {
        this->Input("predictions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("targets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(MseLossCustom);
}

"""

kernel_src="""
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
    uint32_t elemsPerCore;
    uint32_t tileSize;
    uint32_t innerLoops;
    uint32_t nUsed;
    uint32_t programId;

public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR pred_ptr, GM_ADDR target_ptr, GM_ADDR output_ptr, GM_ADDR workspace_ptr,
                                 uint32_t totalElems, uint32_t elemsPerCore, uint32_t tileSize,
                                 uint32_t innerLoops, uint32_t nUsed)
    {
        this->totalElems   = totalElems;
        this->elemsPerCore = elemsPerCore;
        this->tileSize     = tileSize;
        this->innerLoops   = innerLoops;
        this->nUsed        = nUsed;
        this->programId    = AscendC::GetBlockIdx();

        uint32_t start = programId * elemsPerCore;

        // Set global memory buffers
        predGm.SetGlobalBuffer((__gm__ float *)pred_ptr + start, elemsPerCore);
        targetGm.SetGlobalBuffer((__gm__ float *)target_ptr + start, elemsPerCore);
        workspaceGm.SetGlobalBuffer((__gm__ float *)workspace_ptr, nUsed);  // dynamic size
        outputGm.SetGlobalBuffer((__gm__ float *)output_ptr, 1);

        // Initialize input queues for copyin
        pipe.InitBuffer(predQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(targetQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(workspaceInQueue, 1, nUsed * sizeof(float));  // for loading workspace in core 0

        // Initialize output queues for copyout
        pipe.InitBuffer(workspaceOutQueue, 1, this->tileSize * sizeof(float));
        pipe.InitBuffer(outputQueue, 1, this->tileSize * sizeof(float));

        // Initialize intermediate/working buffers for compute
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
    op.Init(predictions, targets, y, workspace, tiling_data.totalElems, tiling_data.elemsPerCore, tiling_data.tileSize, tiling_data.innerLoops, tiling_data.nUsed);
    op.Process();
}

"""
