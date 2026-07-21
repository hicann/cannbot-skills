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
```
host_tiling_src="""

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossCustom, MseLossCustomTilingData)
}

"""

host_operator_src="""

#include "mse_loss_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  MseLossCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  context->SetBlockDim(8);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

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
```python
host_tiling_src="""

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, nElements);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, nTiles);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MseLossCustom, MseLossCustomTilingData)
}

"""

host_operator_src="""

#include "mse_loss_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 1;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  MseLossCustomTilingData tiling;
  const gert::StorageShape* pred_shape = context->GetInputShape(0);
  
  // Calculate total number of elements
  uint32_t nElements = 1;
  for (int i = 0; i < pred_shape->GetStorageShape().GetDimNum(); i++)
    nElements *= pred_shape->GetStorageShape().GetDim(i);
  
  context->SetBlockDim(BLOCK_DIM);
  
  // Tiling parameters
  uint32_t tileLength = 1024;
  uint32_t nTiles = (nElements + tileLength - 1) / tileLength;
  
  tiling.set_nElements(nElements);
  tiling.set_tileLength(tileLength);
  tiling.set_nTiles(nTiles);
  
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;

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