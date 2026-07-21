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
host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxCustomTilingData)
}

"""

host_operator_src="""

#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  SoftmaxCustomTilingData tiling;
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
class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(SoftmaxCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
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
BEGIN_TILING_DATA_DEF(SoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, baseRows);   // base rows per core (floor division)
  TILING_DATA_FIELD_DEF(uint32_t, pivot);       // first 'pivot' cores get baseRows+1
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SoftmaxCustom, SoftmaxCustomTilingData)
}

"""

host_operator_src="""

#include "softmax_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SoftmaxCustomTilingData tiling;
    const gert::StorageShape* input_shape = context->GetInputShape(0);

    // Extract shape dimensions for 2D tensor
    uint32_t nRows = input_shape->GetStorageShape().GetDim(0);
    uint32_t nCols = input_shape->GetStorageShape().GetDim(1);

    // ---- Dynamic core count: query hardware, never hard-code ----
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) { coreNum = 1; }
    uint32_t usedCoreNum = (nRows < coreNum) ? nRows : coreNum;
    context->SetBlockDim(usedCoreNum);

    // ---- Pivot distribution: handles non-divisible row count ----
    uint32_t baseRows = nRows / usedCoreNum;
    uint32_t pivot    = nRows % usedCoreNum;

    tiling.set_tileLength(nCols);
    tiling.set_baseRows(baseRows);
    tiling.set_pivot(pivot);

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
class SoftmaxCustom : public OpDef {
public:
    explicit SoftmaxCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(SoftmaxCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
"""
