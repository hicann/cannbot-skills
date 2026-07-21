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
host_tiling_src="""

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LeakyReluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, baseElems);
  TILING_DATA_FIELD_DEF(uint32_t, pivot);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerLoops);
  TILING_DATA_FIELD_DEF(float, alpha);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LeakyReluCustom, LeakyReluCustomTilingData)
}

"""

host_operator_src="""

#include "leaky_relu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  LeakyReluCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t total_elems = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    total_elems *= x1_shape->GetStorageShape().GetDim(i);

  // Dynamic core count: query Vector (AIV) core count at runtime
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  if (coreNum == 0) { coreNum = 1; }
  uint32_t usedCoreNum = ((uint32_t)total_elems < coreNum) ? (uint32_t)total_elems : coreNum;
  context->SetBlockDim(usedCoreNum);

  // Pivot distribution: first 'pivot' cores get (baseElems + 1) elements
  uint32_t baseElems = (uint32_t)total_elems / usedCoreNum;
  uint32_t pivot     = (uint32_t)total_elems % usedCoreNum;

  // Tiling strategy
  uint32_t tileSize   = 2048;
  uint32_t maxElems   = baseElems + (pivot > 0 ? 1 : 0);
  uint32_t innerLoops = (maxElems + tileSize - 1) / tileSize;

  // Get alpha attribute (first attribute, index 0)
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  const float* negativeSlope = attrs->GetAttrPointer<float>(0);

  tiling.set_baseElems(baseElems);
  tiling.set_pivot(pivot);
  tiling.set_tileSize(tileSize);
  tiling.set_innerLoops(innerLoops);
  tiling.set_alpha(*negativeSlope);

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
class LeakyReluCustom : public OpDef {
public:
    explicit LeakyReluCustom(const char* name) : OpDef(name)
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
        this->Attr("negative_slope").AttrType(OPTIONAL).Float(0.01);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(LeakyReluCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void leaky_relu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
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
BEGIN_TILING_DATA_DEF(LeakyReluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, baseElems);
  TILING_DATA_FIELD_DEF(uint32_t, pivot);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerLoops);
  TILING_DATA_FIELD_DEF(float, alpha);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LeakyReluCustom, LeakyReluCustomTilingData)
}

"""

host_operator_src="""

#include "leaky_relu_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  LeakyReluCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t total_elems = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    total_elems *= x1_shape->GetStorageShape().GetDim(i);

  // Dynamic core count: query Vector (AIV) core count at runtime
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  if (coreNum == 0) { coreNum = 1; }
  uint32_t usedCoreNum = ((uint32_t)total_elems < coreNum) ? (uint32_t)total_elems : coreNum;
  context->SetBlockDim(usedCoreNum);

  // Pivot distribution: first 'pivot' cores get (baseElems + 1) elements
  uint32_t baseElems = (uint32_t)total_elems / usedCoreNum;
  uint32_t pivot     = (uint32_t)total_elems % usedCoreNum;

  // Tiling strategy
  uint32_t tileSize   = 2048;
  uint32_t maxElems   = baseElems + (pivot > 0 ? 1 : 0);
  uint32_t innerLoops = (maxElems + tileSize - 1) / tileSize;

  // Get alpha attribute (first attribute, index 0)
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  const float* negativeSlope = attrs->GetAttrPointer<float>(0);

  tiling.set_baseElems(baseElems);
  tiling.set_pivot(pivot);
  tiling.set_tileSize(tileSize);
  tiling.set_innerLoops(innerLoops);
  tiling.set_alpha(*negativeSlope);

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
class LeakyReluCustom : public OpDef {
public:
    explicit LeakyReluCustom(const char* name) : OpDef(name)
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
        this->Attr("negative_slope").AttrType(OPTIONAL).Float(0.01);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(LeakyReluCustom);
}

"""

kernel_src="""
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
"""
