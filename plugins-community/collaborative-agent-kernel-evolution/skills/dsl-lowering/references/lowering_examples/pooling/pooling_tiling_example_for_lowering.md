### Example input dsl
```python
import torch
import tile.language as tl


@ascend_kernel
def avgpool2d_kernel(
    input_ptr, output_ptr,
    batch_size, height, width, channels,
    out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    padding_h, padding_w,
    tile_w,
    total_tasks,
    tasks_per_core
):
    # Kernel
    # Each core:
    #   - decodes its task_id into (b, oh, ow_group)
    #   - loads tile_w * channels elements into UB
    #   - computes windows_per_load output windows
    #   - handles padding and stride
    #   - safely handles tail windows (ow >= out_w)

    pid = tl.program_id(0)

    task_start = pid * tasks_per_core
    task_end   = min(task_start + tasks_per_core, total_tasks)

    windows_per_load = tile_w // stride_w
    kernel_area = kernel_h * kernel_w
    groups_per_row = (out_w + windows_per_load - 1) // windows_per_load

    # ------------------------------------------------
    # UB buffers
    # ------------------------------------------------
    # bf16 variant: GM data is bfloat16, UB computation in float32
    # Cast bf16<->f32 handled during lowering
    x_ub   = tl.alloc_ub(tile_w * channels, dtype=tl.float32)
    sum_ub = tl.alloc_ub(windows_per_load * channels, dtype=tl.float32)
    cnt_ub = tl.alloc_ub(windows_per_load * channels, dtype=tl.float32)
    out_ub = tl.alloc_ub(windows_per_load * channels, dtype=tl.float32)

    for task_id in range(task_start, task_end):

        # --------------------------------------------
        # decode task
        # --------------------------------------------
        b  = task_id // (out_h * groups_per_row)
        tmp = task_id %  (out_h * groups_per_row)
        oh = tmp // groups_per_row
        ow_group = tmp % groups_per_row

        ow0 = ow_group * windows_per_load

        # --------------------------------------------
        # clear accumulators
        # --------------------------------------------
        with tl.compute():
            tl.duplicate(sum_ub, 0.0, count=windows_per_load * channels)
            tl.duplicate(cnt_ub, 0.0, count=windows_per_load * channels)

        # --------------------------------------------
        # pooling with padding and stride
        # --------------------------------------------
        for kh in range(kernel_h):
            # Compute input row index with padding
            ih = oh * stride_h - padding_h + kh
            
            # Skip if outside valid input range
            if ih < 0 or ih >= height:
                continue

            for w in range(windows_per_load):
                ow = ow0 + w
                if ow >= out_w:
                    continue

                # Compute starting input column for this output window
                w0 = ow * stride_w - padding_w

                # Process each element in the kernel width
                for kw in range(kernel_w):
                    iw = w0 + kw
                    
                    # Skip if outside valid input range
                    if iw < 0 or iw >= width:
                        continue

                    # Load input element
                    base = (
                        b * height * width * channels +
                        ih * width * channels +
                        iw * channels
                    )

                    with tl.copyin():
                        tl.load(
                            input_ptr + base + tl.arange(0, channels),
                            x_ub
                        )

                    # Accumulate sum and count
                    with tl.compute():
                        sum_ptr = sum_ub + w * channels
                        cnt_ptr = cnt_ub + w * channels
                        
                        tl.vadd(sum_ptr, sum_ptr, x_ub, channels)
                        tl.vadd_scalar(cnt_ptr, cnt_ptr, 1.0, channels)

        # --------------------------------------------
        # divide sum by count
        # --------------------------------------------
        with tl.compute():
            tl.vdiv(out_ub, sum_ub, cnt_ub, windows_per_load * channels)

        # --------------------------------------------
        # store
        # --------------------------------------------
        with tl.copyout():
            for w in range(windows_per_load):
                ow = ow0 + w
                if ow < out_w:
                    out_base = (
                        b * out_h * out_w * channels +
                        oh * out_w * channels +
                        ow * channels
                    )
                    tl.store(
                        output_ptr + out_base + tl.arange(0, channels),
                        out_ub + w * channels
                    )


def avgpool2d_host(
    x: torch.Tensor,
    output: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int
):
    batch, height, width, channels = x.shape

    kernel_h = kernel_size
    kernel_w = kernel_size
    stride_h = stride
    stride_w = stride
    padding_h = padding
    padding_w = padding

    # ------------------------------------------------------------
    # Output spatial dimensions
    # ------------------------------------------------------------
    # Standard pooling output size formula:
    #   out_size = floor((in_size + 2*padding - kernel_size) / stride) + 1
    #
    out_h = (height + 2 * padding_h - kernel_h) // stride_h + 1
    out_w = (width + 2 * padding_w - kernel_w) // stride_w + 1

    # ============================================================
    # UB tiling strategy (width dimension)
    # ============================================================
    # tile_w:
    #   Number of *input* width elements that can be processed
    #   in a single load group.
    #
    # For stride = 1, kernel_w = 3:
    #   tile_w = 18 means we can compute multiple overlapping windows
    #
    # For stride = kernel_size (non-overlapping):
    #   tile_w should be a multiple of stride_w for efficiency
    #
    # We choose tile_w based on stride to maximize UB utilization
    # while ensuring good memory coalescing.
    #
    if stride_w == 1:
        tile_w = 18
    else:
        # For larger strides, use a multiple of stride
        tile_w = max(stride_w * 6, kernel_w)

    # windows_per_load:
    #   How many output windows we compute per task iteration
    #
    # For strided pooling, this is approximately tile_w / stride_w
    #
    windows_per_load = max(1, tile_w // stride_w)

    n_cores = tl.num_vec_cores()

    # ============================================================
    # Task decomposition
    # ============================================================
    # We define one "task" as:
    #   - one batch index (b)
    #   - one output row (oh)
    #   - one *group* of output columns
    #
    # Each group computes:
    #   windows_per_load output columns
    #
    groups_per_row = (out_w + windows_per_load - 1) // windows_per_load

    # Total number of independent tasks in the whole workload
    #
    # Task space:
    #   [batch, out_h, groups_per_row]
    #
    total_tasks = batch * out_h * groups_per_row

    # ============================================================
    # Task-to-core mapping
    # ============================================================
    # We use a simple linear partition:
    #
    #   Each core processes a contiguous range of tasks.
    #
    # Ceiling division ensures:
    #   - all tasks are covered
    #   - some cores may process one extra task
    #
    n_used = min(n_cores, total_tasks)
    tasks_per_core = (total_tasks + n_used - 1) // n_used

    # ============================================================
    # Launch kernel
    # ============================================================
    avgpool2d_kernel[n_used](
        x, output,
        batch, height, width, channels,
        out_h, out_w,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        tile_w,
        total_tasks,
        tasks_per_core
    )
```
### Example input AscendC
```
host_tiling_src="""

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AveragePooling2dCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AveragePooling2dCustom, AveragePooling2dCustomTilingData)
}

"""

host_operator_src="""

#include "average_pooling2d_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AveragePooling2dCustomTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  // Dynamic core count
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  if (coreNum == 0) { coreNum = 1; }
  context->SetBlockDim(coreNum);
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
class AveragePooling2dCustom : public OpDef {
public:
    explicit AveragePooling2dCustom(const char* name) : OpDef(name)
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
        this->Attr("kernel_size").Int();
        this->Attr("stride").Int();
        this->Attr("padding").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(AveragePooling2dCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void average_pooling2d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
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
BEGIN_TILING_DATA_DEF(AveragePooling2dCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, outH);
  TILING_DATA_FIELD_DEF(uint32_t, outW);
  TILING_DATA_FIELD_DEF(uint32_t, kernelH);
  TILING_DATA_FIELD_DEF(uint32_t, kernelW);
  TILING_DATA_FIELD_DEF(uint32_t, strideH);
  TILING_DATA_FIELD_DEF(uint32_t, strideW);
  TILING_DATA_FIELD_DEF(uint32_t, paddingH);
  TILING_DATA_FIELD_DEF(uint32_t, paddingW);
  TILING_DATA_FIELD_DEF(uint32_t, tileW);
  TILING_DATA_FIELD_DEF(uint32_t, totalTasks);
  TILING_DATA_FIELD_DEF(uint32_t, tasksPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AveragePooling2dCustom, AveragePooling2dCustomTilingData)
}

"""

host_operator_src="""

#include "average_pooling2d_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AveragePooling2dCustomTilingData tiling;
  const gert::StorageShape* input_shape = context->GetInputShape(0);
  
  // Extract shape dimensions for 4D tensor [batch, height, width, channels]
  uint32_t batchSize = input_shape->GetStorageShape().GetDim(0);
  uint32_t height = input_shape->GetStorageShape().GetDim(1);
  uint32_t width = input_shape->GetStorageShape().GetDim(2);
  uint32_t channels = input_shape->GetStorageShape().GetDim(3);
  
  // Get attributes
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  const int64_t* kernelSizePtr = attrs->GetAttrPointer<int64_t>(0);
  const int64_t* stridePtr = attrs->GetAttrPointer<int64_t>(1);
  const int64_t* paddingPtr = attrs->GetAttrPointer<int64_t>(2);
  
  uint32_t kernelSize = static_cast<uint32_t>(*kernelSizePtr);
  uint32_t stride = static_cast<uint32_t>(*stridePtr);
  uint32_t padding = static_cast<uint32_t>(*paddingPtr);
  
  uint32_t kernelH = kernelSize;
  uint32_t kernelW = kernelSize;
  uint32_t strideH = stride;
  uint32_t strideW = stride;
  uint32_t paddingH = padding;
  uint32_t paddingW = padding;
  
  // Calculate output dimensions
  uint32_t outH = (height + 2 * paddingH - kernelH) / strideH + 1;
  uint32_t outW = (width + 2 * paddingW - kernelW) / strideW + 1;
  
  // Calculate tiling parameters
  uint32_t tileW;
  if (strideW == 1) {
    tileW = 18;
  } else {
    tileW = (strideW * 6 > kernelW) ? strideW * 6 : kernelW;
  }
  
  uint32_t windowsPerLoad = (tileW / strideW > 1) ? tileW / strideW : 1;
  uint32_t groupsPerRow = (outW + windowsPerLoad - 1) / windowsPerLoad;
  uint32_t totalTasks = batchSize * outH * groupsPerRow;

  // Dynamic core count: query Vector (AIV) core count at runtime
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  if (coreNum == 0) { coreNum = 1; }
  uint32_t usedCoreNum = (totalTasks < coreNum) ? totalTasks : coreNum;
  uint32_t tasksPerCore = (totalTasks + usedCoreNum - 1) / usedCoreNum;
  
  // Set tiling data
  tiling.set_batchSize(batchSize);
  tiling.set_height(height);
  tiling.set_width(width);
  tiling.set_channels(channels);
  tiling.set_outH(outH);
  tiling.set_outW(outW);
  tiling.set_kernelH(kernelH);
  tiling.set_kernelW(kernelW);
  tiling.set_strideH(strideH);
  tiling.set_strideW(strideW);
  tiling.set_paddingH(paddingH);
  tiling.set_paddingW(paddingW);
  tiling.set_tileW(tileW);
  tiling.set_totalTasks(totalTasks);
  tiling.set_tasksPerCore(tasksPerCore);
  
  context->SetBlockDim(usedCoreNum);
  
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
class AveragePooling2dCustom : public OpDef {
public:
    explicit AveragePooling2dCustom(const char* name) : OpDef(name)
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
        this->Attr("kernel_size").Int();
        this->Attr("stride").Int();
        this->Attr("padding").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(AveragePooling2dCustom);
}

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void average_pooling2d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
"""