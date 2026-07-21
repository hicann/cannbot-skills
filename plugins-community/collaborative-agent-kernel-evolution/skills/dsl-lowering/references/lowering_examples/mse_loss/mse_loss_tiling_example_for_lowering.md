### Example input dsl
```python
import tile.language as tl

@ascend_kernel
def mse_loss_kernel(
    pred_ptr,                 # [N]
    target_ptr,               # [N]
    workspace,                # [n_used]  per-core partial sums
    output_ptr,               # final scalar
    base_elems,
    pivot,
    tile_size,
    inner_loops,
    total_elems               # for computing mean
):
    pid = tl.program_id(0)
    # Pivot distribution: first 'pivot' cores get one extra element
    my_elems = base_elems + (1 if pid < pivot else 0)
    start = pid * base_elems + min(pid, pivot)

    # ... (UB buffers and computation as before)

    # Phase 2: Core 0 loads all partial sums
    if pid == 0:
        with tl.copyin():
            tl.load(workspace + tl.arange(0, tl.num_programs(0)), workspace_in_ub)

def mse_loss_host(pred: torch.Tensor, target: torch.Tensor, output: torch.Tensor):
    total_elems = pred.numel()

    # Core Partitioning: dynamically query Vector core count
    n_cores    = tl.num_vec_cores()
    n_used     = min(n_cores, total_elems)
    base_elems = total_elems // n_used
    pivot      = total_elems % n_used

    # GM buffer for per-core partial results (size = n_used)
    workspace = torch.empty(n_used, dtype=torch.float32, device=pred.device)

    tile_size = 2048
    max_elems = base_elems + (1 if pivot > 0 else 0)
    inner_loops = (max_elems + tile_size - 1) // tile_size

    mse_loss_kernel[n_used](
        pred, target, workspace, output,
        base_elems, pivot, tile_size, inner_loops, total_elems
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
  context->SetBlockDim(8);  // ← simplified placeholder only; production code must use:
  // auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  // auto aivNum = plat.GetCoreNumAiv();
  // context->SetBlockDim(plat.CalcTschBlockDim(aivNum, 0, aivNum));
  // aicCoreNum=0: pure vector op, no matrix API used (CalcTschBlockDim doc: pass 0 if no matmul)
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  return ge::GRAPH_SUCCESS;
}
}
...
"""
```

### Example output AscendC
```python
host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MseLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalElems);
  TILING_DATA_FIELD_DEF(uint32_t, baseElems);   // base elements per core (floor division)
  TILING_DATA_FIELD_DEF(uint32_t, pivot);        // first 'pivot' cores get baseElems+1
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerLoops);
  TILING_DATA_FIELD_DEF(uint32_t, nUsed);        // actual number of launched cores
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

  // ---- Dynamic core count: query hardware, never hard-code ----
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  if (coreNum == 0) { coreNum = 1; }
  uint32_t usedCoreNum = (totalElems < coreNum) ? totalElems : coreNum;
  context->SetBlockDim(usedCoreNum);

  // ---- Pivot distribution: handles non-divisible total ----
  uint32_t baseElems = totalElems / usedCoreNum;
  uint32_t pivot     = totalElems % usedCoreNum;

  // ---- Tiling strategy ----
  uint32_t maxElems   = baseElems + (pivot > 0 ? 1 : 0);
  uint32_t innerLoops = (maxElems + TILE_SIZE - 1) / TILE_SIZE;

  tiling.set_totalElems(totalElems);
  tiling.set_baseElems(baseElems);
  tiling.set_pivot(pivot);
  tiling.set_tileSize(TILE_SIZE);
  tiling.set_innerLoops(innerLoops);
  tiling.set_nUsed(usedCoreNum);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  // Workspace: n_used floats for per-core partial sums
  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t requiredWorkspaceBytes = usedCoreNum * sizeof(float);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = requiredWorkspaceBytes + sysWorkspaceSize;

  return ge::GRAPH_SUCCESS;
}
}

...  (InferShape / InferDataType / OpDef same as before)

"""

kernel_src="""
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void mse_loss_custom(GM_ADDR predictions, GM_ADDR targets, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}
"""
