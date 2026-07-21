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
#include "kernel_operator.h"

class KernelAvgPool2d {
private:
    AscendC::TPipe pipe;
    // bf16 IO queues (for DataCopy with GM)
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBf16;
    // fp32 compute queues
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    // bf16 output queue
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueBf16;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf, cntBuf;
    AscendC::GlobalTensor<bfloat16_t> inputGm;
    AscendC::GlobalTensor<bfloat16_t> outputGm;
    uint32_t batchSize;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t outH;
    uint32_t outW;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t tileW;
    uint32_t totalTasks;
    uint32_t tasksPerCore;

public:
    __aicore__ inline KernelAvgPool2d() {}
    __aicore__ inline void Init(GM_ADDR input_ptr, GM_ADDR output_ptr, 
                                uint32_t batchSize, uint32_t height, uint32_t width, uint32_t channels,
                                uint32_t outH, uint32_t outW,
                                uint32_t kernelH, uint32_t kernelW,
                                uint32_t strideH, uint32_t strideW,
                                uint32_t paddingH, uint32_t paddingW,
                                uint32_t tileW, uint32_t totalTasks, uint32_t tasksPerCore)
    {
        this->batchSize = batchSize;
        this->height = height;
        this->width = width;
        this->channels = channels;
        this->outH = outH;
        this->outW = outW;
        this->kernelH = kernelH;
        this->kernelW = kernelW;
        this->strideH = strideH;
        this->strideW = strideW;
        this->paddingH = paddingH;
        this->paddingW = paddingW;
        this->tileW = tileW;
        this->totalTasks = totalTasks;
        this->tasksPerCore = tasksPerCore;

        // Set global memory buffers
        uint32_t totalInputElements = batchSize * height * width * channels;
        uint32_t totalOutputElements = batchSize * outH * outW * channels;
        
        inputGm.SetGlobalBuffer((__gm__ bfloat16_t *)input_ptr, totalInputElements);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t *)output_ptr, totalOutputElements);

        // Initialize pipe buffer queues
        uint32_t windowsPerLoad = (tileW / strideW > 1) ? tileW / strideW : 1;
        
        pipe.InitBuffer(inQueueBf16, 1, channels * sizeof(bfloat16_t));
        pipe.InitBuffer(inQueue, 1, channels * sizeof(float));
        pipe.InitBuffer(outQueue, 1, windowsPerLoad * channels * sizeof(float));
        pipe.InitBuffer(outQueueBf16, 1, windowsPerLoad * channels * sizeof(bfloat16_t));
        pipe.InitBuffer(sumBuf, windowsPerLoad * channels * sizeof(float));
        pipe.InitBuffer(cntBuf, windowsPerLoad * channels * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // TODO implemented
    }
};

extern "C" __global__ __aicore__ void average_pooling2d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAvgPool2d op;
    op.Init(x, y, 
            tiling_data.batchSize, tiling_data.height, tiling_data.width, tiling_data.channels,
            tiling_data.outH, tiling_data.outW,
            tiling_data.kernelH, tiling_data.kernelW,
            tiling_data.strideH, tiling_data.strideW,
            tiling_data.paddingH, tiling_data.paddingW,
            tiling_data.tileW, tiling_data.totalTasks, tiling_data.tasksPerCore);
    op.Process();
}
```

### Example output AscendC
```
#include "kernel_operator.h"

class KernelAvgPool2d {
private:
    AscendC::TPipe pipe;
    // bf16 IO queues (for DataCopy with GM)
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBf16;
    // fp32 compute queues
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    // bf16 output queue
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueBf16;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf, cntBuf;
    AscendC::GlobalTensor<bfloat16_t> inputGm;
    AscendC::GlobalTensor<bfloat16_t> outputGm;
    uint32_t batchSize;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t outH;
    uint32_t outW;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t strideH;
    uint32_t strideW;
    uint32_t paddingH;
    uint32_t paddingW;
    uint32_t tileW;
    uint32_t totalTasks;
    uint32_t tasksPerCore;

public:
    __aicore__ inline KernelAvgPool2d() {}
    __aicore__ inline void Init(GM_ADDR input_ptr, GM_ADDR output_ptr, 
                                uint32_t batchSize, uint32_t height, uint32_t width, uint32_t channels,
                                uint32_t outH, uint32_t outW,
                                uint32_t kernelH, uint32_t kernelW,
                                uint32_t strideH, uint32_t strideW,
                                uint32_t paddingH, uint32_t paddingW,
                                uint32_t tileW, uint32_t totalTasks, uint32_t tasksPerCore)
    {
        this->batchSize = batchSize;
        this->height = height;
        this->width = width;
        this->channels = channels;
        this->outH = outH;
        this->outW = outW;
        this->kernelH = kernelH;
        this->kernelW = kernelW;
        this->strideH = strideH;
        this->strideW = strideW;
        this->paddingH = paddingH;
        this->paddingW = paddingW;
        this->tileW = tileW;
        this->totalTasks = totalTasks;
        this->tasksPerCore = tasksPerCore;

        // Set global memory buffers
        uint32_t totalInputElements = batchSize * height * width * channels;
        uint32_t totalOutputElements = batchSize * outH * outW * channels;
        
        inputGm.SetGlobalBuffer((__gm__ bfloat16_t *)input_ptr, totalInputElements);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t *)output_ptr, totalOutputElements);

        // Initialize pipe buffer queues
        uint32_t windowsPerLoad = (tileW / strideW > 1) ? tileW / strideW : 1;
        
        pipe.InitBuffer(inQueueBf16, 1, channels * sizeof(bfloat16_t));
        pipe.InitBuffer(inQueue, 1, channels * sizeof(float));
        pipe.InitBuffer(outQueue, 1, windowsPerLoad * channels * sizeof(float));
        pipe.InitBuffer(outQueueBf16, 1, windowsPerLoad * channels * sizeof(bfloat16_t));
        pipe.InitBuffer(sumBuf, windowsPerLoad * channels * sizeof(float));
        pipe.InitBuffer(cntBuf, windowsPerLoad * channels * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        uint32_t pid = AscendC::GetBlockIdx();
        uint32_t taskStart = pid * tasksPerCore;
        uint32_t taskEnd = taskStart + tasksPerCore;
        if (taskEnd > totalTasks) {
            taskEnd = totalTasks;
        }

        uint32_t windowsPerLoad = (tileW / strideW > 1) ? tileW / strideW : 1;
        uint32_t groupsPerRow = (outW + windowsPerLoad - 1) / windowsPerLoad;

        for (uint32_t taskId = taskStart; taskId < taskEnd; taskId++) {
            // Decode task
            uint32_t b = taskId / (outH * groupsPerRow);
            uint32_t tmp = taskId % (outH * groupsPerRow);
            uint32_t oh = tmp / groupsPerRow;
            uint32_t owGroup = tmp % groupsPerRow;
            uint32_t ow0 = owGroup * windowsPerLoad;

            // Clear accumulators
            Compute1(windowsPerLoad);

            // Pooling with padding and stride
            for (uint32_t kh = 0; kh < kernelH; kh++) {
                int32_t ih = oh * strideH - paddingH + kh;
                
                if (ih < 0 || ih >= height) {
                    continue;
                }

                for (uint32_t w = 0; w < windowsPerLoad; w++) {
                    uint32_t ow = ow0 + w;
                    if (ow >= outW) {
                        continue;
                    }

                    int32_t w0 = ow * strideW - paddingW;

                    for (uint32_t kw = 0; kw < kernelW; kw++) {
                        int32_t iw = w0 + kw;
                        
                        if (iw < 0 || iw >= width) {
                            continue;
                        }

                        uint32_t base = b * height * width * channels +
                                       ih * width * channels +
                                       iw * channels;

                        CopyIn1(base);
                        Compute2(w);
                    }
                }
            }

            // Divide sum by count
            Compute3(windowsPerLoad);

            // Store
            CopyOut1(b, oh, ow0, windowsPerLoad);
        }
    }

private:
    __aicore__ inline void CopyIn1(uint32_t base)
    {
        // Step 1: DataCopy bfloat16 from GM
        AscendC::LocalTensor<bfloat16_t> inputBf16 = inQueueBf16.AllocTensor<bfloat16_t>();
        AscendC::DataCopy(inputBf16, inputGm[base], channels);
        inQueueBf16.EnQue(inputBf16);

        // Step 2: Cast bf16 -> fp32
        AscendC::LocalTensor<bfloat16_t> bf16Local = inQueueBf16.DeQue<bfloat16_t>();
        AscendC::LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        AscendC::Cast(inputLocal, bf16Local, AscendC::RoundMode::CAST_NONE, channels);
        inQueueBf16.FreeTensor(bf16Local);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute1(uint32_t windowsPerLoad)
    {
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        AscendC::LocalTensor<float> cntLocal = cntBuf.Get<float>();
        
        AscendC::Duplicate(sumLocal, 0.0f, windowsPerLoad * channels);
        AscendC::Duplicate(cntLocal, 0.0f, windowsPerLoad * channels);
    }

    __aicore__ inline void Compute2(uint32_t w)
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        AscendC::LocalTensor<float> cntLocal = cntBuf.Get<float>();
        
        uint32_t offset = w * channels;
        AscendC::Add(sumLocal[offset], sumLocal[offset], inputLocal, channels);
        AscendC::Adds(cntLocal[offset], cntLocal[offset], 1.0f, channels);
        
        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void Compute3(uint32_t windowsPerLoad)
    {
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        AscendC::LocalTensor<float> cntLocal = cntBuf.Get<float>();
        AscendC::LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();
        
        AscendC::Div(outputLocal, sumLocal, cntLocal, windowsPerLoad * channels);
        
        outQueue.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut1(uint32_t b, uint32_t oh, uint32_t ow0, uint32_t windowsPerLoad)
    {
        // Step 1: Get fp32 result
        AscendC::LocalTensor<float> outputLocal = outQueue.DeQue<float>();
        
        // Step 2: Cast fp32 -> bf16
        AscendC::LocalTensor<bfloat16_t> bf16Out = outQueueBf16.AllocTensor<bfloat16_t>();
        AscendC::Cast(bf16Out, outputLocal, AscendC::RoundMode::CAST_RINT, windowsPerLoad * channels);
        outQueue.FreeTensor(outputLocal);
        outQueueBf16.EnQue(bf16Out);
        
        // Step 3: DataCopy bf16 to GM
        AscendC::LocalTensor<bfloat16_t> bf16Final = outQueueBf16.DeQue<bfloat16_t>();
        for (uint32_t w = 0; w < windowsPerLoad; w++) {
            uint32_t ow = ow0 + w;
            if (ow < outW) {
                uint32_t outBase = b * outH * outW * channels +
                                  oh * outW * channels +
                                  ow * channels;
                AscendC::DataCopy(outputGm[outBase], bf16Final[w * channels], channels);
            }
        }
        outQueueBf16.FreeTensor(bf16Final);
    }
};

extern "C" __global__ __aicore__ void average_pooling2d_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAvgPool2d op;
    op.Init(x, y, 
            tiling_data.batchSize, tiling_data.height, tiling_data.width, tiling_data.channels,
            tiling_data.outH, tiling_data.outW,
            tiling_data.kernelH, tiling_data.kernelW,
            tiling_data.strideH, tiling_data.strideW,
            tiling_data.paddingH, tiling_data.paddingW,
            tiling_data.tileW, tiling_data.totalTasks, tiling_data.tasksPerCore);
    op.Process();
}
```