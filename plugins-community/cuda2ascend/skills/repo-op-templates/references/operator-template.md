# 算子模板

> 每个算子在 `csrc/ops/<op>/` 下建子目录，复制以下模板填充。`<op>` 取评测集算子原型定义（如 cann-bench `proto.yaml`）的 `operator.name`（小写），目录名与之一致。权威样例见 direct launch 工程样例（如 cann-bench `examples/direct_launch_example/csrc/ops/{add,sqrt}/`）。

## 算子目录结构

```
csrc/ops/<op>/
├── CMakeLists.txt          # 算子自注册（见下方）
├── op_kernel/
│   ├── <op>_kernel.cpp     # bisheng 编译：Kernel 类 + Tiling + extern "C" Launch（见下方）
│   └── <op>_launch.h       # Launch 声明，g++ 可见（见下方）
└── op_plugin/
    └── <op>_plugin.cpp     # g++ 编译：torch.library 注册 + Meta + NPU impl（见下方）
```

## 模板选择规则

direct launch 算子统一一套模板，按算子复杂度调整 Kernel 类内部实现：

| 算子类型 | Kernel 模板调整点 |
|----------|-------------------|
| Element-wise / Activation（L1） | 单输入单输出，CopyIn→Compute→CopyOut 三段式，tiling 按 element count 切 |
| Reduction / Norm（L2） | 多输入或归约，需 ReduceSum/ReduceMax，注意尾块非对齐 |
| Conv / Matmul / MoE（L3） | CUBE 运算（Mmad），多 Tiling 策略，多核切分复杂 |
| Attention / RNN（L4） | 多算子融合，复杂数据流，多 stage 流水 |

> 通用 Ascend C kernel 编写范式（Tiling 设计、Buffer 规划、数据流方法论）参照 `ascendc-tiling-design`；代码架构选型（MemBase/RegBase/SIMT）参照 `repo-knowledge`。

## op_kernel/<op>_kernel.cpp（bisheng 编译）

以 sqrt 为样例的模板。`<op>` / `<Op>` / 输入输出参数 / Kernel 计算逻辑按 `proto.yaml` 替换：

```cpp
/**
 * <Op> 算子 - Kernel + Tiling + Launch
 * 编译：bisheng --npu-arch=<arch> -xasc
 */
#include <tuple>
#include <algorithm>
#include <type_traits>
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"

constexpr static int64_t PIPELINE_DEPTH = 2;

template <typename T>
class Kernel<Op> {
public:
    __aicore__ inline Kernel<Op>() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                int64_t totalLength, int64_t blockLength, uint32_t tileSize)
    {
        // 1. GlobalTensor 绑定（按 blockIdx 偏移）
        xGm_.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
        yGm_.SetGlobalBuffer((__gm__ T *)y + blockLength * AscendC::GetBlockIdx());
        // 2. InitBuffer（TQue / TBuf，按 tileSize * sizeof(T)）
        pipe_.InitBuffer(inQueueX_,  PIPELINE_DEPTH, tileSize * sizeof(T));
        pipe_.InitBuffer(outQueueY_, PIPELINE_DEPTH, tileSize * sizeof(T));
        // 3. 计算 tileNum / tailTileElementNum（处理尾块非对齐）
        int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
        if (currentBlockLength > blockLength) currentBlockLength = blockLength;
        if (currentBlockLength < 0) currentBlockLength = 0;
        elementNumPerTile_ = tileSize;
        tileNum_ = currentBlockLength / elementNumPerTile_;
        tailTileElementNum_ = currentBlockLength - tileNum_ * elementNumPerTile_;
    }

    __aicore__ inline void Process()
    {
        // 三段式流水：CopyIn → Compute → CopyOut，循环 tile + 尾块
        for (int64_t i = 0; i < tileNum_; ++i) {
            CopyIn(i * elementNumPerTile_, elementNumPerTile_);
            Compute(elementNumPerTile_);
            CopyOut(i * elementNumPerTile_, elementNumPerTile_);
        }
        if (tailTileElementNum_ > 0) {
            CopyIn(tileNum_ * elementNumPerTile_, tailTileElementNum_);
            Compute(tailTileElementNum_);
            CopyOut(tileNum_ * elementNumPerTile_, tailTileElementNum_);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        // DataCopyPad 搬入（处理非对齐，禁用 DataCopy）
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        auto xLocal = inQueueX_.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        auto xLocal = inQueueX_.DeQue<T>();
        auto yLocal = outQueueY_.AllocTensor<T>();
        // === 核心计算：按 proto.yaml 公式用 Ascend C 原语实现 ===
        AscendC::<Op>(yLocal, xLocal, count);  // 替换为实际 Ascend C API
        // === 注意 dtype 无原生重载时需 Cast 中转（见 sqrt 样例 fp16/bf16 路径）===
        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t count)
    {
        auto yLocal = outQueueY_.DeQue<T>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> xGm_, yGm_;
    AscendC::TQue<AscendC::TPosition::VECIN,  PIPELINE_DEPTH> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, PIPELINE_DEPTH> outQueueY_;
    int64_t elementNumPerTile_ = 0, tileNum_ = 0, tailTileElementNum_ = 0;
};

// Kernel 函数（<<<>>> launch 语法，bisheng -xasc 模式有效）
template <typename T>
__global__ __aicore__ __vector__ void <op>_kernel(GM_ADDR x, GM_ADDR y,
    int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    Kernel<Op><T> op;
    op.Init(x, y, totalLength, blockLength, tileSize);
    op.Process();
}

// Tiling 计算（返回 numBlocks, blockLength, tileSize）
std::tuple<int64_t, int64_t, int64_t> calc_<op>_tiling_params(int64_t totalLength)
{
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    constexpr static uint32_t FIXED_TILE_ELEMS = 2048;  // 按目标 SoC UB 容量验证
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    numBlocks = std::max(numBlocks, static_cast<int64_t>(1));
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    return std::make_tuple(numBlocks, blockLength, static_cast<int64_t>(FIXED_TILE_ELEMS));
}

// extern "C" Launch 函数（每个 dtype 一个，供 plugin 调用）
extern "C" {
void launch_<op>_kernel_float(GM_ADDR x, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream) {
    <op>_kernel<float><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileSize);
}
void launch_<op>_kernel_half(GM_ADDR x, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream) {
    <op>_kernel<half><<<numBlocks, nullptr, stream>>>(x, y, totalLength, blockLength, tileSize);
}
// bfloat16 按需追加
}
```

> **dtype 中转**：部分 Ascend C API 无 fp16/bf16 原生重载（如 `AscendC::Sqrt`），需 Cast 到 fp32 计算再 Cast 回。详见 direct launch 工程样例（如 cann-bench `examples/direct_launch_example/csrc/ops/sqrt/op_kernel/sqrt_kernel.cpp`）。

## op_kernel/<op>_launch.h（g++ 可见）

Launch 函数声明，供 plugin include：

```cpp
#ifndef <OP>_LAUNCH_H
#define <OP>_LAUNCH_H
#include <cstdint>
#include <tuple>
#ifndef GM_ADDR
#define GM_ADDR void*
#endif

std::tuple<int64_t, int64_t, int64_t> calc_<op>_tiling_params(int64_t totalLength);

extern "C" {
void launch_<op>_kernel_float(GM_ADDR x, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
void launch_<op>_kernel_half(GM_ADDR x, GM_ADDR y,
    int64_t totalLength, int64_t numBlocks, int64_t blockLength, uint32_t tileSize, void* stream);
}
#endif
```

## op_plugin/<op>_plugin.cpp（g++ 编译）

torch.library 注册 + Meta + NPU 实现。**schema 必须与 `proto.yaml` 的 `schema:` 字段逐字一致**：

```cpp
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "../op_kernel/<op>_launch.h"

namespace cann_bench {  // cann_bench 为评测集约定的包名（<pkg>），此处以 cann_bench 为例

// schema 注册 —— 必须与 proto.yaml 的 schema: 逐字一致
TORCH_LIBRARY_FRAGMENT(cann_bench, m)
{
    m.def("<op>(Tensor x) -> Tensor");  // 按 proto.yaml 替换，含 attrs
}

// Meta 函数（输出 shape/dtype 推导）
torch::Tensor <op>_meta(const torch::Tensor &x /*, attrs */) {
    return torch::empty_like(x);  // 按 proto.yaml outputs 推导
}

TORCH_LIBRARY_IMPL(cann_bench, Meta, m)
{
    m.impl("<op>", <op>_meta);
}

// NPU 实现（调 tiling + launch kernel）
torch::Tensor <op>_npu(const torch::Tensor &x /*, attrs */) {
    const c10::OptionalDeviceGuard guard(x.device());
    auto y = <op>_meta(x /*, attrs */);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t totalLength = x.numel();
    int64_t numBlocks, blockLength, tileSize;
    std::tie(numBlocks, blockLength, tileSize) = calc_<op>_tiling_params(totalLength);
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();

    auto acl_call = [=]() -> int {
        auto dtype = x.scalar_type();
        if      (dtype == torch::kFloat32) launch_<op>_kernel_float  (x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        else if (dtype == torch::kFloat16) launch_<op>_kernel_half   (x_ptr, y_ptr, totalLength, numBlocks, blockLength, tileSize, stream);
        // bfloat16 按需追加
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("<Op>", acl_call);  // <Op> 仅作 op name 标签
    return y;
}

TORCH_LIBRARY_IMPL(cann_bench, PrivateUse1, m)
{
    m.impl("<op>", <op>_npu);
}

} // namespace cann_bench
```

> ⚠ **schema 对齐是第一红线**：`m.def(...)` 的签名必须与评测集算子原型定义（如 cann-bench `tasks/levelN/<op>/proto.yaml`）的 `schema:` 字段逐字一致（参数名、类型、默认值、返回值），否则评测注册不上算子。

## CMakeLists.txt（算子自注册）

```cmake
# <Op> 算子自注册
set(<OP>_KERNEL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/<op>_kernel.cpp)
set(<OP>_PLUGIN_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/<op>_plugin.cpp)

register_direct_launch_op(
    "${<OP>_KERNEL_SRCS}" op_kernel
    "${<OP>_PLUGIN_SRCS}" op_kernel
)
```

## cann_bench/__init__.py 追加

每新增算子，在 `cann_bench/__init__.py` 追加导出：

```python
def <op>(...) -> torch.Tensor:
    return torch.ops.cann_bench.<op>(...)
```
