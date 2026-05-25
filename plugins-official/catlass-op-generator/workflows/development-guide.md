# Catlass 算子开发指南

本文件是 catlass 算子直调开发的编码规范与审查清单，扩展自 ops-direct-invoke 通用规范，叠加 catlass 专属约束。

---

## 一、工程结构

```
operators/{operator_name}/
├── docs/
│   ├── DESIGN.md             # Architect 输出（含 catlass 选型）
│   ├── PLAN.md               # Architect 输出（含 catlass 编译选项）
│   ├── REVIEW.md             # Reviewer 输出
│   ├── WALKTHROUGH.md        # 设计串讲
│   ├── environment.json      # 环境检查结果
│   └── perf/round_NNN/       # 性能采集数据
├── op_host/
│   ├── {operator_name}_tiling.h    # TilingData POD（host/kernel 共用）
│   └── {operator_name}.asc         # main + ACL + Tiling 计算
├── op_kernel/
│   ├── {operator_name}.asc         # catlass 拼装类 + kernel 入口
│   └── tiles/{tile_name}.hpp       # 自定义 Tile（如有）
├── scripts/
│   ├── gen_data.py
│   ├── golden.py
│   └── verify_result.py
├── CMakeLists.txt
├── run.sh
└── README.md
```

`./catlass/` 源码必须位于工作区根（与 `operators/` 平级），**禁止**克隆到 `operators/{operator_name}/` 内（C2）。

---

## 二、CMake 编译选项（必备）

```cmake
cmake_minimum_required(VERSION 3.16)
project(catlass_xxx LANGUAGES ASC CXX)

find_package(ASC REQUIRED)

add_executable(catlass_xxx_op
    op_host/catlass_xxx.asc
    op_kernel/catlass_xxx.asc
)

target_compile_options(catlass_xxx_op PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    $<$<COMPILE_LANGUAGE:ASC>:-I${CMAKE_SOURCE_DIR}/../../catlass/include>
    $<$<COMPILE_LANGUAGE:ASC>:-DCATLASS_ARCH=220>
)

# 量化算子追加：
# -DCATLASS_ARCH=2201

target_link_libraries(catlass_xxx_op PRIVATE
    tiling_api
    register
    platform
    m dl
)
```

> ⚠️ `-I<catlass>/include` 与 `-DCATLASS_ARCH=<arch>` 缺一不可（C3）。

---

## 三、op_kernel 写法

### 3.1 catlass 拼装类（顶部命名空间集中）

```cpp
#include <catlass/arch/arch.hpp>
#include <catlass/gemm/gemm_type.hpp>
#include <catlass/gemm/dispatch_policy.hpp>
#include <catlass/gemm/block/block_mmad.hpp>
#include <catlass/gemm/block/gemm_block_swizzle.hpp>
#include <catlass/gemm/kernel/basic_matmul_kernel.hpp>

#include "../op_host/catlass_xxx_tiling.h"   // 仅 POD 结构体头文件

namespace NsCatlassXxx {
using ArchTag        = Catlass::Arch::AtlasA2;
using DispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape    = Catlass::GemmShape<128, 256, 256>;
using L0TileShape    = Catlass::GemmShape<128, 256, 64>;
using AType          = Catlass::Gemm::GemmType<half,  Catlass::layout::RowMajor>;
using BType          = Catlass::Gemm::GemmType<half,  Catlass::layout::RowMajor>;
using CType          = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
using BlockMmad      = Catlass::Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
using BlockEpilogue  = void;  // 或具体组合
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
} // namespace NsCatlassXxx
```

### 3.2 kernel 入口分支

```cpp
extern "C" __global__ __aicore__ void catlass_xxx_op(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_FOR_TILINGKEY(...)   // 按 catlass-op-develop skill 写法
    GET_TILING_DATA(tilingData, tiling);

    if constexpr (TILING_KEY_VAR == 0) {                        // DESIGN.md §2.1 列出的分支条件
        using Kernel = Catlass::Gemm::Kernel::BasicMatmulKernel<
            NsCatlassXxx::BlockMmad,
            NsCatlassXxx::BlockEpilogue,
            NsCatlassXxx::BlockScheduler>;

        GM_ADDR userWs = const_cast<GM_ADDR>(AscendC::GetUserWorkspace(workspace));
        typename Kernel::Params params{
            /* problemShape */ {tilingData.M, tilingData.N, tilingData.K},
            /* gmA */ a, /* layoutA */ {tilingData.M, tilingData.K},
            /* gmB */ b, /* layoutB */ {tilingData.K, tilingData.N},
            /* gmC */ c, /* layoutC */ {tilingData.M, tilingData.N},
            /* userWs */ userWs,
        };
        Kernel{}(params);
    }
    // 其他分支（dtype/转置/Swizzle 组合）...
}
```

### 3.3 强制项（违反 = 审查不通过）

| # | 强制项 | 失败示例 |
|---|--------|---------|
| C4 | 直接 `Kernel{}(params)`，**禁用** `DeviceGemm` 适配器 | `Catlass::DeviceGemm<...> dg(...)` |
| C5 | 不得自实现矩阵乘 / 逐元素 / 拷贝循环 | `for (k = 0; k < K; ++k) c += a*b;` |
| C6 | Workspace 必须 `AscendC::GetUserWorkspace(workspace)` | `SetSysWorkspaceForce(...)` |
| C7 | 不得 `#include` 算子自身的 tiling 实现文件 | `#include "catlass_xxx.tiling.cpp"` |

---

## 四、op_host 写法

### 4.1 TilingData（POD）

```c
// op_host/catlass_xxx_tiling.h
#pragma once
#include <cstdint>

#ifdef __CCE_KT_TEST__
#include "kernel_tiling/kernel_tiling.h"
#endif

BEGIN_TILING_DATA_DEF(CatlassXxxTilingData)
TILING_DATA_FIELD_DEF(uint32_t, M);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, K);
TILING_DATA_FIELD_DEF(uint32_t, usedNumBlocks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CatlassXxxOp, CatlassXxxTilingData)
```

### 4.2 host main + Tiling

- ACL 初始化、device 内存分配
- 计算 TilingData + workspaceSize + tilingKey（按 DESIGN.md §2.1 / §2.2）
- `<<<usedNumBlocks, ...>>>` 启动 kernel
- 结果搬回 host、调 verify
- ACL 清理

---

## 五、测试规范

### 5.1 测试 shape 约束（C9）

- 避免过小 M/N（个位数易触发 AIV UB 越界）
- 优先选 L1 分块 M/N 整数倍

### 5.2 测试级别

| Level | 数据规模 | 用途 |
|-------|---------|------|
| Level 0 | M/N/K = L1 分块整数倍（如 128/256/256） | 基础功能 |
| Level 1 | 1K~4K，覆盖 §2.1 每个 dtype/转置/Swizzle 分支 | 分支覆盖 |
| Level 2 | 极值（K=1, K=L1.K-1 等） | 边界 |

### 5.3 精度阈值

| dtype | rtol | atol |
|-------|------|------|
| FP32 | 1e-5 | 1e-5 |
| FP16 | 1e-3 | 1e-3 |
| BF16 | 1e-2 | 1e-2 |

无 catlass 专属放宽规则。

---

## 六、性能调优规范（Step 6 / 调优场景）

调优必须加载 `/catlass-op-perf-tune`，并遵守：

1. **以 `catlass/docs/1_Practice/10_matmul_optimization.md` 为准**
2. 每次**只动一个变量**（DispatchPolicy / TileShape / Swizzle / Kernel 之一），便于归因
3. 性能下降 → 立即回滚到上一稳定配置
4. PRE/POST 两份 profiler 数据均归档到 `operators/{operator_name}/docs/perf/round_NNN/`

---

## 七、命名规范（C1）

- snake_case 算子名必须含 `catlass` 子串：`catlass_matmul_add` ✓
- CamelCase 类名一致映射：`CatlassMatmulAdd` ✓
- namespace：`NsCatlassMatmulAdd`（避免与 catlass 内部命名冲突）

---

## 八、常见错误清单

| 错误 | 正确写法 |
|------|---------|
| 用 `Catlass::DeviceGemm<...>` 适配器 | 直接 `Kernel{}(params)` |
| 在 op_kernel 写 `for(k=0;k<K;++k) c += a*b` | 用 catlass `BlockMmad` |
| `SetSysWorkspaceForce(...)` | `AscendC::GetUserWorkspace(workspace)` |
| op_kernel `#include "catlass_xxx_tiling.cpp"` | 仅 include `*_tiling.h` POD |
| 算子名 `matmul_add`（无 catlass） | `catlass_matmul_add` |
| `git clone catlass.git operators/catlass_xxx/catlass` | 在工作区根 `git clone catlass.git` |
| CMake 缺 `-DCATLASS_ARCH` | 注入 `-DCATLASS_ARCH=220` |
