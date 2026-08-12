# 03: Compilation Requirements

> **导航**：[02-device-calling.md](./02-device-calling.md) → 本文

## catlass Kernel 编译必需选项

| 选项 | 说明 | 必需？ |
|------|------|:---:|
| `-I<CATLASS_DIR>/include` | catlass 头文件路径 | ✅ |
| `-DCATLASS_ARCH=<架构号>` | 芯片架构号 | ✅ |
| `-DBUILD_CATLASS_MODULE=ON` | 量化算子启用 catlass 子模块 | 量化时 ✅ |

## CATLASS_ARCH 值与芯片对应

| 芯片 | `CATLASS_ARCH` 值 |
|------|-------------------|
| 910b / 910_93 | `2201` |
| 950 | `3510` |

## 注入位置

由工程模板在 op_kernel 库的 `target_compile_options` 中注入：

```cmake
target_compile_options(my_op_kernel PRIVATE
    -I${CATLASS_DIR}/include
    -DCATLASS_ARCH=2201
    # -DBUILD_CATLASS_MODULE=ON   # 量化时取消注释
)
```

**本 skill 不规定**具体的 CMake 语法、变量名、构建命令——这些由工程模板决定。

## op_kernel 头文件包含边界

```cpp
// ✅ 允许包含
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/..."

// ❌ 禁止包含
#include "my_op_tiling.h"       // 算子自身 tiling 文件
#include "my_op_host/..."       // op_host 侧代码
```

**理由**（Δ5）：op_kernel 不应依赖 op_host 的 Tiling 实现细节。TilingData 结构体放在共享头中，op_kernel 通过通用宏（如 `GET_TILING_DATA`）取值。

## 独立 kernel 直调构建（脱离 catlass examples 模板时）

当用 `<<<>>>` kernel 直调（host main 含 launch，如 FA demo / 独立验证工程）且**不走 catlass `examples/` 构建模板**时，自包含 CMake 必须显式注入下列项（examples 模板里这些由 `examples/CMakeLists.txt` 全局提供，独立工程没有，**易漏**）：

| 项 | 做法 | 漏的后果 |
|----|------|----------|
| **`CATLASS_ARCH` 编译宏** | `target_compile_definitions(<op> PRIVATE CATLASS_ARCH=2201)` | catlass 架构头 `copy_l0c_to_gm.hpp` 等 `#if (CATLASS_ARCH==2201)` 不成立 → 不 include 定义 `ScaleGranularity` 等的架构头 → 级联报 `ScaleGranularity undeclared / PackedTileCopyTla too few template args`。**注意：`--npu-arch=dav-2201` 是编译器 flag，≠ `CATLASS_ARCH` 宏，两者都要** |
| `--npu-arch` | `target_compile_options(<op> PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)` | ASC 编译无目标架构 |
| ASC 语言 host 入口 | host main 用 `.asc` 扩展名（自动按 ASC 编译），或 `.cpp` + `set_source_files_properties(<f> PROPERTIES LANGUAGE ASC)` | `<<<>>>` 语法 / AscendC 符号无法编译 |
| include 路径 | `catlass/include` + skill 组件目录（如 `catlass/examples/23_flash_attention_infer/`）+ `$ENV{ASCEND_HOME_PATH}/include`（`acl/acl.h`、`tiling/platform/`）；`kernel_operator.h` 由 ASC 工具链自动提供 | 头文件找不到 |
| link 库 | `ascendc_runtime ascendcl runtime profapi mmpa c_sec error_manager graph_base tiling_api register platform unified_dlog dl m`（`$ASCEND_HOME_PATH/lib64`） | 链接失败 |
| `<<<>>>` 实参类型 | `GM_ADDR = __gm__ uint8_t*`；host `aclrtMalloc` 返回 `void*`，launch 实参须 `(GM_ADDR)dQ` 显式强转（`hwSync` 是 `uint64_t` 不转） | `cannot initialize '__gm__ uint8_t*' with 'void*'` |
| 核数 | `aclGetDeviceCapability(dev, ACL_DEVICE_INFO_AI_CORE_NUM, &v)`（enum 属 `aclDeviceInfo`，**不是** `aclCompileOpt::ACL_AICORE_NUM`）；fallback `PlatformAscendCManager::GetInstance()->GetCoreNumAic()` | 取错 enum → 编译错；blockDim=0 → kernel 不跑 |
| 硬件同步 | `aclrtGetHardwareSyncAddr(&hwSync)` 传给内核首参，内核内 `SetSyncBaseAddr(hwSync)` | AIC/AIV 跨核握手失败 |

最小骨架（FA 类，目标 A2/910B3）：

```cmake
cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(<op> LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(CATLASS_INCLUDE "/path/to/catlass/include")
set(FA_KERNEL_DIR "${CMAKE_SOURCE_DIR}/../../catlass/examples/23_flash_attention_infer")  # catlass 自带 FAInferKernel

add_executable(<op> <op>.asc)
target_compile_definitions(<op> PRIVATE CATLASS_ARCH=2201)          # ★必需，见上表
target_include_directories(<op> PRIVATE ${CATLASS_INCLUDE} ${SKILL_KERNEL_DIR} $ENV{ASCEND_HOME_PATH}/include)
target_compile_options(<op> PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
target_link_directories(<op> PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)
target_link_libraries(<op> PRIVATE ascendc_runtime ascendcl runtime profapi mmpa c_sec
                       error_manager graph_base tiling_api register platform unified_dlog dl m)
```

> 若走 catlass `examples/` 模板（`catlass_example_add_executable`），上述 `CATLASS_ARCH` / `--npu-arch` / 通用 link 库由 `examples/CMakeLists.txt` 全局注入，只需额外加 skill 组件 include 与 `target_link_libraries` 补充库即可（参考 catlass example `82_fa_tnd` / `83_fa_bnsd_causal`）。
