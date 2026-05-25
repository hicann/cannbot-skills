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
