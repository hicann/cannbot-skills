# Layer 0: 硬件与架构

## NpuArch 与芯片

> 来自 `ascendc-npu-arch` skill。

| 芯片 | NpuArch | `__NPU_ARCH__` | SocVersion | 定位 |
|------|---------|:---:|-----------|------|
| Ascend910B1~B4, 910B2C | DAV_2201 | 2201 | ASCEND910B | 主流训练/推理 |
| Ascend910_93 | DAV_2201 | 2201 | ASCEND910B | 训练/推理 |
| Ascend950DT | DAV_3510 | 3510 | ASCEND950 | 新一代 Decode |
| Ascend950PR | DAV_3510 | 3510 | ASCEND950 | 新一代 Prefill |

## Catlass ArchTag

> 来自 `catlass/include/catlass/arch/arch.hpp`。

catlass 用 `ArchTag` 模板参数分发到不同芯片的特化实现。当前定义两种：

| Catlass ArchTag | 对应 NpuArch | 芯片 |
|----------------|-------------|------|
| `Arch::AtlasA2` | DAV_2201 | Ascend910B 系列、Ascend910_93 |
| `Arch::Ascend950` | DAV_3510 | Ascend950DT / Ascend950PR |

```cpp
// Catlass 源码中的定义
namespace Catlass::Arch {
struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;    // ★
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

struct Ascend950 {
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 256 * 1024;   // ← A2 的两倍
    static constexpr uint32_t UB_SIZE = 248 * 1024;
    static constexpr uint32_t BIAS_SIZE = 4 * 1024;
    static constexpr uint32_t FIXBUF_SIZE = 16 * 1024;
};
} // namespace Catlass::Arch
```

## 内存层级（AtlasA2）

> 来自 `catlass/docs/zh/2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md`。

```
GM (Global Memory)
 │
L2 Cache
 │
L1 Buffer (512KB)  ← BlockMmad 把 A/B 从 GM 搬到 L1
 │  ├── L1A pingpong
 │  └── L1B pingpong
 │
L0 Buffer           ← TileMmad 把 L1→L0 然后计算
 ├── L0A (64KB)     ← A 矩阵输入
 ├── L0B (64KB)     ← B 矩阵输入
 └── L0C (128KB)    ← 累加结果，unitflag 开启边算边搬出
 │
UB (192KB)          ← Vector 存储，Epilogue 在此执行逐元素计算
 ├── BIAS (1KB)
 └── FB (7KB)       ← 量化参数、relu 参数等
```

**关键**：L1 Buffer (512KB) 和 UB (192KB) 是**不同**的存储——L1 用于矩阵数据搬运，UB 用于 Vector 计算（Epilogue）。

## 资源约束（AtlasA2）

> 来自 catlass `04_matmul_summary.md` Common 模板 Tiling 建模。

```
L1TileShape(m1, n1, k1):  m1*k1 * sizeof(A) * L1A_STAGES + n1*k1 * sizeof(B) * L1B_STAGES ≤ L1_SIZE
L0TileShape(m0, n0, k0):  m0*k0 * sizeof(A) * L0A_STAGES ≤ L0A_SIZE
                           n0*k0 * sizeof(B) * L0B_STAGES ≤ L0B_SIZE
                           m0*n0 * sizeof(float) * L0C_STAGES ≤ L0C_SIZE
约束: m0 = m1, n0 = n1
```

**以 fp16 + fp32 累加为例**（默认 Pingpong, STAGES=2）：

| 尺寸 | 约束表达式 | 是否满足 |
|------|-----------|---------|
| L1 `<128,256,256>` | `128*256*2*2 + 256*256*2*2 = 384KB` | ≤ 512KB ✓ |
| L0 `<128,256,64>` | `128*64*2*2 = 32KB ≤ 64KB` (L0A/L0B) | ✓ |
| L0C `<128,256>` | `128*256*4*1 = 128KB ≤ 128KB` | ✓ |

## 必须包含的头文件

```cpp
#include "catlass/arch/arch.hpp"          // ArchTag
#include "catlass/catlass.hpp"            // 全局
#include "catlass/gemm/gemm_type.hpp"     // GemmType
#include "catlass/gemm/dispatch_policy.hpp" // DispatchPolicy
#include "catlass/layout/layout.hpp"      // RowMajor/ColumnMajor
```

## CATLASS_ARCH 编译选项

| 芯片 | CATLASS_ARCH 值 |
|------|-----------------|
| 910b / 910_93 | 2201 |
| 950 | 3510 |
