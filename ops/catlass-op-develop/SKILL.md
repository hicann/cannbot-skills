---
name: catlass-op-develop
description: "Generate CATLASS kernel code from design selections. Produce: using chain (BlockMmad/BlockEpilogue/BlockScheduler/Kernel), Kernel::Params construction, Device-side calling code, custom Tile Epilogue header files, MatmulEpilogue and QuantMatmul special handling. Use when implementing op_kernel with catlass templates, writing Device-side kernel calls, creating custom Tile Epilogue, or handling QuantMatmul AIC/AIV coordination."
---

# CATLASS Kernel Code Generation

## Prerequisite: Read Catlass Repository Documentation（强制，先于实现）

在分析和执行具体 catlass 算子实现任务前，**必须先**针对工作区给定的 catlass 目标代码仓库（`./catlass/`）完成以下阅读，与 design skill 共用同一套先验知识：

| 顺序 | 路径 | 目的 |
|------|------|------|
| 1 | `./catlass/README.md` | 了解 catlass 库定位、目录结构、构建/运行方式 |
| 2 | `./catlass/docs/`（含子目录索引与关键设计/API 文档） | 理解算子组装知识、分层设计与实现约束 |
| 3 | `./catlass/examples/` 下设计文档指定的参考样例目录 | 对照样例源码及**样例目录内 README/文档**，确认组件组合与 main() → op_kernel 拆分模式 |

未完成上述阅读，**禁止**进入 using 链拼装与 Device 调用实现。

## Source Code Locations

```
catlass/
├── include/catlass/
│   ├── arch/arch.hpp              # ArchTag 尺寸常量
│   ├── gemm/
│   │   ├── dispatch_policy.hpp    # DispatchPolicy
│   │   ├── block/block_mmad.hpp   # BlockMmad
│   │   ├── block/block_swizzle/   # BlockScheduler
│   │   ├── kernel/                # ★ Kernel 头文件（写代码时核心参考）
│   │   ├── tile/                  # TileCopy, TileMmad
│   │   └── gemm_coord.hpp         # GemmCoord
│   ├── epilogue/
│   │   ├── block/block_epilogue*.hpp  # ★ BlockEpilogue 特化（读槽位、签名）
│   │   ├── tile/tile_elemwise_*.hpp   # ★ Tile 实现（参考签名骨架）
│   │   └── tile/tile_copy.hpp
│   └── layout/layout.hpp         # RowMajor, ColumnMajor
├── examples/                      # 参考实现
│   ├── 00_basic_matmul/basic_matmul.cpp     # ★ 纯 matmul 参考
│   ├── 27_matmul_gelu/matmul_gelu.cpp       # ★ matmul+GELU 参考
│   ├── 12_quant_matmul/                     # ★ 量化参考
│   └── advanced/basic_matmul_aclnn/         # aclnn 工程集成
└── docs/zh/
    ├── 3_API/gemm_api.md                     # Kernel/Block/Tile 分层
    └── 3_API/include/catlass/gemm/kernel/    # Kernel API 文档
```

## Search Strategy

```bash
# Kernel 类型和 Params
rg "struct Params|struct Arguments|struct.*Params" catlass/include/catlass/gemm/kernel/

# Device 调用模式
rg "Kernel\{\}\(params\)|Kernel\{" catlass/examples/

# Epilogue 槽位接口
rg "template.*class.*Epilogue|operator\(\)" catlass/include/catlass/epilogue/

# Tile 签名骨架
rg "struct Tile.*\{|COMPUTE_LENGTH|operator\(\)" catlass/include/catlass/epilogue/tile/

# 量化 Params（scale/perTokenScale）
rg "gmScale|gmPerTokenScale|ptrScale" catlass/include/catlass/gemm/kernel/
```

## When to Use Each Source

- Kernel 组装链理解 → `catlass/docs/zh/3_API/gemm_api.md`（§Kernel API）
- Device 调用模式 → 读 `examples/00_basic_matmul/` 的 using 链
- Epilogue 组装 → 读 `examples/27_matmul_gelu/` 的 BlockEpilogue 组装
- 自定义 Tile 签名 → 查 `catlass/include/catlass/epilogue/tile/` 中现成 Tile 作参考
- Params 字段 → `rg "struct Params" catlass/include/catlass/gemm/kernel/` 直接读源码
- Workspace 取法 → `AscendC::GetUserWorkspace(workspace)`, 见 [architecture/02-device-calling.md](references/architecture/02-device-calling.md)
- **精度脚本（golden/verify）编写 → [precision-verification.md](references/precision-verification.md)**（先经 `ops-precision-standard` 选标准）
- **最优 mmad/epilogue 选型理由 → [catlass-op-design/references/mmad-epilogue-selection.md](../catlass-op-design/references/mmad-epilogue-selection.md)**（实现时据此核对 DESIGN 选型）

---

## Architecture Reference

本 skill 的 `references/` 目录按分层组织：

| 文档 | 内容 |
|------|------|
| [architecture/00-overview.md](references/architecture/00-overview.md) | Kernel 组装全景与 using 链结构 |
| [architecture/01-kernel-assembly.md](references/architecture/01-kernel-assembly.md) | using 链标准模式（无 Epilogue / 有 Epilogue） |
| [architecture/02-device-calling.md](references/architecture/02-device-calling.md) | Device 调用、Params 构造、Workspace 获取 |
| [architecture/03-compilation.md](references/architecture/03-compilation.md) | catlass kernel 编译要求 |
| [patterns/basic-matmul.md](references/patterns/basic-matmul.md) | 纯 matmul 完整代码骨架 |
| [patterns/with-epilogue.md](references/patterns/with-epilogue.md) | + 激活、+ Bias、+ Bias+激活 |
| [patterns/quant-matmul.md](references/patterns/quant-matmul.md) | 量化 Matmul AIC/AIV 协同 |
| [patterns/branch-instantiation.md](references/patterns/branch-instantiation.md) | 多分支 if constexpr 实例化 |
| [rules.md](references/rules.md) | 强制性规则 Δ1–Δ10 |
| [custom-epilogue.md](references/custom-epilogue.md) | 自定义 Tile Epilogue 实现骨架 |
| [precision-verification.md](references/precision-verification.md) | **精度验证脚本（gen_data/golden/verify）编写规则**：对齐官方标准、禁止零容忍小值域门限、golden 镜像内核、int8 用 fp32 BLAS、覆盖实网 shape |
| [shape-constraints.md](references/shape-constraints.md) | 测试 shape 运行期约束 |
| [troubleshooting.md](references/troubleshooting.md) | 常见问题排查 |

## Never / Always

**NEVER**:
- 跳过 `./catlass/README.md`、`./catlass/docs/` 及参考 `examples/` 样例（含样例目录内文档）直接写代码
- 在 op_kernel 中使用 `DeviceGemm` 适配器
- 手写矩阵乘 / 逐元素 / 拷贝循环
- 调用 `SetSysWorkspaceForce`
- `#include` 算子自身的 tiling 实现文件
- 规定算子目录名、文件名、CMake 语法、构建命令
- 把 golden 生成注释掉 / 跳过；让 verify 只覆盖基础 shape 不覆盖实网 shape
- 在 verify 里自创零容忍小值域门限或用全体元素 MARE-max 作硬门限（过零激活会误判）

**ALWAYS**:
- 先阅读 `./catlass/README.md`、`./catlass/docs/` 及参考 `examples/` 样例（含样例目录内文档），再按设计选型写代码
- op_kernel 只用 catlass `Kernel` / `Block*` / `Tile*`
- Device 调用: `Kernel{}(params)`
- Workspace: `AscendC::GetUserWorkspace(workspace)`
- 严格按设计选型实例化每个分支
- 自定义 Tile 对齐目标槽位签名
- verify 判据 = `ops-precision-standard` 选出的官方标准；golden 镜像内核数值路径（fp32 累加→末尾 cast）；int8 GEMM golden 用 fp32 BLAS（`|Cint|<2²⁴` 精确）；gen_data/verify 覆盖基础 + 实网 shape（见 [precision-verification.md](references/precision-verification.md)）
