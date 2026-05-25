---
name: catlass-op-develop
description: "Generate CATLASS kernel code from design selections. Produce: using chain (BlockMmad/BlockEpilogue/BlockScheduler/Kernel), Kernel::Params construction, Device-side calling code, custom Tile Epilogue header files, MatmulEpilogue and QuantMatmul special handling. Use when implementing op_kernel with catlass templates, writing Device-side kernel calls, creating custom Tile Epilogue, or handling QuantMatmul AIC/AIV coordination."
---

# CATLASS Kernel Code Generation

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
| [rules.md](references/rules.md) | 强制性规则 Δ1–Δ8 |
| [custom-epilogue.md](references/custom-epilogue.md) | 自定义 Tile Epilogue 实现骨架 |
| [shape-constraints.md](references/shape-constraints.md) | 测试 shape 运行期约束 |
| [troubleshooting.md](references/troubleshooting.md) | 常见问题排查 |

## Never / Always

**NEVER**:
- 在 op_kernel 中使用 `DeviceGemm` 适配器
- 手写矩阵乘 / 逐元素 / 拷贝循环
- 调用 `SetSysWorkspaceForce`
- `#include` 算子自身的 tiling 实现文件
- 规定算子目录名、文件名、CMake 语法、构建命令

**ALWAYS**:
- op_kernel 只用 catlass `Kernel` / `Block*` / `Tile*`
- Device 调用: `Kernel{}(params)`
- Workspace: `AscendC::GetUserWorkspace(workspace)`
- 严格按设计选型实例化每个分支
- 自定义 Tile 对齐目标槽位签名
