# asc-devkit 目录 → 迁移场景快速映射
> **⚠ 路径校正（2026-08 实测）**：asc-devkit 当前版本的文档实际位于 `docs/zh/api/` 与 `docs/zh/guide/`（多一层 `zh`），且 A5 特性导航文件名为 `docs/zh/asc_950_feature_guide.md`（非 `asc_a5_feature_guide.md`）。本文件中的 `docs/api/`、`docs/guide/` 路径访问时请自动补 `zh` 层。


> 当 agent 不确定去哪个目录时，先查此表。
> 所有路径相对于 $DEVKIT_PATH。

## 快速定位表

| 迁移场景 | 首选目录 | 备选目录 |
|---------|---------|---------|
| 查某 API 在 A5 上的用法 | `docs/api/SIMD-API/` | `include/` 头文件 |
| 查 API 迁移前后的差异 | `docs/guide/跨代迁移兼容性指南/3510架构迁移指导/` | — |
| 查某算子怎么实现 | `examples/` | `docs/guide/算子实践参考/` |
| 查 A5 新增了什么能力 | `docs/asc_a5_feature_guide.md` | `docs/guide/跨代迁移兼容性指南/` |
| 查头文件中函数声明 | `include/` | `impl/` |
| 查 UB 大小/bank 结构 | `docs/guide/编程指南/高级编程/硬件实现/` | — |
| 查性能优化方法 | `docs/guide/算子实践参考/` | `examples/` |
| 查 CMake 编译配置 | `cmake/` | `build.sh` |
| 查 SIMT 线程模型/DCache | `docs/guide/编程指南/编程模型/AI-Core-SIMT编程/` | `docs/api/SIMT-API/` |
| 查 SIMD 编程模型 | `docs/guide/编程指南/编程模型/AI-Core-SIMD编程/` | `docs/api/SIMD-API/` |
| 查 BuiltIn 关键字 | `docs/guide/编程指南/语言扩展层/` | — |
| 查芯片架构映射 | `.agent/skills/asc-npu-arch/` | — |
| 查 API 兼容策略 | `docs/guide/跨代迁移兼容性指南/Ascend-C-API兼容策略.md` | — |
| 查高阶 API（Matmul/Softmax 等） | `docs/api/SIMD-API/高阶API/` | `include/adv_api/` |
| 查 Reg 编程（A5 新增） | `docs/api/SIMD-API/C-API/Reg矢量计算/` | `docs/api/SIMD-API/基础API/Reg矢量计算/` |

## 目录结构速览

```
$DEVKIT_PATH/
├── cmake/               # Ascend C 构建脚本
├── docs/                # 项目文档
│   ├── api/             # API 参考文档（~4700 文件）
│   │   ├── SIMD-API/    # SIMD API（C-API / 基础API / 高阶API）
│   │   ├── SIMT-API/    # SIMT API
│   │   ├── AI-CPU-API/  # AI CPU API
│   │   └── Utils-API/   # 工具 API
│   ├── guide/           # 编程与迁移指南
│   │   ├── 编程指南/     # 编程模型、语言扩展、硬件实现等
│   │   ├── 跨代迁移兼容性指南/ # 2201→3510 迁移（核心！）
│   │   └── 算子实践参考/  # 90+ 算子实现与优化指南
│   └── asc_a5_feature_guide.md  # A5 新特性导航
├── examples/            # 算子样例代码
│   ├── 01_simd_cpp_api/
│   ├── 02_simd_c_api/
│   ├── 03_simt_api/
│   └── 04_aicpu/
├── impl/                # API 实现源码
├── include/             # API 头文件声明
├── tests/               # UT 测试用例
├── tools/               # 工具链
└── .agent/skills/       # asc-devkit 内置 agent skills
```
