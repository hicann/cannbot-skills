# asc-devkit 目录 → 迁移场景快速映射

> **定位**：本文件回答「**该去哪找**」（场景 → 目录）；**怎么查**（优先级与置信度）见 `search-rules.md`；**有哪些文件**（全量详表）见 `knowledge-index.md`。

> **★★ 布局演进与定位协议（本 skill 唯一权威；search-rules / knowledge-index 均指向此处）**
>
> asc-devkit 仓库布局随版本演进：**2026-09 起**文档置于语言层下（`docs/zh/` 中文、`docs/en/` 英文），且目录名与文件名为**英文**；**早期版本**为 `docs/api/`、`docs/guide/` + 中文目录名（如 `编程指南`）。下表及本文件后续表格按早期布局书写，**命中时须按下表换算**；无论哪个版本，**路径以 `$DEVKIT_PATH` 实际内容为准**。
>
> | 主题 | 2026-09 实测路径（相对 `$DEVKIT_PATH`） |
> |---|---|
> | API 文档 | `docs/zh/api/{SIMD-API,SIMT-API,AI-CPU-API,Utils-API}/` |
> | 基础 API | `docs/zh/api/SIMD-API/basic_api/`（`memory_vector_compute` / `reg_vector_compute` / `cube_compute_ISASI` / `sync_control` / `resource_management` / `data_structures` / `cache_control` / `scalar_compute` / `Kernel-Tiling` / `debug_interface` / `tool_interface` / `special_register_access`） |
> | C-API | `docs/zh/api/SIMD-API/c_api/`（`vector_compute` / `reg_compute` / `cube_compute` / `cube_datamove` / `vector_datamove` / `sync` / `atomic` / `spr` / `scalar_compute` / `cache_ctrl`） |
> | 高阶 API | `docs/zh/api/SIMD-API/adv_api/` |
> | Tensor API | `docs/zh/api/SIMD-API/tensor_api/` |
> | 编程指南 | `docs/zh/guide/programming_guide/`（`programming_model` / `language_extension` / `advanced_programming` / `compilation_and_execution` / `debug_and_tuning` / `library_api`） |
> | **跨代迁移指南** | `docs/zh/guide/cross_gen_migration_guide/`（`overview.md` / `ascend_c_api_compatibility.md` / `3510_arch_migration/2201_to_3510_arch_changes.md` / `3510_arch_migration/2201_to_3510_guide/`） |
> | **A5/3510 新特性** | `docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md`（旧名 `asc_a5_feature_guide.md` / `asc_950_feature_guide.md` 均**已不存在**） |
> | 算子实践参考 | `docs/zh/guide/operator_practice/`（`simd_operator_impl` / `simd_operator_optimization` / `simt_operator_impl` / `simt_operator_optimization` / `simd_simt_hybrid_operator_impl` / `simd_simt_hybrid_optimization` / `best_practices` / `performance_analysis` / `functional_debug` / `typical_operator_cases`） |
> | 样例代码 | `examples/01_simd_cpp_api/`（`06_compatibility_guide/` 为跨代兼容样例：`data_copy_l1togm` / `pattern_transformation` / `matmul_s4` / `fill` / `set_loaddata_boundary` / `subnormal` / `scatter` / `fixpipe_params_switch`）、`02_simd_c_api/`、`03_simt_api/`、`04_aicpu/`、`05_simd_simt_hybrid/` |
> | 头文件 / 实现 / 构建 | `include/{basic_api,c_api,adv_api,simt_api,tensor_api,utils}/`、`impl/`、`cmake/`、`build.sh` |
>
> **定位协议（三级）**：① 先 `ls $DEVKIT_PATH/docs/` 确认语言层与目录名；② 路径不存在时 `find $DEVKIT_PATH/docs -iname '*关键词*'`（英文关键词，如 `migration`、`optimization`、`sync`）；③ 仍无命中再 `rg "关键词" $DEVKIT_PATH/docs -l`。**禁止**在不确认路径存在的情况下按本表硬拼路径。



> 当 agent 不确定去哪个目录时，先查此表。
> 所有路径相对于 $DEVKIT_PATH。

## 快速定位表

| 迁移场景 | 首选目录 | 备选目录 |
|---------|---------|---------|
| 查某 API 在 A5 上的用法 | `docs/zh/api/SIMD-API/` | `include/` 头文件 |
| 查 API 迁移前后的差异 | `docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/` | — |
| 查某算子怎么实现 | `examples/` | `docs/zh/guide/operator_practice/` |
| 查 A5 新增了什么能力 | `docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md` | `docs/zh/guide/cross_gen_migration_guide/` |
| 查头文件中函数声明 | `include/` | `impl/` |
| 查 UB 大小/bank 结构 | `docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/` | — |
| 查性能优化方法 | `docs/zh/guide/operator_practice/` | `examples/` |
| 查 CMake 编译配置 | `cmake/` | `build.sh` |
| 查 SIMT 线程模型/DCache | `docs/zh/guide/programming_guide/programming_model/ai_core_simt_programming/` | `docs/zh/api/SIMT-API/` |
| 查 SIMD 编程模型 | `docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/` | `docs/zh/api/SIMD-API/` |
| 查 BuiltIn 关键字 | `docs/zh/guide/programming_guide/language_extension/` | — |
| 查芯片架构映射 | `.agent/skills/asc-npu-arch/` | — |
| 查 API 兼容策略 | `docs/zh/guide/cross_gen_migration_guide/ascend_c_api_compatibility.md` | — |
| 查高阶 API（Matmul/Softmax 等） | `docs/zh/api/SIMD-API/adv_api/` | `include/adv_api/` |
| 查 Reg 编程（A5 新增） | `docs/zh/api/SIMD-API/c_api/reg_compute/` | `docs/zh/api/SIMD-API/basic_api/reg_vector_compute/` |

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
