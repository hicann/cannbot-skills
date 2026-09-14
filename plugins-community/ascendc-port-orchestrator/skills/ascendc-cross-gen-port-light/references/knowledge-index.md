# 全量仓知识索引（asc-devkit）

> **★ 外部仓布局与定位协议（MUST 先读）**：见 `references/devkit-path-map.md` 顶部——含 2026-09 实测的「主题 → 路径」映射表与三级定位协议（`ls` 确认语言层 → `find -iname` → `rg`）。**禁止**在不确认路径存在的情况下硬拼路径。


> 本索引基于 asc-devkit 仓库目录结构，帮助 agent 快速定位所需材料。
> 所有路径均相对于 $DEVKIT_PATH（git clone 后的仓库根目录）。
> 仓库地址：https://gitcode.com/cann/asc-devkit

## 一、API 文档索引

### 1.1 SIMD C-API（语言扩展层，基于指针编程）

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| Memory 矢量计算 | docs/zh/api/SIMD-API/c_api/vector_compute/ | 传统基于 LocalTensor 的 C-API |
| Reg 矢量计算 | docs/zh/api/SIMD-API/c_api/reg_compute/ | 基于 RegTensor 的 C-API（A5 新增） |
| 矢量数据搬运 | docs/zh/api/SIMD-API/c_api/vector_datamove/ | GM↔UB, GM↔L0A/L0B 等 |
| 矩阵数据搬运 | docs/zh/api/SIMD-API/c_api/cube_datamove/ | L1/L0C 相关搬运 |
| 矩阵计算 | docs/zh/api/SIMD-API/c_api/cube_compute/ | Cube/Mmad 相关 |
| 原子操作 | docs/zh/api/SIMD-API/c_api/atomic/ | Atomic 操作 |
| 标量计算 | docs/zh/api/SIMD-API/c_api/scalar_compute/ | 标量运算 |
| 其他操作 | docs/zh/api/SIMD-API/c_api/utils/ | asc_init 等 |
| 同步控制 | docs/zh/api/SIMD-API/c_api/sync/ | 同步机制 |
| 系统变量 | docs/zh/api/SIMD-API/c_api/spr/ | 系统级变量 |
| 缓存控制 | docs/zh/api/SIMD-API/c_api/cache_ctrl/ | Cache 操作 |
| 数据结构 | docs/zh/api/SIMD-API/c_api/defs/ | C-API 数据结构 |
| C-API 总览 | docs/zh/api/SIMD-API/c_api/C-API.md | C-API 概述 |
| 通用说明和约束 | docs/zh/api/SIMD-API/c_api/general_description_and_constraints.md | 通用约束条件 |

### 1.2 基础 API（C++ Tensor 编程）

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| Memory 矢量计算 | docs/zh/api/SIMD-API/basic_api/memory_vector_compute/ | 基于 Tensor 的 Memory API |
| Reg 矢量计算 | docs/zh/api/SIMD-API/basic_api/reg_vector_compute/ | 基于 Tensor 的 Reg API |
| 数据结构 | docs/zh/api/SIMD-API/basic_api/defs/ | Tensor/Layout/Pointer 等 |
| 矩阵计算（ISASI） | docs/zh/api/SIMD-API/basic_api/cube_compute（ISASI）/ | 分离式矩阵计算 |
| 矩阵计算（TensorAPI） | docs/zh/api/SIMD-API/basic_api/cube_compute（TensorAPI）/ | Tensor 式矩阵计算 |
| Cube 分组管理 | docs/zh/api/SIMD-API/basic_api/cube_group_mgmt_ISASI/ | Cube 管理 |
| 原子操作 | docs/zh/api/SIMD-API/basic_api/atomic/ | 基础 API 原子操作 |
| 同步控制 | docs/zh/api/SIMD-API/basic_api/sync/ | 基础 API 同步 |
| 标量计算 | docs/zh/api/SIMD-API/basic_api/scalar_compute/ | 基础 API 标量 |
| 缓存控制 | docs/zh/api/SIMD-API/basic_api/cache_ctrl/ | 基础 API Cache |
| 资源管理 | docs/zh/api/SIMD-API/basic_api/resource_management/ | 资源分配管理 |
| 调试接口 | docs/zh/api/SIMD-API/basic_api/debug_interface/ | 调试用 API |
| 工具接口 | docs/zh/api/SIMD-API/basic_api/tool_interface/ | 辅助工具 |
| 特殊寄存器访问 | docs/zh/api/SIMD-API/basic_api/special_register_access/ | SPR 访问 |
| Kernel-Tiling | docs/zh/api/SIMD-API/basic_api/Kernel-Tiling/ | Tiling 接口 |
| 基础 API 总览 | docs/zh/api/SIMD-API/basic_api/basic_api.md | 基础 API 概述 |
| 数据搬运导览 | docs/zh/api/SIMD-API/basic_api/data_move_guide/ | 搬运 API 导航 |

### 1.3 高阶 API

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 数学计算 | docs/zh/api/SIMD-API/adv_api/math_compute/ | Exp/Ln/Sqrt 等 |
| 归约操作 | docs/zh/api/SIMD-API/adv_api/reduction_operations/ | ReduceMax/Min/Sum |
| 排序操作 | docs/zh/api/SIMD-API/adv_api/sort_operations/ | Sort/MrgSort |
| 矩阵计算 | docs/zh/api/SIMD-API/adv_api/cube_compute/ | Matmul 高阶 |
| 卷积计算 | docs/zh/api/SIMD-API/adv_api/convolution_compute/ | Conv 高阶 |
| 归一化操作 | docs/zh/api/SIMD-API/adv_api/normalization/ | Softmax/LayerNorm 等 |
| 数据过滤 | docs/zh/api/SIMD-API/adv_api/data_filter/ | TopK/Gather 等 |
| 激活函数 | docs/zh/api/SIMD-API/adv_api/activation_functions/ | Relu/Gelu 等 |
| 量化操作 | docs/zh/api/SIMD-API/adv_api/quantization/ | Quant/Dequant |
| 张量变换 | docs/zh/api/SIMD-API/adv_api/tensor_transform/ | Transpose/Reshape |
| 索引计算 | docs/zh/api/SIMD-API/adv_api/index_compute/ | Index 计算 |
| 随机函数 | docs/zh/api/SIMD-API/adv_api/random_functions/ | Random 生成 |
| HCCL 通信类 | docs/zh/api/SIMD-API/adv_api/HCCL_communication/ | 分布式通信 |
| 高阶 API 总览 | docs/zh/api/SIMD-API/adv_api/adv_api.md | 高阶 API 概述 |

### 1.4 SIMT API

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 原子操作 | docs/zh/api/SIMT-API/atomic_operations/ | SIMT Atomic |
| Warp 函数 | docs/zh/api/SIMT-API/Warp_functions/ | Warp 级操作 |
| 协作组 | docs/zh/api/SIMT-API/cooperative_groups/ | Cooperative Groups |
| 数学函数 | docs/zh/api/SIMT-API/math_functions/ | SIMT 数学运算 |
| 访存函数 | docs/zh/api/SIMT-API/memory_access_functions/ | SIMT 内存访问 |
| 同步与内存栅栏 | docs/zh/api/SIMT-API/sync_and_memory_fence/ | SIMT Sync |
| 地址空间谓词函数 | docs/zh/api/SIMT-API/address_space_predicate_functions/ | 地址空间判断 |
| 地址空间转换函数 | docs/zh/api/SIMT-API/address_space_conversion_functions/ | 地址空间转换 |
| SIMT API 总览 | docs/zh/api/SIMT-API/SIMT-API.md | SIMT API 概述 |
| SIMD 与 SIMT 混合编程简介 | docs/zh/api/SIMT-API/SIMD_SIMT_hybrid_programming_intro/ | 混合编程概述 |

### 1.5 其他 API

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| AI CPU API | docs/zh/api/AI-CPU-API/ | AI CPU 专用 API |
| Utils API | docs/zh/api/Utils-API/ | 工具类 API |
| API 列表总览 | docs/zh/api/api_list.md | 全量 API 分类表 |
| API README | docs/zh/api/README.md | API 文档目录 |

## 二、编程指南索引

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 编程指南目录 | docs/zh/guide/index.md | 编程指南主目录（368 行，链接所有章节） |
| SIMD 编程模型 | docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/ | SIMD 编程全貌 |
| SIMD 抽象硬件架构 | docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/abstract_hardware_architecture.md | 硬件抽象层说明 |
| SIMD 基于 Tensor CPP | docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/cpp_tensor_programming/ | C++ Tensor 编程 |
| SIMD 基于指针 C | docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/c_pointer_programming/ | C 指针编程 |
| SIMD TPipe/TQue 框架 | docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/tpipe_tque_programming/ | 框架编程 |
| SIMD 核函数 | docs/zh/guide/programming_guide/programming_model/ai_core_simd_programming/kernel_function.md | Kernel 函数说明 |
| SIMT 编程模型 | docs/zh/guide/programming_guide/programming_model/ai_core_simt_programming/ | SIMT 编程全貌 |
| SIMT 线程架构 | docs/zh/guide/programming_guide/programming_model/ai_core_simt_programming/thread_architecture.md | SIMT 线程模型 |
| SIMT 内存层级 | docs/zh/guide/programming_guide/programming_model/ai_core_simt_programming/memory_hierarchy.md | DCache 等 |
| SIMT 同步机制 | docs/zh/guide/programming_guide/programming_model/ai_core_simt_programming/synchronization.md | SIMT 同步 |
| SIMT 核函数 | docs/zh/guide/programming_guide/programming_model/ai_core_simt_programming/kernel_function.md | SIMT Kernel |
| SIMT 编程示例 | docs/zh/guide/programming_guide/programming_model/ai_core_simt_programming/programming_examples.md | SIMT 样例 |
| SIMD BuiltIn 关键字 | docs/zh/guide/programming_guide/language_extension/simd_builtin_keywords.md | SIMD 语言扩展 |
| SIMT BuiltIn 关键字 | docs/zh/guide/programming_guide/language_extension/simt_builtin_keywords.md | SIMT 语言扩展 |
| 混合 BuiltIn | docs/zh/guide/programming_guide/language_extension/simd_simt_hybrid_builtin_keywords.md | 混合语言扩展 |
| SIMD C API | docs/zh/guide/programming_guide/language_extension/simd_language_extension_c_api.md | SIMD C 扩展 |
| SIMT C API | docs/zh/guide/programming_guide/language_extension/simt_language_extension_c_api.md | SIMT C 扩展 |
| 硬件实现 | docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/ | 各 NPU 版本硬件细节 |
| SuperKernel | docs/zh/guide/programming_guide/advanced_programming/super_kernel/ | SuperKernel 编程 |
| 高级特性 | docs/zh/guide/programming_guide/advanced_programming/advanced_ai_core_programming_model/ | 高级编程特性 |
| 编译与运行 | docs/zh/guide/programming_guide/compilation_and_execution/ | BiSheng 编译器、RTC 等 |
| 调试调优 | docs/zh/guide/programming_guide/debug_and_tuning/ | CPU 孪生调试、NPU 调试 |
| 类库 API | docs/zh/guide/programming_guide/library_api/ | 基础/高阶 API 概述 |

## 三、跨代迁移指南（核心！）

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 迁移概述 | docs/zh/guide/cross_gen_migration_guide/overview.md | 迁移核心原则 |
| API 兼容策略 | docs/zh/guide/cross_gen_migration_guide/ascend_c_api_compatibility.md | 各层级 API 兼容性矩阵 |
| 220x→351x 架构变更 | docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.md | 数据通路/计算/存储/同步变更详述 |
| 基础 API 迁移 | docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_guide/ | 逐 API 迁移对照 |
| 高阶 API 迁移 | docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_guide/ | 高阶 API 变更 |
| 算子编译迁移 | docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_guide/ | --npu-arch 等编译变更 |
| A5 特性导航 | docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md | 13 个 A5 新特性索引 |
| 如何选择 API | docs/zh/asc_how_to_choose_api.md | 多层级 API 选择指南 |

## 四、算子实践与样例

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 算子实践参考目录 | docs/zh/guide/operator_practice/ | 90+ 优化指南 |
| SIMD 算子实现 | docs/zh/guide/operator_practice/simd_operator_impl/ | 矢量/矩阵/融合算子实现 |
| SIMD 性能优化 | docs/zh/guide/operator_practice/simd_operator_optimization/ | 访存/流水线/计算/Tiling 优化 |
| SIMT 算子实现 | docs/zh/guide/operator_practice/simt_operator_impl/ | SIMT 算子实现 |
| SIMT 性能优化 | docs/zh/guide/operator_practice/simt_operator_optimization/ | SIMT 性能优化 |
| SIMD+SIMT 混合实现 | docs/zh/guide/operator_practice/simd_simt_hybrid_operator_impl/ | 混合编程实现 |
| SIMD+SIMT 混合优化 | docs/zh/guide/operator_practice/simd_simt_hybrid_optimization/ | 混合编程优化 |
| 优秀实践 | docs/zh/guide/operator_practice/best_practices/ | FlashAttention/GroupedMatmul 等 |
| 典型算子实践案例 | docs/zh/guide/operator_practice/typical_operator_cases/ | 端到端案例 |
| 功能调试 | docs/zh/guide/operator_practice/functional_debug/ | 调试技巧 |
| 性能分析 | docs/zh/guide/operator_practice/performance_analysis/ | 性能分析方法 |
| SIMD C++ 样例 | examples/01_simd_cpp_api/ | SIMD C++ API 样例 |
| SIMD C 样例 | examples/02_simd_c_api/ | SIMD C API 样例 |
| SIMT 样例 | examples/03_simt_api/ | SIMT API 样例 |
| AICPU 样例 | examples/04_aicpu/ | AI CPU 样例 |

## 五、源码与头文件

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 高阶 API 声明 | include/adv_api/ | 高阶 API 头文件 |
| 基础 API 声明 | include/basic_api/ | 基础 API 头文件 |
| C API 声明 | include/c_api/ | C 语言扩展层头文件 |
| SIMT API 声明 | include/simt_api/ | SIMT API 头文件 |
| Tensor API 声明 | include/tensor_api/ | Tensor API 头文件 |
| AI CPU API 声明 | include/aicpu_api/ | AI CPU 头文件 |
| 工具类声明 | include/utils/ | 工具类头文件 |
| 高阶 API 实现 | impl/adv_api/ | 高阶 API 实现 |
| 基础 API 实现 | impl/basic_api/ | 基础 API 实现 |
| C API 实现 | impl/c_api/ | C 语言扩展层实现 |
| SIMT API 实现 | impl/simt_api/ | SIMT API 实现 |
| Tensor API 实现 | impl/tensor_api/ | Tensor API 实现 |
| AI CPU API 实现 | impl/aicpu_api/ | AI CPU 实现 |
| 工具类实现 | impl/utils/ | 工具类实现 |
| CMake 构建 | cmake/ | Ascend C 构建脚本 |
| 工具链 | tools/ | 编译/打包工具 |

## 六、Agent Skills（asc-devkit 内置）

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| NPU 架构事实 | .agent/skills/asc-npu-arch/ | chip→arch→SocVersion 映射、dtype 事实 |
| API UT 生成 | .agent/skills/asc-api-ut-gen/ | API 单元测试生成系统 |
| 文档规范检查 | .agent/skills/asc-doc-checker/ | 文档写作规范 |
