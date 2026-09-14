# 全量仓知识索引（asc-devkit）
> **⚠ 路径校正（2026-08 实测）**：asc-devkit 当前版本的文档实际位于 `docs/zh/api/` 与 `docs/zh/guide/`（多一层 `zh`），且 A5 特性导航文件名为 `docs/zh/asc_950_feature_guide.md`（非 `asc_a5_feature_guide.md`）。本文件中的 `docs/api/`、`docs/guide/` 路径访问时请自动补 `zh` 层。


> 本索引基于 asc-devkit 仓库目录结构，帮助 agent 快速定位所需材料。
> 所有路径均相对于 $DEVKIT_PATH（git clone 后的仓库根目录）。
> 仓库地址：https://gitcode.com/cann/asc-devkit

## 一、API 文档索引

### 1.1 SIMD C-API（语言扩展层，基于指针编程）

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| Memory 矢量计算 | docs/api/SIMD-API/C-API/Memory矢量计算/ | 传统基于 LocalTensor 的 C-API |
| Reg 矢量计算 | docs/api/SIMD-API/C-API/Reg矢量计算/ | 基于 RegTensor 的 C-API（A5 新增） |
| 矢量数据搬运 | docs/api/SIMD-API/C-API/矢量数据搬运/ | GM↔UB, GM↔L0A/L0B 等 |
| 矩阵数据搬运 | docs/api/SIMD-API/C-API/矩阵数据搬运/ | L1/L0C 相关搬运 |
| 矩阵计算 | docs/api/SIMD-API/C-API/矩阵计算/ | Cube/Mmad 相关 |
| 原子操作 | docs/api/SIMD-API/C-API/原子操作/ | Atomic 操作 |
| 标量计算 | docs/api/SIMD-API/C-API/标量计算/ | 标量运算 |
| 其他操作 | docs/api/SIMD-API/C-API/其他操作/ | asc_init 等 |
| 同步控制 | docs/api/SIMD-API/C-API/同步控制/ | 同步机制 |
| 系统变量 | docs/api/SIMD-API/C-API/系统变量/ | 系统级变量 |
| 缓存控制 | docs/api/SIMD-API/C-API/缓存控制/ | Cache 操作 |
| 数据结构 | docs/api/SIMD-API/C-API/数据结构/ | C-API 数据结构 |
| C-API 总览 | docs/api/SIMD-API/C-API/C-API.md | C-API 概述 |
| 通用说明和约束 | docs/api/SIMD-API/C-API/通用说明和约束.md | 通用约束条件 |

### 1.2 基础 API（C++ Tensor 编程）

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| Memory 矢量计算 | docs/api/SIMD-API/基础API/Memory矢量计算/ | 基于 Tensor 的 Memory API |
| Reg 矢量计算 | docs/api/SIMD-API/基础API/Reg矢量计算/ | 基于 Tensor 的 Reg API |
| 数据结构 | docs/api/SIMD-API/基础API/数据结构/ | Tensor/Layout/Pointer 等 |
| 矩阵计算（ISASI） | docs/api/SIMD-API/基础API/矩阵计算（ISASI）/ | 分离式矩阵计算 |
| 矩阵计算（TensorAPI） | docs/api/SIMD-API/基础API/矩阵计算（TensorAPI）/ | Tensor 式矩阵计算 |
| Cube 分组管理 | docs/api/SIMD-API/基础API/Cube分组管理（ISASI）/ | Cube 管理 |
| 原子操作 | docs/api/SIMD-API/基础API/原子操作/ | 基础 API 原子操作 |
| 同步控制 | docs/api/SIMD-API/基础API/同步控制/ | 基础 API 同步 |
| 标量计算 | docs/api/SIMD-API/基础API/标量计算/ | 基础 API 标量 |
| 缓存控制 | docs/api/SIMD-API/基础API/缓存控制/ | 基础 API Cache |
| 资源管理 | docs/api/SIMD-API/基础API/资源管理/ | 资源分配管理 |
| 调试接口 | docs/api/SIMD-API/基础API/调试接口/ | 调试用 API |
| 工具接口 | docs/api/SIMD-API/基础API/工具接口/ | 辅助工具 |
| 特殊寄存器访问 | docs/api/SIMD-API/基础API/特殊寄存器访问/ | SPR 访问 |
| Kernel-Tiling | docs/api/SIMD-API/基础API/Kernel-Tiling/ | Tiling 接口 |
| 基础 API 总览 | docs/api/SIMD-API/基础API/基础API.md | 基础 API 概述 |
| 数据搬运导览 | docs/api/SIMD-API/基础API/数据搬运导览/ | 搬运 API 导航 |

### 1.3 高阶 API

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 数学计算 | docs/api/SIMD-API/高阶API/数学计算/ | Exp/Ln/Sqrt 等 |
| 归约操作 | docs/api/SIMD-API/高阶API/归约操作/ | ReduceMax/Min/Sum |
| 排序操作 | docs/api/SIMD-API/高阶API/排序操作/ | Sort/MrgSort |
| 矩阵计算 | docs/api/SIMD-API/高阶API/矩阵计算/ | Matmul 高阶 |
| 卷积计算 | docs/api/SIMD-API/高阶API/卷积计算/ | Conv 高阶 |
| 归一化操作 | docs/api/SIMD-API/高阶API/归一化操作/ | Softmax/LayerNorm 等 |
| 数据过滤 | docs/api/SIMD-API/高阶API/数据过滤/ | TopK/Gather 等 |
| 激活函数 | docs/api/SIMD-API/高阶API/激活函数/ | Relu/Gelu 等 |
| 量化操作 | docs/api/SIMD-API/高阶API/量化操作/ | Quant/Dequant |
| 张量变换 | docs/api/SIMD-API/高阶API/张量变换/ | Transpose/Reshape |
| 索引计算 | docs/api/SIMD-API/高阶API/索引计算/ | Index 计算 |
| 随机函数 | docs/api/SIMD-API/高阶API/随机函数/ | Random 生成 |
| HCCL 通信类 | docs/api/SIMD-API/高阶API/HCCL通信类/ | 分布式通信 |
| 高阶 API 总览 | docs/api/SIMD-API/高阶API/高阶API.md | 高阶 API 概述 |

### 1.4 SIMT API

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 原子操作 | docs/api/SIMT-API/原子操作/ | SIMT Atomic |
| Warp 函数 | docs/api/SIMT-API/Warp函数/ | Warp 级操作 |
| 协作组 | docs/api/SIMT-API/协作组/ | Cooperative Groups |
| 数学函数 | docs/api/SIMT-API/数学函数/ | SIMT 数学运算 |
| 访存函数 | docs/api/SIMT-API/访存函数/ | SIMT 内存访问 |
| 同步与内存栅栏 | docs/api/SIMT-API/同步与内存栅栏/ | SIMT Sync |
| 地址空间谓词函数 | docs/api/SIMT-API/地址空间谓词函数/ | 地址空间判断 |
| 地址空间转换函数 | docs/api/SIMT-API/地址空间转换函数/ | 地址空间转换 |
| SIMT API 总览 | docs/api/SIMT-API/SIMT-API.md | SIMT API 概述 |
| SIMD 与 SIMT 混合编程简介 | docs/api/SIMT-API/SIMD与SIMT混合编程简介/ | 混合编程概述 |

### 1.5 其他 API

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| AI CPU API | docs/api/AI-CPU-API/ | AI CPU 专用 API |
| Utils API | docs/api/Utils-API/ | 工具类 API |
| API 列表总览 | docs/api/Ascend-C-API列表.md | 全量 API 分类表 |
| API README | docs/api/README.md | API 文档目录 |

## 二、编程指南索引

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 编程指南目录 | docs/guide/index.md | 编程指南主目录（368 行，链接所有章节） |
| SIMD 编程模型 | docs/guide/编程指南/编程模型/AI-Core-SIMD编程/ | SIMD 编程全貌 |
| SIMD 抽象硬件架构 | docs/guide/编程指南/编程模型/AI-Core-SIMD编程/抽象硬件架构.md | 硬件抽象层说明 |
| SIMD 基于 Tensor CPP | docs/guide/编程指南/编程模型/AI-Core-SIMD编程/基于Tensor的CPP编程/ | C++ Tensor 编程 |
| SIMD 基于指针 C | docs/guide/编程指南/编程模型/AI-Core-SIMD编程/基于指针的C语言编程/ | C 指针编程 |
| SIMD TPipe/TQue 框架 | docs/guide/编程指南/编程模型/AI-Core-SIMD编程/基于TPipe-TQue框架编程/ | 框架编程 |
| SIMD 核函数 | docs/guide/编程指南/编程模型/AI-Core-SIMD编程/核函数.md | Kernel 函数说明 |
| SIMT 编程模型 | docs/guide/编程指南/编程模型/AI-Core-SIMT编程/ | SIMT 编程全貌 |
| SIMT 线程架构 | docs/guide/编程指南/编程模型/AI-Core-SIMT编程/线程架构.md | SIMT 线程模型 |
| SIMT 内存层级 | docs/guide/编程指南/编程模型/AI-Core-SIMT编程/内存层级.md | DCache 等 |
| SIMT 同步机制 | docs/guide/编程指南/编程模型/AI-Core-SIMT编程/同步机制.md | SIMT 同步 |
| SIMT 核函数 | docs/guide/编程指南/编程模型/AI-Core-SIMT编程/核函数.md | SIMT Kernel |
| SIMT 编程示例 | docs/guide/编程指南/编程模型/AI-Core-SIMT编程/编程示例.md | SIMT 样例 |
| SIMD BuiltIn 关键字 | docs/guide/编程指南/语言扩展层/SIMD-BuiltIn关键字.md | SIMD 语言扩展 |
| SIMT BuiltIn 关键字 | docs/guide/编程指南/语言扩展层/SIMT-BuiltIn关键字.md | SIMT 语言扩展 |
| 混合 BuiltIn | docs/guide/编程指南/语言扩展层/SIMD与SIMT混合编程BuiltIn关键字.md | 混合语言扩展 |
| SIMD C API | docs/guide/编程指南/语言扩展层/SIMD语言扩展层C-API.md | SIMD C 扩展 |
| SIMT C API | docs/guide/编程指南/语言扩展层/SIMT语言扩展层C-API.md | SIMT C 扩展 |
| 硬件实现 | docs/guide/编程指南/高级编程/硬件实现/ | 各 NPU 版本硬件细节 |
| SuperKernel | docs/guide/编程指南/高级编程/SuperKernel/ | SuperKernel 编程 |
| 高级特性 | docs/guide/编程指南/高级编程/高级特性/ | 高级编程特性 |
| 编译与运行 | docs/guide/编程指南/编译与运行/ | BiSheng 编译器、RTC 等 |
| 调试调优 | docs/guide/编程指南/调试调优/ | CPU 孪生调试、NPU 调试 |
| 类库 API | docs/guide/编程指南/类库API/ | 基础/高阶 API 概述 |

## 三、跨代迁移指南（核心！）

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 迁移概述 | docs/guide/跨代迁移兼容性指南/概述.md | 迁移核心原则 |
| API 兼容策略 | docs/guide/跨代迁移兼容性指南/Ascend-C-API兼容策略.md | 各层级 API 兼容性矩阵 |
| 220x→351x 架构变更 | docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201到3510架构变更.md | 数据通路/计算/存储/同步变更详述 |
| 基础 API 迁移 | docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201迁移3510指导/基础API迁移指导.md | 逐 API 迁移对照 |
| 高阶 API 迁移 | docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201迁移3510指导/高阶API迁移指导.md | 高阶 API 变更 |
| 算子编译迁移 | docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201迁移3510指导/算子编译迁移指导.md | --npu-arch 等编译变更 |
| A5 特性导航 | docs/asc_a5_feature_guide.md | 13 个 A5 新特性索引 |
| 如何选择 API | docs/asc_how_to_choose_api.md | 多层级 API 选择指南 |

## 四、算子实践与样例

| 子主题 | 全量仓路径 | 说明 |
|--------|-----------|------|
| 算子实践参考目录 | docs/guide/算子实践参考/ | 90+ 优化指南 |
| SIMD 算子实现 | docs/guide/算子实践参考/SIMD算子实现/ | 矢量/矩阵/融合算子实现 |
| SIMD 性能优化 | docs/guide/算子实践参考/SIMD算子性能优化/ | 访存/流水线/计算/Tiling 优化 |
| SIMT 算子实现 | docs/guide/算子实践参考/SIMT算子实现/ | SIMT 算子实现 |
| SIMT 性能优化 | docs/guide/算子实践参考/SIMT算子性能优化/ | SIMT 性能优化 |
| SIMD+SIMT 混合实现 | docs/guide/算子实践参考/SIMD与SIMT混合算子实现/ | 混合编程实现 |
| SIMD+SIMT 混合优化 | docs/guide/算子实践参考/SIMD与SIMT混合算子性能优化/ | 混合编程优化 |
| 优秀实践 | docs/guide/算子实践参考/优秀实践/ | FlashAttention/GroupedMatmul 等 |
| 典型算子实践案例 | docs/guide/算子实践参考/典型算子实践案例/ | 端到端案例 |
| 功能调试 | docs/guide/算子实践参考/功能调试/ | 调试技巧 |
| 性能分析 | docs/guide/算子实践参考/性能分析/ | 性能分析方法 |
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
