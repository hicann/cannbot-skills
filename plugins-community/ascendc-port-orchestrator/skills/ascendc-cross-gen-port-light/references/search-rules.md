# 文档查找规则与置信度评估
> **⚠ 路径校正（2026-08 实测）**：asc-devkit 当前版本的文档实际位于 `docs/zh/api/` 与 `docs/zh/guide/`（多一层 `zh`），且 A5 特性导航文件名为 `docs/zh/asc_950_feature_guide.md`（非 `asc_a5_feature_guide.md`）。本文件中的 `docs/api/`、`docs/guide/` 路径访问时请自动补 `zh` 层。


> 本文件定义 agent 在迁移过程中查找技术文档的标准流程。
> 查找时必须遵循三级优先级，并在每次查找后评估置信度。

## 查找优先级（从高到低）

### Level 1：本地 Skill 文件（最高优先）

以下主题**必须先查本地 skill 文件**，这些是 skill 独有的迁移工作流和实现指南：

| 主题 | 本地路径 |
|------|---------|
| 迁移实现方案（L1） | `references/impl/l1-guide.md` |
| API 映射表（L2 量化路径等） | `references/impl/api-mapping.md` |
| 构建脚本模板 | `references/build_system/build_and_install.sh.template` |
| 精度测试流程 | `references/precision-testing/` 下全部文件 |
| 阶段工作流 | `stages/` 下对应 stage 文件 |

### Level 2：全量仓精确查找（按索引路由）

以下主题**必须按 `knowledge-index.md` 路由到全量仓**（`$DEVKIT_PATH`）：

| 主题 | 全量仓首选路径 |
|------|-------------|
| API 接口详情（函数签名、参数、约束、示例） | `$DEVKIT_PATH/docs/api/SIMD-API/` 下对应子目录 |
| 迁移兼容性 / API 变更对照 | `$DEVKIT_PATH/docs/guide/跨代迁移兼容性指南/` |
| A5 架构变更详情 | `$DEVKIT_PATH/docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201到3510架构变更.md` |
| 编程模型（SIMD/SIMT/混合） | `$DEVKIT_PATH/docs/guide/编程指南/编程模型/` |
| 硬件实现细节（NPU 3510 参数） | `$DEVKIT_PATH/docs/guide/编程指南/高级编程/硬件实现/` |
| 算子实现参考与优化技巧 | `$DEVKIT_PATH/docs/guide/算子实践参考/` |
| A5 新特性导航 | `$DEVKIT_PATH/docs/asc_a5_feature_guide.md` |
| API 选择指南 | `$DEVKIT_PATH/docs/asc_how_to_choose_api.md` |
| 语言扩展层 BuiltIn 关键字 | `$DEVKIT_PATH/docs/guide/编程指南/语言扩展层/` |

### Level 3：全量仓探索查找（兜底）

当 Level 1/2 未找到时，按以下顺序逐步扩大搜索范围：

1. **搜索样例代码**：`$DEVKIT_PATH/examples/` 目录下按编程模型分类查找
   - SIMD C++ → `examples/01_simd_cpp_api/`
   - SIMD C → `examples/02_simd_c_api/`
   - SIMT → `examples/03_simt_api/`
   - AICPU → `examples/04_aicpu/`

2. **搜索头文件原型**：`$DEVKIT_PATH/include/` 目录下用 `rg` 搜索函数名
   ```bash
   rg "函数名" $DEVKIT_PATH/include/ --type cpp -l
   ```

3. **搜索实现细节**：`$DEVKIT_PATH/impl/` 目录下用 `rg` 搜索函数名
   ```bash
   rg "函数名" $DEVKIT_PATH/impl/ --type cpp -l
   ```

4. **全局关键词搜索**：在全量仓根目录 `rg` 搜索
   ```bash
   rg "关键词" $DEVKIT_PATH/ --type md -l
   rg "关键词" $DEVKIT_PATH/ --type cpp -l
   ```

## 置信度评估标准

| 置信度 | 条件 | 行动 |
|--------|------|------|
| **HIGH** | 找到精确匹配的文档/代码，信息完整 | 直接使用，继续当前阶段 |
| **MEDIUM** | 找到相关文档但信息不完全，或存在版本差异 | 结合多文件综合判断，标注不确定点 |
| **LOW** | 未找到直接相关材料，或搜索结果模糊 | 停止当前路径，扩大搜索或上报用户 |

### 置信度评估方法

- 从 `knowledge-index.md` 直接路由到目标文件 → HIGH
- 从 Level 3 的 rg 搜索命中 → 需要验证相关性，通常 MEDIUM
- 搜索结果超过 10 个文件仍无法定位 → LOW，停止搜索上报用户

## 触发全量仓深度搜索的场景（显式列表）

以下场景**必须主动去全量仓查找**，不能仅依赖本地 skill 文件：

1. **算子实现参考**：迁移时需要参考同类算子的实现
   → 查 `$DEVKIT_PATH/examples/` 和 `$DEVKIT_PATH/docs/guide/算子实践参考/`

2. **API 函数签名验证**：确认 A5 上某 API 的确切原型和参数列表
   → 查 `$DEVKIT_PATH/include/` 头文件

3. **性能优化技巧**：UB bank conflict、pipeline 优化、DoubleBuffer 等
   → 查 `$DEVKIT_PATH/docs/guide/算子实践参考/SIMD算子性能优化/`

4. **数据类型支持确认**：某 API 在 A5 上是否支持某 dtype
   → 查 API 文档的 `<cann-filter>` 标签或 `$DEVKIT_PATH/docs/api/` 下对应文件

5. **架构特性细节**：NPU 3510 的具体硬件参数（UB 大小、bank 结构等）
   → 查 `$DEVKIT_PATH/docs/guide/编程指南/高级编程/硬件实现/`

6. **编译构建问题**：CMake 配置、编译选项
   → 查 `$DEVKIT_PATH/cmake/` 和 `$DEVKIT_PATH/build.sh`

7. **SIMT 编程细节**：线程架构、DCache、同步机制
   → 查 `$DEVKIT_PATH/docs/guide/编程指南/编程模型/AI-Core-SIMT编程/`

8. **高阶 API 用法**：Matmul/Softmax/ReduceSum 等高阶接口
   → 查 `$DEVKIT_PATH/docs/api/SIMD-API/高阶API/`

## LOADED 标记规范

从全量仓读取文件后，MUST 输出 LOADED 标记：

```
[LOADED] $DEVKIT_PATH/<相对路径>
```

示例：
```
[LOADED] $DEVKIT_PATH/docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201到3510架构变更.md
[LOADED] $DEVKIT_PATH/docs/api/SIMD-API/C-API/Reg矢量计算/reg_vector/asc_abs.md
```
