# 文档查找规则与置信度评估

> **★ 外部仓布局与定位协议（MUST 先读）**：见 `references/devkit-path-map.md` 顶部——含 2026-09 实测的「主题 → 路径」映射表与三级定位协议（`ls` 确认语言层 → `find -iname` → `rg`）。**禁止**在不确认路径存在的情况下硬拼路径。


> 本文件定义 agent 在迁移过程中查找技术文档的标准流程。
> 查找时必须遵循三级优先级，并在每次查找后评估置信度。

## 查找优先级（从高到低）

### Level 1：本地 Skill 文件（最高优先）

本 skill 自带的迁移工作流与实现指南**必须先查**——完整清单见 `SKILL.md` 的「Reference 文件索引」（stages/ 阶段文件、impl/ 实现指南、build_system/ 构建模板、precision-testing/ 精度资产）。这些是 skill 独有内容，全量仓中没有对应物。

### Level 2：全量仓精确查找（按索引路由）

按两个索引文件的**分工**路由，不在此重复列路径：

- **不知道该去哪个目录** → `references/devkit-path-map.md`「快速定位表」（迁移场景 → 首选/备选目录）
- **已知主题、要查具体文件** → `references/knowledge-index.md`（按 devkit 目录组织的全量详表：API 文档 / 编程指南 / 跨代迁移指南 / 算子实践 / 源码头文件）

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
   → 查 `$DEVKIT_PATH/examples/` 和 `$DEVKIT_PATH/docs/zh/guide/operator_practice/`

2. **API 函数签名验证**：确认 A5 上某 API 的确切原型和参数列表
   → 查 `$DEVKIT_PATH/include/` 头文件

3. **性能优化技巧**：UB bank conflict、pipeline 优化、DoubleBuffer 等
   → 查 `$DEVKIT_PATH/docs/zh/guide/operator_practice/simd_operator_optimization/`

4. **数据类型支持确认**：某 API 在 A5 上是否支持某 dtype
   → 查 API 文档的 `<cann-filter>` 标签或 `$DEVKIT_PATH/docs/zh/api/` 下对应文件

5. **架构特性细节**：NPU 3510 的具体硬件参数（UB 大小、bank 结构等）
   → 查 `$DEVKIT_PATH/docs/zh/guide/programming_guide/advanced_programming/hardware_implementation/`

6. **编译构建问题**：CMake 配置、编译选项
   → 查 `$DEVKIT_PATH/cmake/` 和 `$DEVKIT_PATH/build.sh`

7. **SIMT 编程细节**：线程架构、DCache、同步机制
   → 查 `$DEVKIT_PATH/docs/zh/guide/programming_guide/programming_model/ai_core_simt_programming/`

8. **高阶 API 用法**：Matmul/Softmax/ReduceSum 等高阶接口
   → 查 `$DEVKIT_PATH/docs/zh/api/SIMD-API/adv_api/`

## LOADED 标记规范

从全量仓读取文件后，MUST 输出 LOADED 标记：

```
[LOADED] $DEVKIT_PATH/<相对路径>
```

示例：
```
[LOADED] $DEVKIT_PATH/docs/zh/guide/cross_gen_migration_guide/3510_arch_migration/2201_to_3510_arch_changes.md
[LOADED] $DEVKIT_PATH/docs/zh/api/SIMD-API/c_api/reg_compute/
```
