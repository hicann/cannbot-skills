# 阶段 1：评估定级（含 API 差异风险扫描）

**前置条件**：`[GATE-0] ENV_CONFIRMED` 已输出。

## LOADED 检查点

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/impl/api-diff-guide.md` | `[LOADED] api-diff-guide` |
| 2 | `references/impl/cube-migration-guide.md`（**cube 类算子** MUST，见 Step 1.2） | `[LOADED] cube-migration-guide` |

**★ 本文件 MUST 在算子源码读取前加载。** API 差异指南定义了精度/兼容性/性能三个维度的风险扫描清单。

## Step 1.1：读取算子源码

MUST 读取以下文件：

| 文件 | 检查内容 |
|------|----------|
| `_def.cpp` | SoC 列表、DataType 配置 |
| kernel `.cpp` / `.h` | API 使用模式（Memory-based / Register-based）、条件编译、**跨核同步原语（CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll、NotifyNextBlock/WaitPreBlock）** |
| `op_host/CMakeLists.txt` | 是否已有 ascend950 分支 |
| Tiling `.cpp` / `.h` | 是否有 `IsRegbaseSocVersion` 判断 |

## Step 1.2：查阅全量仓参考（按需）

在判定迁移层级之前，如果需要了解算子的参考实现或 A5 新特性，按 `references/search-rules.md` 路由到全量仓：

- **查看同类算子样例**：`$DEVKIT_PATH/examples/` 目录下按编程模型分类查找
- **确认 A5 新特性**：读 `$DEVKIT_PATH/docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md`
- **查阅算子实践参考**：`$DEVKIT_PATH/docs/zh/guide/operator_practice/` 下查找同类算子的实现和优化案例
- **Cube 类算子**（Step 1.1 中发现 Mmad/LoadData/Fixpipe/DataCopyCO12DstParams 等 cube API）：MUST 先读 `references/impl/cube-migration-guide.md`（输出 `[LOADED] cube-migration-guide`），再按 search-rules 路由到全量仓兼容性样例（`06_compatibility_guide/` 下 data_copy_l1togm / pattern_transformation / matmul_s4 / fill / set_loaddata_boundary）

读取全量仓文件后输出：`[LOADED] $DEVKIT_PATH/<相对路径>`

## Step 1.3：★ API 差异风险扫描

**基于已加载的 `api-diff-guide.md`，对算子源码做三维度风险扫描。** 扫描结果记入算子源码分析摘要。

### 1.3.1 精度风险扫描（Subnormal）

按 `api-diff-guide.md` §1.4 清单扫描：

| 扫描项 | 检测方法 | 命中后的处理 |
|--------|---------|-------------|
| 是否调用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal | grep 源码中的 API 名 | 标记为 Subnormal 风险算子 |
| **★ 是否存在结构性 subnormal 排除（策略 0，最先检查）** | 追受影响操作数生成链，找结构性下界证明：max 归一锚点（`exp(0)=1` → 分母 ≥1）/ invalid line 哨兵修复 / 数学下界推导（详见 `api-diff-guide.md` §1.4 策略 0） | **给出证明链（含代码行号）→ 保持 INTRINSIC，无需 API 级适配**；禁止无证明直接跳过 |
| 输入数据是否可能包含 subnormal | **★ 基于证据分析**（见下方"Subnormal 风险证据要求"） | 必须选择适配策略（eps/模板参数/分支） |
| 是否在性能关键路径 | 判断算子是否为高频算子（RMSNorm/RoPE/Softmax） | 性能关键路径优先用 eps 规避 |
| 是否已有 eps 处理 | 检查源码中 `+ epsilon` / `+ eps` / 分母保护常量 | **★ 仅有 eps 不能免处理**——MUST 按 `api-diff-guide.md` §1.6 检查 eps 在所有支持 dtype 下是否可表示 |

**★ Subnormal 风险证据要求（禁止仅凭业务经验判断）**：

当算子使用 Sqrt/Div/Reciprocal/Rsqrt/Exp/Ln 时，"输入是否可能包含 subnormal"的判断 MUST 基于以下证据链，不得仅凭"优化器输入通常不会出现 subnormal"等业务经验结论：

1. **dtype 分析**：列出算子支持的所有 dtype（FP16/BF16/FP32）及其 minimum normal、minimum subnormal 值
2. **输入范围分析**：从算子接口语义、典型使用场景推导每个输入的值域范围
3. **中间计算分析**：追踪计算链中每一步的中间结果值域（如 `a * b` 的值域、`a²` 的值域、`sqrt(a)` 的输入值域）
4. **常量分析**：检查算子中使用的所有标量常量（eps/epsilon/分母保护值/数值稳定常量），针对每个 dtype 检查：
   - 是否可表示（在 normal 范围内）
   - 是否为 subnormal
   - 是否低于 minimum subnormal（会下溢为 0）
   - 转换后是否变为 0
   - 是否可能导致除零/NaN
   （详见 `api-diff-guide.md` §1.6）
5. **计算链分析**：追踪从输入到 Sqrt/Div 输入的完整计算链，判断中间结果是否可能落入 subnormal 区间；**受影响操作数存在结构性下界（max 归一锚点/哨兵修复/数学推导）时优先给出证明（策略 0）**

**无法证明无风险时，按存在风险处理**——选择适配策略（eps 规避 / PRECISION_1ULP_FTZ_FALSE / high_precision 分支）。

**Subnormal 决策流程**（见 `api-diff-guide.md` §1.5）：

```
算子是否使用 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt？
├─ 否 → 无需处理 Subnormal
└─ 是 → 【策略 0】是否存在结构性 subnormal 排除？（max 归一锚点 / 哨兵修复 / 数学下界，MUST 证据链）
    ├─ 是 → 保持 INTRINSIC（高性能），分析摘要记录证明链
    └─ 否 → 输入数据是否可能包含 Subnormal 浮点数？
        ├─ 否 → 使用默认 INTRINSIC 模式（高性能，Subnormal→0）
        └─ 是 → 算子是否在性能关键路径？
            ├─ 是 → 优先 eps 规避；不可行则开 high_precision 分支
            └─ 否 → 直接使用 PRECISION_1ULP_FTZ_FALSE
```

### 1.3.2 兼容性风险扫描

按 `api-diff-guide.md` §2.3 清单扫描：

| 扫描项 | 检测方法 | 命中后的处理 |
|--------|---------|-------------|
| 是否使用 int4b_t 作为 Mmad 输入 | grep `int4b_t` + `Mmad` / `Matmul` | 需在 cast 成 int8 后再计算 |
| 是否使用 4:2 稀疏 | grep `Sparse` / `4:2` | A5 不支持 4:2 稀疏，需移除或替换实现 |

### 1.3.3 性能风险扫描

按 `api-diff-guide.md` §3 评估算子是否命中三类性能退化路径：

**扫描项 A：TransDataTo5HD / vnchwconv 退化（P0）**

| 扫描项 | 检测方法 | 风险等级 |
|--------|---------|---------|
| 算子是否调用 `TransDataTo5HD` | grep `TransDataTo5HD` | 按算子类别查 `api-diff-guide.md` §3.2.3 影响排序 |
| 是否纯转置/格式转换类算子 | 算子名含 Transpose/Pad/Unfold | P0 极高 |
| 是否在 tile 主循环中调用 | 分析 TransDataTo5HD 调用位置 | 主循环中风险更高 |
| 是否需要正反两次转置 | 分析 ND→NZ→ND 等 | 双向转换风险更高 |

**扫描项 B：高阶 ReduceSum / VSLDB 退化（P0）**

| 扫描项 | 检测方法 | 风险等级 |
|--------|---------|---------|
| 算子是否调用高阶 `ReduceSum` | grep `ReduceSum`（非 Block/Whole/Repeat/Pair/MicroAPI） | 按算子类别查 `api-diff-guide.md` §3.3.4 影响排序 |
| 是否使用 Pattern::Reduce::AR | grep `Reduce::AR` / 分析 tiling | AR 模式才命中 VSLDB |
| 是否可能命中 R=16/24 窄分支 | 分析 tiling 后归约轴宽度 | R=16/24 命中 VSLDB，吞吐减半 |

**扫描项 C：BilinearInterpolation 退化（P1）**

| 扫描项 | 检测方法 | 风险等级 |
|--------|---------|---------|
| 算子是否调用 `BilinearInterpolation` | grep `BilinearInterpolation` | 按算子类别查 `api-diff-guide.md` §3.4.5 影响排序 |
| 是否为 GridSample/ROIAlign/Deformable | 算子名匹配 | P0 理论风险 |

## Step 1.4：判定迁移层级

按 SKILL.md 决策树判定 L1/L2/L3。

### L2 特有判定信号

| 信号 | 说明 |
|------|------|
| 性能关键路径 | RMSNorm/RoPE/Softmax（vector 类）、Attention/Matmul（cube 类）等高频算子——**不区分编程模型类别**，cube 类性能关键算子同样触发 L2 |
| kernel 中有 `Cast<fp8/hifloat8, float>` | 950 新增类型必须 MicroAPI |
| kernel 中有 `ReduceSumCustom` | 950 应替换为 ReduceSum MicroAPI |
| kernel 中有 `DataCopyPad` 非对齐存储 | 950 应替换为 DataCopyUnAlign |
| FP32→INT8 量化 | 950 需三步量化 + Pack |

### cube 类算子的 L2 语义（定级 L2 后 MUST）

**L2 对 cube 算子 ≠ Vector 算子的 RegBase 整体重写**，而是三个独立判定 + 组合执行：

1. **AIC 侧计算路径**：MUST 评估低阶直跑（LoadData2DParamsV2 / Fixpipe / Mmad，见 `cube-migration-guide.md` 改动 2/3/4），并给出"保留高阶 Matmul API / 弃用走低阶"的结论与理由——**禁止默认沿用高阶 API 不做评估**；
2. **跨核同步协议**：按 `cube-migration-guide.md` 改动 6 全量重写（模式语义 / flagId / PIPE）；
3. **AIV 侧 vector 计算**：评估 RegBase 重写（`l2-guide.md`）。**评估前置方法——查 VF 封装实现归属**：接口头 → `base_impl.h` 的 `__NPU_ARCH__` 分流，三类结论：
   - 3510 有 regbase / 3510 专用实现（如 SoftmaxFlashV2 → `regbase/3510/softmax_flashv2_impl.h`、DropOut → `dropout_3510_impl.h`）→ **保留接口即可，自动获得 3510 实现，无需重写**——结论须写明"框架已提供 3510 实现"，禁止写成"保留 Memory 代码"；
   - 3510 仅有 membase 实现 → 评估迁移；
   - 3510 无实现 → 必须重写。

三部分缺一不可。"论证了保留高阶 API 不死锁"只完成 ②，不等于完成 L2。

### L3 特有判定信号

| 信号 | 说明 |
|------|------|
| Scatter/Gather 操作 | 按索引读写 GM |
| 索引逻辑简单 | 无需复杂计算 |
| 无需 UB 中转 | SIMT 直接访问 GM |

## Step 1.5：同步协议盘点

**MANDATORY**：Step 1.1 中发现任何跨核同步原语（CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll、NotifyNextBlock/WaitPreBlock）时，MUST 执行本步；未发现时跳到 Step 1.6。

跨核同步协议**按原语逐个核对**，不要整体判定。四步法：

**① 枚举同步点**：列出 kernel 中所有核间同步原语调用，每个同步点记录：

| 字段 | 内容 |
|------|------|
| 同步点位置 | 文件 + 行号 + 所在函数 |
| 原语 | CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll 等 |
| 模式号 | 如 `CrossCoreSetFlag<2, PIPE_MTE3>` 的模式 2 |
| flagId 表达式 | 如 `SYNC_C1_V1_FLAG[(taskId + 1) & 1][blockIdx & 1]`——**含核身份索引（taskId/blockIdx/subblock 号参与索引）时重点标记** |
| 调用方角色 | AIC / AIV / 核内 subblock 序号（`ASCEND_IS_AIC`/`ASCEND_IS_AIV` 分支）。无核类型分支的"全执行"结构（所有核无条件执行 bmm/vector/写输出）在 A2/A3 常见，950 分离模式下矩阵/矢量路径的执行核不同（见 `cube-migration-guide.md`「分离模式核能力模型」），需在迁移方案中说明各代码段的核归属；保留高阶 Matmul API 时必查双主模式约束（`cube-migration-guide.md`「保留高阶 Matmul API 的必查项」） |

**② 推断参与集合假设**：写出每个同步点"谁等谁"的粒度——单对（AIC↔指定 AIV）/ 组内全集合（AIC↔全部 AIV）/ 全核（所有 AIC 或所有 AIV）。**flagId 含核身份索引的，即隐式假设了"flagId 级配对粒度"**（用不同 flagId 区分不同对端）。

**③ 对照目标平台语义**：读 `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md` 与 `CrossCoreSetFlag_ISASI.md`，核对：
- 所用模式在目标平台的参与集合定义与 flagId 计数器驱动条件（模式 0 = 全核、模式 1 = 组内全部 AIV、模式 2 = AIC↔组内全部 AIV、模式 4 = AIC↔单个 AIV）
- 型号支持范围（**模式 4 仅 950PR/950DT 支持**）
- **模式号相同 ≠ 语义相同**：同一模式跨架构可能语义变化（如模式 2 从 flagId 级配对收紧为全集合），必须以目标平台文档为准，不得假设与旧平台一致
- **参数写法按目标平台文档生效**：旧平台被忽略的参数在目标平台可能生效——无参调用（如裸 `CrossCoreWaitFlag(flagId)`）在 950 按默认模式 0（全核）执行，必须与配对 SetFlag 的模式号显式核对；"不写参数，语义也在变"（实战：910b 无参 WaitFlag 照常工作、950 上生效致模式错位死锁，见 `references/impl/cube-migration-guide.md` 改动 6 同步协议沿用决策规则）
- **参与集合恒定**：同一 flagId 的 Set/Wait 是否在所有参与核的**无条件路径**上执行——被 needExec/needPair/if 条件分支包裹、循环 0 次（分片边界）、空闲核都可能造成参与核缺 Set（同上，沿用决策规则第 4 条）

**④ 处置决策**：每个同步点给出结论——

| 结论 | 条件 |
|------|------|
| 沿用 | **4 项全部静态论证通过**：① 集合粒度 ≥ 目标平台文档定义粒度（含等号）② 模式在目标平台支持 ③ 参数写法按目标平台文档生效 ④ 参与集合恒定（Set/Wait 在所有参与核的无条件路径上） |
| 禁用（走标准路径） | 4 项中任一项无法静态论证（如粒度 < 文档粒度、无参调用默认语义错位、Set 可能缺位）→ 死锁风险 |
| 改用其他模式 | 需支持单对粒度时改用模式 4——但须核对型号支持范围（仅 950PR/DT） |

**通用判定规则：假设集合粒度 < 目标平台文档定义的集合粒度 → 死锁风险 → 禁用该路径走标准流程，禁止原样移植。默认值反转：沿用是论证结论，不是赌注——4 项中任一项无法静态论证即禁用/降级走标准路径，禁止"沿用 + 运行时验证"式风险转移（死锁失败从方案阶段 1 次静态检查前移到运行时排查=成本后置；完整前置判定框架（5 项静态论证 + 参与集合推导表）见 `references/impl/cube-migration-guide.md` 改动 6 同步协议沿用决策规则）。**

## Step 1.6：生成算子源码分析摘要

**MANDATORY**：无论算子有无文档，都 MUST 基于被迁移源码生成摘要，并在对话中展示完整内容，同时写入文件 `${OP_DIR}/analysis_<op_name>.md` 持久化保存。

摘要内容：

```markdown
# 算子源码分析摘要：算子名

## 1. 算子功能描述
- 计算逻辑（从 kernel 代码逆向提取）
- 算子类别：elementwise / reduction / attention / matmul / 其他

## 2. 接口签名
- 输入参数（名称、类型、shape 约束）
- 输出参数（名称、类型、shape 计算规则）
- 属性参数

## 3. 支持的数据类型
- 从 _def.cpp 和 kernel 代码提取
- 列出：FP16 / BF16 / FP32 / INT8 / FP8 / HiFloat8 等

## 4. 计算逻辑伪代码
- 从 Compute 函数提取 API 调用序列
- 标注升精度路径、条件编译分支

## 5. 关键 API 使用
- 列出所有 CANN API 调用（DataCopy/Cast/Reduce/Sort 等）

## 6. 边界条件与特殊值
- 输入域约束、NaN/Inf 处理

## 7. ★ API 差异风险扫描结果
### 7.1 精度（Subnormal）
- 是否使用 Exp/Ln/Sqrt/Rsqrt/Div/Reciprocal：是/否
- ★ 是否存在结构性 subnormal 排除（策略 0）：是/否（是时 MUST 附证明链：max 归一锚点/哨兵修复路径+代码行号/数学下界推导）
- ★ 输入是否可能包含 subnormal：是/否（MUST 附证据链：dtype 分析 + 输入范围 + 中间计算 + 常量分析 + 计算链分析）
- ★ eps/常量 dtype 可表示性检查：列出所有 eps/常量值，针对每个 dtype 标注是否可表示/normal/subnormal/下溢为 0/可能导致除零
- 选定适配策略：结构性排除（策略 0）/ eps 规避 / API 模板参数 / high_precision 分支 / 无需处理
### 7.2 兼容性
- 是否使用 int4b_t 作为 Mmad 输入：是/否
- 是否使用 4:2 稀疏：是/否
### 7.3 性能
- 是否调用 TransDataTo5HD：是/否（风险等级 P0-P3）
- 是否调用高阶 ReduceSum（AR 模式）：是/否（是否命中 R=16/24）
- 是否调用 BilinearInterpolation：是/否
- 高风险 shape 清单：（列出需回归的 shape）

## 8. 同步协议盘点结果（有跨核同步原语时）
- 同步点清单：位置 / 原语 / 模式号 / flagId 表达式 / 调用方角色
- 参与集合假设与目标平台语义对照结论
- 处置决策：沿用 M 个 / 禁用 K 个 / 改用 R 个（按 Step 1.5 ④）
- 输出格式：`[SYNC-AUDIT] 同步点 N 个：沿用 M 个 / 禁用 K 个 / 改用 R 个`
```

### 逆向分析方法

> 完整五步逆向方法论（接口签名提取 → 计算逻辑逆向 → 输入域约束 → 边界条件 → PyTorch 对标）见 `references/precision-testing/source-code-reverse-analysis.md`。

| 方法 | 适用场景 |
|------|---------|
| 代码逆向分析 | 所有算子——从 Compute 函数逐行提取 |
| 同类算子类比 | 算子名可推断功能时（如 `fast_gelu` 类比 `gelu`） |
| 测试用例反推 | 有已有 UT/ST 时 |
| PyTorch 对标 | 有 PyTorch 同名接口时 |

**有文档算子**：先从源码提取，再与文档交叉对比，记录差异。测试用例始终以源码为准。

## Step 1.7：输出迁移方案

基于 Step 1.4 判定的层级、Step 1.3 的 API 差异扫描结果、Step 1.5 的同步协议盘点和 Step 1.6 的源码分析摘要，生成迁移方案并在对话中展示：

| 项目 | 内容 |
|------|------|
| 迁移层级 | L1/L2/L3 |
| 涉及文件 | 列出所有需要新增/修改的文件 |
| 改造范围 | 简述需要改什么（如"移除 V220 guards"、"Compute 函数改为 RegBase"） |
| 同步协议处置 | Step 1.5 各同步点的沿用/禁用/改用结论（有同步点时必填） |
| 预期风险 | 如 UB 预算、Subnormal 处理、量化路径、同步原语集合粒度错配（卡死/超时）等 |
| ★ API 差异适配项 | 精度/兼容性/性能三个维度的适配清单 |
| ★ 性能回归 shape | 性能退化高风险 shape 清单（供阶段 2 优化适配使用） |

迁移方案在对话中展示后即可进入阶段 2（展示供用户审阅留痕，无需等待确认）。

## Gate 输出条件

- [ ] 算子源码已全部读取
- [ ] **★ `api-diff-guide` 已加载并输出 LOADED Token**
- [ ] **★ API 差异三维度风险扫描已完成**
- [ ] 迁移层级已判定并标注理由
- [ ] 同步协议盘点已完成（有跨核同步原语时：同步点已枚举、参与集合假设已推断、目标平台语义已对照、处置决策已输出）
- [ ] 算子源码分析摘要（含 API 差异扫描结果与同步协议盘点结果）已在对话中展示并写入文件
- [ ] 迁移方案（含 API 差异适配项与同步协议处置）已在对话中展示
- [ ] 明确了算子支持的 dtype 和 shape 范围
- [ ] **★ 已列出性能回归高风险 shape 清单**

**全部通过 → 输出 `[GATE-1] LEVEL=L1/L2/L3` → 进入阶段 2**
