# 路线 2201→3510：A2/A3（DAV_2201）→ A5（DAV_3510）

本文件是路线包 `routes/2201-to-3510/` 的入口，承载该架构对的全部路线专属内容，由 `../../SKILL.md`「Step R：迁移路线判定」路由进入。
**回退校验**：若实际源/目标平台不是 DAV_2201→DAV_3510，立即停止并退回 `../../SKILL.md` Step R 重新判定，禁止沿用本文件的差异清单与决策树。
进入本路线后，本文件的约束与 `../../SKILL.md` 的全局约束、Gate 协议同等强制；阶段执行细则在本包 `stages/`、路线专属知识在本包 `references/`、路线无关公共资产在 `../../references/`。本包自包含，禁止引用其他路线包的内容。

## 阶段执行流程

**MUST 按顺序执行。每个阶段开始时先读取本包对应 stage 文件，完成时输出 Gate Token（判据见 `../../SKILL.md` Gate 协议）。**

| 阶段 | 文件（本包内） | Gate Token |
|------|------|------------|
| 0 环境验证 | `stages/stage_0_env.md` | `[GATE-0] ENV_CONFIRMED` |
| 1 评估定级 | `stages/stage_1_assess.md` | `[GATE-1] LEVEL=L1/L2/L3`（层级判定规则见本文决策树） |
| 2 代码改造 | `stages/stage_2_implement.md` | `[GATE-2] FILES_MODIFIED=[...]` |
| 3 编译安装 | `stages/stage_3_build.md` | `[GATE-3] BUILD=PASS INSTALL=PASS` |
| 4 精度验证 | `stages/stage_4_precision.md` | `[GATE-4] PRECISION=N/N_PASS`（仅当 N_FAIL==0 时允许输出 PASS；否则输出 `PRECISION=N/M_PASS BLOCKED`） |
| 5 性能验证 | `stages/stage_5_performance.md` | `[GATE-5] PERF=COLLECTED`（仅当 10 项证据全部存在时允许输出；否则输出 `PERF=NOT_COLLECTED BLOCKED`） |

## 本路线覆盖的两大迁移能力

**① A5 API 差异迁移适配**（所有层级 MUST 执行），覆盖三个维度：
- **精度差异**：A5 裁剪 subnormal，基础算数 API（Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt）在 subnormal 场景精度丢失
- **兼容性差异**：Mmad 不支持 int4 类型，量化 matmul 需 cast 成 int8
- **性能差异**：vnchwconv/VSLDB 吞吐下降、BilinearInterpolation 实现路径变化

详见 `references/api-diff-guide.md`。

**② Cube 类算子迁移**（kernel 含 Mmad/LoadData/Fixpipe 等 cube API 时）：Cube 数据通路（L1/L0A/L0B/L0C）、分形变化（ZZ→NZ）、跨核同步协议等与 Vector 正交的硬件差异，详见 `references/cube-migration-guide.md`（配套排查见 `references/cube-debug-lessons.md`）。

## 迁移层级决策树（含 API 差异适配）

```
算子是否满足以下任一条件？
  ├─ 性能关键路径 (RMSNorm/RoPE/Softmax 等 vector 类；Attention/Matmul 等 cube 类计算型算子)
  ├─ 量化 Cast 链路复杂 (FP32→FP8/HiFloat8/INT8)
  ├─ 需要溢出模式控制
  └─ 950 新增数据类型 (FP8/HiFloat8)
  │
  ├─ 是 → L2: RegBase API 重写（cube 类算子的 L2 = AIC 低阶直跑评估 + 同步协议重写 + AIV RegBase 评估三部分组合，语义见 `stages/stage_1_assess.md` Step 1.4）
  │
  └─ 否，是否满足以下全部？
      ├─ Scatter/Gather 操作
      ├─ 索引逻辑简单
      ├─ 无需 UB 中转
      └─ 线程并行度高
      │
      ├─ 是 → L3: SIMT 优化
      └─ 否 → L1: 基础适配

  ★ 无论选择哪个层级，都 MUST 额外执行 API 差异适配（见下方）
```

### API 差异适配（所有层级 MUST 执行）

无论算子判定为 L1/L2/L3，都 MUST 在阶段 1 评估时做 API 差异风险扫描，在阶段 2 改造时做适配处理：

| 差异维度 | 扫描时机 | 适配时机 | 参考文件 |
|---------|---------|---------|---------|
| 精度（Subnormal） | 阶段 1 | 阶段 2 | `references/api-diff-guide.md` §1 |
| 兼容性（int4/Mmad） | 阶段 1 | 阶段 2 | `references/api-diff-guide.md` §2 |
| 性能（vnchwconv/VSLDB/BilinearInterpolation） | 阶段 1 | 阶段 2（优化） | `references/api-diff-guide.md` §3 |

## 路线专属 Reference 索引

### API 差异适配（★ 本路线所有层级 MUST READ）

| 文件 | 用途 | LOADED Token |
|------|------|-------------|
| `references/api-diff-guide.md` | A5 API 差异（精度/兼容性/性能）适配指南 | `[LOADED] api-diff-guide` |

### 实现指南（按层级 × 算子类别加载）

| 层级 | 类别 | MUST READ | 补充参考 |
|------|------|-----------|---------|
| L1 | Vector | `references/l1-guide.md` | — |
| L1 | Cube | `references/l1-guide.md`（host 侧步骤通用）+ `references/cube-migration-guide.md` | — |
| L2 | Vector | `references/l2-guide.md`（含 Reg 级搬移 VF 封装约束与编译错误速查） | `references/api-mapping.md`；Reg API 速览见 `references/l2-guide.md`「API 速览与知识真源」 |
| L2 | Cube | `references/cube-migration-guide.md`（AIC 侧：低阶直跑/同步协议）+ `references/l2-guide.md`（AIV 侧 Vector 路径，适用边界见其头部声明） | `references/api-mapping.md`（AIV 侧）；Reg API 速览见 `references/l2-guide.md`「API 速览与知识真源」 |
| L3 | 通用 | `references/l3-guide.md` | — |

**Cube 类算子**（kernel 含 Mmad/LoadData/LoadDataWithTranspose/Fixpipe/DataCopyCO12DstParams/CrossCoreSetFlag 等 cube API）：迁移时 MUST 同时阅读 `references/cube-migration-guide.md`——Cube 数据通路（L1/L0A/L0B/L0C）、分形变化（ZZ→NZ）、跨核同步机制与 Vector 差异正交，独立成章；**该 guide 不替代对应层级指南，仅覆盖 AIC 侧差异，AIV 侧 Vector 路径仍用层级指南的既有经验**（各环节配套文件见 cube-guide「配套经验引用表」）。

### 实战经验（迁移沉淀，按需加载）

| 文件 | 内容 | LOADED Token（在 `stages/stage_2_implement.md` 声明） |
|------|------|------|
| `references/cube-debug-lessons.md` | 测试与 debug 排查：死锁/卡死/结果错误按「错误信号 → 排查手段 → 解决方法」排查（无同步路径对照法、mask 约定与消费粒度、取证打印、路径级验证）（阶段 4 测试失败/精度异常时 MUST 查阅） | `[LOADED] cube-debug-lessons` |
| `references/multi-stage-guide.md` | 多阶段（2+ 算法阶段共享 UB/workspace）流水模式与 UB 峰值计算（阶段 2 按需） | `[LOADED] multi-stage-guide` |
| `references/ub-budget-guide.md` | UB 预算估算与溢出排查（**UB 用量与 SIMT DCache 预留的唯一权威表**；阶段 2 设计 tile 时、阶段 4/5 遇 UB out-of-range 时查阅） | `[LOADED] ub-budget-guide` |

## 路线专属约束（与 SKILL.md 全局约束同等强制）

1. **★ 精度测试 MUST 包含 subnormal 场景用例**（当算子使用 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt 时）
2. **★ 性能测试 MUST 包含 API 差异高风险 shape 回归用例**（按 `references/api-diff-guide.md` §3 触发 shape 设计：R=16/24、小 tile、非连续 stride）
3. **★ eps / epsilon / 常量 MUST 进行 dtype 可表示性检查**——源码中存在 `+eps` 不能直接证明 eps 保护有效。MUST 对所有支持 dtype 检查 eps 值是否可表示、是否为 normal/subnormal/低于 subnormal、转换后是否为 0、是否可能导致除零/NaN（见 `references/api-diff-guide.md` §1.6）
4. **★ Subnormal 风险分析 MUST 基于证据**——不得仅凭业务经验判断"不会出现 subnormal"。MUST 基于 dtype、输入范围、中间计算、常量、eps 值、minimum normal/subnormal、计算链给出证据；**受影响操作数存在结构性下界（max 归一锚点/invalid line 哨兵修复/数学推导）时优先按策略 0 给出证明（保持 INTRINSIC，见 `references/api-diff-guide.md` §1.4）**。无法证明无风险时按存在风险处理（见 `stages/stage_1_assess.md` Step 1.3.1）
5. **★ FP16 精度失败 MUST 进入五阶段调试流程**——FP16 FAIL + FP32 PASS 时，禁止直接标注"FP16 已知限制"。MUST 执行 Phase 1→5 调试，给出根因和归类（见 `stages/stage_4_precision.md` Step 4.6.2）
6. **kernel 含跨核同步原语（CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll、NotifyNextBlock/WaitPreBlock）时，MUST 按 `stages/stage_1_assess.md` Step 1.5 逐同步点盘点参与集合粒度并对照目标平台文档**（**模式号相同 ≠ 语义相同**，以目标平台为准）

## 路线专属「绝对不要做」（每条含替代方案）

| 禁止 | 应该做 |
|------|--------|
| **★ 忽略 Subnormal 风险** | **扫描 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt 用法，按 `references/api-diff-guide.md` §1 适配** |
| **★ 忽略 int4 Mmad 兼容性** | **扫描 int4b_t + Mmad 用法，按 `references/api-diff-guide.md` §2 cast 成 int8** |
| **★ 性能回归用平均随机 shape** | **按 `references/api-diff-guide.md` §3 触发 shape 设计（R=16/24、小 tile、非连续 stride）** |
| **★ 仅凭业务经验判断"不会出现 subnormal"** | **基于 dtype/输入范围/中间计算/eps 值/计算链给出证据，无法证明无风险时按存在风险处理** |
| **★ eps 值不检查 dtype 可表示性就声称"已有 eps 保护"** | **对所有支持 dtype 检查 eps 是否可表示/normal/subnormal/下溢为 0/导致除零（`references/api-diff-guide.md` §1.6）** |
| **★ FP16 FAIL + FP32 PASS 时跳过五阶段调试** | **MUST 执行 Phase 1→5，给出根因、是否 A5 特有、是否修复及理由（见 `stages/stage_4_precision.md` Step 4.6.2）** |
| 原样保留依赖旧架构同步原语语义的跨核同步协议（flagId 配对、跨核 Set/Wait） | 按 `stages/stage_1_assess.md` Step 1.5 逐个同步点盘点参与集合粒度，对照目标平台模式语义与型号支持范围；无法论证成立则禁用该路径走标准流程 |
| 断言新旧平台核映射/核比例差异（如"910b 是 1:1、950 是 1:2"） | MIX 比例是算子配置项（`__mix__(1,N)` / `KERNEL_TYPE_MIX_AIC_1_2`），非平台属性；差异以官方迁移指导（`2201_to_3510_arch_changes.md`）与 API 文档为准 |
