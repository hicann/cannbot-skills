---
name: ascendc-cross-gen-port-light
description: AscendC 算子轻量迁移 skill（ascendc-cross-gen-port 的无 golden 轻量入口）：把已有 DAV_2201（arch22）平台的 AscendC 算子工程按 Stage 0-4 阶段门禁改造迁移到 DAV_3510（arch35）平台，精度标杆由 agent 逆向源码自合成并与 A5 实测双向互检，不经编排引擎。当用户无 KernelBench golden 输入、或希望基于已有 910b/910_93 算子修改后快速迁移到 950/arch35 时使用；需引擎驱动的端到端自动移植（自动构建/精度/性能闭环与报告）请改用 ascendc-cross-gen-port。
version: 3.3
date: 2026-08-18
---

# AscendC 算子跨架构迁移（DAV_2201 → DAV_3510，含 API 差异适配）

本 Skill 是**路由器**，不包含实现细节。所有细节在 `stages/` 和 `references/` 中。

本 skill 覆盖的核心迁移能力：

**① A5 API 差异迁移适配**（所有层级 MUST 执行），覆盖三个维度：
- **精度差异**：A5 裁剪 subnormal，基础算数 API（Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt）在 subnormal 场景精度丢失
- **兼容性差异**：Mmad 不支持 int4 类型，量化 matmul 需 cast 成 int8
- **性能差异**：vnchwconv/VSLDB 吞吐下降、BilinearInterpolation 实现路径变化

详见 `references/impl/api-diff-guide.md`。

## 外部依赖：asc-devkit 全量仓

本 skill 依赖 asc-devkit 全量仓提供 API 文档、样例代码和头文件。

| 属性 | 值 |
|------|-----|
| **仓库地址** | https://gitcode.com/cann/asc-devkit |
| **用途** | API 文档（4700+）、算子样例（1800+）、头文件（600+）、实现源码（4400+） |
| **路径变量** | `$DEVKIT_PATH`（在 Stage 0 中由用户指定或 git clone 后记录） |

### 依赖初始化

在 Stage 0 环境检查阶段，agent 必须确认全量仓已 clone 并记录路径。
详见 `stages/stage_0_env.md` 中的「全量仓初始化」步骤。

### 文档查找协议

1. **必读索引**：每次需要查找文档前，先参考 `references/knowledge-index.md`
2. **分级查找**：按 `references/search-rules.md` 定义的优先级查找
3. **目录映射**：不确定去哪个目录时，查 `references/devkit-path-map.md`
4. **置信度评估**：每次查找后评估置信度（HIGH/MEDIUM/LOW），LOW 时扩大搜索或上报用户
5. **LOADED 标记**：从全量仓读取文件后，输出 `[LOADED] $DEVKIT_PATH/<相对路径>`

### 全量仓提供的核心资源

| 资源类型 | 路径 | 说明 |
|---------|------|------|
| API 文档 | `$DEVKIT_PATH/docs/api/` | SIMD/SIMT/高阶 API 完整文档 |
| 编程指南 | `$DEVKIT_PATH/docs/guide/编程指南/` | 编程模型、语言扩展、硬件实现 |
| 迁移指南 | `$DEVKIT_PATH/docs/guide/跨代迁移兼容性指南/` | 2201→3510 迁移详述 |
| 算子实践 | `$DEVKIT_PATH/docs/guide/算子实践参考/` | 90+ 实现与优化指南 |
| 算子样例 | `$DEVKIT_PATH/examples/` | SIMD/SIMT/AICPU 样例代码 |
| 头文件 | `$DEVKIT_PATH/include/` | API 声明 |
| 实现源码 | `$DEVKIT_PATH/impl/` | API 实现 |

## 迁移层级决策树（含 API 差异适配）

```
算子是否满足以下任一条件？
  ├─ 性能关键路径 (RMSNorm/RoPE/Softmax 等 vector 类；Attention/Matmul 等 cube 类计算型算子)
  ├─ 量化 Cast 链路复杂 (FP32→FP8/HiFloat8/INT8)
  ├─ 需要溢出模式控制
  └─ 950 新增数据类型 (FP8/HiFloat8)
  │
  ├─ 是 → L2: RegBase API 重写（cube 类算子的 L2 = AIC 低阶直跑评估 + 同步协议重写 + AIV RegBase 评估三部分组合，语义见 stage_1 Step 1.4）
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
| 精度（Subnormal） | 阶段 1 | 阶段 2 | `references/impl/api-diff-guide.md` §1 |
| 兼容性（int4/Mmad） | 阶段 1 | 阶段 2 | `references/impl/api-diff-guide.md` §2 |
| 性能（vnchwconv/VSLDB/BilinearInterpolation） | 阶段 1 | 阶段 2（优化） | `references/impl/api-diff-guide.md` §3 |

## 阶段执行流程

**MUST 按顺序执行。每个阶段开始时先读取对应 stage 文件（见上方表格），完成时输出 Gate Token。**

| 阶段 | 文件 | Gate Token |
|------|------|------------|
| 0 环境验证 | `stages/stage_0_env.md` | `[GATE-0] ENV_CONFIRMED` |
| 1 评估定级 | `stages/stage_1_assess.md` | `[GATE-1] LEVEL=L1/L2/L3` |
| 2 代码改造 | `stages/stage_2_implement.md` | `[GATE-2] FILES_MODIFIED=[...]` |
| 3 编译安装 | `stages/stage_3_build.md` | `[GATE-3] BUILD=PASS INSTALL=PASS` |
| 4 精度验证 | `stages/stage_4_precision.md` | `[GATE-4] PRECISION=N/N_PASS`（仅当 N_FAIL==0 时允许输出 PASS；否则输出 `PRECISION=N/M_PASS BLOCKED`） |

## Reference 文件索引

### API 差异适配（★ 本 skill 新增，所有层级 MUST READ）

| 文件 | 用途 | LOADED Token |
|------|------|-------------|
| `references/impl/api-diff-guide.md` | A5 API 差异（精度/兼容性/性能）适配指南 | `[LOADED] api-diff-guide` |

### 实现指南（按层级 × 算子类别加载）

| 层级 | 类别 | MUST READ | 补充参考 |
|------|------|-----------|---------|
| L1 | Vector | `references/impl/l1-guide.md` | — |

### 精度测试（阶段 4 MUST READ）

| 文件 | 用途 |
|------|------|
| `references/precision-testing/pytorch-binding-build-guide.md` | PyTorch 绑定构建（唯一标准方式） |
| `references/precision-testing/torch_aclnn_helper.h.template` | EXEC_NPU_CMD 桥接头文件模板 |
| `references/precision-testing/OPS_PRECISION_STANDARDS.md` | 精度标准（混合容差 rtol/atol + 双门限，真源 ops-precision-standard） |
| `references/precision-testing/test_op_precision_aclnn_template.py.template` | pytest 测试模板 |

**L2 改造时 MUST 先读取 RegBase 最佳实践的 `references/regbase_development_guide.md`，再开始代码改写。禁止凭记忆写 RegBase 代码。**

## Gate 协议

1. **先读 stage 文件，再执行工作**——每个阶段开始时 MUST 读取对应的 stage 文件
2. **LOADED Token**——stage 文件中要求加载的 reference 文件，加载后 MUST 输出 `[LOADED] <文件名>`
3. **GATE Token**——阶段完成时 MUST 输出对应 `[GATE-X]` token，包含实际结果值
4. **禁止跳阶段**——没有上一阶段的 GATE Token，禁止执行下一阶段
5. **禁止跳过 LOADED**——没有所有要求的 LOADED Token，禁止继续当前阶段
6. **★ API 差异 LOADED**——阶段 1 MUST 加载 `api-diff-guide`，输出 `[LOADED] api-diff-guide`
7. **★ Evidence-based Gate（全局约束）**——任何 Gate 不得仅依据以下因素判定 PASS：
   - Todo completed / checkbox checked
   - Agent 自我声明（"阶段已完成"、"预计没问题"）
   - 静态代码分析（不实际执行编译/测试）
   - "已知限制"标注（不经过根因分析就标注为限制）

   每个 Gate MUST 有 Skill 明确要求的实际证据（编译产物、测试输出、CSV 数据、报告文件等）。证据不足时 MUST 保持 `NOT VERIFIED` / `NOT PASSED` / `BLOCKED` 状态，不得输出 PASS。

8. **★ GATE-4 严格通过条件**——`[GATE-4] PRECISION=N/N_PASS` 仅当满足以下全部条件时允许输出：
   - 所有计划用例均已执行（无 SKIPPED / NOT RUN）
   - N_PASS == N_TESTS（N_FAIL == 0）
   - 未通过修改 threshold / 跳过 case / 修改输入数据等方式规避失败
   - 若存在 FAIL：MUST 输出 `[GATE-4] PRECISION=N/M_PASS BLOCKED`（M < N），并进入根因分析流程（见 `stage_4_precision.md` Step 4.6）

## 全局约束

1. 未完成阶段 0 环境验证前，禁止执行任何编译或测试操作
2. 所有 Shell 命令 MUST 使用阶段 0 记录的变量（`${CANN_SET_ENV}`、`${PYTHON_PATH}`）
3. 禁止自行探测 CANN 路径——检测不到时 MUST 向用户询问
4. 精度测试 MUST 通过 `torch.ops.npu.算子名(...)` 调用（EXEC_NPU_CMD 标准方式）
5. 禁止使用 PyTorch 原生同名接口做精度验证
6. 禁止手动写 aclnn C API 调用（`aclnnXxxGetWorkspaceSize` + `aclnnXxx`）
7. 编译通过不算完成——MUST 通过精度验证
8. 禁止自行拼编译命令——MUST 使用 `build_and_install.sh` 脚本
9. 精度测试用例数 MUST ≥ 30
10. **★ 精度测试 MUST 包含 subnormal 场景用例**（当算子使用 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt 时）
11. **★ 性能测试 MUST 包含 API 差异高风险 shape 回归用例**
12. **★ 精度测试中任何 NaN/Inf 输出 MUST 触发根因分析**——在根因确定且归类完成前，禁止输出 GATE-4 PASS。禁止直接用"已知限制"跳过分析（见 `stage_4_precision.md` Step 4.6.1）
13. **★ eps / epsilon / 常量 MUST 进行 dtype 可表示性检查**——源码中存在 `+eps` 不能直接证明 eps 保护有效。MUST 对所有支持 dtype 检查 eps 值是否可表示、是否为 normal/subnormal/低于 subnormal、转换后是否为 0、是否可能导致除零/NaN（见 `api-diff-guide.md` §1.6）
14. **★ Subnormal 风险分析 MUST 基于证据**——不得仅凭业务经验判断"不会出现 subnormal"。MUST 基于 dtype、输入范围、中间计算、常量、eps 值、minimum normal/subnormal、计算链给出证据；**受影响操作数存在结构性下界（max 归一锚点/invalid line 哨兵修复/数学推导）时优先按策略 0 给出证明（保持 INTRINSIC，见 `api-diff-guide.md` §1.4）**。无法证明无风险时按存在风险处理（见 `stage_1_assess.md` Step 1.3.1）
15. **★ FP16 精度失败 MUST 进入五阶段调试流程**——FP16 FAIL + FP32 PASS 时，禁止直接标注"FP16 已知限制"。MUST 执行 Phase 1→5 调试，给出根因和归类（见 `stage_4_precision.md` Step 4.6.2）
16. **★ 所有 Gate 均为 evidence-based**——禁止仅依据 Todo completed / Agent 自我声明 / 静态分析 / "预计没问题"判定 PASS。证据不足时保持 NOT VERIFIED / NOT PASSED / BLOCKED（见 Gate 协议第 7 条）
17. **kernel 含跨核同步原语（CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll、NotifyNextBlock/WaitPreBlock）时，MUST 按 stage_1 Step 1.5 逐同步点盘点参与集合粒度并对照目标平台文档**（**模式号相同 ≠ 语义相同**，以目标平台为准）
18. **禁止臆造新旧平台架构差异**——MIX 比例（1:1/1:2）、核映射等以官方迁移指导与 API 文档为准，不得凭推断断言

## 绝对不要做（每条含替代方案）

| 禁止 | 应该做 |
|------|--------|
| 跳过阶段 0 环境验证 | 执行 `stages/stage_0_env.md` 完整流程，输出 GATE-0 |
| 自行探测 CANN 路径（`find`/`locate`/扫描 `/home`） | 检测不到时向用户询问，不得猜测 |
| 在 Shell 命令中写死路径 | 使用阶段 0 记录的 `${CANN_SET_ENV}`、`${PYTHON_PATH}` 变量 |
| 用 `torch.gather`/`torch.acosh` 做精度测试 | 用 `torch.ops.npu.算子名(...)`（EXEC_NPU_CMD 绑定） |
| 用 `import ascend_kernel` 或 `torch.ops.aclnn.*` | 用 `torch.ops.npu.算子名(...)`（注册到 npu 命名空间） |
| 手写 `aclnnXxxGetWorkspaceSize` + `aclnnXxx` | 用 `EXEC_NPU_CMD(aclnnXxx, ...)` 宏（参考 `torch_aclnn_helper.h`） |
| 同一算子调用两次 `add_modules_sources` | 合并为一次调用，所有参数放在同一条中 |
| 用 `""` 作为 `TILING_DIR` 元素（被 CMake 吞掉） | 用 `"default"` 代替空字符串 |
| 用 `-soc=` 单横线 | 用 `--soc=ascend950` 双横线 |
| L2 中 FP32→INT8 一步 Cast | 三步：FP32→INT16→FP16→INT8（见 `api-mapping.md`） |
| L2 中 `SetCtrlSpr` 后不恢复 | 先 `GetCtrlSpr` 保存，计算完 `SetCtrlSpr` 恢复 |
| 绑定用 `.asc` 文件或 ASC 编译器 | 用 `.cpp` + 纯 C++ 编译（`LANGUAGES CXX`） |
| 用 `TORCH_LIBRARY(custom_ops, m)` 注册 | 用 `TORCH_LIBRARY_FRAGMENT(npu, m)` + `TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)` |
| 假设 aclnn 接口名 = 底层 L0 算子名 | 读 aclnn L2 源码（`op_host/op_api/aclnn_*.cpp`）确认 950 调度路径 |
| **★ 忽略 Subnormal 风险** | **扫描 Exp/Ln/Sqrt/Div/Reciprocal/Rsqrt 用法，按 `api-diff-guide.md` §1 适配** |
| **★ 忽略 int4 Mmad 兼容性** | **扫描 int4b_t + Mmad 用法，按 `api-diff-guide.md` §2 cast 成 int8** |
| **★ 性能回归用平均随机 shape** | **按 `api-diff-guide.md` §3 触发 shape 设计（R=16/24、小 tile、非连续 stride）** |
| **★ 仅凭业务经验判断"不会出现 subnormal"** | **基于 dtype/输入范围/中间计算/eps 值/计算链给出证据，无法证明无风险时按存在风险处理** |
| **★ eps 值不检查 dtype 可表示性就声称"已有 eps 保护"** | **对所有支持 dtype 检查 eps 是否可表示/normal/subnormal/下溢为 0/导致除零（`api-diff-guide.md` §1.6）** |
| **★ 精度测试出现 NaN/Inf 后直接标注"已知限制"** | **MUST 执行根因分析（除零/溢出/下溢/eps/dtype 转换），归类后才可标注** |
| **★ FP16 FAIL + FP32 PASS 时跳过五阶段调试** | **MUST 执行 Phase 1→5，给出根因、是否 A5 特有、是否修复及理由** |
| **★ Gate 仅依据 Todo completed / 自我声明判定 PASS** | **MUST 有实际证据（编译产物/测试输出/CSV/报告文件），证据不足时保持 BLOCKED** |
| 原样保留依赖旧架构同步原语语义的跨核同步协议（flagId 配对、跨核 Set/Wait） | 按 stage_1 Step 1.5 逐个同步点盘点参与集合粒度，对照目标平台模式语义与型号支持范围；无法论证成立则禁用该路径走标准流程 |
| 断言新旧平台核映射/核比例差异（如"910b 是 1:1、950 是 1:2"） | MIX 比例是算子配置项（`__mix__(1,N)` / `KERNEL_TYPE_MIX_AIC_1_2`），非平台属性；差异以官方迁移指导（`2201到3510架构变更.md`）与 API 文档为准 |
