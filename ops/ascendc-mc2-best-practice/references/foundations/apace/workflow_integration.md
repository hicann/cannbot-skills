# CANNBot Step 1→7 在 apace MC2 场景下的工作流映射

本文档把父级 `plugins-official/ops-direct-invoke/AGENTS.md` 的 7 步流程具体化到 apace 通算融合算子场景。CANNBot 主控和 Architect/Developer/Reviewer 三类 Subagent 在每个 Step 进入前都应先读对应小节，明确"本阶段在 apace 场景下要做什么、门禁是什么"。

> 父级流程定义以 `plugins-official/ops-direct-invoke/AGENTS.md` 为准；本文件只补充 apace 场景的差异化要点，不重写父级规则。
>
> **与 skill 四步模型的对应**：plugin Step 1 ≈ skill Step 1（[`workflow/step1-project-setup.md`](workflow/step1-project-setup.md)）；plugin Step 2/2.5 ≈ skill Step 2+3（[`workflow/step3-design.md`](workflow/step3-design.md)）；plugin Step 3-7 ≈ skill Step 4（[`workflow/step4-implementation.md`](workflow/step4-implementation.md)）。**plugin 场景下以本文档为主要消费文档**，四步文档为模型参考；路线模型（`apace_native`/`apace_custom`/`unsupported`）、R1-R20 红线（[`review-checklist.md`](review-checklist.md)）与场景注册表（[`scenarios/`](scenarios/)）为两边共用的合同。

## 目录

1. [Step 1：环境检查](#step-1环境检查门禁)
2. [Step 2：设计](#step-2设计architect)
3. [Step 2.5：设计串讲](#step-25设计串讲)
4. [Step 3：开发](#step-3开发developer)
5. [Step 4：审查](#step-4审查reviewer)
6. [Step 5：修复循环](#step-5修复循环)
7. [Step 6：精度与性能验收](#step-6精度与性能验收)
8. [Step 7：完成汇报](#step-7完成汇报)
9. [后续阅读](#后续阅读)

---

## Step 1：环境检查（门禁）

### apace 场景额外校验

| 校验项 | 达标条件 | 失败处理 |
|--------|---------|---------|
| NPU 架构 | 必须为 dav-3510 | 非 3510 直接终止 |
| CANN 版本 | 已在 CANN 9.2.0 验证；更低版本未验证 | 低于 9.2.0 提示未验证风险，由用户决策是否继续 |
| 多卡环境 | 至少 rankNum 张卡可用 | 单卡无法运行通算融合算子 |
| HCCL 可用 | `hccl.h` 头文件存在 | 提示用户检查 CANN 安装 |
| CANN 内置 apace | 内置 apace 目录 `test -d` 实测存在（路径形态见 [`workflow/step1-project-setup.md`](workflow/step1-project-setup.md) §1） | 提示用户检查 CANN 安装版本 |
| ops-transformer 可拉取（可选） | 网络可访问 gitcode.com | 仅跟踪 master 契约时需要；缺失不阻塞 |

> **路径实测纪律**：environment.md 中记录的每条路径（bisheng、头文件、.so、apace/blaze 框架目录）必须用 `test -x`（可执行文件）或 `test -f`/`test -d` 实测通过后才标 ✅——凭记忆填写的路径（如漏掉中间目录层级）会在编译阶段才暴露。记录精确到可直接使用的完整路径。

### 门禁

环境检查全部通过 → 进入 Step 2。任何一项失败禁止进入 Step 2。

---

## Step 2：设计（Architect）

### DESIGN.md 必须额外包含的小节

#### §约束显式确认

| 约束 | 验收条件 |
|:---|:---|
| ① 禁止 `__schedmode__(1)` | 核配比由 `KERNEL_TYPE_MIX_AIC_1_1` 保证 |
| ② Matmul 走 Blaze 模板 | 使用 `BlockMmad` + `BlockScheduler` + `MatmulWithScaleMx`，无 `AscendC::Matmul` |
| ③ 禁止修改 `block/` 和 `tiling/` | 只在 `kernel/<op>/` 下创建文件 |
| ④ 直调仅 UDMA | HCCL windows 不支持直调 |

#### §切分策略

- **两阶段 `splitAxisTileCnt` 策略**：精度调试阶段用 `tileCnt=1` 串行基线；性能调优阶段扫描 `{1,2,4,8}`（上限受 flag 计数约束：compute-first `T ≤ 15`、flagId 取 tid 时 `commTurn ≤ 16`，见 [`fusion.md`](fundamentals/fusion.md) §6.2.10/§3.3，超出档位非法）
- 切分轴按**算子数据分布语义**确定（输入分布轴与输出分布轴分别冻结，见 [`compute.md`](fundamentals/compute.md) §4.7 语义判定）——通信原语只决定数据搬运方向，**不决定切分轴**；同一原语可对应不同切分（如 AllToAll PUT 既可服务输入 K 切分，也可服务 compute-first 的输入不切分/输出 M 分布）。未明确时回 grill-protocol 维度 9 澄清
- Win 区空间预算（两段式）：data 段 `rankSize × rankDataBytes` + scale 段 `rankSize × scaleKaSize × axisM`（见 [`operator-anatomy.md`](operator-design/operator-anatomy.md) §3.5 / [`fusion.md`](fundamentals/fusion.md) §6.1）

#### §golden 语义（每卡输入/输出契约）

**验收条件**：明确每张卡的输入（本地数据 + 远端数据来源）与输出语义、切分轴（M/N/K 哪根轴按 rank 切）、聚合方式——这是 gen_data.py golden 的依据，切分轴写错则精度验证整体失效。

#### §API 验证清单

**验收条件**：DESIGN.md 选用的每个 API 标注「验证来源（官方文件:行号）+ 在当前 CANN 版本的可用性验证状态」。未验证 API 禁止入设计（生产教训：旧 CANN 版本存在符号缺失致链接失败的案例，设计阶段未发现则开发期返工）。CANN 版本以 Step 1 environment.md 记录为准，编译/运行前核对 `ASCEND_HOME_PATH` 指向同一版本。

#### §AIV/AIC 分工图

**验收条件**：明确哪些 work 在 AIV（通信），哪些在 AIC（计算），同步点在哪里（CrossCore flag 的 flagId 和 PIPE）。

### Architect 加载顺序

1. 读 [`architecture.md`](fundamentals/architecture.md) 建立整体心智模型
2. 通信侧不确定 → 读 [`communication.md`](fundamentals/communication.md)
3. 计算侧不确定 → 读 [`compute.md`](fundamentals/compute.md)
4. tiling 结构 → 读 [`operator-anatomy.md`](operator-design/operator-anatomy.md) §3
5. localMatmul 模式选择 → 读 [`fusion.md`](fundamentals/fusion.md) §5

### 门禁

- 双文件齐全（DESIGN.md + PLAN.md）；`unsupported` 路线仅 DESIGN.md
- DESIGN.md 包含"约束确认"小节（4 项均勾选 ✅）
- DESIGN.md 包含 golden 语义与 API 验证清单小节
- 切分策略参数有可解释的依据
- 路线决策已记录（`apace_native` / `apace_custom` / `unsupported`；apace_custom 须含 `selected_scenario`）

---

## Step 2.5：设计串讲

### 串讲关注点

- `[REUSE]`/`[MODIFY]` 标记是否合理（不能把 `[REUSE]` 错标成 `[MODIFY]`）
- AIC 的 `MatmulProcess` 是否正确遍历所有 rank
- Win 区空间预算是否够
- localMatmul 模式选择是否合理（参见 [`fusion.md`](fundamentals/fusion.md) §5）

### 收敛

严格 1 轮串讲；分歧写入 WALKTHROUGH.md `## 设计串讲仲裁`。

---

## Step 3：开发（Developer）

### 验收条件

基础工程验收（工程结构/共享层零修改/编译/冒烟/精度/文档同步）按 [`development-guide.md`](operator-design/development-guide.md) §5 验收清单逐项执行；apace 场景增量项：

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | host 前置校验 | 整除/对齐/核数下限/Win 容量/flag 计数峰值等校验在 fork/建链前完成，非法输入拒绝 launch（[`development-guide.md`](operator-design/development-guide.md) §3.5 清单模式） |
| 2 | UB 静态区隔离 | 通信 commBuf/barrierBuf 与 TPipe 管理 buffer 物理隔离（静态偏移或 guard TBuf，[`communication.md`](fundamentals/communication.md) 陷阱 #9） |
| 3 | 分阶段 bring-up | 新链路（bias/新通信对象/新归约）先退化对照（zero 值、T=1、rank=2）隔离验证再铺开 |
| 4 | 精度标准 | 以各算子 ST `verify_result.py` 为准（官网 `tests/st/`：all_to_all `rtol=atol=1e-2`；all_gather bit-exact 或 ≤1 ULP（bf16 raw uint16 比较）） |

### 开发阶段红线

- **禁止**使用 `__schedmode__(1)` 或 `[[bisheng::core_ratio(1,1)]]`
- **禁止**新增 `Hccl::` 高阶 API 调用
- **禁止**包含 asc-devkit 的 matmul 头（`AscendC::Matmul`）
- **禁止**修改 `block/` 和 `tiling/` 目录下的共享层文件
- **禁止**修改 `localMatmul` 等关键参数后不同步更新 DESIGN.md
- 改完每个 `[MODIFY]` 文件后立即跑一次精度验证做冒烟

> 详细改造验收标准见 [`development-guide.md`](operator-design/development-guide.md)。

---

## Step 4：审查（Reviewer）

### 必查清单

按 [`review-checklist.md`](review-checklist.md) 逐项检查（R1~R20，红线定义以 [`review-checklist.md`](review-checklist.md) 为规范源，含操作化检查方法、常见 FAIL 原因与修复方向）。违反任意红线项 = FAIL。

### 门禁

- REVIEW.md 判定 `PASS` 或 `PASS WITH NOTES` → 进入 Step 6
- REVIEW.md 判定 `FAIL` → 进入 Step 5 修复循环

---

## Step 5：修复循环

最多 3 轮。每轮：Developer 修复 → Reviewer 复审。

### 常见"修不动"问题与修复路径

| 问题 | 根因 | 修复路径 |
|:---|:---|:---|
| CrossCore flag idx 不配对 | AIC SetFlag idx ≠ AIV WaitFlag idx | 核对 flag 编排表（见 [`fusion.md`](fundamentals/fusion.md) §3） |
| CommContext 填充错误 | 手动赋值 channelHandles/commBufferAddrs | 必须由 `CommChannelBuilder::CreateDeviceContext` 填充 |
| tiling 不匹配 | 切换 tileCnt 后复用旧 tiling | 必须重新调 `GetTilingData` |
| splitKNum 配置错 | `localMatmul=1` 应 `rankSize-1`，`0/2` 应 `rankSize` | 修正 splitKNum（见 [`fusion.md`](fundamentals/fusion.md) §5） |
| localMatmul=1 MTE 异常（507015） | LOCAL fixpipe 未排空时 REMOTE AtomicAdd 读旧值 | RunLocalMatmul 和 RunMatmul 之间加 `PipeBarrier<PIPE_ALL>()`（详见 [`fusion.md`](fundamentals/fusion.md) §5） |

> ⚠️ **禁止添加 PipeBarrier 后不重试直接回退到 localMatmul=2** — PipeBarrier 的开销远小于通算并行带来的收益（生产实测性能差距显著）。

3 轮仍未通过 → 暂停上报用户。

---

## Step 6：精度与性能验收

### 6a 精度验收（Reviewer）

Reviewer 独立运行精度测试，输出精度验收报告 `docs/precision/summary.txt`：

- 全部 dtype 达标 → 进入 6b
- 精度不达标 → 回到 Step 5 修复循环（收敛计数器重置为 0，额外允许最多 3 轮；REVIEW.md 全局轮次编号从末尾最后一轮递增继续），Developer 修复后重新走 Step 5b → Step 6

### 6b 性能采集（Developer）

#### 验收条件

| # | 验收项 | 达标条件 |
|:---|:---|:---|
| 1 | 采集模式 | msprof task-based 采集 |
| 2 | L2 cache flush | 采集前必须刷 L2 cache（256MB 级大 buffer 拷贝；官网 PUT ST 有大 buffer 拷贝占位但未接线，apace 场景落地模板用 `heavy_add_kernel`，见 [`host-and-testing.md`](operator-design/host-and-testing.md) §4 与 [`../shared/profiling_mc2.md`](../../shared/profiling_mc2.md)），前一轮热度会污染本轮指标 |
| 3 | 多卡数据后处理 | 以官方 `parse_prof.py` 口径为准：跳过前 3 轮 warmup + 剔除 >1.2×min 离群后每卡取平均 → 多卡取最大值（见 [`host-and-testing.md`](operator-design/host-and-testing.md) §6.1 #2） |
| 4 | tileCnt 扫描 | 扫描 `tileCnt ∈ {1,2,4,8}`（上限受 flag 计数约束：compute-first `T ≤ 15`、flagId 取 tid 时 `commTurn ≤ 16`，见 [`fusion.md`](fundamentals/fusion.md) §6.2.10/§3.3），选 Task Duration 最小者 |
| 5 | 数据归档 | `docs/perf/round_NNN/` 存在且含多个 `PROF_*` 子目录 |
| 6 | 性能达标 | 满足 R15 投产级门槛：真实大 shape × R=2/4 双档 × 三路径对标归档（判据见 [`host-and-testing.md`](operator-design/host-and-testing.md) §6） |

> 官网 `tests/st/{op}/run.sh --perf` 提供 msprof 性能采集模式，产出由 `scripts/parse_prof.py --all` 解析，可复用该链路。
>
> 详细采集流程见 [`../shared/profiling_mc2.md`](../../shared/profiling_mc2.md)。

### 门禁

精度验收报告 `docs/precision/summary.txt` 已归档且全部达标 + 性能数据已归档 → 进入 Step 7。

---

## Step 7：完成汇报

CANNBot 主控汇总以下信息给用户：

- **最终判定**：PASS / PASS WITH NOTES
- **总分**：Reviewer 100 分制
- **代码路径**：`operators/{op_name}/`
- **精度概要**：各 dtype 达标状态（读取 `docs/precision/summary.txt`）
- **性能概要**：Task Duration、主导流水、通信隐藏率
- **关键问题列表**：审查和修复阶段遗留的 NOTE

---

## 后续阅读

| 文档 | 何时读 |
|:---|:---|
| [`architecture.md`](fundamentals/architecture.md) | 第一次了解 apace 三层架构 |
| [`development-guide.md`](operator-design/development-guide.md) | 工程搭建时，定位改造验收标准 |
| [`communication.md`](fundamentals/communication.md) | 通信接口与机制 |
| [`compute.md`](fundamentals/compute.md) | 计算接口与 kernel 模式 |
| [`operator-anatomy.md`](operator-design/operator-anatomy.md) | 算子骨架（tiling/Impl/入口规则） |
| [`fusion.md`](fundamentals/fusion.md) | 通算融合组合模式 |
| [`host-and-testing.md`](operator-design/host-and-testing.md) | host 序列与 ST 工程 |
| [`../shared/pipeline_tuning.md`](../../shared/pipeline_tuning.md) | 通算并行调优 |
| [`../shared/profiling_mc2.md`](../../shared/profiling_mc2.md) | 性能采集详细流程 |
