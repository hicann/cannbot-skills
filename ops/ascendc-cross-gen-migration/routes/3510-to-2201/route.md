# 路线 3510→2201：A5（DAV_3510）→ A2/A3（DAV_2201）

本文件是路线包 `routes/3510-to-2201/` 的入口，承载该架构对的全部路线专属内容，由 `../../SKILL.md`「Step R：迁移路线判定」路由进入。

**回退校验**：若实际源/目标平台不是 DAV_3510→DAV_2201，立即停止并退回 `../../SKILL.md` Step R 重新判定，禁止沿用本文件的差异清单与决策树。

进入本路线后，本文件的约束与 `../../SKILL.md` 的全局约束、Gate 协议同等强制；阶段执行细则在本包 `stages/`、路线专属知识在 `references/`、脚本在 `scripts/`、模板在 `assets/`、迁移数据在 `experiments/`。本包自包含，禁止引用其他路线包的内容。

# Ascend 算子架构迁移（A5 → A2/A3）

把 A5 算子"降"到 A2 不是换编译参数，而是**能力裁剪 + 数据通路改道 + 预算重算 + 性能重建**。A5 上更难写、更快的能力（RegBase、SIMT、L0C→UB 直通、FP8 计算通路）在 A2 上不可用；A5 上被删除或降速的设施（L1→GM、GM→L0A/L0B、int4 Mmad、4:2 稀疏、高吞吐 `vnchwconv`）在 A2 上重新可用。迁移要同时做两件事：**删掉 A2 没有的**，**用回 A2 有的**。

## 本技能解决什么 / 不解决什么

| 解决 | 不解决 |
|---|---|
| 已有 A5 算子源码 → A2/A3 可编译、可运行 | 从零设计一个新算子（用 `ops-design-paradigms` / `ascendc-tiling-design`） |
| A5 专有能力的等价替换与"无法等价"项的用户决策 | 跨厂商 / 跨代 NPU 之外的迁移 |
| arch22 工程接线（目录、soc、config、CMake、def） | 用 A5 硬件验证 A2 性能（禁止） |
| 预算重算（UB 248→192KB、L0C 256→128KB、BT 4→1KB） | 修改 CANN 编译器或补齐 A2 硬件缺失能力 |
| A2 侧精度验证与性能调优、证据化交付 | 保证与 A5 输出 bit 级一致（A5 的 FTZ 行为本身不可逆） |

**平台边界**：本技能只覆盖 `DAV_3510 → DAV_2201`（A5 → A2/A3）。目标若是 310P（DAV_2002）或其他架构，停止并说明不在覆盖范围，禁止套用本技能的差异表。

## 何时使用 / 何时不用

**典型触发问法**（命中任一即适用）：

- "把这个 950 上跑通的算子迁到 910B / A2"
- "A5 代码在 910B 上编译不过 / 精度不对 / 性能差"
- "arch35 的算子改 arch22"
- "把 950 的 mask / 残块 / 量化能力下沉到 A2"

**不适用**：A2→A5 迁移走 `routes/2201-to-3510/`；目标 310P 或其他代际；从零设计新算子（用设计类技能）。

**一句话判定**：源 = `DAV_3510`，目标 = `DAV_2201` → 本技能；否则停止并换技能。

## 输入契约（阶段 0 核对，不满足就停）

| 输入形态 | 判定 | 处理 |
|---|---|---|
| A5 算子工程源码（含 `op_kernel/`、`op_host/`、`build.sh` 或 CMake） | 推荐 | 直接进入阶段 1 扫描 |
| A5 算子设计文档 + 参考实现（无源码） | 可接受 | 先按文档重建 A5 语义基线（公式、dtype、shape、内存布局、边界行为），再迁移；文档缺失项必须向用户确认 |
| 只有算子名 / 公式 / 一句需求 | 不足 | 停止，索取 dtype、shape、layout、参考语义与目标性能口径 |

**必须拿到的事实**（任一缺失则询问，不猜）：目标芯片（910B / 910_93，"A2"与"A3"同属 DAV_2201）、CANN 版本与安装路径、工程构建入口、支持的 dtype 清单、验收 shape 与性能目标口径、A2 真机或可用的等价环境。

## 知识依赖（单向消费，不复制真源）

| 知识入口 | 本技能消费什么 | 何时加载 |
|---|---|---|
| `npu-arch` | 芯片型号 ↔ NpuArch ↔ `__NPU_ARCH__` / arch-token 映射；A2/A5 Buffer、核数、L2 参数真值 | 阶段 0、1 判定目标与预算 |
| `ascendc-api-best-practices` | A2 MemBase API 签名与限制、跨代 API 差异、Subnormal 语义 | 阶段 2 改写、阶段 4 精度异常 |
| `ascendc-regbase-best-practice` | RegBase 代码形态识别（要删掉什么） | 阶段 1 盘点、阶段 2 拆解 |
| `ascendc-simt-best-practices` | SIMT 代码形态与可替代的 SIMD/AICPU 方案 | 阶段 1 盘点、阶段 2 决策 |
| `ascendc-tiling-design` | A2 tiling 切分与 Buffer 规划方法论 | 阶段 2 预算重算 |
| `ops-precision-standard` | 混合容差阈值、标杆构造、用例设计（**精度阈值的唯一真源**） | 阶段 4 |
| `ops-profiling` | msprof 采集、arch22 指标解读、可比较计时 | 阶段 5 |
| `ascendc-performance-best-practices` / `ascendc-perf-optimize` | A2 调优经验与参考实现 | 阶段 5 |
| `ascendc-env-check` | CANN/NPU 环境检查与 `NpuArch` 实测 | 阶段 0 |

> 真源纪律：芯片映射只在 `npu-arch` 维护、精度阈值只在 `ops-precision-standard` 维护。本技能不复制这些表；发现真源与本技能冲突时以真源为准并记录勘误。

## 迁移等级判定（阶段 1 产出，`scripts/scan_a5_only.py` 给出初判）

```
命中 RegBase/MicroAPI 或 SIMT 代码形态？
  ├─ 是 → D3 重写级：计算主体必须按 MemBase SIMD 重写（+ 预算重算 + 精度重建）
  └─ 否 → 命中 A5 专有数据通路/同步（L0C→UB、UB→L1、SSBuffer、BufferID、双主、模式 4、NDDMA/CCU）
          或 A5 专有数据类型（FP8/HiFloat8/MX）？
          ├─ 是 → D2 通路级：改数据流、接口与 dtype 策略（可能伴随局部重写）
          └─ 否 → D1 接线级：只做 arch35→arch22 工程接线与死代码清理
```

- **D3 一定包含 D2 与 D1 的工作**；等级取最高命中项。
- **A5 专有数据类型属于"决策项"而非机械替换项**：A2 没有对应硬件格式时，等价性由算子语义决定（换 dtype / 拆量化 / 拒绝支持），必须经用户确认并记录（见 `references/a5-only-inventory.md`）。
- 扫描器只做静态初判，不替代人工确认；`verify` 级命中（如 `LoadData2DParamsV2` 是否在 A2 头文件存在）必须查头文件/文档确认。

## 小范围能力下沉快速路径（feature down-port）

当 A5 已实现、A2 只是被门禁挡住某个**既有语义**（典型：GBSA 的 `mask_type=0` 非因果、`residual_block_mode=1` 隐式尾块），不必按 7 阶段全量重做。先花 10–20 分钟做两件事：① `python3 scripts/preflight_env.py` 确认 CANN/依赖/构建工具/vendor 库没有硬缺失；② **证明缺口**：构建包内是否含目标 soc kernel、ACLNN 直调是否真的失败。若直调 `getws=0, run=0` 则说明 A2 本来可用，不要按迁移做。确认缺口后按"四处门禁 + 一条验证"推进：

1. **host tiling 门禁**：定位 `socVer_ == 950` 之类的限制，确认放开条件（dtype / quant / layout / mask）。
2. **aclnn / metadata 门禁**：同一参数在 `op_api` 层通常还有一份校验，必须同步放开，否则主算子仍被拦。
3. **tilingKey**：新增目标组合（dtype × mask × residual × LSE），键位与既有 bitfield 一致，旧 key 不动。
4. **arch22 kernel**：照 arch35 的 `if constexpr (isCausal / hasResidualBlock)` 语义改可见范围、尾块追加、块计数上限；**不改 arch35 与公共模板**。
5. **验证下限**：干净构建 + 可见用例 + 至少 1 个自造残块/非因果用例（可见用例覆盖不到时），并记录参考来源。

**改动边界**：只动 arch22、tilingKey、两处 host 门禁；arch35 仅在接口签名必须同步时改一行，不做重构。

**先分清两类任务**（决定改动量，避免把 kernel 扩展误当门禁放开）：

| 类型 | 判定 | 改动 |
|---|---|---|
| 门禁型 | arch22 kernel 已有该语义分支，只是 host/metadata 拒绝 | 放开两处门禁，补用例 |
| 内核扩展型 | 目标布局/组合没有 tilingKey 或 arch22 分支（如新增 BSND layout、新 residual 语义） | 门禁 + tilingKey + arch22 kernel 分支；先列文件清单再动手 |

**构建成本**：大算子（注意力反向类）单算子全量构建可能 1–2 小时。首次构建后保留 `build/` 做增量；只有切换 soc/arch 或缓存明显污染时才 `rm -rf build`。文档/README 改动不触发重编。

**命令块（不要反复全量构建）**：

```bash
chmod +x build.sh
bash build.sh --pkg --soc=ascend910b --ops=<op>,<op_metadata>
./build_out/*.run --quiet --install-path=<独立目录>
source <独立目录>/vendors/<vendor>/bin/set_env.bash
python3 <可见用例 runner>
```

### 简单算子效率路径（S 类，先按这个顺序开工）

目标：**≤25 min、≤25M tokens** 完成一个简单搬运/寄存器类算子的 A2 迁移。

1. 只读 `SKILL.md` + `assets/simple-arch22-kernel.md`，不先大范围读 arch35 代码；
2. **先写四件套**：小 POD tiling 结构 → A2 host tiling → `REGISTER_TILING_DATA_CLASS` → 独立 kernel 入口；
3. kernel 只改必要文件，一次构建（≤3 次）；
4. 直调一次 + `msprof` 一次，Task Duration 不高于参考 5% 即停止。

禁止“先复用大 tiling 跑通、再重构成小结构”：实测这种返工多花 2–3 倍时间与 tokens。

**等待纪律（最容易被忽视的耗时来源）**：构建用一次前台 `timeout 600–900 bash build.sh ... > log 2>&1; tail -5 log`，不要用 `sleep 90/180` 反复轮询；后台构建时先做下一项编辑，只用 `pgrep` 查一次。禁止 `find /` 或全盘 grep，检索限定在仓库/安装目录并加 `-maxdepth`。

### 任务规模决定流程深度

| 规模 | 判定 | 流程 |
|---|---|---|
| S 单点 | 只缺 AddConfig/注册，或单个 kernel 文件的小改动（如 masked_causal_conv1d） | Step 1.0 缺口验证 + 最小注册/内核 + 一个 smoke；不写文档、UT、性能 |
| M 门禁/扩展 | 门禁 + tilingKey + arch22 分支（GBSA/BSA/SFAG） | 快速路径，预算 ≤200 次工具调用 / ≤40 min |
| L 整算子 | 完全没有 arch22 内核（block_attn_res_prepare） | 7 阶段裁剪 + 增量构建，允许更长预算 |

S 类若 30 min 内没有跑通 smoke，按 M 处理；不要把 S 类做成完整交付文档。

### 预算与必读（防止过度流程）

- **预算**：门禁型 ≤ 120 次工具调用 / ≤ 20 min；内核扩展型 ≤ 200 次 / ≤ 40 min。超出仍未收敛时停止扩大调研，先交付最小可验证改动。
- **必读最小集**：本文件 + `references/feature-downport-playbook.md`；只在命中具体报错时查 `references/pitfalls.md`。不要读 stages 0–6 的完整流程，也不要写分析/交付模板。
- **Gate 最小集**：快速路径覆盖 7 阶段流程，只需 `GATE-0`（环境简版）、`GATE-3`（构建安装）、`GATE-4`（精度）；不要求 GATE-1/2/5/6 文档。
- **构建纪律**：禁止反复 `rm -rf build` 全量重建；批量改完再构建一次；S 类 ≤3 次、M 类 ≤6 次；到上限仍未通过时停止重建，先读日志定位根因（此时继续构建通常只是重复失败）。
- **验收边界**：验收前不新增 UT/文档；已有 UT 与改动冲突时只做最小修复。
- **简单搬运类硬性要求**：A5 侧是通用模板时，必须在 arch22 文件内自行实现主循环（不得只包一层通用类）；验收看核时，不看文件是否存在。
- **小拷贝实现约束（≤4 KB/token）**：固定 16 KiB `TBuf` + 两次 `DataCopyPad` + `PipeBarrier<PIPE_ALL>`、按字节拷贝；不要用 TQue/双缓冲、逐 token 事件链、按 UB 全量分配的大 buffer（收益需以本机 profile 为准，见 `experiments/spc.md`）。
- **停止条件**：构建通过 + 目标用例通过 + 回归通过即停止；UT、文档、性能阶段仅在用户明确要求时补。

完整案例与常见错法见 `references/feature-downport-playbook.md`；没有 pytest/UT 或 custom 包分发不确定时，按 `references/verification-harness.md` 搭 ACLNN 直调验证。


## 与 `2201-to-3510` 路线的关系（禁止反向套用）

本路线是 **A5→A2/A3**，与升级路线 `routes/2201-to-3510/` 方向相反，以下概念**不适用**：

| 升级路线的概念 | 本路线对应 |
|---|---|
| L1 基础适配 / L2 RegBase 改造 / L3 SIMT 优化 | **不适用**；本路线使用 D1/D2/D3（按 A5 专有能力依赖范围定级） |
| stage 6 = A5 新特性使能 | **不适用**；本路线 stage 6 = 交付（`stage_6_delivery.md`） |
| 先接线、再逐层改 RegBase/SIMT | **顺序相反**：必须先完成 A5 专有构造盘点（stage 1 + `scripts/scan_a5_only.py`），blocking 清零后再接线/改代码 |

**前置步骤（硬性）**：`scripts/scan_a5_only.py --root <工程> --fail-on blocking` 返回 0 且 `RESIDUAL=0` 后，才允许进入 stage 2 的代码改造与 arch22 接线。

**构建命令（完整）**：

```bash
source <cann>/set_env.sh
cd <op_repo>
bash build.sh --pkg --ops=<op> --soc=ascend910b 2>&1 | tee build_910b.log
```

## 阶段执行流程

**MUST 按顺序执行；每个阶段开始时先读对应 stage 文件，完成时输出 Gate Token。没有上一阶段 Gate 不得进入下一阶段。**（例外：走「小范围能力下沉快速路径」时，按该节的必读最小集与 Gate 最小集执行，不读 stages 0–6。）

| 阶段 | 文件 | Gate Token |
|---|---|---|
| 0 环境与输入确认 | [stages/stage_0_env.md](stages/stage_0_env.md) | `[GATE-0] ENV_CONFIRMED target=<soc> arch=2201` |
| 1 A5 依赖盘点与定级 | [stages/stage_1_assess.md](stages/stage_1_assess.md) | `[GATE-1] LEVEL=D1/D2/D3 BLOCKING=<n> DECISIONS=<n>` |
| 2 迁移改造 | [stages/stage_2_implement.md](stages/stage_2_implement.md) | `[GATE-2] MIGRATION=PASS FILES=<n> RESIDUAL=0` |
| 3 编译与安装 | [stages/stage_3_build.md](stages/stage_3_build.md) | `[GATE-3] BUILD=PASS INSTALL=PASS` |
| 4 精度验证 | [stages/stage_4_precision.md](stages/stage_4_precision.md) | `[GATE-4] PRECISION=N/N_PASS`（N_FAIL≠0 时输出 `PRECISION=N/M_PASS BLOCKED`） |
| 5 性能验证与调优 | [stages/stage_5_performance.md](stages/stage_5_performance.md) | `[GATE-5] PERF=COLLECTED TARGET=<met\|not-met\|blocked> EVIDENCE=<n>` |
| 6 交付 | [stages/stage_6_delivery.md](stages/stage_6_delivery.md) | `[GATE-6] DELIVERY=READY REPORT=<path>` |

阶段 4/5 是**交付门槛**，不是可选项："可运行"= GATE-3 + GATE-4，"高性能"= GATE-5 达标或给出硬件受限的证据与整改项。用户明确只要"能跑"时，MUST 在报告中标注 `PERF=NOT_REQUIRED_BY_USER` 并保留阶段 5 的基线数据，不得悄悄跳过。

## Gate 协议

1. **先读 stage，再干活**：每阶段开始 MUST 读取对应 `stages/stage_*.md`。
2. **LOADED 标记**：stage 要求加载的参考文件，加载后 MUST 输出 `[LOADED] <文件名>`。
3. **证据化 Gate**：禁止仅凭"改完了"、"应该没问题"、Todo 勾选、静态分析判定 PASS。每个 Gate 必须有可复核产物（扫描报告、编译日志/产物、精度输出、msprof 数据、报告文件）。
4. **状态不许美化**：证据不足时输出 `BLOCKED` / `NOT VERIFIED`；部分通过输出实际分子分母（如 `PRECISION=27/30_PASS`），禁止四舍五入成 PASS。
5. **失败不隐藏**：任一阶段失败 MUST 记录失败现象、已排除假设与下一步，不得用"已知限制"替代根因分析。

## 全局约束

1. 阶段 0 未确认目标芯片、CANN 环境与 A2 可执行环境前，禁止编译、安装、测试。
2. 禁止把 A5 专有构造留在 arch22 编译路径里"先编过再说"（`#if __NPU_ARCH__ == 3510` 死分支必须删除或改为公共代码）。
3. 禁止用 A5 设备或 A5 数据冒充 A2 结果；A2 无设备时 Gate 保持 BLOCKED 并明确缺什么。
4. 禁止为通过 Gate 放宽阈值、删用例、改输入范围、缩小 dtype 覆盖；精度阈值以 `ops-precision-standard` 为准。
5. 任何 dtype/功能迁移（FP8→bf16、SIMT→AICPU、int4 改 int8 等）MUST 记录：原语义、替换方案、误差影响、用户确认结论。
6. 禁止硬编码 Buffer 容量与核数（192KB/128KB/24 核等只是典型值）：运行时用 `PlatformAscendC::GetCoreMemSize` / `GetCoreNumAiv` 等获取。
7. 性能结论 MUST 来自 arch22 实测；arch35 的 RegBase/VF/RVEC 指标不能用于推断 A2 瓶颈。
8. 交付物 MUST 包含可复现命令（构建、安装、精度、性能）与变更账本（文件 → 改了什么 → 依据）。
9. 不得声称"A2 与 A5 数值 bit 级一致"：A5 的 FTZ（subnormal→0）不可逆，A2 恢复 IEEE subnormal，二者结果本就有差异。
10. 动手前先列目标文件清单（arch22、tilingKey、两处 host 门禁）；`arch35` 只读，越界改动视为返工，需在变更账本中给出理由。

## 绝对不要做（每条含替代方案）

| 禁止 | 应该做 |
|---|---|
| 直接 `--soc=ascend910b` 编译 arch35 代码，指望"能过就过" | 先按阶段 1 盘点，删/改 A5 专有构造后再接线 |
| 把 `arch35/` 目录改名为 `arch22/` 就宣布完成 | 目录名是架构承诺：内容必须真的只剩 arch22 可用实现 |
| 保留 `__VEC_SCOPE__` / `MicroAPI::` / `RegTensor` 到 arch22 | 按 `references/api-migration-recipes.md` 重写为 `LocalTensor` + count 语义与尾块处理 |
| 把 SIMT `VF_CALL` 直接注释掉留空实现 | 明确选择 SIMD 重写 / AICPU 实现 / 不支持三种结论之一 |
| 用"编译器会报错再说"代替静态盘点 | 阶段 1 先跑 `scripts/scan_a5_only.py`，把阻塞项列清 |
| FP8/HiFloat8 张量直接当 bf16 指针 reinterpret | 走量化/反量化语义，记录精度影响并经用户确认 |
| 沿用 A5 的 40KB SIMT DCache 预留而不重算 | A2 无 SIMT DCache，按 `GetCoreMemSize(UB)` 重算 tiling（常见 192KB） |
| 沿用 A5 的 tiling 常量（248KB/256KB/28 核） | 用运行时接口取值，按 `scripts/a2_ub_budget.py` 复核 |
| 为 A5 性能退化做过的规避改动原样保留 | 审计并回退：A2 上 `vnchwconv` 高吞吐、int4/4:2 可用、L1↔GM 可用 |
| 只跑一个 shape 就宣布"高性能" | 按阶段 5 的证据清单采集：多 shape、中位数计时、瓶颈占比 |
| 把精度差异归因为"硬件差异"就结束 | 分桶定位（dtype 表示粒度 / FTZ 残留 / 算法路径差异），给出证据 |

## 参考索引

| 文件 | 用途 |
|---|---|
| [references/knowledge-index.md](references/knowledge-index.md) | 知识地图、来源与置信度、按阶段加载清单 |
| [references/input-contract.md](references/input-contract.md) | 输入形态、必需事实、拒绝条件与澄清话术 |
| [references/a5-only-inventory.md](references/a5-only-inventory.md) | A5 专有能力清单：识别信号 → A2 处置 → 严重级 → 验证方式 |
| [references/api-migration-recipes.md](references/api-migration-recipes.md) | 代码级改写配方（RegBase→MemBase、SIMT、dtype、同步、Cube、FTZ） |
| [references/resource-budget.md](references/resource-budget.md) | 预算差异与重算公式、UB/L0C/BT/L1 对照、`a2_ub_budget.py` 用法 |
| [references/build-and-config.md](references/build-and-config.md) | arch22 工程接线、构建命令、常见编译错误 |
| [references/precision-verification.md](references/precision-verification.md) | 迁移场景的精度验证设计（dtype 等价、FTZ 回退、边界） |
| [references/performance-tuning-a2.md](references/performance-tuning-a2.md) | A2 性能调优手册（口径、指标、手段、收敛规则） |
| [references/pitfalls.md](references/pitfalls.md) | 迁移专属故障模式与处置 |
| [references/feature-downport-playbook.md](references/feature-downport-playbook.md) | 小范围能力下沉方法（四处门禁、边界、验证下限）+ 案例索引（实验记录在 `experiments/`） |
| [references/verification-harness.md](references/verification-harness.md) | 无 pytest/UT 或 custom 分发不确定时的 ACLNN 直调验证、依赖自检与常见坑 |
| [scripts/scan_a5_only.py](scripts/scan_a5_only.py) | A5 专有构造静态扫描（JSON/Markdown 报告 + 等级初判） |
| [scripts/a2_ub_budget.py](scripts/a2_ub_budget.py) | UB 预算核算与 A5→A2 tile 缩放建议 |
| [assets/migration-analysis.md.template](assets/migration-analysis.md.template) | 阶段 1 分析报告模板 |
| [assets/migration-report.md.template](assets/migration-report.md.template) | 阶段 6 交付报告模板 |
| [experiments/gbsa.md](experiments/gbsa.md) | GBSA 实验记录：门禁原文、tilingKey、精度、A/B 成本 |
| [experiments/sfa.md](experiments/sfa.md) | SFA 实验记录：LSE+PA_BSND 门禁、基线行为、评测命令 |
| [experiments/bsa.md](experiments/bsa.md) | BSA 实验记录：BSND 门禁、tilingKey/arch22 改动、ACLNN 直调配方 |
| [experiments/sfag.md](experiments/sfag.md) | SFAG 实验记录：sinks 仅 950、arch22 改动与收益 |
| [experiments/block_attn_res_prepare.md](experiments/block_attn_res_prepare.md) | block_attn_res_prepare 实验记录：A2 构建注册/内核、收益 |
| [experiments/mcc.md](experiments/mcc.md) | MaskedCausalConv1d 实验记录：A2 注册、因果卷积精度、收益 |

## 官方来源与置信度

- **HIGH（本仓已核）**：`npu-arch` 的芯片/Buffer/核数/稀疏表；`ascendc-api-best-practices` 的跨代 API 差异与 Subnormal 语义；`ops-precision-standard` 的容差；`ops-profiling` 的指标字段。
- **MEDIUM（版本相关，落地前复核）**：CANN 版本相关的宏名、API 重载、构建参数、`Fixpipe` 结构体命名。
- **必须现场验证**：A2 头文件是否存在某 API、某 dtype 枚举是否可用、目标 CANN 的 `--soc` 拼写与 `AddConfig` 字符串。验证方式：`grep` 目标 CANN 的 `include/` 与 `opp/` 目录、查官方文档、`ascendc-docs-search`。

> 任何"我记得 A2 支持/不支持 X"都必须落到上述证据，不能凭经验断言。
| [assets/simple-arch22-kernel.md](assets/simple-arch22-kernel.md) | 简单搬运类算子的专用 arch22 kernel 模板 |
| [experiments/spc.md](experiments/spc.md) | 简单搬运类下沉 A2 的实现路径对比与核时收益 |
- **tiling 结构（简单算子必做）**：不要复用 A5/arch35 的大 tiling 结构，按四步做 A2 专用小结构：① 定义只含必需字段的 POD 结构；② host tiling 只填这些字段；③ `REGISTER_TILING_DATA_CLASS` 注册；④ 独立 kernel 入口读取。复用大结构实测约 11.0 µs，小结构约 8.9–9.7 µs（见 `experiments/spc.md`）。
- **性能验收**：简单算子必须 msprof 对比参考实现，Task Duration 高于参考 5% 以上视为未完成。
| [experiments/baru.md](experiments/baru.md) | 复杂 Attention 反向算子：A2 注册、标量暂存区、尾块修复与 28/28 数据 |

> GATE-3 必须在 A2 真机（`NpuArch=2201`）完成；无 A2 设备只能 `BUILD=BLOCKED REASON=NO_A2_DEVICE`，A5 机器不能替代 A2 验证。
| [references/sfag-sinks-port.md](references/sfag-sinks-port.md) | SFAG sinks 反向移植指引（参考实现、公式、atomic、验证） |
