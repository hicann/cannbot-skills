# Subagent 调用参数详情

本文档是 PerformanceOptimizationAgent 调用各阶段 Subagent 的**唯一执行手册**。每个 Step 包含：调用参数、关联 Skill、完成验证、约束提醒。

---

## Step 1：性能数据采集与分析

### Subagent 调用参数

```
{
  "description": "性能数据采集与分析",
  "subagent_type": "ascendc-perf-analysis-expert",
  "prompt": "
请对以下待调优的 demo 代码进行性能数据采集与分析，按 **运行 demo 采集数据 → 性能分析 → 输出调优方案** 三阶段顺序执行：

- 源码目录：{code_dir}
- 测试用例文件：{cases_csv}（CSV 格式，包含所有测试用例的 shape、dtype、attrs 等参数。**若 case 数量超过 20 个，仅取前 20 个**）
- 输出目录：{output_dir}（性能数据、报告等所有产出物落盘到此目录下）

---

### 阶段一：运行 demo 并采集性能数据

1. 编译运行 `{code_dir}` 中的 demo 代码，确保可正常执行
2. **对 `{cases_csv}` 中全部用例**，使用 `/ops-profiling` 逐个 case 采集上板性能数据（msprof aiv_time）
   - **禁止只选代表性用例**：每个 case 都是用户关注的，必须全部采集
   - 所有测试用例均须在 NPU 上运行（禁止 host 侧短路绕过 kernel）
3. 从 profiling 数据中获取每个 case 的 aicore/aiv 耗时，记录数据来源（profiling），供后续分析使用

【输出：阶段一】
- **全部 case** 的性能数据采集产物（profiling 目录），存放到 `{output_dir}/perf_per_case/` 下
- 每个 case 的 aiv_time 汇总表
- 采集失败时停止，不进入阶段二

---

### 阶段二：性能分析

1. 加载 `/ascendc-perf-optimize`，进行建模和流水分析，形成多个性能调优方案
2. **逐 case 分析**：对 `{cases_csv}` 中的**每一个 case** 进行性能瓶颈分析：
   - 给出该 case 的 bound 类型（VEC/MEM/SCALAR BOUND）和具体瓶颈指标
   - 将瓶颈特征相同的 case 归并到同一组，在组内统一说明
   - **每个 case 必须在报告中出现**，不允许遗漏或仅以"同上"替代
3. case 归并分组维度（按优先级）：dtype → bound 类型 → shape 规模（S/M/L）→ 特殊值特征

【输出：阶段二】
- 多个性能调优方案
- 全部 case 的瓶颈分析结果（按组归并展示，但每个 case 独立列出）
- 分析失败时停止，不进入阶段三

---

### 阶段三：输出《性能调优方案》

> ⚠️ 以阶段二分析结论为输入；**禁止**重做分析或建模。

1. **输出《性能调优方案》报告**：
   - **最多 3 个性能调优方案**，按预期效果从高到低排序
   - 每个方案包含：优化目标、调优后的 Tiling 参数、调优策略、参考的 skill 路径、**模板骨架**（模板代码路径 + 关键结构摘要）、**模板分支条件**（该方案适用的 case 分支判定条件）
   - **逐 case 覆盖表**：列出哪些 case 将被该方案优化，以及每个 case 的预期改进方向和**所属模板分支**
   - 报告最后一节须包含"case 覆盖清单"：以表格列出全部 case，标注每个 case 归属的瓶颈组、适用方案、预期效果
   - **方案融合规则**：
     - 多个优化方向若涉及**不同模板且分支条件互斥**（如 AR 路径 vs ARA 路径，或 R ≤ 阈值 vs R > 阈值），**应融合为一个方案**（一个 kernel 含多个模板分支），而非拆成多个独立方案
     - 若多个方向涉及**同一模板的同一分支**（即对同一组 case 有不同优化策略），则**选最优策略拆为多方案并列**，由 Step 2 实测对比
     - 融合方案的 case 覆盖表须标注每个 case 走哪个模板分支


【输出：阶段三】
- 《性能调优方案》：方案概览 + 具体措施 + skill 路径引用 + **逐 case 覆盖清单**
- **报告必须落盘**到 `{output_dir}/性能调优方案.md`
- 自由发挥处显式标注「知识库暂未收录」

---

【验收标准】
- 三阶段按序完成，不跳步、不倒序
- 《性能调优方案》可含多个可行方案，显式标注覆盖维度
- 各阶段阈值与判据以当次打开的 skill 文件为准
- **全部 case 均已采集性能数据，且在报告中出现**
- 禁止修改算子源码
  "
}
```

---

## Step 2：方案实施

Step 2 由 **PerformanceOptimizationAgent（主 agent）** 分两个阶段完成：先并行分发方案实施，再统一采集性能并生成报告。

### 阶段 2a：并行方案实施（主 agent → 多个 impl subagent）

主 agent 读取《性能调优方案》，对其中每个方案各启动一个 `ascendc-perf-impl-expert` subagent，**并行执行**：

> ⚠️ **模板优先**：若《性能调优方案》中标注了"✅可直接拷贝使用"的模板 .h 文件，prompt 中须将模板文件完整路径传递给 impl expert，并明确要求"直接拷贝使用，禁止从头重写"。impl expert 遇到 MicroAPI 编译报错时须先尝试命名空间别名等最小修复，禁止直接降级。

```
{
  "description": "实施方案：<方案名称>",
  "subagent_type": "ascendc-perf-impl-expert",
  "prompt": "
请实施以下单个性能调优方案：

- 算子源码目录：{code_dir}
- 方案描述：<从《性能调优方案》中提取该方案的完整内容，含模板代码路径、关键骨架结构、模板分支条件、逐 case 模板映射>
- 方案标识：<方案名称，用于目录命名>
- 测试用例文件：{cases_csv}（CSV 格式，用于精度验证）
- 输出目录：{output_dir}（优化代码等产出物放到此目录下）

**模板优先指令**：
- 若方案中标注了模板 .h 文件路径且分类为"✅可直接拷贝使用"，**必须直接拷贝这些 .h 文件到项目 op_kernel/ 目录**，仅做最小适配（命名空间别名、TilingData 字段补充、include 路径修正），禁止从头重写
- 模板内的 MicroAPI VF 计算（RegTensor、MaskReg、LoadAsFp32、StoreFromFp32 等）必须原样保留，禁止降级为高层 API
- 遇到 MicroAPI / Reg API 编译报错时，必须按以下顺序尝试修复：
  1. 命名空间别名（如模板用 `AscendC::MicroAPI::`，实际 CANN 为 `AscendC::Reg::`，加 `namespace MicroAPI = Reg;`）
  2. API 签名适配（对照 CANN 实际头文件调整参数数量/类型）
  3. 头文件 include 路径修正
  4. 仅当以上修复均无效时，才降级对应分支，并在返回结果中明确列出尝试过的修复手段和失败原因

【任务】
按 `ascendc-perf-impl-expert.md` 执行流程，完成：复制目录 → 实现代码 → 编译 → 精度验证。
目录命名：`{output_dir}/{算子名}_optimized_<方案标识>/`（从 `{code_dir}` 复制源码到此目录，在此目录上修改）
精度验证：对 `{cases_csv}` 中**全部 case** 进行精度验证，不遗漏
完成后返回：优化代码目录路径、编译状态、精度验证结果（逐 case PASS/FAIL 明细）、**模板使用情况**（哪些模板被直接拷贝使用、哪些被降级、降级原因及尝试过的修复手段）。不采集性能、不生成报告。
  "
}
```

主 agent 等待所有 subagent 完成后，进入阶段 2b。

### 阶段 2b：统一性能采集与报告（主 agent 执行）

1. **统一 msprof 采集**：对基线 + 所有优化方案目录，按 `{cases_csv}` **逐个 case** 用 msprof 采集 aiv_time
   - **禁止只选代表性用例**：必须覆盖全部 case
   - 记录每个 case 在各个方案下的 aiv_time(us)
2. **精度确认**：确认所有方案对全部 case 精度均通过
3. **生成《性能调优报告》**并落盘到 `{output_dir}/性能调优报告.md`：
   - **逐 case 内核时间对比表**（msprof aiv_time）：行 = 全部 case，列 = 基线 + 各方案
   - **逐 case 分析**：对每个 case 明确写出：
     - 当前存在的性能瓶颈是什么（bound 类型 + 关键指标）
     - 采用了什么优化手段（引用具体方案）
     - 加速比提升了多少（基线 aiv_time / 优化后 aiv_time）
   - 瓶颈特征相同的 case **可以归并到同一组统一说明瓶颈和手段**，但每个 case 的加速比数据**必须独立列出**
   - 方案对照表和改进幅度
   - **最佳方案标注**：按全部 case 几何平均加速比，明确标注本轮优化提升最明显的方案及其代码目录。该目录作为多轮优化下一轮的基线起点
   - 标注采集方式与数据路径

【验收标准】
- 所有方案编译通过、精度验证通过
- 报告含**每个 case** 的 msprof aiv_time 对比表和逐 case 分析（瓶颈 → 手段 → 加速比）
- 所有 case 均在 NPU 上运行
- 报告已落盘

---

## Step 3：CANN-Bench 评测接入（可选）

### 触发条件

用户提供以下三项信息时，主 agent 在 Step 2 完成后进入 Step 3：

| 参数 | 必需 | 说明 |
|------|------|------|
| cann_bench 代码路径 | **是** | CANN-Bench 仓库根目录，包含 `tasks/` 和评测脚本 |
| 评测环境信息 | **是** | 目标 A5 芯片型号（如 Ascend 950）、CANN 版本、可用核数等 |
| 接入 cannbench 的决策 | **是** | 用户明确要求"接入 cannbench 评测"或"跑 A5 加速比" |

### 阶段 3a：确认 A5 基线数据

> 详细方案见 [`workflows/cannbench-a5-baseline-collection.md`](cannbench-a5-baseline-collection.md)

1. 检查 cann_bench 代码路径下是否已有该算子的 A5 基线数据（通常位于 `tasks/metadata/Ascend950DT_9582.json` 或 `scripts/baseline/output/`）
2. **若 A5 数据存在**：直接读取，记录为基线参考值
3. **若 A5 数据不存在**：
   - 使用 `scripts/collect_baseline.py` 在 A5 真机采集（见上述文档 §4）
   - 产出合并写入 `tasks/metadata/Ascend950DT_9582.json`
   - 记录基线 `baseline_perf_us` / `t_hw_us` 等核心指标

### 阶段 3b：原始基线 + 优化算子串行接入 CANN-Bench

> ⚠️ **关键**：必须同时接入**原始基线代码**和**优化后代码**两套，在同一 CANN-Bench 框架下评测，才能剥离"接入方式差异"，得出优化带来的净收益。
>
> ⚠️ CANN-Bench 是单仓库，不能并行操作。由一个 agent **按顺序**先接入基线代码、再接入优化代码。

主 agent 启动 **一个** agent（`ascendc-perf-impl-expert` 或通用 agent），执行以下任务：

```
{
  "description": "CANN-Bench 基线+优化串行接入",
  "subagent_type": "ascendc-perf-impl-expert",
  "prompt": "
请将原始基线算子和优化后算子**依次**接入 CANN-Bench 评测框架，使用 direct_launch_example 方式。先完成基线接入，再处理优化代码。

- 基线代码目录：{code_dir}（用户最初提供的待调优 demo 代码目录）
- 优化代码目录：{optimized_code_dir}（Step 2 产出的最佳优化方案目录）
- cann_bench 代码路径：{cann_bench_dir}
- 评测环境信息：{env_info}
- 测试用例文件：{cases_csv}
- 输出目录：{output_dir}（评测报告等产出物落盘到此目录下）

【任务】

===== 第一步：接入原始基线代码 =====

1. **理解 direct_launch_example 接入方式**：
   - 参考 cann_bench 仓库中已有的 direct_launch_example（如 `tasks/level1/exp/` 下的接入代码）
   - direct_launch_example 的核心文件：`CMakeLists.txt`（含 `add_executable`）、`op_host/<op>.asc`（host 直调入口）、`op_kernel/<op>_kernel.asc`（kernel 实现）、`scripts/run_all_cases.py`（批量测试脚本）

2. **接入基线代码**：
   a. 将基线代码目录复制一份，作为 CANN-Bench 接入的工作副本（不污染原始目录）
   b. 确保 `CMakeLists.txt`、`run.sh`、`build.sh` 等构建脚本正确配置（--npu-arch=dav-3510）
   c. 将 `{cases_csv}` 中的测试用例转换为 cann_bench 的 `cases.csv` 格式
   d. 编译并运行，确认所有 case 可正常执行且精度通过

3. **评测基线**：
   a. 运行 cann_bench 评测脚本（跑分命令参考 `workflows/cannbench-a5-baseline-collection.md` §8.1），对全部 case 采集 A5 性能数据
   b. 记录基线代码的逐 case elapsed_us
   c. 将评测 JSON 报告保存到指定路径

===== 第二步：接入优化后代码 =====

4. **接入优化代码**：
   a. 以优化代码目录为基础，确保 `CMakeLists.txt`、`run.sh`、`build.sh` 等构建脚本正确配置（--npu-arch=dav-3510）
   b. 使用与第一步相同的 `{cases_csv}` 测试用例
   c. 编译并运行，确认所有 case 可正常执行且精度通过

5. **评测优化代码**：
   a. 运行 cann_bench 评测脚本（跑分命令参考 `workflows/cannbench-a5-baseline-collection.md` §8.1），对全部 case 采集 A5 性能数据
   b. 计算每个 case 的加速比（A5 基线 / 优化后 elapsed_us）

6. **输出结果**：
   - 基线代码评测 JSON 报告路径（含逐 case elapsed_us）
   - 优化代码评测 JSON 报告路径（含逐 case elapsed_us）
   - 编译状态和精度验证结果（两套各自）
   - 不生成最终报告
  "
}
```

主 agent 等待 agent 完成后，进入阶段 3c。

### 阶段 3c：生成 CANN-Bench 评测报告

主 agent 在两个 agent 均完成后，统一生成报告：

1. **采集 A5 性能数据**：
   - 从 Agent 1（基线）的评测 JSON 报告中提取基线代码的逐 case elapsed_us
   - 从 Agent 2（优化）的评测 JSON 报告中提取优化代码的逐 case elapsed_us
2. **生成《CANN-Bench 评测报告》**并落盘：
   - **逐 case A5 加速比表**：每行一个 case，列至少包含：
     - A5 官方基线 baseline_perf_us（来自 metadata）
     - 原始基线代码 CANN-Bench elapsed_us
     - 原始基线 vs A5 官方加速比 = baseline_perf_us / 原始基线 elapsed_us
     - 优化后代码 CANN-Bench elapsed_us
     - 优化后 vs A5 官方加速比 = baseline_perf_us / 优化后 elapsed_us
     - 优化后 vs 原始基线加速比 = 原始基线 elapsed_us / 优化后 elapsed_us（净优化收益）
     - SOL% perf_score
   - **评分汇总**：按评分公式计算各 case 得分及总分（参考 `workflows/cannbench-a5-baseline-collection.md` §8.2-§8.3）
   - **方案对照**：与 Step 2 的优化方案关联，说明优化效果在大规模基准上的验证情况
   - **优化净收益分析**：对比"原始基线→A5官方"和"优化后→A5官方"两列加速比，分析优化在 CANN-Bench 框架下的真实提升幅度
   - 标注所有数据来源（A5 基线数据路径 / 原始基线 profiling 路径 / 优化后 profiling 路径）

【验收标准】
- A5 基线数据已确认（已存在或新采集）
- 原始基线代码和优化后代码均已接入 cann_bench 并编译通过
- 全部 case 精度验证通过（两套代码各自通过）
- 报告含每个 case 的三列加速比（原始基线 vs A5 / 优化后 vs A5 / 优化后 vs 原始基线）
- 报告已落盘到 `{output_dir}/CANN-Bench评测报告.md`

---

## 报告格式通用规范

所有报告必须包含以下字段，供 PerformanceOptimizationAgent 解析判断：

```markdown
**状态**: ✅完成 / ❌失败 / ⏭️跳过

**阶段**: 性能数据采集与分析 / 方案实施 / CANN-Bench 评测接入

**摘要**: 一句话描述本阶段结论

**详细内容**: （各阶段特定内容）
```

### 逐 case 覆盖格式（适用于各阶段报告）

报告中涉及 case 列表时，必须使用以下格式，确保全部 case 覆盖无遗漏：

```markdown
| Case | Shape | dtype | 瓶颈类型 | 适用方案 | 基线 aiv(us) | 优化后 aiv(us) | 加速比 | 备注 |
|------|-------|-------|---------|---------|-------------|---------------|--------|------|
| 1 | xxx | fp16 | VEC BOUND | 方案A | 14.1 | 6.0 | 2.35x | LUT 优化 |
| 2 | xxx | fp32 | VEC BOUND | 方案B | 37.6 | 26.7 | 1.41x | Taylor8 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
```
