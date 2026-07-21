---
name: cake-evo
description: Evolutionary AscendC operator generation — spawn parallel variants with different optimization strategies and select the best. Use as the top-level orchestrator for multi-round kernel optimization. 触发：需要多轮并行进化以优化内核性能时。
---

## What I do

通过并行生成多个AscendC内核变体、评估性能并迭代进化来生成高性能内核。

**与cake-evo agent的区别**: 此skill在主Claude Code窗口中运行，可以直接使用Task工具启动并行子agent，避免了nested agent的限制。

## Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  步骤 1   收集配置                                                   │
│           ├── 模式 A: 描述模式（从自然语言描述开始）                  │
│           └── 模式 B: 基线内核模式（从现有 AscendC 内核开始）         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  步骤 2   自动检测环境                                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  步骤 3   执行共享前置步骤（只执行一次）                              │
│           ├── 模式 A: op_desc → reference → functional → ascend-call │
│           └── 模式 B: 读取基线 → 评估 → 复制到共享目录              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  步骤 4   进化轮次循环（r = 1 … max_rounds）        ◄───────┐        │
│           4.1 创建轮次目录 + 复制共享文件                     │        │
│           4.2 并行启动 cake-partial 子Agent（同一消息内）     │        │
│           4.3 收集并分类结果（好 / 中 / 差）                  │        │
│           4.4 显示轮次摘要                                    │        │
│           4.5 检查终止条件 ──────────────────────────────────┘        │
│               ├── 达到目标加速比？ → 退出循环                         │
│               └── 达到最大轮数？  → 退出循环                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  步骤 5   输出最终结果（最优实现 + 完整轮次报告）                     │
└─────────────────────────────────────────────────────────────────────┘
```

执行此skill时，你（主Claude Code）负责整个编排流程并自动运行, 不要暂停（除非遇到错误需要人工干预）。
Before taking any action, use your `todo` tool to add every step of this workflow to your task list. You must complete the entire todo list before yielding to the user.
---

### 步骤1: 收集配置

CAKE进化支持两种启动模式。当用户请求进化内核生成时，首先确认运行模式:

#### 模式A: 描述模式 (从自然语言描述开始)
用户提供算子的自然语言描述，从头生成所有文件。

#### 模式B: 基线内核模式 (从现有AscendC内核开始)
用户提供现有AscendC内核项目目录的路径，先评估基线性能，再从基线出发进行进化优化。

**⚠️ 关键约束 - 接口保留原则**:
> 在基线内核模式下，进化过程**不得修改**算子的输入/输出/参数接口。所有优化只能在内核实现层面进行（计算逻辑、分块策略、内存布局、双缓冲等），不得改变算子签名。

在确认模式后，询问以下配置（或从用户消息中提取）：
- **算子名称**: 简短标识符 (例如: "FastGELU", "Tril")
- **[描述模式] 算子描述**: 操作的自然语言描述
- **[基线模式] 基线内核路径**: 现有AscendC内核项目目录路径
- **最大轮数**: 最大进化轮数 (默认: 2，推荐: 2-3)
- **并行数量**: 每轮并行候选数 (默认: 3，推荐: 3-5)
- **目标加速比**: 期望达到的目标加速比 (默认: 3x)

---

### 步骤2: 自动检测环境

```bash
which npu-smi
```

- 输出: "✅ 检测到CANN环境，使用本地编译模式"
- 验证: `echo $ASCEND_HOME_PATH`
- 如果未检测到npu-smi，告诉用户需要配置CANN环境

---

### 步骤3: 执行共享前置步骤（只执行一次）

生成时间戳并创建共享目录：
```bash
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p output/{op_name}_evo_${timestamp}/shared
```

根据输入模式选择对应的准备方式:

---

#### 模式A: 描述模式

**你自己**（不要启动子agent）按顺序调用以下skill，所有输出保存到 `output/{op_name}_evo_{timestamp}/shared/`：

1. 调用 `op-desc-generation` skill
2. 调用 `reference-generation` skill
3. 调用 `functional-conversion` skill
4. 调用 `ascend-call-generation` skill，告知其输出目录为shared目录

5. **生成CMake项目到shared目录**:

   `ascend-call-generation` 内部的 `gen_project.py` 默认将CMake项目输出到 `output/{op_name}/` 而非shared目录。调用skill完成后，必须使用 `--output-dir` 将CMake项目直接生成到shared目录：
   ```bash
   .venv/bin/python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascend-call-generation/scripts/gen_project.py \
       {op_name} $(pwd)/output/{op_name}_evo_{timestamp}/shared/{op_name}_project.json \
       --output-dir $(pwd)/output/{op_name}_evo_{timestamp}/shared/
   ```
   > **⚠️ 路径注意**: `project_json` 参数必须使用绝对路径（或 `$(pwd)/...`），相对路径会导致 msopgen 找不到文件。

6. **如果上游首次读写了 api_description，则在 `op-desc-generation` 阶段先复制到 shared/work_dir**:
   - 统一复制为 `api_description.md`
   - 后续 round 会从 `shared/` 复制到各自 work_dir，因此评测默认也能在 work_dir 下看到该文件

---

#### 模式B: 基线内核模式

**步骤3B-1: 读取基线项目文件**

读取基线内核项目中的关键文件:
- `{baseline_kernel_path}/{op_name}Custom/op_kernel/{op_name}_custom.cpp` — AscendC内核代码
- `{baseline_kernel_path}/{op_name}Custom/op_host/{op_name}_custom.cpp` — Host代码
- `{baseline_kernel_path}/{op_name}_op_desc.json` — 算子描述
- `{baseline_kernel_path}/{op_name}_reference.py` — PyTorch参考实现
- `{baseline_kernel_path}/{op_name}_functional.py` — Functional API

**步骤3B-2: 推导算子描述**

- 如果 `{op_name}_op_desc.json` 已存在 → 直接使用
- 否则 → 从内核代码和host代码推导描述（严格保留输入/输出/参数接口）

**步骤3B-3: 评估基线内核性能**

在启动任何进化轮次之前，必须先评估基线内核的性能以建立参考基准:
1. 按照预加载的 `ascendc-evaluation` skill执行评估
2. 将结果保存到 `output/{op_name}_evo_{timestamp}/baseline_evaluation.json`
3. 向用户展示基线性能报告

**步骤3B-4: 复制基线文件到 shared/**

```bash
cp -r {baseline_kernel_path}/* output/{op_name}_evo_{timestamp}/shared/
```

检查并补全缺失文件（按需生成）:
- 缺少 `{op_name}_op_desc.json` → 运行 `op-desc-generation`
- 缺少 `{op_name}_reference.py` → 运行 `reference-generation`
- 缺少 `{op_name}_functional.py` → 运行 `functional-conversion`
- 缺少 `{op_name}_custom.py`、`{op_name}.cpp` 或 `{op_name}Custom/` → 运行 `ascend-call-generation`

---

完成后 `shared/` 应包含:
- `{op_name}_op_desc.json`
- `{op_name}_reference.py`
- `{op_name}_functional.py`
- `{op_name}_custom.py`
- `{op_name}.cpp`
- `{op_name}Custom/` (CMake项目)
- `api_description.md`（如果上游流程已拿到该文档，建议由 `op-desc-generation` 先复制到 shared/work_dir）

---

### 步骤4: 执行进化轮次

对于每轮 r (从1到max_rounds)：

#### 4.1 创建轮次目录并复制共享文件

```bash
for p in $(seq 0 $((parallel_num - 1))); do
    mkdir -p output/{op_name}_evo_{timestamp}/round_{r}/parallel_{p}
    cp -r output/{op_name}_evo_{timestamp}/shared/* \
          output/{op_name}_evo_{timestamp}/round_{r}/parallel_{p}/
done
```

#### 4.2 并行启动 cake-partial 子agent

**⚠️ 关键：必须在同一条消息中发送所有 Task 工具调用以实现真正的并行执行。**

对每个并行索引 p (0 到 parallel_num-1)，在**同一条消息**中同时发送所有 Task 调用：
- `subagent_type`: `cake-partial`
- `description`: `Generate kernel variant {p}`
- `run_in_background`: `true`
- `prompt`: 见下方模板

**本地模式**子agent prompt模板：

```
Optimize AscendC kernel for {op_name} operator.

Operator Description:
{op_description}

Output Directory: output/{op_name}_evo_{timestamp}/round_{r}/parallel_{p}

The following shared files have already been generated and copied to the output directory:
- {op_name}_op_desc.json
- {op_name}_reference.py
- {op_name}_functional.py
- {op_name}_custom.py (DO NOT regenerate)
- {op_name}.cpp (DO NOT regenerate)
- {op_name}Custom/ (CMake project with host code)

CRITICAL: DO NOT regenerate or modify these files. Start directly from DSL baseline generation.

{inspirations_text}

Follow the preloaded skill instructions to execute these steps in sequence:
1. dsl-baseline-generation - Generate DSL baseline (read existing files from output directory)
2. dsl-lowering - Apply DSL lowering (local). Before writing any AscendC code:
   a. Read `references/meta_prompts/strategy_index.md`
   b. Select strategy IDs relevant to this operator type
   c. Read each referenced `strategies/XXX.md` detail file
   d. Apply selected patterns in tiling_pass, init_pass, process_pass, process_nonaligned_pass
3. cake-code-review - Check coding red-line and review/fix AscendC code
4. ascendc-evaluation - Evaluate locally

After each step, record current states in a log file in the output directory.
Return evaluation results: compilation success, precision, and speedup.
```

**启动后**，使用 TaskOutput 逐个收集结果：
- `block`: `true`
- `timeout`: `1200000`（20分钟）
- 如果超时（TaskOutput返回超时），使用 `TaskStop` 终止该子agent，标记为失败，继续下一个

**关于 `{inspirations_text}` 的填充规则**:
- **描述模式，第1轮**: `inspirations_text` 为空字符串
- **描述模式，第2轮及以后**: 从上一轮好/中/差层采样实现
- **基线内核模式，第1轮**: 必须包含基线内核代码摘要和性能数据作为灵感
- **基线内核模式，第2轮及以后**: 同描述模式，从上一轮结果中采样

#### 4.3 收集并分类结果

收到任何子agent 完成通知后，**无论该 agent 的返回文本是否完整**（可能因上下文耗尽而截断），
**必须直接读取** `output/{op_name}_evo_{timestamp}/round_{r}/parallel_{p}/evaluation_results.json`
作为权威评估结果。不得依赖 agent 消息中的文字描述。

从 JSON 文件提取：
- `compilation_success`, `precision_passed`, `speedup`, `base_time_ms`, `gen_time_ms`

按加速比分类（30%/40%/30%）：
- **好层**（前30%）: 最佳性能
- **中层**（中间40%）: 平均性能
- **差层**（后30%）: 较弱性能

#### 4.4 显示轮次摘要

```
轮次 {r} 摘要:
  总实现数: {total}
  编译成功: {n}/{total}
  精度通过: {n}/{total}
  最佳加速比: {best}x
  平均加速比: {avg}x

分类:
  好层: {n} 个实现
  中层: {n} 个实现
  差层: {n} 个实现
```

#### 4.5 检查终止条件

- 达到目标加速比 → 停止
- 没有成功实现 → 停止
- 达到最大轮数 → 停止
- 否则 → 构建灵感文本，进入下一轮

**灵感文本**（用于下一轮 `{inspirations_text}`）：

从各层采样灵感（好层1个 + 中层1个）：
```
Based on previous round results, use these implementations as inspiration:

Best implementation (speedup: {best}x):
[DSL code from best parallel dir]

Medium implementation (speedup: {medium}x):
[DSL code from medium parallel dir]

Explore different approaches from these to improve performance.
```

---

### 步骤5: 最终结果

```
进化完成！

前3个实现:
1. round_{r}_parallel_{p}: {best}x 加速比
2. ...
3. ...

最佳实现路径: output/{op_name}_evo_{timestamp}/round_{r}/parallel_{p}/
```

6. Continue to the next step in agent workflow

---

## Key Notes

1. **并行关键**: 必须在同一条消息中发送所有 Task 调用，否则不是真正并行
2. **共享文件**: 步骤1-4只执行一次，所有变体复制共享文件，不重新生成
3. **子agent类型**: 始终使用 `cake-partial`，不要使用 `cake` 或 `cake-evo`
4. **gen_project.py 路径规则**:
   - `project_json` 参数必须是**绝对路径**（用 `$(pwd)/...` 或完整路径）；相对路径会因 msopgen 在子目录中运行而失败
   - 使用 `--output-dir $(pwd)/output/{op_name}_evo_{timestamp}/shared/` 将CMake项目直接输出到shared目录，避免事后手动复制
5. **目录结构**:
   ```
   output/{op_name}_evo_{timestamp}/
   ├── shared/                     # 共享文件（只生成一次）
   ├── baseline_evaluation.json    # [仅基线模式] 基线性能
   ├── skill_trace.json            # 主 agent skill 调用追踪
   ├── skill_trace_aggregate.json  # 所有变体 skill 追踪聚合
   ├── round_1/
   │   ├── parallel_0/
   │   │   └── skill_trace.json    # 该变体 skill 调用追踪
   │   ├── parallel_1/
   │   └── round_1_results.json
   └── round_2/
   ```
