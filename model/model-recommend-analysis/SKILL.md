---
name: model-recommend-analysis
description: 推荐模型昇腾NPU性能分析与优化，根据Profiling（必选）和Dump图（可选），分析瓶颈点、分析图结构相似结构，找到融合算子和pass机会；并给出优化建议。触发场景：推荐模型在昇腾NPU上性能不达标，需要分析MindStudio Profiler输出定位瓶颈并给出优化建议，或客户推荐业务迁移到昇腾需要调优吞吐和时延。
---

# 推荐优化 Skill

## 适用场景

- 推荐模型在昇腾NPU上性能不达标
- 需要分析MindStudio Profiler输出，定位瓶颈并给出优化建议
- 客户推荐业务迁移到昇腾，需要调优吞吐和时延
- **若提供了GE build图(Pbtxt)或 PyTorch fxgraph，可识别Top相似结构，给出融合pass/融合算子优化建议**

## 核心指标

推荐推理优化目标是**满足时延要求下的最大吞吐**，不是单纯的降时延。

```
单卡吞吐=BS × 多实例并发数 × 1000 ÷ AVG(H2D耗时+ModelExecute耗时+D2H耗时)
```

关键约束：当客户时延预期 > 单实例迭代耗时(H2D耗时+ModelExecute耗时+D2H耗时)，才有多实例并行的空间

推荐训练优化目标是**最大吞吐，即单实例迭代耗时**。单迭代耗时优化和推荐推理同，但是多实例和控核一般不适用训练。

## 核心WorkFlow

**输入要求：** Profiling 数据为必选输入，Dump 图为可选输入。

- **步骤1：** 识别是 GE Profiling 还是 PyTorch Profiling
- **步骤2：** 执行 Profiling 分析脚本，计算 H2D 耗时、ModelExecute 耗时（算子耗时和、调度耗时等）、D2H 耗时、Top 算子耗时、Top 接口耗时，识别是否含动态 shape 等
- **步骤3（可选）：** 若提供了 Dump 图（pbtxt 或 fxgraph `*output_code*.py` ），执行图分析脚本，结合 Profiling 分析 Top 重复子图结构（含算子名、shape、dtype），给出融合 pass/融合算子优化建议
  - 若未提供 Dump 图，则仅基于 Profiling 分析结果给出优化建议，跳过重复子图分析
- **步骤4：** 汇总 Profiling 分析（和 Dump 图分析），输出 Markdown 分析报告
- **步骤5：** 校验分析报告。必须遵守的原则：1、确保数据和 Profiling（及 Dump 图）对齐；2、优化措施中的环境变量、配置参数、接口调用等，必须真实存在、可实施可验证，详见“优化建议校验规则”
- **步骤6：** 校验通过后，删除临时文件，如 profiling_report.md 和 dump_report.md

## 优化建议校验规则

优化建议中涉及的环境变量、配置参数、接口调用等实施细节，**必须严格遵循以下校验规则**，确保建议可实施、可验证、不产生误导。

### 规则1：禁止凭记忆编写实施参数

- **禁止**直接凭记忆或经验编写环境变量名、参数值、接口签名
- Agent 必须通过**联网查询昇腾官方文档**或**参考链接**确认参数真实存在后再写入建议
- 若无法联网验证，则建议中**不得给出具体参数**，改为引导用户查阅文档

### 规则2：区分执行模式

生成实施示例前，**必须先确认当前模型的执行模式**（从 Profiling 分析结果中获取）：

| 执行模式                                    | 判断依据                                                                         | 适用措施                                                     | 不适用措施                                   |
| ------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| **图模式 (GE/TorchAir/ATC AutoFuse)** | op_statistic 含（`autofuse_`或 `autofused_` 前缀） + 有 ModelExecute         | 图模式环境变量、GE/TorchAir/ATC 图编译 config、AutoFuse Pass | Eager 专用接口                               |
| **图模式 (Inductor + AscendC)**       | op_statistic 含 （`autofuse_`或 `autofused_` 前缀） + 无 ModelExecute        | Inductor 环境变量、AscendC 后端配置                          | GE 专用环境变量                              |
| **图模式 (GE/TorchAir)**              | api_statistic 含 ModelExecute (无融合算子)                                       | 图模式环境变量、GE/TorchAir 图编译 config                    | Eager 专用接口                               |
| **图模式 (Inductor + Triton)**        | op_statistic 含`triton_poi_fused_`/`triton_per_fused_`/`triton_unk_fused_` 前缀融合算子          | Inductor 环境变量、Triton 后端配置                           | GE 专用环境变量                              |
| **图模式 (Inductor + DVM)**           | op_statistic 含`dvm_` 前缀融合算子                                             | Inductor 环境变量、DVM 后端配置                              | GE 专用环境变量                              |
| **图模式 (有Dump图)**                 | 提供了 Dump 图文件 (pbtxt / output_code.py)                                      | 图模式环境变量、图编译 config                                | Eager 专用接口                               |
| **非图模式 (Eager)**                  | 无自动融合算子、无 ModelExecute、无 Dump 图文件，且 OP Name 全部含`aclnn` 前缀 | torch.compile 转图、手动多流 (torch_npu.npu.Stream)          | 图模式环境变量（MAX_RUNTIME_CORE_NUMBER 等） |
| **无法判断**                          | 无自动融合算子、无 ModelExecute、无 Dump 图文件，且不能确认是否 Eager            | 引导用户确认执行模式后再给建议                               | 不给具体环境变量或接口建议                   |

> **关键约束**：
>
> - 有自动融合算子 → **一定是图模式**
> - 走GE/TorchAir (有 ModelExecute) → **就是图模式**
> - 有 Dump 图文件 → **基本是图模式**
> - 其他无法判断时，**直接报"无法判断"，不要瞎猜**
> - 图模式下仍可能有部分 aclnn 算子（torch.compile 未完全覆盖），不影响图模式判定
> - 图模式环境变量（如 `MAX_RUNTIME_CORE_NUMBER`、`ENABLE_DYNAMIC_SHAPE_MULTI_STREAM`）**仅对图模式生效**，Eager 模式下无效。若当前为 Eager 模式，必须先建议转为图模式（如 `torch.compile`），再建议使用图模式环境变量。

### 规则3：实施示例必须标注来源

每条包含具体参数的建议，必须标注参数来源：

```
来源类型：
  - [昇腾官方文档]：URL 链接
  - [CANN 代码仓]：gitcode 链接
  - [Skill 内置知识]：本 SKILL.md 中的优化方向章节
```

### 规则4：参考链接清单

所有优化措施涉及的官方文档链接统一维护在本文档末尾「[参考链接](#参考链接)」区域。Agent 生成建议时应优先查阅该区域，正文以 `[名称]` 方式引用，不再内联完整 URL。

### 规则5：不可验证时的降级策略

当 Agent 无法联网验证某个参数时，按以下降级策略处理：

1. **降级为方向性建议**：不给具体参数，改为"建议查阅 [文档链接] 获取最新的环境变量配置"
2. **标注未验证**：在建议后标注 `(参数需查阅官方文档确认)`
3. **禁止臆造**：绝对不可编造环境变量名或参数值


## 脚本说明

### 脚本清单

| 脚本                            | 功能                                | 输入                            | 输出          |
| ------------------------------- | ----------------------------------- | ------------------------------- | ------------- |
| `scripts/profiling_parser.py` | Profiling 性能数据解析              | MindStudio Profiler 输出目录    | Markdown 报告 |
| `scripts/graph_analyzer.py`   | 图结构分析（支持 pbtxt 和 fxgraph） | 图文件 + Profiling 目录（可选） | Markdown 报告 |

### Profiling 分析

```bash
python scripts/profiling_parser.py <profiling_dir> -o profiling_report.md
```

### 图结构分析（可选）

脚本自动识别文件格式：`.pbtxt` 走纯文本解析，`.py` 走 AST 解析，无需手动指定。

```bash
# 结合 Profiling 分析（按总耗时排序）
python scripts/graph_analyzer.py <graph_file> --profiling-dir <profiling_dir> -o dump_report.md

# 仅分析图结构（无 Profiling 时，按算子数×重复次数排序）
python scripts/graph_analyzer.py <graph_file> -o dump_report.md
```

#### 图类型自动识别

| 图源         | 文件格式                                  | 识别方式                                           | 解析方法                                                                    |
| ------------ | ----------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------- |
| GE Build图   | `*.pbtxt`                               | 扩展名 + 内容关键词(`ir_version`/`graph {`)    | 纯文本解析，无需 protobuf 库                                                |
| PyTorch fx图 | `*output_code*.py` 或 `*_runnable.py` | 扩展名 + 内容关键词(`class Repro`/`torch.ops`/`def call`) | AST 解析，提取`forward`/`call` 中的 `torch.ops.<ns>.<op>.<overload>(...)` 调用，支持模块级 `def call`（无 class）、`with` 块内算子 |

#### 重复子图分析逻辑

- **子图比较**：使用算子类型（op_type）序列做 n-gram 比较。GE 图去掉 `ge:` 前缀；fxgraph 的 `op_type` 去掉 overload 后缀（`aten.mul.Tensor`→`aten.mul`），`raw_op_type` 保留全名用于展示
- **子图输出**：每个重复子图实例输出完整算子名（用于定位 dump 图位置）、OP Type、Input/Output Shape、Input/Output Dtype（pbtxt）或输入参数/源码行号/原始调用代码（fxgraph）
- **排序策略**：有 Profiling 按总耗时（avg_instance_time × repeat_count）降序；无 Profiling 按算子数×重复次数降序
- **去重策略**：同长度模式按 op_type 组成签名去重 + 跨长度模式按实例位置覆盖去重

#### Profiling 与图节点耗时匹配策略

图节点与 Profiling 算子耗时的匹配优先级（从高到低）：

| 优先级 | 策略           | 说明                                               | 适用场景                         |
| ------ | -------------- | -------------------------------------------------- | -------------------------------- |
| 1      | 算子名精确匹配 | Profiling OP Name == 图节点名                      | pbtxt + fxgraph                  |
| 2      | 算子名模糊匹配 | 包含关系                                           | pbtxt + fxgraph                  |
| 3      | aclnn 前缀匹配 | Profiling 中`aclnn{算子名}` → 图节点名          | fxgraph（aten IR → aclnn 对应） |
| 4      | 算子类型匹配   | op_type 平均耗时                                   | pbtxt + fxgraph                  |
| 5      | 前后算子名关联 | 用图节点的 producer/consumer 名在 Profiling 中搜索 | pbtxt + fxgraph                  |
| 6      | shape 匹配     | 用输入 shape 签名匹配                              | pbtxt                            |

#### 优化建议生成原则

优化建议**全部基于分析结果动态生成**，不硬编码特定算子对。`graph_analyzer.py` 聚焦重复子图结构分析，生成以下建议：

1. **重复子图融合建议**：基于 Top 10 重复子图，按重复次数和总耗时生成
2. **已有融合分析**：检测`*op_statistic*.csv`中"OP Type"是否含`autofuse_`、 `autofused_`、`triton_per`、  `triton_poi`、`triton_unk_fused`、`dvm_` 前缀。有则表示已开启 AutoFuse（GE/TorchAir/Inductor+AscendC/ATC）、Triton 融合（Inductor+Triton）或 DVM 融合（Inductor+DVM）；无则提示需结合 Profiling 进一步分析是否需要开启

## Profiling分析方法

### H2D耗时计算

**H2D分析方法：**

- 方法1：直接从api_statistic*.csv 中 `InputCopy` 统计；或者msprof_*.json中InputCopy；
- 方法2：如果方法1解析不到，估计是客户自己调用rtMemcpy/rtMemcpyAsync/rtMemcpyBatch/rtMemcpyBatchAsync处理的，所以如果方法1解析不到，则可通过msprof_*.json 分析统计memcpy函数耗时来计算；

**说明：**
1、优先使用方法1分析；
2、方法2分析H2D耗时时，需要结合迭代次数，计算单迭代H2D耗时；
3、H2D 是一次迭代中 ModelExecute前面的memcpy总耗时，包括多次函数调用的间隙；

### ModelExecute耗时计算

- `step_trace_*.csv` 或 `step_trace_time.csv`中的"Iteration Time(us)" 值求和，再除以 "Iteration Time(us)" 数量，得到"Iteration耗时"；或者`api_statistic_*.csv`文件中含 API Name="ModelExecute" 行对应的Avg(us)，即为"Iteration耗时"
- Ascend Profiler 的 `step_trace_time.csv` 无"Iteration Time(us)"列，使用"Computing"列作为NPU活跃耗时（迭代计算时间），同时保留 Stage(Wall)/Free 拆解展示NPU空闲占比。**注意：不要用"Stage"列作为迭代耗时，Stage包含Free(空闲)时间，会导致算子耗时比严重失真**
- `op_statistic_*.csv` 的"Total Time(us)" 值求和，再除以迭代次数，得到"Iteration算子耗时和"；或者从`op_summary_*.csv`或`kernel_details.csv`中"Task Duration(us)" 求和，再除以迭代次数，得到"Iteration算子耗时和"

#### 首次编译迭代排除

当获取到逐次迭代耗时且迭代次数 > 1 时，脚本自动检测首次迭代是否含编译耗时：

- **判断条件**：首次迭代耗时 > 后续迭代平均耗时的 **5x**
- **触发动作**：将首次迭代单独标为"首次编译迭代耗时"，不参与平均值计算
- **报告呈现**：核心指标汇总表新增一行"首次编译迭代耗时"，数据来源标注"已排除，不计入平均值"
- **不影响场景**：首次迭代耗时未达 5x 阈值时，正常计算全部迭代的平均值

#### 迭代次数推断（无step_trace时的回退策略）

当 step_trace 中有"Stage"和"Computing"列时（step_trace_time.csv格式），脚本逐行收集 Computing 值作为迭代NPU活跃耗时（每行一次迭代），迭代次数 = 行数。当 `step_trace` 和 `api_statistic` 均不可用时，脚本从 `op_summary`/`kernel_details` 推断迭代次数，优先级：

| 优先级 | 策略                     | 说明                                                         |
| ------ | ------------------------ | ------------------------------------------------------------ |
| 1      | api_statistic sync count | aclrtSynchronizeStream/Device 的 Count 值                    |
| 2      | Infer ID 唯一值数        | op_summary 含`Infer ID` 列，唯一值数 = 迭代次数            |
| 3      | Step Id 唯一值数         | kernel_details 含`Step Id` 列，唯一值数 = 迭代次数         |
| 4      | Op Name 众数出现次数     | GE Profiling: 非aclnn算子名在单次迭代中唯一，众数 = 迭代次数 |

> **注意**：kernel_details `Name` 和 op_summary `OP Type` 不能用于推断迭代数，因为这些实体在单次迭代内出现多次（如 MatMulV2 每迭代 101 个），众数 ≠ 迭代次数。
> 若迭代次数推断为0但有算子耗时数据，脚本自动设为1次迭代并用算子耗时和估算 `iter_avg`。

### D2H耗时计算

**D2H分析方法：**

- 方法1：直接从api_statistic*.csv 中 `OutputCopy` 统计；或者msprof_*.json中OutputCopy
- 方法2：如果方法1解析不到，估计是客户自己调用rtMemcpy/rtMemcpyAsync/rtMemcpyBatch/rtMemcpyBatchAsync处理的，所以如果方法1解析不到，则可通过msprof_*.json 分析统计memcpy函数耗时来计算；

**说明：**
1、优先使用方法1分析；
2、方法2分析D2H耗时时，需要结合迭代次数，计算单迭代D2H耗时；
3、D2H 是一次迭代中 ModelExecute后面的memcpy总耗时，包括多次函数调用的间隙；

### 如何判断是否含动态shape

- 根据op_summary*.csv 中的"OP State" 有dynamic，就表示有动态shape
- 根据msprof_*.json中有ModelExecute，且有infershape函数或者aclnn接口调用，就表示有动态shape

## 优化方向（3大方向）

### H2D和D2H优化

- 措施1：批量H2D/D2H。适用场景："多次H2D，调用间隙大"。预期收益：减少多次拷贝，带来的传输头开销。**约束：需修改业务侧代码，将多次 aclrtMemcpy 合并为 aclrtMemcpyBatch；图模式下可通过 GE 的 Data 算子合并输入**
- 措施2：Pinned内存。适用场景："每次H2D都做虚实地址转化"。预期收益：大幅提升H2D效率。**约束：PyTorch 场景通过 torch_npu.npu.PinMemPoolManager 或 aclrtMallocHost 分配 Pinned 内存；非通用环境变量，需代码改造**
- 措施3：Embedding层优化，减少Input大小。适用场景："同一个特征，重复传输，比如用户行为序列"。预期收益：减少Input数据量。**约束：需业务侧代码改造，合并重复特征的 H2D 传输**
- 措施4：异步H2D。适用场景："H2D占比大，比如10%+，且多次H2D"。预期收益：部分H2D和计算overlap，减少H2D占比。**约束：PyTorch 场景通过 torch_npu.npu.Stream 创建独立 stream 异步拷贝；GE 图模式通过多 Data 算子 + Stream 并行， 参数为“ge.compile.h2dOverlappedWithCompute=1”；需确保数据依赖正确**

### NN计算优化

- 措施1：算子自动融合。适用场景：vector占比大，算子数多，算子平均耗时短。预期收益：减少MTE搬运，减少调度次数，缓解调度bound。**约束：GE/ATC 图模式通过 `export AUTOFUSE_FLAGS="--enable_autofuse=true"` 开启，Reduce/Concat 融合需额外 `--autofuse_enable_pass=reduce,concat`；`--autofuse_enable_pass` 的可用值必须以 AutoFuse 官方文档为准（常见值：`reduce`、`concat`、`matmul`、`split`、`gather`、`transpose`、`scatter`、`slice`，其中matmul表示CV融合；其中910系列只支持reduce/concat，950系列往后支持所有参数值），**严禁臆造 pass 名（如 `pointwise` 不是合法值）**；PyTorch 场景通过 `torch.compile(model)` 开启 Inductor 编译，并通过环境变量 `export TORCHINDUCTOR_NPU_BACKEND=ascendc` 选择高性能 AscendC 后端（该环境变量必须在 `torch.compile()` 调用之前设置）；其他可选后端：`default`（Triton 模式）、`mlir`、`dvm`；Eager 模式无法直接开启，需先转图模式；参考：[AutoFuse]、[TORCHINDUCTOR_NPU_BACKEND]**
- 措施2：手写融合算子和Pass。适用场景：重复结构且耗时较大。预期收益：需要实测。**约束：GE 图模式通过自定义 Pass 注册（GE REGISTER_PASS）；AscendC 算子通过 Ascend C API 开发融合算子；需 CANN 算子开发能力。参考：[图Pass开发]**
- 措施3：单算子tiling key优化。适用场景：op_summary*.csv中的算子的aic_scalar_ratio 或 aiv_scalar_ratio 占比超过30%。预期收益：算子性能提升，显著降低scalar耗时占比。**约束：需联系算子开发团队修改 tiling 策略，非用户侧可配置；可通过 AscendC 修改算子的 tiling key 选择逻辑**
- 措施4：静态图下沉。适用场景：动态shape可有限分档或shape不变场景。预期收益：需要实测，提升会很大。**约束：GE 图模式通过 `ge.exec.dynamicImageSize` 或 `--dynamic_dims` (ATC) 配置分档；PyTorch 通过 `torch.compile(model, dynamic=False)` 转静态图；需确认 shape 确实可分档或固定**
- 措施5：多线程并行调度。适用场景：动态图执行场景。预期收益：减少host调度开销。**约束：环境变量 `export MAX_RUNTIME_CORE_NUMBER=3` 仅对图模式生效；TorchNPU，通过TASK_QUEUE_ENABLE=1，开启流水调度；配置后需配合绑核使用。参考：[多线程调度]**
- 措施6：调度线程绑核。适用场景：ARM CPU 动态调度耗时高场景。预期收益：~10%+，减少线程跨核切换。**约束：通过环境变量 `CPU_AFFINITY_CONF=<mode>,npu<id>:<start>-<end>` 配置 Host 侧 CPU 核绑定，mode=1 粗粒度/mode=2 细粒度，推荐细粒度；这是 CPU 侧绑核，与 AICore 控核（ge.aicoreNum）是不同的措施；图模式多线程调度时需配合在首次迭代前绑核。参考：[torch_npu绑核]**
- 措施7：混合调度。适用场景：动态图热点shape可静态分档，其他还走动态图执行场景。预期收益：减少host调度耗时。**约束：PyTorch 通过 torch.compile 配合 dynamic shape 分档实现；GE 图模式通过ge.compileHybridMMode或tfa参数compile_hybrid_mode设置；需确认热点 shape 可枚举**
- 措施8：AICPU算子转Aicore。适用场景：AICPU算子，且对应有等价Aicore可以替换。预期收益：显著提升算子性能，对应算子性能提升~50%。**约束：需 CANN 算子开发团队用 AscendC 重写 AICPU 算子为 AICore 实现；非用户侧可配置；通过 op_summary 中 Task Type=AI_CPU 识别目标算子**
- 措施9：静态图多流并行。适用场景：单实例场景，时延已超出业务阈值，但NPU使用率低，带宽使用率低。预期收益：CV并行，预期有5%左右收益；自动多流并行，预期有30%左右收益。**约束：图模式通过 `ge.autoMultistreamParallelMode` 开启，参考：[GE options]；Eager 模式需用 torch_npu.npu.Stream 手动多流或先转图模式。**

**Profiling判断:**

- `aic_scalar_ratio >= 30% or aiv_scalar_ratio >=30%` -> tiling key 选择问题，联系算子开发优化
- `cube_utilization < 20%` -> shape 太小，需要增大BS
- `aic_mac_ratio < 30%` -> 计算不是瓶颈，需要考虑自动或手动融合，减少搬运或调度
- Iteration算子耗时和 / Iteration耗时 < 90% -> 调度bound，需优化动态调度耗时
- `Task Type = AI_CPU` -> 即为AICPU算子

### 多实例并行

该措施适用于推荐推理，训练慎用。

- 措施1：多实例并行。适用场景：时延预算 > 单迭代耗时，且单实例下，NPU使用率低。预期收益：~2x吞吐提升，实际需实测。**约束：通过多线程创建多个推理实例实现；每个实例需独立的 Device 资源或通过时间片轮转共享；需确保显存容量充足。参考：[推荐推理最佳实践]**
- 措施2：AICore 控核。适用场景：在多实例并行下，需要配置算子编译时使用的 AICore 核数，防止实例之间抢占 AICore 资源激烈，导致 device 调度变长。预期收益：~20%，需实测，主要收益来自避免 AICore 资源竞争。**约束：GE 图模式通过 GE 图编译参数 `ge.aicoreNum` 配置（如 "8|8" 表示两个实例各分配 8 核 AICore）；PyTorch 场景当前通过 `torch_npu.npu.set_device_limit()` 和 `torch_npu.npu.set_stream_limit()` 接口实现控核，未来计划支持环境变量 `NPU_DEVICE_LIMIT`。参考：[GE options]、[推荐推理最佳实践]、[PyTorch环境变量]**


## 输入文件格式

MindStudio Profiler输出目录，脚本自动识别以下文件名：

| 数据类型                                         | 文件名模式                                                |
| ------------------------------------------------ | --------------------------------------------------------- |
| 算子耗时                                         | `*op_summary_*.csv` （含前缀）或 `kernel_details.csv` |
| 算子汇总                                         | `*op_statistic_*.csv`（含前缀）                         |
| Host API                                         | `*api_statistic_*.csv` 或 `api_statistic.csv`         |
| 迭代耗时                                         | `step_trace_*.csv` 或 `step_trace_time.csv`           |
| 迭代耗时和HostAPI耗时，算子耗时，H2D/D2H耗时汇总 | `msprof_*.json` 或 `trace_view.json`                  |

> **多文件选择**：当目录下有多个 `*op_summary*.csv`（如含 `_no_op_name`、`_output_` 后缀），脚本自动评分选取含 `Op Name` 列且数据非空的最优文件。

图结构分析输入文件：

| 图源         | 文件格式                                  | 解析方式                                                                                |
| ------------ | ----------------------------------------- | --------------------------------------------------------------------------------------- |
| GE Build图   | `*.pbtxt`                               | 纯文本解析，无需protobuf库；提取`node { name, op_type, input, output, shape, dtype }` |
| PyTorch fx图 | `*output_code*.py` 或 `*_runnable.py` | AST 解析，提取`class Repro.forward` 中的 `torch.ops.<ns>.<op>.<overload>(...)` 调用 |

> **多个图文件选择**：当前目录下有多个`*.pbtxt`，自动优选含`_Build`的文件，这是GE 编译后的图；优选`*output_code*.py` 文件。如果 `_Build.pbtxt` 文件有多个，则多个文件需要都分析。

## 输出报告格式

- 文件格式：Markdown。需要保障在各Markdown渲染器中都能正常显示。
- 输出路径：report 目录，如果无report目录，则创建；如无权限，则走Agent默认处理策略
- Profiling 报告（必选）：1）Profiling 文件发现、2）核心指标汇总表（H2D/ModelExecute/D2H 耗时及占比；含是否含动态shape、是否开启自动融合；若首次迭代含编译耗时则单独列出"首次编译迭代耗时"行）、3）算子类型分布表（含耗时占比）、4）Top 20 耗时算子表（含shape，dtype，aic/aiv 占比，mte 占比；需要根据算子类型+shape+dtype 去重）、5）问题算子汇总表（按 问题类型+OpName+OpType+TaskType+InputShapes+OutputShapes 聚合去重，列：问题类型 | Op Name | OP Type | Task Type | Input Shapes | Output Shapes | 重复次数/iter | 总耗时/iter(us)，Top 50 按总耗时降序）、6）Host耗时分析表（Host API耗时Top15, 包含Host API，Level，per-iter(us)， calls/iter，总耗时）
- 图分析报告（可选，提供图文件时生成）：1）基本信息（含图拓扑结构摘要：最大深度/宽度/分支因子/根叶节点数/环路检测）、2）重复子图分析表（含算子名/shape/dtype）+ Top 5 子图实例详情表（含算子名/OP Type/Shape/Dtype/输入参数/源码行号）
- 优化建议列表：Agent 根据脚本分析出的问题，匹配"优化方向"中的适用场景，结合昇腾社区资料给出优化建议。每条建议包含：对应的问题、适用的优化措施、触发条件、预期收益、实施示例（如开关配置、接口调用示例）。优化建议合入最终报告的最后一章

**重要规则：**

- 优化建议按优先级排序（高/中/低），每条建议包含：触发条件、预期收益、实施示例
- 优化建议全部基于分析结果动态生成，不硬编码特定算子对；
- 优化建议需要结合昇腾社区资料，给出实施示例，如开关配置示例，调用的接口示例等
- 如果有融合算子和pass的优化措施，需要给出示例代码，以便参考，可使用cannbot已有融算子和图pass开发skill处理
- 优化措施，必须真实存在，如参数，环境变量，配置，接口等必须真实存在，且使用方法正确
- 自动融合优化建议：如果没有图分析报告，也可根据 `*op_summary*.csv`的中算子执行序，给出建议；自动融合一般都支持CV融合（cube算子后面接Vector算子，Vector->Cube不支持），VV融合（vector算子间融合），norm类算子融合，reduce类一般支持前向融合（即reduce在后面）；支持搬运类（gather（含Embedding）、transpose、split/slice/strideslice、concat）、reduce类、matmul/batchmatmul类、eltwise类、broadcast类等融合，其中除了matmul/batchmatmul类是cube算子，其他都是vector算子。`--fusion_switch_file` 不是自动融合配置，是内置手写pass开关配置。
- 静态shape图下，不能出现CPU绑核、多线线程调度等动态图下的优化措施 
- 优化措施如果已经实施，需要确认是否还有进一步优化空间；如果有，则给出进一步优化空间；如果无，则不在优化措施中显示。融合算子名中有autofuse_**_mm_** 或 Catlass，说明已经开启了CV融合；其他已开启的融合能力，可从自动融合算子名中，猜出，一般自动融合名格式为“前缀_算子类型1_算子类型2_..._算子类型N”
- 如果判断是否可能已经做了控核：当前算子使用的最大核数和实际最大物理核数比，如果前者小于后者，则可能已做控核；如果等于，则未控核。算子使用最大核数计算方法：按任务类型获取算子最大使用核数，任务类型是  `*op_summary*.csv`  中"Task Type" 或   `*kernel_details*.csv` 中获取“Accelerator Core”，核数是 "Block Dim" 或 "Block Num"， 然后按任务类型，找到不同任务类型的最大使用核数。注意AI_VECTOR_CORE/AI_CORE 可分别控核，所以需要分别计算并判断。是哪款芯片，profiling和dump图无法分析，需要plog或者询问用户，还是无法获取哪款芯片多少最大核数，则控核优化措施需要标注为[可能的优化措施]
- 优化措施，需要区别是TFA/GE/ATC, Torch inductor, torchair等

## 参考链接

正文以 `[名称]` 方式引用，完整链接见下表：

| 名称                        | 链接                                                                                                                                                              |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [AutoFuse]                  | https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/graphdevg/autofuse_1_0004.html                                                        |
| [TORCHINDUCTOR_NPU_BACKEND] | https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/ENV/docs/zh/environment_variable_reference/TORCHINDUCTOR_NPU_BACKEND.md                         |
| [多线程调度]                | https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/envvar/envref_07_0034.html                                                           |
| [动态shape多流并发]         | https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/envvar/envref_07_0033.html                                                           |
| [环境变量总览]              | https://www.hiascend.com/document/detail/zh/canncommercial/latest/maintenref/envvar/envref_07_0001.html                                                           |
| [推荐推理最佳实践]          | https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/graphdevg/atlasag_25_0101.html                                                        |
| [图Pass开发]                | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/graphdevg/docs/zh/user_guides/graph_dev/custom_pass_development/introduction.md |
| [torch_npu绑核]             | https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/ENV/docs/zh/environment_variable_reference/CPU_AFFINITY_CONF.md                                 |
| [PyTorch环境变量]           | https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/ENV/docs/zh/environment_variable_reference/env_variable_list.md                                 |
| [GE options]                | https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendgraphapi/atlasgeapi_07_0150.html                                                      |
| [CANN代码仓]                | https://gitcode.com/cann                                                                                                                                          |
| [Ascend代码仓]              | https://gitcode.com/Ascend                                                                                                                                        |
| [CANN社区]                  | https://www.hiascend.com/cann                                                                                                                                     |
