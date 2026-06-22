# NPU 模型推理端到端优化工作流

## 概述

本工作流编排三个专业化 subagent（model-infer-analyzer / model-infer-implementer / model-infer-reviewer）对目标模型按阶段执行 NPU 推理优化。

```
阶段 0: 模型分析与建立基线
阶段 1: 并行化改造（多卡部署时）
阶段 2: KVCache + FA
阶段 3: 融合算子
阶段 4: 量化适配改造（如阶段 0 已完成量化初评估且用户启用）
阶段 5: 图模式适配
阶段 6: 优化总结
```

每个优化阶段遵循统一流程：分析 → 方案确认 → 实施 → 验证 → 阶段总结。

---

## 重要原则

- **严格按阶段流程执行**：逐阶段推进，每阶段完成分析→用户确认→实施→验证→总结的完整流程后才能进入下一阶段，不可跳过或并行
- **保持模型完整性**：不为通过验证而简化模型实现、删减功能或降低优化标准
- **主 agent 只做编排**：主 agent 负责派发 subagent、呈现报告、用户确认，不直接修改模型代码或自行实施优化。FAIL 后必须派发 implementer 修复，不能自己改代码
- **implementer 自验证检查**：implementer 返回后，读 progress.md 自验证 section，确认五项（参考 skill、代码加载、编译、推理、输出）完整且与常驻区环境一致（如常驻区有 NPU 环境则推理项不应为空）。缺失或矛盾 → 拒绝重派。不自行审查代码替代验证
- **验证前提检查**：任何验证结果的前提是被测代码确实被执行。验证前先确认修改后的代码被加载和走到（如检查日志中的模型路径、关键优化标记）
- **异常把控**：主 agent 对各阶段 subagent 报告中的异常保持敏感（分析结论不合理、性能数据与改动预期不符、精度异常等），不接受未经调查的报告，要求 subagent 重新调查。输出不可读（重复 token、乱码、空文本、全 EOS）是硬 FAIL，不可降级——性能优化不改变计算正确性；reviewer 报告 PASS 但主 agent 审核发现硬指标异常时，应判 FAIL 或向用户确认
- **subagent 派发规范**：dispatch prompt 严格只包含模板代码块内的字段和占位符，除用户明确要求外，不以任何形式附加上下文（如分析结论、技术方案、实施流程、部署配置等）。subagent 通过读取 progress.md 获取上下文，主 agent 不在 dispatch 中转述

---

## 共享状态文件

> **agent 产物位置约定**：所有 agent 流程产物（progress.md / progress_history.md / optimization_report.md / baseline/ 等）统一归入模型目录下的 `agentic/` 子目录，模型根目录只保留模型本身的代码、配置与入口（modeling / config / infer.sh / requirements / README）。下文中所有 `progress.md` / `baseline/` 等裸引用均指 `{model_dir}/agentic/` 下的同名文件。

`{model_dir}/agentic/progress.md`：常驻区（阶段 0 分析 + 进度概览表）+ 工作区（当前阶段记录）。初始模板见 `templates/progress_template.md`。

`{model_dir}/agentic/progress_history.md`：历史归档。除阶段 6 优化总结外，默认仅 Grep 查找；阶段 6 允许一次性 Read 全文用于生成总报告。

**读写规则**：常驻区由 阶段 0 写入，后续只有主 agent 更新概览表。工作区由各 subagent 追加，写入前先读取现有内容。

**阶段推进**：每阶段验证通过后，主 agent 更新概览表 → 调用 `scripts/archive_progress.py` 归档工作区 → 清空工作区。阶段 0 不归档。

---

## 工作流程

### 阶段 0：模型分析与建立基线

#### 0.1 信息收集

若用户未提供以下信息，主 agent 使用提问工具向用户确认：
- **模型工作目录**：模型代码所在路径（如 `cann-recipes-infer/models/xxx`）
- **模型来源**：HuggingFace 链接、本地权重路径、或仓库内已有
- **权重路径**：已下载的权重位置（如未下载可后续处理）
- **部署模式**：默认 `框架部署`（cann-recipes-infer 仓库主推，接入 `executor/core/`）。仅在用户明确表示要在仓外使用 / 不接框架时改为 `独立部署`

#### 0.2 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
任务: 模型架构全面分析
模型来源: {HuggingFace 链接 / 本地路径 / 仓库内已有}
分析内容:
  - 架构类型（LLM / MoE / Diffusion / 多模态）
  - 网络结构拆解（Embedding → Transformer Blocks → Output Head）
  - Prefill / Decode 分支差异
  - 关键模块：Attention 类型（GQA/MHA/MLA）、FFN/MoE 结构、特殊模块。架构识别必须基于实际 config 值（config.json / model.config），不能仅从代码类定义推断——注意可配置开关（如 use_mla、n_routed_experts）
  - 运行环境：通过 `asys info -r=status` 确认 NPU 型号（运行前需 source CANN 包路径：source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash）；通过 `npu-smi info` 确认单卡 HBM 容量，记录量化模式、执行模式、部署卡数
  - 模型当前状态：确认代码是否存在且可运行（有 infer.sh 且能跑通）、agentic/baseline/baseline_metadata.json 是否存在。报告状态（可运行/不可运行/需多卡），不自行采集基线数据
  - 若模型不可运行，记录具体原因和缺失项
输出:
  - 使用 templates/progress_template.md 创建 {model_dir}/agentic/progress.md，将分析结果写入常驻区（模型信息、并行策略、进度概览）
  - 使用 templates/optimization_report_template.md 初始化 {model_dir}/agentic/optimization_report.md
```

#### 0.3 分析确认与状态分流

主 agent 将 analyzer 返回的分析结果呈现给用户确认，根据模型状态确定路径：

- **a. 模型可运行** → 进入 0.4 采集基线
- **b. 模型无法运行（代码缺失或适配不完整）** → 进入 0.4 框架适配
- **c. 模型需多卡部署（单卡显存不足）** → 进入 0.4 或直接进入阶段 1

#### 0.4 框架适配与基线建立

根据 0.3 确定的路径派发 implementer：

**路径 a（模型可运行）**：

```
必须使用 skill: model-infer-migrator
工作目录: {model_dir}
部署模式: {框架部署 / 独立部署}
任务: 部署基线采集
```

implementer 返回后，提取 baseline_metadata.json 摘要写入 progress.md 常驻区 → 进入 0.5 量化候选确认；若用户不启用量化，则进入阶段 2

**路径 b（模型无法运行）**：

```
必须使用 skill: model-infer-migrator
工作目录: {model_dir}
部署模式: {框架部署 / 独立部署}
任务: 框架适配 + 部署基线建立
模型来源: {HuggingFace 链接 或 本地路径}
权重路径: {如已知}
```

implementer 返回后：
- 若输出 baseline_metadata.json → 提取摘要写入 progress.md 常驻区 → 进入 0.5 量化候选确认；若用户不启用量化，则进入阶段 2
- 若标记"需多卡" → 进入阶段 1 并行化

**路径 c（模型需多卡部署）**：
- 若代码已存在（仓库内已有框架适配的模型）→ 进入 0.5 量化候选确认；若用户不启用量化，则进入阶段 1，并行基线在并行跑通后建立
- 若代码不存在 → 先按路径 b 派发 migrator 完成单卡框架适配（完整的 modeling + configuration + YAML + infer.sh，并注册到 cann-recipes-infer/executor/core/support_models.py，作为并行化改造的代码基础），migrator 标记"需多卡"后进入 0.5 量化候选确认；若用户不启用量化，则进入阶段 1

#### 0.5 量化候选确认与方案初评估

> 本步骤属于阶段 0 的扩展分析，只评估量化方案，不修改代码。
> 量化初评估结论作为后续并行、融合、图模式和量化改造的参考输入。
> 若用户不启用量化候选，则跳过本步骤和阶段 4 量化适配改造，直接进入后续非量化优化阶段。

阶段 0 完成模型分析、且模型代码具备可读的框架适配基础后，主 agent 先向用户确认是否启用量化候选。

1. 若用户不启用量化，记录决策，跳过 0.5 和阶段 4。
2. 若用户启用量化，继续确认是否已有 `quant_export_dir`。
3. 若已有量化产物，用户提供 `quant_export_dir`，进入量化方案初评估。
4. 若尚无量化产物，记录“量化产物未交付”，跳过 0.5 和阶段 4；用户后续交付 `quant_export_dir` 后，可从 0.5 重新进入量化方案初评估。在产物返回前，主流程不得进入量化改造。

`quant_export_dir` 是 infer 侧消费的上游量化产物目录，不是模型工作目录；它可以来自用户已有产物、AMCT 或其它量化产物生成流程的输出。

量化初评估前，主 agent 需确认用户已交付完整量化产物目录。典型交付物包括量化配置文件（如 `config.json`）、权重索引文件（如 `model.safetensors.index.json`）、量化权重文件（如 `*.safetensors`）和量化交付说明（如 `deploy_quantization.md`）。具体契约由 `model-infer-quantization` skill 检查；缺关键交付物时，不派发改造，只要求补充量化算法或产物契约。

派发 model-infer-analyzer：

```
工作目录: {model_dir}
必须使用 skill: model-infer-quantization
任务: 量化方案初评估
量化产物目录: {quant_export_dir}
分析内容:
  - 检查量化产物契约、结构匹配、接入分级和部署形态估算
  - 按模型结构判断，不按模型名称套用参考
  - 不修改模型代码，不采集基线
  - 若契约不满足，输出量化算法或产物契约补充诉求
```

主 agent 执行：

1. 若量化初评估报告明确“无法安全接入”“需补充量化产物契约”或“建议暂不量化”，记录补充诉求，后续阶段按非量化路径推进。
2. 若量化初评估报告明确可接入，写入 progress.md 常驻区，至少包含契约结论、结构参考卡、接入分级、显存/部署判断、对后续阶段的影响；若评估显示量化后可能收敛卡数，只作为并行决策输入，不在本步骤修改代码。
3. 向用户确认是否保留量化改造候选；若保留，则后续阶段在做并行、融合、图模式决策时参考该量化信息。

---

### 阶段 1：并行化改造

> 单卡模型跳过本阶段。并行策略影响后续所有阶段的代码结构（通信组、TP 切分、EP 路由），必须先于 KVCache/FA 改造完成。
> 本阶段仍按非量化部署路径实施和验证；阶段 0.5 的量化初评估只作为显存、卡数和后续量化候选的参考输入。

#### 1.1 确认部署需求

主 agent 使用提问工具向用户确认：

1. **部署卡数和节点配置**：总卡数、每节点几卡（影响 TP 上限）
2. **目标场景**：高吞吐 / 低时延 / 均衡
3. **实际序列长度**：决定是否需要 CP / KVP
4. **batch size 需求**（如有）

确认后结合 阶段 0 的模型参数、硬件信息和量化初评估（如有）做快速可行性检查：
- 卡数 × 单卡显存是否能容纳模型参数；若 progress.md 已有量化初评估，同时给出浮点口径和量化口径
- MoE 模型的专家数是否 ≥ 卡数（否则 EP 不可行）
- 序列长度 × KV Cache 是否超出总显存

有矛盾则向用户反馈，调整卡数、batch size 或序列长度后再派发分析。

#### 1.2 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
必须使用 skill: model-infer-parallel-analysis
任务: 并行策略分析（至少包含以下内容）
部署需求:（填入 1.1 确认的结果）
分析内容:
  - 提取模型参数和模块链路
  - 基于单套 parallel_config 分析整体并行策略，结合目标场景权衡 Prefill/Decode
  - 定量估算（显存、通信量），确定 parallel_config 具体值
  - 输出 2-3 个候选方案并排序
```

#### 1.3 方案确认

主 agent 使用提问工具向用户确认关键决策：

1. **整体并行策略**：纯 TP / MoE EP / 模块级差异化并行？
2. **各模块 TP 度**：attn_tp / dense_tp / moe_tp / embed_tp / lmhead_tp / oproj_tp
3. **长序列附加配置**：是否引入 CP / KVP？
4. 其他需确认的细节（AFD、EPLB 等进阶配置）

用户确认后进入实施。

#### 1.4 启动实施 subagent

派发 model-infer-implementer：

```
工作目录: {model_dir}
必须使用 skill: model-infer-parallel-impl
部署模式: {框架部署 / 独立部署}
任务: 并行化改造（至少包含以下内容）
阶段要点:
  - 按已确认的 parallel_config 实施
  - 通信组创建 → 逐模块并行层替换 → Embed/LMHead 并行 → YAML 配置 → 权重处理
自验证:
  - 编译通过、多卡推理无 crash、吐字正常（可读、不重复、非全零、不提前 EOS）
  - 实施记录写入 progress.md 工作区
```

> **验收 gate**：执行"implementer 自验证检查"（见重要原则）。不通过 → 拒绝重派。

#### 1.5 启动验证 subagent

派发 model-infer-reviewer：

```
工作目录: {model_dir}
部署模式: {框架部署 / 独立部署}
任务: 并行化改造后验证
约束: 禁止修改模型代码和自行调试，仅返回验证报告
至少包含以下验证:
验证内容:
  - 精度: 运行多卡推理，输出与基线对比
  - 性能: 运行推理，对比改造前后的 Prefill 耗时和 Decode 单步耗时
  - 结果写入 progress.md 工作区（精度验证 + 性能验证 section）
检查项:
  - parallel_config 各参数已正确实施（YAML 配置与代码一致）
  - 各 rank 吐字正常（可读、不重复、非全零、不提前 EOS）
```

#### 1.6 Profiling 策略校准（TODO：待适配）

> 当前暂不可用，跳过此步骤。后续适配后启用。

派发 model-infer-analyzer：

```
工作目录: {model_dir}
必须使用 skill: model-infer-parallel-analysis（第五步 Profiling 校准）
任务: 并行策略 profiling 校准
分析内容:
  - 通信占比是否合理（< 20%）
  - 各 Rank 耗时是否均衡（MoE EP 场景）
  - 显存峰值是否与估算一致
  - 若偏差过大，给出调整建议
```

#### 1.7 阶段总结

主 agent 执行：
1. reviewer 报告 FAIL → 派发 model-infer-implementer 修复 → 重新验证，最多 5 轮
2. profiling 校准发现策略问题或性能未达预期 → 回到 1.2 调整 parallel_config 重新确认（TODO：待 profiling 适配后启用）
3. 若 阶段 0 标注"无基线"（模型需多卡才能运行），并行验证通过后派发 model-infer-implementer 采集基线：

    ```
    必须使用 skill: model-infer-migrator
    工作目录: {model_dir}
    部署模式: {框架部署 / 独立部署}
    任务: 部署基线采集
    ```

    implementer 返回后，提取 baseline_metadata.json 摘要写入 progress.md 常驻区。后续阶段的性能对比以此次采集的基线为准，忽略并行验证阶段的性能数据。
4. 全部通过后，综合结果输出阶段总结报告
5. 向用户确认当前阶段优化，确认后提交 commit，进入下一阶段

---

### 阶段 2：KVCache 静态化 + FA 算子替换

> 框架部署模式默认推荐路径：**PA + FA + TND**（标准 LLM）/ **PA + FA + TND_NTD + MLA absorb**（MLA 模型）/ **PA + FA + TND + 滑窗约束**（含滑窗层）。migrator 阶段产出的 Legacy KV + HF 原版 SDPA + BSH 是过渡骨架，本阶段主要负责升级到默认推荐路径。具体选型由 analyzer 输出，详见 `docs/design/kv_cache_design.md` 与 model-infer-kvcache skill。
> 后续阶段 3 的融合算子优化专注于 Attention Core 之外的模块。
> 本阶段仍按非量化部署路径实施和验证，不切换量化权重或量化运行配置。

#### 2.1 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
部署模式: {框架部署 / 独立部署}
必须使用 skill: model-infer-kvcache（关注第一层快速选型 + 第二层数据结构与算子）
任务: KVCache 模式分析和选型
分析内容:
  - 框架部署默认走 PA + FA + TND（MLA 用 TND_NTD）；评估是否需要偏离默认
  - 确定每个 cache 的 attn_type（FullAttention / SlidingWindow）
  - 是否启用 MLA absorb、是否多 attn_type 混合
  - FA 算子版本（v1 / v2）与 layout 选择
  - 估算块数 / 单卡显存占用
```

#### 2.2 方案确认

主 agent 使用提问工具向用户逐条确认关键决策（框架部署默认推荐 PA + FA + TND，仅在架构限制 / 算子不兼容 / 特殊性能目标下偏离）：

1. **KVCache 模式**：默认 PA；仅在特殊场景回退连续缓存
2. **attn_type**：FullAttention / SlidingWindow（含混合层场景）
3. **MLA absorb 路径**（仅 MLA 模型）：是否启用 absorb（影响 FA 入参和 rope 处理）
4. **FA 算子 + layout**：v1 / v2；标准 LLM 默认 TND，MLA 默认 TND_NTD
5. 其他偏离默认的方案细节（如 KVP / 多 attn_type 混合等）

用户确认后进入实施。

#### 2.3 启动实施 subagent

派发 model-infer-implementer：

```
工作目录: {model_dir}
部署模式: {框架部署 / 独立部署}
必须使用 skill: model-infer-kvcache（关注第三层实施流程对应大节）
任务: 阶段 2 KVCache + FA 改造（至少包含以下内容）
阶段要点:
  - 按已确认的 KVCache 模式 / attn_type / FA 算子方案实施
  - 框架部署：cache_entries 配置 + get_cache_info 实现 + attention forward 内 block_table[self.attn_type] / slot_mapping[self.attn_type] 字典访问
  - 独立部署：Runner 自管 KV + block_table / slot_mapping + ForwardMetaData 构造
  - 写入算子用 npu_scatter_nd_update_（MLA 用 npu_kv_rmsnorm_rope_cache 融合）
  - 阶段分支用 forward_metadata.is_prefill；hidden_states 全程保持 [TotalTokens, hidden_size] 二维（不要 reshape 成 BS）
自验证:
  - 编译通过、推理无 crash、吐字正常（可读、不重复、非全零、不提前 EOS）
  - 实施记录写入 progress.md 工作区
```

> **验收 gate**：执行"implementer 自验证检查"（见重要原则）。不通过 → 拒绝重派。

#### 2.4 启动验证 subagent

派发 model-infer-reviewer：

```
工作目录: {model_dir}
部署模式: {框架部署 / 独立部署}
任务: KVCache + FA 改造后验证
约束: 禁止修改模型代码和自行调试，仅返回验证报告
至少包含以下验证:
验证内容:
  - 精度: 运行推理，Prefill/Decode 输出与基线对比
  - 性能: 运行推理，对比改造前后的 Prefill 耗时和 Decode 单步耗时
  - 性能验证: 若工作目录下有 agentic/baseline/baseline_metadata.json，用它作为性能对比基准
  - 结果写入 progress.md 工作区（精度验证 + 性能验证 section）
检查项:
  - KVCache 模式选型、静态化实现及 FA 算子替换已完成（框架部署：cache_entries + get_cache_info + forward 接入 block_table[attn_type]；独立部署：Runner 自管 KV + 自管 forward_metadata）
  - Prefill 和 Decode 阶段缓存逻辑差异已正确处理（阶段分支用 forward_metadata.is_prefill）
  - hidden_states 全程保持 [TotalTokens, hidden_size] 二维形态
```

#### 2.5 阶段总结

主 agent 执行：
1. reviewer 报告 FAIL → 派发 model-infer-implementer（调试 KVCache/FA 精度问题，使用 model-infer-precision-debug skill）→ 重新派发 model-infer-reviewer 验证，最多 5 轮
2. 5 轮仍未解决 → 回退问题模块或整阶段改动 → 向用户报告阻塞点，请求决策
3. 调试发现需更换 KVCache 方案 → 回到 2.3 重新实施
4. 精度达标但性能未提升 → 派发 model-infer-analyzer（排查性能问题：部署配置、前置处理开销、测试方法、NPU 利用率等）→ 将分析和建议呈现给用户决策
5. 全部通过后，综合 analyzer/implementer/reviewer 的结果，输出阶段总结报告
6. 向用户确认当前阶段优化，确认后提交 commit，进入下一阶段

---

### 阶段 3：融合算子优化

> 若阶段 2 已完成 FA 算子替换，则本阶段跳过 FA 算子本身，但 Attention 子链路（RoPE 融合、KV write 融合、QK Norm 等）仍需分析和优化。
> 本阶段仍按非量化部署路径实施和验证；融合结果会在阶段 4 复核是否兼容量化。

#### 3.1 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
必须使用 skill: model-infer-fusion（关注分析匹配部分，步骤 1-4）
任务: 融合算子匹配分析
分析内容:
  - 拆解模型各模块，识别可替换的计算模式
  - 匹配仓库已有模型的融合算子用法
  - 若阶段 2 已完成 FA 替换（见 progress.md），跳过 FA 算子本身，但 Attention 子链路仍需分析
  - 覆盖所有模块：Attention 子链路（RoPE、KV write、QK Norm）、MoE、FFN、Norm 等
  - 输出候选替换清单（原算子 → NPU 融合算子 → 替换理由）
```

#### 3.2 方案确认

主 agent 使用提问工具向用户分模块确认关键决策：

1. **Attention 子链路替换**：RoPE 融合、KV write 融合、QK Norm 等（注意：仅跳过已在阶段 2 完成的 FA 算子，子链路仍需分析）
2. **MoE / FFN 模块替换**：MoE routing、grouped_matmul、激活函数融合等
3. **Norm / 其他模块替换**：RMSNorm、残差流融合等
4. **跳过模块的理由**：是否认可各跳过理由？（不能仅因"改动大"跳过）
5. 其他需用户确认的方案细节（如替换优先级、特殊算子参数选择等）

用户确认后进入实施。

#### 3.3 启动实施 subagent

派发 model-infer-implementer：

```
工作目录: {model_dir}
必须使用 skill: model-infer-fusion（关注实施替换部分，步骤 5）
任务: 阶段 3 融合算子优化（至少包含以下内容）
阶段要点:
  - 若阶段 2 已完成 FA 替换（见 progress.md），跳过 FA 算子本身，Attention 子链路仍需优化
  - 覆盖所有模块：Attention 子链路、MoE、FFN、Norm 等
自验证:
  - 编译通过、推理无 crash、吐字正常（可读、不重复、非全零、不提前 EOS）
  - 实施记录写入 progress.md 工作区
```

> **验收 gate**：执行"implementer 自验证检查"（见重要原则）。不通过 → 拒绝重派。

#### 3.4 启动验证 subagent

派发 model-infer-reviewer：

```
工作目录: {model_dir}
任务: 融合算子替换后验证
约束: 禁止修改模型代码和自行调试，仅返回验证报告
至少包含以下验证:
验证内容:
  - 精度: 运行推理，每个替换模块独立对比替换前后的输出
  - 性能: 运行推理，整体 Prefill/Decode 耗时对比替换前
  - 性能验证: 若工作目录下有 agentic/baseline/baseline_metadata.json，用它作为性能对比基准
  - 结果写入 progress.md 工作区（精度验证 + 性能验证 section）
检查项:
  - 所有模块的分析与替换决策已完成
  - 每个替换模块均有精度和性能对比结果
  - 跳过的模块有硬约束理由（不能仅因"改动大"跳过）
```

#### 3.5 阶段总结

主 agent 执行：
1. reviewer 报告 FAIL → 派发 model-infer-implementer（修复对应模块的融合算子精度问题）→ 重新派发 model-infer-reviewer 验证，最多 5 轮
2. 5 轮仍未解决 → 回退问题模块或整阶段改动 → 向用户报告阻塞点，请求决策
3. 精度达标但性能未提升 → 派发 model-infer-analyzer（排查性能问题：部署配置、前置处理开销、测试方法、NPU 利用率等）→ 将分析和建议呈现给用户决策
4. 全部通过后，综合 analyzer/implementer/reviewer 的结果，输出阶段总结报告
5. 向用户确认当前阶段优化，确认后提交 commit，进入下一阶段

---

### 阶段 4：量化适配改造

> 本阶段在融合算子改造后、图模式适配前执行。图模式应尽量捕获最终 dtype、layout、kernel 和 cache 路径，因此量化改造优先先于图模式完成。
> 若阶段 0 未提供量化产物或用户暂不启用量化，则跳过本阶段进入图模式适配。
> 本阶段只接入已交付的量化方案和权重，不重新设计上游量化算法，不擅自扩展未确认的量化路线。
> 本阶段严格按既定量化方案和量化权重验证，不因报错擅自修改量化方案；若融合算子不支持量化，按用户确认的回退原则处理。
> 阶段 4 实施并验证通过前，当前默认部署路径仍是非量化融合后模型；只有用户接受量化实测结果后，量化路径才作为阶段 5 输入。

#### 4.1 量化接入前变更复核

本步骤只判断前置阶段的后续改动是否破坏量化初评估结论，不重做完整初评估。主 agent 先读取量化初评估报告：

- 若量化初评估报告缺失或已过期，回到 0.5 重新做量化方案初评估，不在本步骤补做完整分析。
- 若量化初评估报告可用，派发 model-infer-analyzer 做轻量变更复核：

```
工作目录: {model_dir}
必须使用 skill: model-infer-quantization
任务: 量化接入前变更复核
量化产物目录: {quant_export_dir}
分析内容:
  - 快速复核前置阶段相对量化初评估的变化是否影响量化接入
  - 聚焦权重加载、runtime object、模块前缀、dtype/layout/cache、post-load 和融合回退决策
  - 若无影响，明确沿用量化初评估结论
  - 不重做完整初评估，不修改模型代码，不采集基线
```

若变更复核明确不满足量化条件，本阶段不派发 implementer，只输出阻塞点和补充诉求，随后由用户确认是否跳过量化进入阶段 5。

#### 4.2 改造前用户决策

主 agent 基于量化初评估报告、量化接入前变更复核结论（如有）和量化前最新基线，将复核后的最新量化方案呈现给用户确认。

若用户暂不量化或选择保留融合并跳过量化，记录原因，跳过到阶段 5。
若用户选择停止并补量化方案，输出阻塞点、融合量化需求和产物契约诉求，不继续后续阶段。
若用户确认继续量化改造，进入实施。

#### 4.3 启动实施 subagent

仅当用户选择“继续量化改造”时派发 model-infer-implementer：

```
工作目录: {model_dir}
必须使用 skill: model-infer-quantization
任务: 阶段 4 量化适配改造与量化权重加载
量化产物目录: {quant_export_dir}
量化前最新基线: {latest_baseline}
任务要点:
  - 按用户确认的量化路线接入配置、权重、runtime 映射和 post-load 处理
  - 严格使用既定量化方案和量化权重，不因报错修改 target / ignore / 张量语义
  - 若融合算子不支持量化，按用户确认的回退原则处理，并记录清单
  - 获取量化部署基线；若量化后单卡可满足，按确认部署口径验证
自验证:
  - 量化模型可加载，Prefill / Decode 跑通，输出可读且有量化路径生效证据
  - 融合算子回退清单、原始错误和后续融合量化需求已记录
```

> **验收 gate**：执行"implementer 自验证检查"（见重要原则）。不通过 → 拒绝重派。

#### 4.4 启动验证 subagent

派发 model-infer-reviewer：

```
工作目录: {model_dir}
任务: 阶段 4 量化适配验证
约束: 禁止修改模型代码和自行调试，仅返回验证报告
至少包含以下验证:
验证内容:
  - 量化权重确已加载，且日志或状态文件能证明量化路径真实生效
  - Prefill / Decode 至少跑通一次，输出可读、不重复、非全零、不提前 EOS
  - 精度: 运行推理，对比量化前后的输出可用性和关键误差
  - 性能: 运行推理，记录量化部署基线；与非量化基线和量化前最新基线对比
  - 若量化后可单卡部署，则以单卡部署结果作为主要验证口径
检查项:
  - 每个量化模块均有生效证据
  - 回退到原有非融合算子的模块已显式记录
  - 无擅自修改量化方案、target、ignore 或权重张量语义的行为
  - 量化初评估报告和量化接入前变更复核结论中的冲突项已给出处理方式
```

#### 4.5 实测后用户决策与阶段总结

主 agent 执行：
1. reviewer 报告 FAIL → 派发 model-infer-implementer（修复量化权重加载、映射或精度问题）→ 重新派发 model-infer-reviewer 验证，最多 5 轮
2. 5 轮仍未解决 → 回退问题模块或整阶段改动 → 向用户报告阻塞点，请求决策
3. 精度达标但性能未提升 → 派发 model-infer-analyzer（排查量化路径是否真实生效、部署配置、测试方法、NPU 利用率等）→ 将分析和建议呈现给用户决策
4. 量化基线产出后，向用户呈现实测结果和三种后验决策：采用当前量化方案继续进入图模式 / 保留融合算子并跳过量化进入图模式 / 修正量化方案后迭代验证
5. 若用户接受当前量化方案，将量化部署基线、量化后部署口径、融合算子回退清单作为阶段 5 图模式适配输入
6. 若用户选择跳过量化，记录跳过原因和实测收益；进入阶段 5 前必须确保运行配置和代码路径回到非量化融合后模型，不把量化改动作为默认执行路径
7. 若用户选择修正量化方案后迭代验证，记录当前阻塞点和补充诉求，等待新量化产物后回到 4.1 复核
8. 综合输出量化改造方案、改造要点、问题及解决方案、融合算子量化需求和收益结论
9. 将可复用经验沉淀到 `model-infer-quantization/references/quantization-structure-cards.md`；只有新增结构才创建新结构卡
10. 向用户确认当前阶段优化，确认后提交 commit，进入下一阶段

---

### 阶段 5：图模式适配优化

> 图模式基于阶段 4 之后用户接受的执行路径适配：若接受量化，则验证量化路径下的 Decode 图模式；若跳过量化，则验证非量化融合后路径。

#### 5.1 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
必须使用 skill: model-infer-graph-mode（关注分析和方案设计部分）
任务: 图模式适配方案分析
分析内容:
  - 分析模型中的图中断点（dynamic shape、数据依赖控制流等）
  - 评估图模式适配方案（npugraph_ex / GE 图模式）
  - 若阶段 4 已接受量化方案，基于量化 dtype、layout、kernel 和 cache 路径分析图中断点
  - 图模式仅适用于 Decode 阶段，Prefill 禁止使用
```

#### 5.2 方案确认

主 agent 使用提问工具向用户逐条确认关键决策：

1. **图模式后端**：npugraph_ex / GE 图模式？
2. **图中断点处理**：analyzer 识别的图中断点及解决方案是否认可？
3. 其他需用户确认的方案细节（如 mark_static 处理、编译缓存策略等）

用户确认后进入实施。

#### 5.3 启动实施 subagent

派发 model-infer-implementer：

```
工作目录: {model_dir}
必须使用 skill: model-infer-graph-mode（关注实施和代码改造部分）
任务: 阶段 5 图模式适配（至少包含以下内容）
阶段要点:
  - 图模式仅适用于 Decode 阶段，Prefill 禁止使用
  - 若阶段 4 已接受量化方案，图模式必须基于量化路径实施，不回退到非量化路径
自验证:
  - 编译通过、图编译无 graph break、推理无 crash、吐字正常（可读、不重复、非全零、不提前 EOS）
  - 实施记录写入 progress.md 工作区
```

> **验收 gate**：执行"implementer 自验证检查"（见重要原则）。不通过 → 拒绝重派。

#### 5.4 启动验证 subagent

派发 model-infer-reviewer：

```
工作目录: {model_dir}
任务: 图模式适配后验证
约束: 禁止修改模型代码和自行调试，仅返回验证报告
至少包含以下验证:
验证内容:
  - 精度: 运行推理，图模式 Decode 输出与 eager 模式对比
  - 性能: 运行推理，本阶段增量 + 相对原始基线的累计变化；若阶段 4 已接受量化方案，同时记录相对量化基线的增量
  - 性能验证: 若工作目录下有 agentic/baseline/baseline_metadata.json，用它作为性能对比基准
  - 结果写入 progress.md 工作区（精度验证 + 性能验证 section）
检查项:
  - Decode 阶段已启用图模式
  - Prefill 阶段未使用图模式
  - 若阶段 4 已接受量化方案，能证明图模式运行在量化路径上
  - 性能同时记录本阶段增量和累计变化
```

#### 5.5 阶段总结

主 agent 执行：
1. reviewer 报告 FAIL → 派发 model-infer-implementer（修复图模式适配问题，如图中断、精度偏差）→ 重新派发 model-infer-reviewer 验证，最多 5 轮
2. 5 轮仍未解决 → 回退问题模块或整阶段改动 → 向用户报告阻塞点，请求决策
3. 精度达标但性能未提升 → 派发 model-infer-analyzer（排查性能问题：部署配置、前置处理开销、测试方法、NPU 利用率等）→ 将分析和建议呈现给用户决策
4. 全部通过后，综合 analyzer/implementer/reviewer 的结果，输出阶段总结报告
5. 向用户确认当前阶段优化，确认后提交 commit，进入下一阶段

---

### 阶段 6：优化总结

使用 `templates/optimization_report_template.md` 模板，将完整的优化报告写入模型 agentic 目录：

```
models/{model_name}/agentic/optimization_report.md
```

报告章节按 `templates/optimization_report_template.md` 模板组织（成果型，按内容主题非阶段时间线）：
概述、模型结构、性能基线、并行切分、KVCache 与 Attention、算子融合、量化适配、图模式、累计性能演进、算子需求、当前未覆盖项。其中并行切分（单卡部署时）、量化适配（未启用时）、算子需求（无 CANN 期望支持点时）三段无内容时整段删除。

各章节遵循模板顶部「可读性约束」：按主题组织，禁用"阶段 N"前缀，禁止"用户确认状态 / 精度判定"等过程字段，过程记录留在 progress_history.md。

> 报告内容从 progress.md（常驻区）+ progress_history.md（Read 全文，一次性）中提取整理。生成时严格遵循 `templates/optimization_report_template.md` 顶部「写入规范」：清扫 progress 中的过程语言（试错时序、状态徽章、流程编号、编排状态等），技术内容保留，过程信息留在 progress_history.md 不进入 report。
>
> 执行过程中发现的 skill 流程缺失、描述不清、约束缺失、参考过时等问题，在归档前于 progress.md 工作区末尾以 "Skill 反馈" 条目汇总一句，随工作区一并归档进 progress_history.md，不进入对外的 optimization_report.md。

---
