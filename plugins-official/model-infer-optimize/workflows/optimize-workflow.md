# NPU 模型推理端到端优化工作流

## 概述

本工作流编排三个专业化 subagent（model-infer-analyzer / model-infer-implementer / model-infer-reviewer）对目标模型按阶段执行 NPU 推理优化。

```
阶段 0: 模型分析与建立基线
阶段 1: 并行化改造（多卡部署时）
阶段 2: KVCache + FA
阶段 3: 融合算子
阶段 4: 图模式适配
阶段 5: 优化总结
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

`{model_dir}/progress.md`：常驻区（阶段 0 分析 + 进度概览表）+ 工作区（当前阶段记录）。初始模板见 `templates/progress_template.md`。

`{model_dir}/progress_history.md`：历史归档。除阶段 5 优化总结外，默认仅 Grep 查找；阶段 5 允许一次性 Read 全文用于生成总报告。

**读写规则**：常驻区由 阶段 0 写入，后续只有主 agent 更新概览表。工作区由各 subagent 追加，写入前先读取现有内容。

**阶段推进**：每阶段验证通过后，主 agent 更新概览表 → 调用 `scripts/archive_progress.py` 归档工作区 → 清空工作区。阶段 0 不归档。

---

## 工作流程

### 阶段 0：模型分析与建立基线

#### 0.1 信息收集

若用户未提供以下信息，主 agent 使用提问工具向用户确认：
- **模型工作目录**：模型代码所在路径（如 `cann-recipes-infer/models/xxx`）
- **模型来源**：HF 链接、本地权重路径、或仓库内已有
- **权重路径**：已下载的权重位置（如未下载可后续处理）

#### 0.2 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
任务: 模型架构全面分析
模型来源: {HF 链接 / 本地路径 / 仓库内已有}
分析内容:
  - 架构类型（LLM / MoE / Diffusion / 多模态）
  - 网络结构拆解（Embedding → Transformer Blocks → Output Head）
  - Prefill / Decode 分支差异
  - 关键模块：Attention 类型（GQA/MHA/MLA）、FFN/MoE 结构、特殊模块。架构识别必须基于实际 config 值（config.json / model.config），不能仅从代码类定义推断——注意可配置开关（如 use_mla、n_routed_experts）
  - 运行环境：通过 `npu-smi info -m` 确认 NPU 型号，通过 `npu-smi info -t memory -i <device_id>` 确认单卡 HBM 容量，记录量化模式、执行模式、部署卡数
  - 模型当前状态：确认代码是否存在且可运行（有 infer.sh 且能跑通）、baseline/baseline_metadata.json 是否存在。报告状态（可运行/不可运行/需多卡），不自行采集基线数据
  - 若模型不可运行，记录具体原因和缺失项
输出:
  - 使用 templates/progress_template.md 创建 {model_dir}/progress.md，将分析结果写入常驻区（模型信息、并行策略、进度概览）
  - 使用 templates/optimization_report_template.md 初始化 optimization_report.md
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
必须使用 skill: /model-infer-migrator
工作目录: {model_dir}
任务: 部署基线采集
```

implementer 返回后，提取 baseline_metadata.json 摘要写入 progress.md 常驻区 → 进入阶段 2

**路径 b（模型无法运行）**：

```
必须使用 skill: /model-infer-migrator
工作目录: {model_dir}
任务: 框架适配 + 部署基线建立
模型来源: {HF 链接 或 本地路径}
权重路径: {如已知}
```

implementer 返回后：
- 若输出 baseline_metadata.json → 提取摘要写入 progress.md 常驻区 → 进入阶段 2
- 若标记"需多卡" → 进入阶段 1 并行化

**路径 c（模型需多卡部署）**：
- 若代码已存在（仓库内已有框架适配的模型）→ 直接进入阶段 1 并行化，基线在并行跑通后建立
- 若代码不存在 → 先按路径 b 派发 migrator 完成单卡框架适配（完整的 modeling + Runner + infer.py + infer.sh + YAML 单卡配置，作为并行化改造的代码基础），migrator 标记"需多卡"后再进入阶段 1

---

### 阶段 1：并行化改造

> 单卡模型跳过本阶段。并行策略影响后续所有阶段的代码结构（通信组、TP 切分、EP 路由），必须先于 KVCache/FA 改造完成。

#### 1.1 确认部署需求

主 agent 使用提问工具向用户确认：

1. **部署卡数和节点配置**：总卡数、每节点几卡（影响 TP 上限）
2. **目标场景**：高吞吐 / 低时延 / 均衡
3. **实际序列长度**：决定是否需要 CP / KVP
4. **batch size 需求**（如有）

确认后结合 阶段 0 的模型参数和硬件信息做快速可行性检查：
- 卡数 × 单卡显存是否能容纳模型参数（阶段 0 已知参数量和量化模式）
- MoE 模型的专家数是否 ≥ 卡数（否则 EP 不可行）
- 序列长度 × KV Cache 是否超出总显存

有矛盾则向用户反馈，调整卡数、batch size 或序列长度后再派发分析。

#### 1.2 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
必须使用 skill: /model-infer-parallel-analysis
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

1. **整体并行策略**：纯 TP / EP+TP 混合 / 模块差异化？
2. **各模块 TP 度**：attn_tp / dense_tp / moe_tp / embed_tp / lmhead_tp / oproj_tp
3. **长序列附加配置**：是否引入 CP / KVP？
4. 其他需确认的细节（AFD、EPLB 等进阶配置）

用户确认后进入实施。

#### 1.4 启动实施 subagent

派发 model-infer-implementer：

```
工作目录: {model_dir}
必须使用 skill: /model-infer-parallel-impl
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
必须使用 skill: /model-infer-parallel-analysis（第五步 Profiling 校准）
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
    必须使用 skill: /model-infer-migrator
    工作目录: {model_dir}
    任务: 部署基线采集
    ```

    implementer 返回后，提取 baseline_metadata.json 摘要写入 progress.md 常驻区。后续阶段的性能对比以此次采集的基线为准，忽略并行验证阶段的性能数据。
4. 全部通过后，综合结果输出阶段总结报告
5. 向用户确认当前阶段优化，确认后提交 commit，进入下一阶段

---

### 阶段 2：KVCache 静态化 + FA 算子替换

> KVCache 模式选型（连续缓存 / PA / MLA 压缩）会影响 FA 算子的调用方式和参数配置。
> 后续阶段 3 的融合算子优化专注于 Attention Core 之外的模块。

#### 2.1 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
必须使用 skill: /model-infer-kvcache（关注选型分析部分）
任务: KVCache 模式分析和选型
分析内容:
  - 根据模型 Attention 架构选择 KVCache 模式（连续缓存 / PA / MLA）
  - 确定对应的 FA 算子版本和 layout
  - 评估 Prefill / Decode 差异对缓存策略的影响
```

#### 2.2 方案确认

主 agent 使用提问工具向用户逐条确认关键决策：

1. **KVCache 模式**：连续缓存 / PA / MLA？（影响后续所有缓存和 FA 实现）
2. **MLA absorb 路径**（仅 MLA 模式）：是否使用 absorb？（影响 FA 入参和 rope 处理方式）
3. **FA 算子选择**：如 npu_fused_infer_attention_score / npu_fused_infer_attention_score_v2 等
4. 其他需用户确认的方案细节（如数据 layout、Prefill/Decode 差异策略等）

用户确认后进入实施。

#### 2.3 启动实施 subagent

派发 model-infer-implementer：

```
工作目录: {model_dir}
必须使用 skill: /model-infer-kvcache（关注实施和代码模板部分）
任务: 阶段 2 KVCache + FA 改造（至少包含以下内容）
阶段要点:
  - 按已确认的 KVCache 模式和 FA 算子方案实施
  - 实施缓存初始化、更新逻辑、FA 入参构造
  - 适配 Prefill / Decode 差异
  - 静态化完成后替换 Attention Core 为 FA 算子
自验证:
  - 编译通过、推理无 crash、吐字正常（可读、不重复、非全零、不提前 EOS）
  - 实施记录写入 progress.md 工作区
```

> **验收 gate**：执行"implementer 自验证检查"（见重要原则）。不通过 → 拒绝重派。

#### 2.4 启动验证 subagent

派发 model-infer-reviewer：

```
工作目录: {model_dir}
任务: KVCache + FA 改造后验证
约束: 禁止修改模型代码和自行调试，仅返回验证报告
至少包含以下验证:
验证内容:
  - 精度: 运行推理，Prefill/Decode 输出与基线对比
  - 性能: 运行推理，对比改造前后的 Prefill 耗时和 Decode 单步耗时
  - 性能验证: 若工作目录下有 baseline/baseline_metadata.json，用它作为性能对比基准
  - 结果写入 progress.md 工作区（精度验证 + 性能验证 section）
检查项:
  - KVCache 模式选型、静态化实现及 FA 算子替换已完成
  - Prefill 和 Decode 阶段缓存逻辑差异已正确处理
```

#### 2.5 阶段总结

主 agent 执行：
1. reviewer 报告 FAIL → 派发 model-infer-implementer（调试 KVCache/FA 精度问题，使用 /model-infer-precision-debug）→ 重新派发 model-infer-reviewer 验证，最多 5 轮
2. 5 轮仍未解决 → 回退问题模块或整阶段改动 → 向用户报告阻塞点，请求决策
3. 调试发现需更换 KVCache 方案 → 回到 2.3 重新实施
4. 精度达标但性能未提升 → 派发 model-infer-analyzer（排查性能问题：部署配置、前置处理开销、测试方法、NPU 利用率等）→ 将分析和建议呈现给用户决策
5. 全部通过后，综合 analyzer/implementer/reviewer 的结果，输出阶段总结报告
6. 向用户确认当前阶段优化，确认后提交 commit，进入下一阶段

---

### 阶段 3：融合算子优化

> 若阶段 2 已完成 FA 算子替换，则本阶段跳过 FA 算子本身，但 Attention 子链路（RoPE 融合、KV write 融合、QK Norm 等）仍需分析和优化。

#### 3.1 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
必须使用 skill: /model-infer-fusion（关注分析匹配部分，步骤 1-4）
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
必须使用 skill: /model-infer-fusion（关注实施替换部分，步骤 5）
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
  - 性能验证: 若工作目录下有 baseline/baseline_metadata.json，用它作为性能对比基准
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

### 阶段 4：图模式适配优化

#### 4.1 启动分析 subagent

派发 model-infer-analyzer：

```
工作目录: {model_dir}
必须使用 skill: /model-infer-graph-mode（关注分析和方案设计部分）
任务: 图模式适配方案分析
分析内容:
  - 分析模型中的图中断点（dynamic shape、数据依赖控制流等）
  - 评估图模式适配方案（npugraph_ex / GE 图模式）
  - 图模式仅适用于 Decode 阶段，Prefill 禁止使用
```

#### 4.2 方案确认

主 agent 使用提问工具向用户逐条确认关键决策：

1. **图模式后端**：npugraph_ex / GE 图模式？
2. **图中断点处理**：analyzer 识别的图中断点及解决方案是否认可？
3. 其他需用户确认的方案细节（如 mark_static 处理、编译缓存策略等）

用户确认后进入实施。

#### 4.3 启动实施 subagent

派发 model-infer-implementer：

```
工作目录: {model_dir}
必须使用 skill: /model-infer-graph-mode（关注实施和代码改造部分）
任务: 阶段 4 图模式适配（至少包含以下内容）
阶段要点:
  - 图模式仅适用于 Decode 阶段，Prefill 禁止使用
自验证:
  - 编译通过、图编译无 graph break、推理无 crash、吐字正常（可读、不重复、非全零、不提前 EOS）
  - 实施记录写入 progress.md 工作区
```

> **验收 gate**：执行"implementer 自验证检查"（见重要原则）。不通过 → 拒绝重派。

#### 4.4 启动验证 subagent

派发 model-infer-reviewer：

```
工作目录: {model_dir}
任务: 图模式适配后验证
约束: 禁止修改模型代码和自行调试，仅返回验证报告
至少包含以下验证:
验证内容:
  - 精度: 运行推理，图模式 Decode 输出与 eager 模式对比
  - 性能: 运行推理，本阶段增量 + 相对原始基线的累计变化
  - 性能验证: 若工作目录下有 baseline/baseline_metadata.json，用它作为性能对比基准
  - 结果写入 progress.md 工作区（精度验证 + 性能验证 section）
检查项:
  - Decode 阶段已启用图模式
  - Prefill 阶段未使用图模式
  - 性能同时记录本阶段增量和累计变化
```

#### 4.5 阶段总结

主 agent 执行：
1. reviewer 报告 FAIL → 派发 model-infer-implementer（修复图模式适配问题，如图中断、精度偏差）→ 重新派发 model-infer-reviewer 验证，最多 5 轮
2. 5 轮仍未解决 → 回退问题模块或整阶段改动 → 向用户报告阻塞点，请求决策
3. 精度达标但性能未提升 → 派发 model-infer-analyzer（排查性能问题：部署配置、前置处理开销、测试方法、NPU 利用率等）→ 将分析和建议呈现给用户决策
4. 全部通过后，综合 analyzer/implementer/reviewer 的结果，输出阶段总结报告
5. 向用户确认当前阶段优化，确认后提交 commit，进入下一阶段

---

### 阶段 5：优化总结

使用 `templates/optimization_report_template.md` 模板，将完整的优化报告写入模型目录：

```
models/{model_name}/optimization_report.md
```

报告应包含：
- 模型分析概要
- 各阶段的优化措施、精度验证结果、性能验证结果
- 功能问题记录表（问题描述、影响范围、处理方式、状态）
- 性能问题记录表（瓶颈描述、优化措施、优化前后数据、增益）
- 累计优化效果（相对原始基线的总提升）
- 遗留问题与后续建议
- Skill 反馈（执行过程中发现的 skill 流程缺失、描述不清、约束缺失、参考过时等问题）

> 报告内容从 progress.md（常驻区）+ progress_history.md（Read 全文，一次性）中提取整理。

---
