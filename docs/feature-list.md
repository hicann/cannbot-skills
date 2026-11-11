# 功能清单

全量 Skills 与 Agents 清单，按开发领域汇总。各 Skill 的典型使用样例见 [Skills 使用样例](skills-usage.md)。

📖 [安装指南](installation-guide.md) · [使用样例](skills-usage.md) · [架构设计](architecture-design.md) · [README](../README.md)

---

## Ascend C 算子开发

### Skills

<!-- skills-table-start -->
| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **ascendc-api-best-practices** | API 使用最佳实践、参数限制 | — |
| **npu-arch** | NPU 架构知识、芯片型号映射 | — |
| **ascendc-docs-search** | API 文档索引 + 在线搜索 | — |
| **ascendc-env-check** | NPU 设备查询、CANN 环境验证 | — |
| **ascendc-kirin-env-check** | 麒麟平台（Kirin NPU）端侧开发环境检查：开发依赖、CANN 双能力线（atc→omc / camodel sim 直调）、hdc 设备识别 | — |
| **ops-kirin-device-verify** | 麒麟设备端到端验证：编译安装 vendor 包、生成 ONNX/裸 bin 与 golden、atc 转 OMC、HAP 安装与 hdc 推送、推理启动、输出回传与精度比对 | — |
| **ascendc-cross-gen-migration** | 已有 A2/A3 AscendC 算子按 Stage 0–5 迁移到 A5；从源码自合成标杆并完成精度和性能验证 | — |
| **ascendc-tiling-design** | Tiling 和 Kernel 设计方法论，按算子类别分类 | — |
| **ascendc-precision-debug** | 精度调试，症状-原因速查、常见陷阱 | — |
| **ascendc-runtime-debug** | 运行时错误调试，错误码解析（161xxx/361xxx/561xxx） | — |
| **ascendc-crash-debug** | 卡死/崩溃/内存错误调试；AIC/AIV Error 需要定位具体指令时按症状路由到反汇编工具 | — |
| **ascendc-sync-audit** | 信号同步检验与修正，SetFlag/WaitFlag/CrossCore 配对、缺失、错序、flagId 冲突 | — |
| **ascendc-ut-develop** | UT 单元测试用例开发与覆盖率增强 | — |
| **ascendc-st-design** | aclnn 接口测试用例设计、L0 / L1 测试用例生成 | — |
| **ascendc-code-review** | 代码检视方法论、5 大类别规范 | [查看](skills-usage.md#ascendc-code-review) |
| **ascendc-whitebox-design** | 白盒测试用例设计与生成 | — |
| **ascendc-registry-invoke-template** | 完整自定义算子工程模板，提供标准工程结构、代码模板、UT/ST 样例和多芯片架构参考 | — |
| **ascendc-registry-invoke-to-direct-invoke** | 注册调用算子转 `<<<>>>` kernel 直调 | [查看](skills-usage.md#ascendc-registry-invoke-to-direct-invoke) |
| **ascendc-direct-invoke-to-registry-invoke** | `<<<>>>` kernel 直调转注册调用算子 | [查看](skills-usage.md#ascendc-direct-invoke-to-registry-invoke) |
| **ascendc-direct-invoke-template** | Kernel 直调工程模板，提供验证过的样例工程和修改指南 | — |
| **ops-theoretical-performance-estimation** | 设计期性能建模与方案比较：Ascend950 RegBase 纯 Vector Roofline 估算，以及内置 OpModel 建模、DAG 寻优、空泡分析与设计文档工作流 | [入口](../ops/ops-theoretical-performance-estimation/SKILL.md) |
| **ops-profiling** | NPU 性能采集与分析，CSV 指标解读、瓶颈定位、优化建议 | — |
| **ops-perf-routing-plan** | Shape 级 tiling 路由规划：理论下限估算、worst-first 分组、tiling 可行性校验与冻结路由计划 | [入口](../ops/ops-perf-routing-plan/SKILL.md) |
| **ops-precision-standard** | 算子精度标准，按 dtype 分类提供 atol/rtol 精度比对标准 | — |
| **ascendc-docs-gen** | 算子文档写作与 ACLNN API 完整生成，按需加载取证、审阅及交付流程 | [入口](../ops/ascendc-docs-gen/SKILL.md) |
| **ascendc-api-docs-gen** | Ascend C API 文档写作参考，按需加载源码完整生成流程 | [入口](../ops/ascendc-api-docs-gen/SKILL.md) |
| **ascendc-api-docs-review** | Ascend C API 文档（Markdown）质检：结构完整性、规范性、语言通顺性 | — |
| **ascendc-docs-review** | aclnn 算子 API 说明文档质检：规范性、完备性、可读性、正确性 | — |
| **ops-simulator** | NPU 仿真器技能。提供 CANN Simulator 的使用指导，包括精度仿真、性能仿真、流水线分析。 | — |
| **ascendc-blaze-best-practice** | Matmul/Cube/GEMM/BMM 单算子直调生成（Blaze/tensor_api 路径），覆盖模板选型、改造、Tiling 及排错 | — |
| **ascendc-tensor-api-best-practice** | 在 Ascend 950 / DAV_3510 上用户明确指定 Tensor API 时的实现路线权威源（硬约束，不由 Blaze 优先选型改写），Tensor API 为主、C API 为辅，覆盖 API 分工、Layout、内存预算、工具链、精度与性能 | — |
| **ascendc-blaze-migration** | 将 Ascend 950 Matmul 家族核函数等价迁移到 Blaze/tensor_api，冻结 tiling 与架构，完成原调用链逐字节一致性、性能和双仓 PR 交付 | — |
| **ascendc-performance-best-practices** | 按算子族组织的性能优化经验与参考代码总结 | — |
| **ascendc-regbase-best-practice** | DAV_3510 RegBase 算子 API 约束、实现结构、常见陷阱及真实参考算子 | — |
| **cann-env-setup** | 昇腾 NPU CANN 安装与环境配置指导 | — |
<!-- skills-table-end -->

### Agents

| Agent | 功能 |
|-------|------|
| **ascendc-ops-architect** | 算子架构师，支持需求分析和方案设计两种场景 |
| **ascendc-ops-developer** | 算子开发者，支持代码实现、编译测试和精度验证 |
| **ascendc-ops-tester** | 算子测试者，支持ST/UT用例生成与执行 |
| **ascendc-ops-reviewer** | 代码检视专家，支持快速检视和全功能检视两种模式 |

直调插件的 Agent 清单已迁至 [cannbot 中的 ops-direct-invoke](https://gitcode.com/cann/cannbot/tree/master/plugins-official/ops-direct-invoke)，不再作为本仓本地 Agent 列出。

## AISS Tiling 参数求解

> 以下 Skill 位于 `community/ops/` 目录，属社区贡献（AISS Group at HIT），实验/非正式版本，可能存在平台或能力限制。

### Skills（1 个）

| Skill | 功能 |
|-------|------|
| **aiss-tiling-solver** | AISS-TilingSolver 工具自动求解最优 Tiling 参数（MatMul / Vector），覆盖下载安装、构造 JSON 输入、运行求解、结果解读与故障排查 |

## TileLang2AscendC 算子开发

> 以下 Skill 位于 `community/ops/` 目录，属实验/非正式版本，可能存在平台或能力限制。

### Skills

| Skill | 功能 | 限制 | 使用样例 |
|-------|------|------|---------|
| **tilelang2ascend-tilelang-designer** | TileLang 算子设计表达，支持 Attention/Matmul/Norm/Sort 等复杂算子 |  |  |
| **tilelang2ascend-translator** | 将 TileLang 设计转译为 AscendC kernel |  |  |
| **tilelang2ascend-operator-project-init** | AscendC kernel 工程初始化与算子注册 |  |  |
| **tilelang2ascend-case-simplifier** | 算子测试用例精简 |  |  |
| **tilelang2ascend-precision-tuning** | AscendC 算子精度调试与修复 |  |  |
| **tilelang2ascend-trace-recorder** | 算子任务执行 trace 记录 |  |  |
| **cuda2ascend-simt** | CUDA 算子迁移到 Ascend C SIMT，支持 standalone sample / torch_npu / pybind 三类交付形态 | 仅支持 Ascend 950 PR；不支持 native JIT、torch 复数 dtype、device 侧 FP64、CUDA 生态库、协作组等 | [查看](skills-usage.md#cuda2ascend-simt) |

### Agents

| Agent | 功能 |
|-------|------|
| **tilelang2ascendc-kernel-generator** | 从 PyTorch Model 出发，端到端完成算子设计表达和 AscendC kernel 落地（简单算子 ops-direct-invoke / 复杂算子 TileLang） |

## Catlass 算子开发

### Skills

| Skill | 功能 |
|-------|------|
| **catlass-op-design** | 分析算子需求并选型 CATLASS 组件（ArchTag、DispatchPolicy、TileShape、BlockMmad 等） |
| **catlass-op-develop** | 根据设计选型生成 CATLASS kernel 代码（模板拼装 + Device 调用 + host 侧框架） |
| **catlass-op-perf-tune** | 调优 CATLASS kernel 性能（TileShape、DispatchPolicy、Swizzle、Kernel 类型） |

### 五阶段工作流

[cannbot 中的 catlass-op-generator](https://gitcode.com/cann/cannbot/tree/master/plugins-official/catlass-op-generator) 使用本仓 `catlass-cpp-interface`、`catlass-cpp-reference`、`catlass-cpp-design`、`catlass-cpp-develop`、`catlass-cpp-test` 和横向知识查询 `catlass-cpp-knowledge`，覆盖通用 CATLASS 算子与注意力家族。

## PyPTO 算子开发

### Skills

| Skill | 功能 |
|-------|------|
| **pypto-orchestration-manual** | 编排者入口：团队原则 / 成员名册 / 强制规则 |
| **pypto-intent-understand** | 需求意图理解与规格生成 |
| **pypto-op-plan** | 需求规划：相似样例搜索与可行性评估 |
| **pypto-api-explore** | API 可行性探索与分析 |
| **pypto-docs-search** | API 文档 / 参考实现 / golden / 错误码检索 |
| **pypto-golden-generate** | Golden 参考实现生成 |
| **pypto-op-design** | 算子方案设计生成（含模块拆解，并入原 pypto-op-construct 能力） |
| **pypto-op-develop** | 算子代码实现与测试 |
| **pypto-op-verify** | 验证 runner 与 `detailed_tensor_compare` 用法 |
| **pypto-op-review** | 逐算子 PyPTO 调用提取（调试用） |
| **pypto-precision-compare** | 精度中间结果对比分析 |
| **pypto-precision-debug** | 精度问题代码层排查 |
| **pypto-general-debug** | 卡住/不明失败的调试路由 |
| **pypto-op-perf-tune** | 算子性能分析与自动调优（含泳道图子技能） |
| **pypto-op-perf-panko-manual** | PANKO 搜索式自动调优契约（world model + evaluator，用户要求 PANKO 时启用） |
| **pypto-op-knowledge** | 算子开发经验表 / 问题查找表 |
| **pypto-memory-template** | 算子级 `MEMORY.md` 模板 |

编排入口与 Agent 由 [cannbot 应用仓](https://gitcode.com/cann/cannbot/tree/master/plugins-official/pypto-op-orchestrator) 维护。

## PyPTO-Pro 算子开发

> 以下 Skill 位于 `community/ops/` 目录，属实验/非正式版本，可能存在平台或能力限制。

### Skills（1 个）

| Skill | 功能 |
|-------|------|
| **pypto-pro-kernel** | 原生 pypto_pro kernel 编写、验证、调试、native-task 计时与性能优化（@pl.jit / VF / SIMT / Cube API，含可运行样例与静态检查） |

> PyPTO-Pro 的编排入口等插件资产统一维护在 cannbot 仓，本仓仅保留 skill 本体。

## TileLang 算子开发

### Skills

| Skill | 功能 |
|-------|------|
| **tilelang-env-check** | TileLang-Ascend 环境检查与配置验证 |
| **tilelang-submodule-pull** | 自动拉取 tilelang 仓库及其三方子模块代码 |
| **tilelang-op-design** | 算子设计文档生成 |
| **tilelang-op-develop** | 基于设计文档生成算子实现代码与测试 |
| **tilelang-op-test-design** | 算子测试设计与测试覆盖率分析 |
| **tilelang-api-best-practices** | TileLang Ascend API 使用最佳实践 |
| **tilelang-programming-model-guide** | Developer/Expert 模式选择与 pass_configs 配置指南 |
| **tilelang-perf-optimization** | 性能调优与性能劣化模式检查 |
| **tilelang-review** | 代码格式检查与自动修复 |

### Agents

| Agent | 功能 |
|-------|------|
| **tilelang-op-analyst** | 需求理解与算子设计 |
| **tilelang-op-developer** | 代码生成、测试与精度调试 |
| **tilelang-op-perf-tuner** | 性能分析、瓶颈定位与调优 |

## Triton 算子开发

### Skills

| Skill | 功能 |
|-------|------|
| **triton-task-extractor** | 从用户输入中提取算子，构建任务文件 |
| **triton-op-designer** | 设计高质量算法，指导代码生成 |
| **triton-op-coding** | 根据设计生成 Triton 内核代码 |
| **triton-op-verifier** | 验证算子精度和性能测试 |
| **triton-latency-optimizer** | 逐步优化 Triton 代码性能 |

### Agents

| Agent | 功能 |
|-------|------|
| **triton-op-generator** | Triton 算子端到端生成与优化（插件已迁移至 cannbot 仓，Skills 保留在本仓 `ops/`） |

## Runtime

### Skills

<!-- skills-table-start -->
| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **runtime_migration** | Runtime 接口迁移，提供兼容层与直接迁移两种方式，含设备/内存/流/事件/IPC/库管理/VMM 适配与错误码映射 | — |
| **runtime-llt-generator** | 为 Runtime 接口、函数、分支、缺陷或当前代码变更生成并验证 LLT | — |
<!-- skills-table-end -->

> 仅用于用户自身合法拥有或已获授权代码的处理，不用于未授权第三方代码的复制、翻译或转换。

---

## torch.compile 图模式

### Skills

| Skill | 功能 |
|-------|------|
| **torch-npugraph-ex-knowledge** | npugraph_ex（aclgraph）模式使用指南，涵盖模式配置、FX Pass、编译缓存、多流并行、内存复用等 |
| **torch-custom-ops-guide** | 自定义算子入图完整指南，覆盖 torch.library.custom_op / Library 注册、Meta 推导等全流程 |
| **torch-npugraph-ex-dfx-triage** | npugraph_ex DFX 问题分诊入口，统一采集日志并路由到编译/运行/精度/性能四个专科 |
| **torch-npugraph-ex-compile-error-diagnosis** | npugraph_ex 编译期报错诊断，覆盖 graph break / BackendCompilerFailed / Meta 推导失败等 |
| **torch-npugraph-ex-runtime-error-diagnosis** | npugraph_ex 运行时报错诊断，覆盖 ACL replay / aclnn / HCCL / OOM 等 |
| **torch-npugraph-ex-performance-diagnosis** | npugraph_ex 性能诊断，聚焦 reinplace 未命中导致的冗余 tensor move |
| **torch-npugraph-ex-template** | npugraph_ex 模式 MRE 代码模板，含标准编译模板和编译缓存模板 |

## NPU 模型推理优化

### Skills

| Skill | 功能 |
|-------|------|
| **model-infer-migrator** | 框架适配与部署基线建立 |
| **model-infer-parallel-analysis** | 并行策略分析（TP/EP/DP） |
| **model-infer-parallel-impl** | 并行切分实施 |
| **model-infer-kvcache** | KVCache 优化 + FA 替换 |
| **model-infer-fusion** | torch_npu 融合算子分析与替换 |
| **model-infer-quantization** | compressed-tensors 量化适配改造 |
| **model-infer-graph-mode** | torch.compile 图模式适配 |
| **model-infer-precision-debug** | NPU 推理精度诊断 |
| **model-infer-runtime-debug** | NPU 运行时错误诊断 |
| **model-infer-multi-stream** | 多流并行优化 |
| **model-infer-prefetch** | 权重预取适配 |
| **model-infer-superkernel** | SuperKernel 适配 |
| **model-infer-profiling** | 用 `torch_npu.profiler` 采集 PyTorch 推理性能数据（Level1 + PipeUtilization） |
| **model-infer-perf-breakdown** | 把 kernel_details / trace 按模型结构切成 component 实例并分桶，输出 wall/bubble 中位数与异常 layer 单页 HTML |
| **model-infer-harmony** | 麒麟 NPU 端侧（Kirin9030）ASR 模型 4bit 量化、omg 模型转换与 CANNPAK 打包 |

基础优化、SOTA 探索工作流及配套 Agent 由 [cannbot 中的 model-infer-optimize](https://gitcode.com/cann/cannbot/tree/master/plugins-official/model-infer-optimize) 维护，不再作为本仓本地 Agent 列出；上列原子 Skill 继续在本仓维护。

## 推荐模型推理优化

> 以下 Skill 位于 `community/model/` 目录，属实验/非正式版本，可能存在平台或能力限制。
> 对应的编排插件（agent / workflow / hooks / 内置 GE_SESSION_DEMO 框架）维护在 cannbot 仓 `plugins-community/recommend-infer-optimize`。

### Skills（18 个）

| Skill | 功能 |
|-------|------|
| **model-recommend-ascend-env-baseline** | 环境基线自检与工作流路由（条目 0）：NPU / CANN / GE 组件核对、GE_SESSION_DEMO 编译，约定全局产物目录 |
| **model-recommend-baseline-run** | 单卡基线运行（条目 2）：prepare / compile / run 三步采集首基线时延、QPS、HBM 峰值与精度验证 |
| **model-recommend-concurrency-strategy-analysis** | 并发与限核策略分析（条目 3）：给出 THREAD_NUM / CORE_NUM / GRAPH_MAX_PARALLEL / batch_size 组合并 sweep 校准 |
| **model-recommend-custom-op-development** | 自定义算子开发与 GE 接入（条目 12）：msopgen 工程 → 算子包部署 → onnx 节点入图 → 精度与性能验证 |
| **model-recommend-flash-attention-fusion** | FA 族融合（条目 7）：确认 GE 内置 FlashAttention 算子支持性，经图融合 pass 替换 attention 子图 |
| **model-recommend-fusion-operator-adaptation** | 既有算子与图优化适配（条目 7）：图编译内置优化、onnx 等价改写、精度模式路由，不新增算子实现 |
| **model-recommend-fusion-planning-estimation** | 融合规划与理论收益估计（条目 10）：Roofline bound 判定与候选排序，决定 kernel 开发优先级 |
| **model-recommend-ge-graph-optimization** | GE 图模式与 shape 策略（条目 6）：all_static / all_dynamic / custom 选型、编译选项调优、dump_graph 验证 |
| **model-recommend-kernel-generation** | AscendC Kernel 生成（条目 11）：msopgen 工程与 Matmul 高阶 API，产出可上机 kernel 并完成 GE 集成 |
| **model-recommend-memory-access-optimize** | 访存优化（条目 9）：查表 / Gather / 大 GEMM / elementwise 访存链的等价改写与 kernel 侧双缓冲预取 |
| **model-recommend-multi-thread-optimization** | 多线程与图并行（条目 8）：RunGraphAsync 并发提交、图实例并行与限核实施，sweep 三阶段寻优 |
| **model-recommend-optimization-loop** | 持续优化循环编排（贯穿条目 5→14）：选动作 → 执行 → 采集 → 分析 → 收益判定 → 动作表更新，loop_state.json 管状态 |
| **model-recommend-performance-analysis** | 性能分析（条目 5）：三表报告（kernel / overlap / 融合模式）、bound 判定与优化建议 |
| **model-recommend-precision-comparison** | 精度对比门禁（条目 4）：标杆建立、verify.py 判定、dump_data 逐节点定位 |
| **model-recommend-profiling-deep-analysis** | Profiling 四项专项（条目 5）：算子执行时间、多流掩盖、H2D/D2H 通信、host bound 分析 |
| **model-recommend-regression-baseline-mgmt** | 回归验收与基线管理（条目 14）：优化台账、逐项三关验收、perf_ledger、最终报告与下一步动作表 |
| **model-recommend-runtime-debug** | 运行时诊断（条目 13）：GE 初始化 / onnx 解析 / aicore timeout / OOM / 卡死 / 错误码与 plog 排查 |
| **model-recommend-structure-analysis** | 模型结构与显存分析（条目 1）：onnx 结构档案、算子构成（AIC/AIV）、链路清单、显存估算与可执行判定 |

## NPU 模型训练调测

> 方法论适用于 torchtitan-npu 等 PyTorch on NPU 训练框架。精度诊断 Skill 通过 `model-train-skills` 技能包安装（`npx skills add https://gitcode.com/cann/cannbot-skills.git --skill <skill-name>`），端到端流程可使用 [`model-train-precision-diagnose`](https://gitcode.com/cann/cannbot/tree/master/plugins-community/model-train-precision-diagnose) Plugin。

### Skills

| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **model-train-precision-numerical-mismatch** | E01 正向、E02 反向有限值偏差的首差异定位与标杆对齐 | [查看](skills-usage.md#model-train-precision-numerical-mismatch) |
| **model-train-precision-nonfinite** | E04 前向、反向及 Optimizer/Scaler 边界的 NaN/Inf/Overflow 定位 | [查看](skills-usage.md#model-train-precision-nonfinite) |
| **model-train-precision-determinism** | E08 固定输入、权重、Seed 和确定性配置后的重复运行不一致定位 | [查看](skills-usage.md#model-train-precision-determinism) |
| **model-train-oom-analysis** | NPU 训练 OOM 内存诊断与优化。按「日志分类 → 静态估算 → snapshot 深度分析 → 优化建议」流程定位内存问题 | [查看](skills-usage.md#model-train-oom-analysis) |
| **model-train-log-visualization** | 训练日志可视化，从 stdout 解析 loss/grad_norm/内存/吞吐等指标并绘图，支持两份日志对比 | [查看](skills-usage.md#model-train-log-visualization) |

**边界说明**：

- 有限值偏差诊断必须提供可比标杆（标杆环境或标杆数据）；无法判断症状类型时，先使用 `model-train-precision-diagnose` Plugin 路由。
- 纯 OOM（无精度问题）→ 使用 `model-train-oom-analysis`。
- 纯推理精度问题 → 使用 `model-infer-precision-debug`。
- 纯算子精度问题 → 使用 `ascendc-precision-debug`。

## Skill 治理工具

### Skills

| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **cannbot-skill-reviewer** | 审查新增或修改的 `SKILL.md` 是否符合 CANNBot 入库要求，输出自动门禁、九维评分、阻塞项和整改建议 | [查看](skills-usage.md#cannbot-skill-reviewer) |

## GitCode 协作工具

### Skills

| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **gitcode-pr-handler** | 根据 GitCode PR 代码变更重新生成标题（约定式提交）与描述（沿用仓库 PR 模板）并写回 PR | [查看](skills-usage.md#gitcode-pr-handler) |
| **gitcode-issue-gen** | 自动判断两条路径：(PR路径) 从 PR diff 生成关联 Issue 并完成双向关联；(手动路径) 交互式收集信息生成 Issue 草稿，经确认后提交 | [查看](skills-usage.md#gitcode-issue-gen) |
| **gitcode-issue-handler** | GitCode 单 Issue/批量 Issue 状态机：分诊、可配置自动首响与负责人转交、稳定复现、最小修复、测试、经确认的代码交付与报告 | [查看](skills-usage.md#gitcode-issue-handler) |
| **gitcode-toolkit** | GitCode API/Token/URL/日志/变更展示 + Git 克隆/diff/log/remote + PR 创建工作流共享参考（内部参考，不直接触发） | [查看](skills-usage.md#gitcode-toolkit) |

## CANN / 端侧工具链

> 以下 Skill 位于 `tools/` 目录，提供 CANN / 端侧工具链使用指导（维测类仅 Ascend EP 形态，RC 形态不支持；hdc 属 HarmonyOS SDK）。

| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **tool-asys** | asys 一键式故障信息收集：collect/launch/info/health/diagnose/analyze（trace/coredump/stackcore/coretrace/UB/AI Core Error 六类文件解析）/config/profiling | — |
| **tool-msaicerr** | msaicerr 故障分析：AI Core Error 问题定位（info.txt）、Dump 文件解析为 .bin/.npy、bin 转 npy 类型转换、内置算子样例环境检查 | — |
| **tool-msnpureport** | msnpureport Device 侧日志导出（单次/常驻连续）、维测配置查询与设置（日志级别/Coremask/加速器复位/singlecommit）、导出目录结构解读 | — |
| **tool-atc** | ATC 模型转换：框架网络模型/单算子描述文件转 om 离线模型、高频参数选型、转换报错排查（EZ3003/E19999/E20002/E90003 等），支撑算子开发「离线模型 → 设备验证」环节 | — |
| **tool-hdc** | HarmonyOS Device Connector（hdc）命令行：设备连接管理、HAP 安装/卸载、文件传输、设备端 shell 与日志抓取、错误码速查，支撑端侧验证的设备侧操作 | — |
| **tool-exception-dump** | AI Core Error 系统化诊断工作流（34 步）：环境与输入检查、场景判定与报错行定位、错误类型解码、单算子复现、证据链汇聚与修复建议，输出六段式证据链诊断报告，与 tool-msaicerr/tool-asys 分工协作 | — |
| **tool-npu-compute** | 使用 npu-compute 选择和采集 NPU 性能指标、确认采集结果、解包 `.npu-rep`，并分析带宽、流水线、存储访问、缓存和资源冲突等数据；支持由多个采集项组成的指标集合（Set）和可单独选择的采集项（Section） | [查看](skills-usage.md#tool-npu-compute) |
| **tool-npu-check** | `memcheck` 设备全局内存（GM）访问检查、`synccheck` 同步操作配对检查、命令执行及检查报告解读 | — |
| **npu-objdump-toolkit** | npu-objdump/msobjdump 算子编译产物分析：查看元数据与 ELF、提取 device ELF、反汇编 Ascend AICore 指令，并将异常 PC 定位到 OPP 对象、MIX AIC/AIV 物理符号和窄指令窗口 | — |
| **tool-aml** | 为 oam-tools-diag 的 AICore STL 诊断模块（aicore_stl_detect）新增算子：自动生成 .cpp（host 规格+下发+注册）、.asc（kernel 下发包装）与 aic/aiv .cce 文件，并完成 host so 编译与注册校验 | — |

## 领域目录入口（自 README 索引表移入）

> 目录级入口，非官方插件；官方插件索引保留在 [README](../README.md)。

| 领域 | 说明 | 入口 |
|------|------|------|
| **GE** | 图模式相关问题定位 | [graph/](../graph/) |
| **Runtime** | Runtime LLT 开发与接口迁移 | [runtime/](../runtime/) |
| **CANN 工具链** | 故障信息收集、Device 日志导出、AI Core Error 分析、npu-compute 性能采集与报告分析 | [tools/](../tools/) |
| **科学计算模型迁移** | 框架级代码 NPU 迁移（环境门禁/脚本适配/精度性能对比） | [community/model/science-model-npu-migration/](../community/model/science-model-npu-migration/) |
| **治理与协作** | Skill 审查、GitCode PR/Issue 自动化 | infra/ |
| **文档通用质检** | docs-common-review 统一入口按门禁/全量/自定义三种模式编排 9 项检查（markdownlint、标签闭合、链接、资源、拼写、文件命名、标点、合规、AI 语义），聚合统一 JSON 质检报告（级别统计+门禁判定+耗时统计），并支持家族基准回归；各原子 Skill 仅承接点名的单项专项检查 | [infra/docs-common-review/](../infra/docs-common-review/) |
