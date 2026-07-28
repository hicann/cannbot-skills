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
| **ascendc-tiling-design** | Tiling 和 Kernel 设计方法论，按算子类别分类 | — |
| **ascendc-precision-debug** | 精度调试，症状-原因速查、常见陷阱 | — |
| **ascendc-runtime-debug** | 运行时错误调试，错误码解析（161xxx/361xxx/561xxx） | — |
| **ascendc-crash-debug** | 卡死/崩溃调试，Kernel 挂起、Coredump 分析 | — |
| **ascendc-ut-develop** | UT 单元测试用例开发与覆盖率增强 | — |
| **ascendc-st-design** | aclnn 接口测试用例设计、L0 / L1 测试用例生成 | — |
| **ascendc-code-review** | 代码检视方法论、5 大类别规范 | [查看](skills-usage.md#ascendc-code-review) |
| **ascendc-task-focus** | 任务聚焦，解决长任务"迷失在中间"的问题 | — |
| **ascendc-whitebox-design** | 白盒测试用例设计与生成 | — |
| **ascendc-registry-invoke-template** | 完整自定义算子工程模板，提供标准工程结构、代码模板、UT/ST 样例和多芯片架构参考 | — |
| **ascendc-registry-invoke-to-direct-invoke** | 注册调用算子转 `<<<>>>` kernel 直调 | [查看](skills-usage.md#ascendc-registry-invoke-to-direct-invoke) |
| **ascendc-direct-invoke-to-registry-invoke** | `<<<>>>` kernel 直调转注册调用算子 | [查看](skills-usage.md#ascendc-direct-invoke-to-registry-invoke) |
| **ascendc-direct-invoke-template** | Kernel 直调工程模板，提供验证过的样例工程和修改指南 | — |
| **ops-profiling** | NPU 性能采集与分析，CSV 指标解读、瓶颈定位、优化建议 | — |
| **ops-precision-standard** | 算子精度标准，按 dtype 分类提供 atol/rtol 精度比对标准 | — |
| **ascendc-docs-gen** | 算子文档写作参考，支持需求分析、详细设计等多个标准模版 | — |
| **ops-simulator** | NPU 仿真器技能。提供 CANN Simulator 的使用指导，包括精度仿真、性能仿真、流水线分析。 | — |
| **ascendc-blaze-best-practice** | Matmul/Cube/GEMM/BMM 单算子直调生成（Blaze/tensor_api 路径），覆盖模板选型、改造、Tiling 及排错 | — |
| **ascendc-blaze-migration** | 将 Ascend 950 Matmul 家族核函数等价迁移到 Blaze/tensor_api，冻结 tiling 与架构，完成原调用链逐字节一致性、性能和双仓 PR 交付 | — |
| **ascendc-performance-best-practices** | 按算子族组织的性能优化经验与参考代码总结 | — |
| **ascendc-regbase-best-practice** | DAV_3510 RegBase 算子 API 约束、实现结构、常见陷阱及真实参考算子 | — |
| **cann-env-setup** | 昇腾 NPU CANN 安装与环境配置指导 | — |
| **aiss-tiling-solver** | AISS-TilingSolver 工具自动求解最优 Tiling 参数，覆盖安装、输入构造、运行求解、结果解读 | — |
<!-- skills-table-end -->

### Agents

| Agent | 功能 |
|-------|------|
| **ascendc-ops-architect** | 算子架构师，支持需求分析和方案设计两种场景 |
| **ascendc-ops-developer** | 算子开发者，支持代码实现、编译测试和精度验证 |
| **ascendc-ops-tester** | 算子测试者，支持ST/UT用例生成与执行 |
| **ascendc-ops-reviewer** | 代码检视专家，支持快速检视和全功能检视两种模式 |
| **ascendc-kernel-architect** | Kernel直调架构师，支持需求分析、API验证、方案设计 |
| **ascendc-kernel-developer** | Kernel直调开发者，支持代码实现、编译测试、性能采集、文档编写 |
| **ascendc-kernel-reviewer** | Kernel直调审查者，支持独立构建验证、7维度评分、精度验证 |

## TileLang2AscendC 算子开发

> 以下 Skill 位于 `plugins-community/tilelang2ascendc-ops-generator/skills/` 目录，属实验/非正式版本，可能存在平台或能力限制。

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

### Agents

| Agent | 功能 |
|-------|------|
| **catlass-op-architect** | Catlass 算子架构设计专家，需求分析与组件选型 |
| **catlass-op-generator** | Catlass 算子开发实现专家，模板拼装与框架代码生成 |
| **catlass-op-reviewer** | Catlass 算子代码审查专家，构建验证与 C1-C11 检视 |

## PyPTO 算子开发

### Skills（17 个）

| Skill | 功能 |
|-------|------|
| **pypto-orchestration-manual** | 编排者入口：团队原则 / 成员名册 / 强制规则 |
| **pypto-intent-understand** | 需求意图理解与规格生成 |
| **pypto-op-plan** | 需求规划：相似样例搜索与可行性评估 |
| **pypto-api-explore** | API 可行性探索与分析 |
| **pypto-docs-search** | API 文档 / 参考实现 / golden / 错误码检索 |
| **pypto-golden-generate** | Golden 参考实现生成 |
| **pypto-op-design** | 算子方案设计生成 |
| **pypto-op-construct** | 模块语义拆解与逐模块构建 |
| **pypto-op-develop** | 算子代码实现与测试 |
| **pypto-op-verify** | 验证 runner 与 `detailed_tensor_compare` 用法 |
| **pypto-op-review** | 逐算子 PyPTO 调用提取（调试用） |
| **pypto-precision-compare** | 精度中间结果对比分析 |
| **pypto-precision-debug** | 精度问题代码层排查 |
| **pypto-general-debug** | 卡住/不明失败的调试路由 |
| **pypto-op-perf-tune** | 算子性能分析与自动调优（含泳道图子技能） |
| **pypto-op-knowledge** | 算子开发经验表 / 问题查找表 |
| **pypto-memory-template** | 算子级 `MEMORY.md` 模板 |

### Agents（8 个 + 编排者）

编排者 `pypto-op-orchestrator`（AGENTS.md 注入，全流程唯一 owner）按 7 阶段调度 8 个子代理：

| Agent | 功能 |
|-------|------|
| **pypto-op-planner** | 需求规划：SPEC.md / API_REPORT.md |
| **pypto-op-mathematician** | Golden 参考实现 |
| **pypto-op-architect** | 架构设计：拆解决策、Tile、Loop 结构 |
| **pypto-op-designer** | 模块拆解与契约、staged 文件布局 |
| **pypto-op-coder** | 单文件 kernel 编码 |
| **pypto-op-verifier** | 裁判：对抗测试 + 精度/layout 校验 |
| **pypto-op-debugger** | 失败定位与补丁方案 |
| **pypto-op-optimizer** | 性能分析与自动调优 |

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
| **triton-op-generator** | Triton 算子端到端生成与优化 |

## Runtime

### Skills

<!-- skills-table-start -->
| Skill | 功能 | 使用样例 |
|-------|------|---------|
| **runtime_migration** | Runtime 接口迁移，提供兼容层与直接迁移两种方式，含设备/内存/流/事件/IPC/库管理/VMM 适配与错误码映射 | — |
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

### Agents

| Agent | 功能 |
|-------|------|
| **model-infer-analyzer** | 模型分析、方案设计、并行策略推荐 |
| **model-infer-implementer** | 代码改造、调试修复 |
| **model-infer-reviewer** | 精度验证、性能对比 |
| **model-infer-sota-scenario** | 构造可复现推理输入、跑通精度基线、定可机判的判定口径（sota-approach） |
| **model-infer-sota-profiling-instrumenter** | 用 model-infer-profiling 为场景插桩/启用采集并产出 profile（sota-approach） |
| **model-infer-sota-profile-analyzer** | 用 model-infer-perf-breakdown 跑性能分析，产出时间分布 + 逐算子实测/理论 gap（sota-approach） |
| **model-infer-sota-candidate** | 按候选来源并行发现优化候选，产出候选 Plan 草案（sota-approach） |
| **model-infer-sota-implementer** | 按 Plan 用单点技术 skill 实施单个优化并保留 enable 开关、自验证（sota-approach） |
| **model-infer-sota-reviewer** | 只复核不改码，验证 Plan 是否真实生效、精度/性能是否达验收口径（sota-approach） |

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
| **gitcode-issue-handler** | GitCode Issue 端到端处置，按内容自动选择 PR 代码变更路径或 Comment 答复路径 | [查看](skills-usage.md#gitcode-issue-handler) |
| **gitcode-toolkit** | GitCode API/Token/URL/日志/变更展示 + Git 克隆/diff/log/remote + PR 创建工作流共享参考（内部参考，不直接触发） | [查看](skills-usage.md#gitcode-toolkit) |

## 社区 Plugins

> 以下 Plugin 位于 `plugins-community/` 目录，属社区维护，可能存在平台或能力限制。

| Plugin | 功能 | 使用样例 |
|--------|------|---------|
| **science-model-npu-migration** | 框架级代码 NPU 迁移全流程：环境门禁、脚本适配、精度/性能对比与 mig_docs 交付归档；覆盖 torch_npu / MindSpore Ascend | — |
