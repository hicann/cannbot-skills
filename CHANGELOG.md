## 🔥 更新日志
### 【2026-07-21】
#### 新特性 New Features
- 【AscendC/Blaze 迁移】新增 `ascendc-blaze-migration` Skill。面向 Ascend 950 Matmul 家族，冻结 tiling 与 Blaze 架构，完成组件决策、原调用链逐字节一致性、性能验收和双仓 PR 交付；包含安装登记与 RotateQuant 闭环案例。

### 【2026-07-20】
#### 测试框架 Test Framework
- 【ST测试框架】评测用例格式迁移：将全部 95 个 `evals/evals.md` 转换为 `evals/evals.json` 统一格式。新增 `scripts/convert_evals.py` 转换脚本和 `evals_json_parser.py` JSON 解析器，修改 `common.py`/`conftest.py`/`main.py`/`test_*basic.py` 等 8 个文件以读取 JSON 格式。新格式兼容 ST 框架（结构化 type+pattern）和 skill-creator（纯文本 description）双消费者。

#### 问题修复 Bug Fix
- 【文档/安装部署】清理全仓库 `ops-lab/` 历史废弃目录的残留引用（issue #435），涉及目录结构图、路径引用、扫描配置及脚本逻辑等。

### 【2026-07-18】
#### 架构重构 Architecture Refactoring
- 【ST测试框架】重构评测用例组织架构：将集中式 `tests/system/cases/` 目录下的 95 个评测用例迁移到对应 Skill/Team 目录下的 `evals/evals.md`，实现测试用例与 Skill 代码内聚，便于维护和版本控制。
- 【测试框架】修改 `common.py`、`conftest.py`、`main.py`，替换为分布式发现逻辑，从 Skill/Team 本地目录扫描和加载评测用例。

### 【2026-07-13】
#### 问题修复 Bug Fix
- 【cannbot-insight】加固 `start.sh -c` 的 CLI 参数边界，命令与附加参数逐项传递，避免词分割和路径名展开。

### 【2026-07-06】
#### 特性增强 Feature Enhancement
- 【安装部署】全部 11 个插件（官方 + 社区）适配 CodeArts（华为云码道）工具，`init.sh` 安装脚本与 `quickstart.md` 文档新增 `codearts` 参数支持，总体已支持 OpenCode / Claude / Trae / Cursor / Copilot / CodeArts。

### 【2026-07-02】
#### 新特性 New Features
- 【模型训练】新增 3 个训练调测 Skill：model-train-accuracy-debug（精度异常定界）、model-train-log-visualization（训练日志可视化）、model-train-oom-analysis（OOM 内存诊断）。方法论适用于 torchtitan-npu 等 PyTorch on NPU 训练框架。

### 【2026-07-01】
#### 新特性 New Features
- 【Runtime】新增 Skill：`runtime_migration`，面向 Runtime 接口迁移，提供兼容层与直接迁移两种方式，含设备/内存/流/事件/IPC/库管理/VMM 适配实现与错误码映射；明确仅用于用户合法拥有或已获授权代码。

### 【2026-06-29】
#### 社区贡献 Community
- 【模型推理】新增 Skill：`science-model-npu-migration`，提供昇腾 NPU 框架级代码迁移全流程：环境门禁、脚本适配、精度/性能对比与 `mig_docs` 交付归档；覆盖 `torch_npu` / MindSpore Ascend，不含 ATC/OM 离线模型转换。

### 【2026-06-26】
#### 新特性 New Features
- 【模型推理】新增 Skill：`model-infer-harmony`，覆盖麒麟 NPU 端侧（Kirin9030 / HarmonyOS）ASR 模型 4bit W4A16 量化、omg 离线模型转换与 CANNPAK 打包全流程。
- 【算子直调，ops-direct-invoke】Step 6 增加精度验收（Reviewer 独立执行）。
- 【ST测试】新增 ops-direct-invoke Sigmoid 算子生成 ST 用例，配套引入 code_gen 评测模板与评分体系。

#### 问题修复 Bug Fix
- 【ops-registry-invoke】修复 Trae CN 环境下 skill 加载失败告警，移除 skill 工具硬编码，改用 `/skill-name` 通用调用方式。
- 【docs】移除 opbase 仓库已失效的外部文档链接。

### 【2026-06-25】
#### 新特性 New Features
- 【install-helper】新增交互式安装助手 CLI 工具（`@cannbot-ai/install-helper@1.0.3`），支持插件/Skill 安装、更新、卸载、健康检查等完整生命周期管理。发布到 npm，支持 `curl | bash`、`npx`、`npm install -g` 三种安装方式。
- 【安装部署】新增顶层 `install.sh` 一键安装脚本。

#### 文档 Documentation
- 【README】全面优化安装描述：双轨制安装（完整插件安装 + Agent Skills 标准 `npx skills add`）、安装方式表格化、手动安装脚本增加具体示例。

### 【2026-06-24】
#### 新特性 New Features
- 【仿真 ops-simulator】新增基于 `summary.json` 的性能瓶颈分析能力（瓶颈定位 + 优化建议），新增 5 个 reference 文档（性能指标参考、AIC/AIV/通用 issue 手册、issue 模板）。

#### 重构 Refactor
- 【算子直调，ops-direct-invoke】拆分 Developer 串讲模式为独立 Design Reviewer Agent（`ascendc-kernel-design-reviewer`）。

#### 问题修复 Bug Fix
- 【ops】`ops-precision-standard` 精度标准文档修复。

### 【2026-06-23】
#### 特性增强 Feature Enhancement
- 【ops-registry-invoke】集成 infra Skills 依赖，统一环境检查与基础设施能力。
- 【ops】补充 CV 融合算子流水间同步的知识。

### 【2026-06-22】
#### 新特性 New Features
- 【模型推理】新增 Skill：model-infer-quantization，接入 compressed-tensors 量化方案，覆盖权重/激活/KVCache 量化适配，并编排进 model-infer-optimize 量化阶段。
- 【模型推理】migrator / kvcache / parallel-impl 等主链 Skill 按 cann-recipes-infer 仓库框架重写，参考资料拆分框架部署与独立部署两种形态。

#### 特性增强 Feature Enhancement
- 【仿真 ops-simulator】新增基于 `summary.json` 的性能瓶颈分析能力：Quick Diagnosis 决策流程（核数 → 负载均衡 → 利用率 → Bound 类型 → 带宽交叉检查），新增 5 个 reference（指标阈值参考、general / AIC / AIV issue 手册、issue 模板）；支持 `cache` / `bandwidth` 缺失时的 fallback 信号。

#### 测试框架 Test Framework
- 【ST】增强门禁检查 — 添加 `--all` 参数、跳过报告生成、报告路径重构。

### 【2026-06-18】
#### 特性增强 Feature Enhancement
- 【triton】改进 GPU kernel 迁移策略。
- 【ops-registry-invoke-template】补充 matmul_blaze_example 样例。

#### 测试框架 Test Framework
- 【ST】为 `graph/` 和 `infra/` 共 10 个 Skill 添加 ST 测试用例，优化脚本日志打印。
- 【ST】新增 `ascendc-docs-search` 和 `ascendc-api-best-practices` 的 ST 评测用例（10 Cases）。
- 【benchmark】修复 opencode headless 模式下不 dispatch subagent 的问题。

### 【2026-06-17】
#### 新特性 New Features
- 【cuda-direct-invoke】新增 test gate harness。
- 【Matmul】新增 MM/GMM Skill。

#### 特性增强 Feature Enhancement
- 【ops-direct-invoke / ops-direct-invoke-flash】集成 infra Skills，统一环境检查与基础设施能力。
- 【ops-direct-invoke-flash】补充算子脚手架工程并简化用户界面。
- 【ops】新增 Blaze GroupMatMul delta 文档。
- 【ops-registry-invoke】支持调用ascendc-blaze-best-practice生成matmul类算子，并补充CV融合算子样例工程。
- 【模型推理】新增 Skill：model-infer-quantization，接入 compressed-tensors 量化方案，覆盖权重/激活/KVCache 量化适配，并编排进 model-infer-optimize 量化阶段。
- 【模型推理】migrator / kvcache / parallel-impl 等主链 Skill 按 cann-recipes-infer 仓库框架重写，参考资料拆分框架部署与独立部署两种形态。

#### 问题修复 Bug Fix
- 【code-review】REVIEW.md 多轮审查保留完整审计记录。

### 【2026-06-16】
#### 新特性 New Features
- 【TileLang】新增 TileLang 插件，支持通过 TileLang 表达完成 ASC 算子自动生成。
- 【aclnn 算子开发】更新代码检视流程为大规模并行检视流程。

#### 测试框架 Test Framework
- 【ST】为 24 个 ops Skill 添加 ST 测试用例，新增 `gate_check --repeat` 参数。

### 【2026-06-15】
#### 特性增强 Feature Enhancement
- 【Matmul】SWAT 算法更新策略并适配量化，新增参数分析能力。
- 【Kirin】补充端侧环境检查文档。

#### 测试框架 Test Framework
- 【ST】为 ops/ 下 Skill 添加 ST 测试用例。

### 【2026-06-13】
#### 文档 Documentation
- 【ops-direct-invoke-flash】新增 init.sh 安装脚本与 README 文档。

#### 测试框架 Test Framework
- 【TileLang】更新 ST 测试用例。
- 【model-infer-optimize】新增 CI evals 评测。

### 【2026-06-12】
#### 特性增强 Feature Enhancement
- 【ops】blaze Skill 支持后融合能力。
- 【st-test】统一 skill/team 报告，新增 Ascend Platform 平台过滤。

### 【2026-06-11】
#### 新特性 New Features
- 【benchmark】新增算子评测框架到 tests/benchmark。
- 【triton-op-generator】插件算子生成性能优化。

#### 特性增强 Feature Enhancement
- 【catlass-op-generator】改进 skills 和 agents。
- 【gitcode-toolkit】PR 创建流程新增 git 提交身份校验步骤。

#### 测试框架 Test Framework
- 【tests】新增 [file_contains] 检查类型，修复 CI 连接中断问题，修复 opencode 会话导出截断问题。

#### 问题修复 Bug Fix
- 【npu-arch】补全 Scalar 执行单元，修正 SIMD/SIMT VF 描述 (#263)。

### 【2026-06-12】
#### 新特性 New Features
- 【模型推理】新增 Skill：`model-infer-profiling`，用 `torch_npu.profiler` 正确采集 PyTorch 推理性能数据（强约束 `ExperimentalConfig(Level1 + PipeUtilization)`，否则 `kernel_details.csv` 缺列）。
- 【模型推理】新增 Skill：`model-infer-perf-breakdown`，把 `kernel_details.csv` + `trace_view.json` 按模型结构切成 component 实例并分桶，输出 wall/bubble 中位数与异常 layer 的单页 HTML。
- 【模型推理】新增编排插件 `model-infer-sota-approach`（`plugins-official/`）：baseline 之上由 profiling 驱动的多方向探索式优化编排 Team，与 `model-infer-optimize` 前后衔接、互补不替代。

#### 特性增强 Feature Enhancement
- 【模型推理，model-infer-multi-stream】从「自带 Plan 编排」改造为「多流技术知识包」，聚焦多流的分析 / 实现 / 验证规则，Plan / round 编排交由 `model-infer-sota-approach`。

### 【2026-06-10】
#### 新特性 New Features
- 【FA 算子】新增 FA 算子开发基础能力。
- 【TileLang】将 TileLang 移入 plugins-official。
- 【ops-qa-suite】创建 ops-qa-suite 能力。

#### 特性增强 Feature Enhancement
- 【算子直调，ops-direct-invoke】支持 Kirin 系列芯片开发，当前支持 Simulator 开发方式：**ascendc-direct-invoke-template** 直调模板新增 Kirin Vector 算子样例工程；**ascendc-env-check** 新增 Simulator 可运行性检查步骤；**npu-arch** 新增 Kirin 端侧硬件参数；**ops-direct-invoke workflow** 的 verify_environment.sh 新增 Simulator 支持情况检查。
- 【代码检视】代码检视能力迭代增强。

#### 测试框架 Test Framework
- 【ST】新增 npu-arch / ascendc-registry-invoke-template / ascendc-whitebox-design 三个 Skill 的 ST 评测用例。

### 【2026-06-09】
#### 特性增强 Feature Enhancement
- 【ascendc-docs-search】搜索接口迁移至 intelligent/search POST 接口。
- 【cuda2ascend-simt】改进触发描述。

#### 测试框架 Test Framework
- 【tests】修复多个 Bug、对齐三方规范并优化规则体系、UT 测试运行性能优化。

### 【2026-06-07】
#### 特性增强 Feature Enhancement
- 【PyPTO】完善算子开发编排能力并登记发布。

### 【2026-06-06】
#### 特性增强 Feature Enhancement
- 【PyPTO 算子】更新算子设计 Skill（pypto-op-design），重构设计方案生成与模板。
- 【PyPTO 编排】优化算子开发 Subagent（analyst / developer）。

### 【2026-06-05】
#### 新特性 New Features
- 【TileLang】整理 agent 结构，更新已有 skill 内容，新增环境检查和性能调优特性。

#### 特性增强 Feature Enhancement
- 【ascendc-precision-debug】以 reference 形式合入 ascendc-dumptensor Skill，补齐 kernel 内插桩调试能力，保持原子性、可独立增删。
- 【pypto-op-perf-tune】更新性能调优 Skill。

#### 测试框架 Test Framework
- 【ST】新增 ascendc-docs-gen / ascendc-ut-develop / ascendc-direct-invoke-to-registry-invoke 三个 Skill 的 ST 评测用例（28 Cases）。
- 【tests】version check 从 FAIL 降级为 WARN。

### 【2026-06-04】
#### 新特性 New Features
- 【ops-direct-invoke-flash】新增 Ascend C Kernel 从零构建插件。
- 【安装部署】支持 VS Code Copilot 一键安装。

#### 重构 Refactor
- 【ST】ST 框架稳定性修复、文档同步与评测用例全面启用。
- 【ops-direct-invoke】工作流中环境检查从 shell 改为 env-check 和 npu-arch skill。

#### 测试框架 Test Framework
- 【ST】评测稳定性优化与 Team 测试能力支持。

#### 问题修复 Bug Fix
- 【model-infer】恢复 reference repo 配置并修复 skill 链接。

### 【2026-06-03】
#### 新特性 New Features
- 【ops-direct-invoke-flash】新增 Ascend C Kernel 从零构建 Skill（`/ops-direct-invoke-flash`，位于 `plugins-official/`）：从 CPU 函数 / 数学公式 / 代码片段 / 文本描述出发，文档先行设计、分阶段实现、本地 + 远程 NPU 验证，并由子 Agent 评审。支持 Ascend950 / dav-3510 的 AscendC::Reg 原生编程。
- 【ops-profiling】新增 msprof 性能采集方法。
- 【性能优化】构建性能优化策略 Skill。

#### 重构 Refactor
- 【ops-direct-invoke】统一 subagent_type 引用格式，移除 `ops-direct-invoke:` 插件前缀，改为短名称格式（如 `@ascendc-kernel-architect`），简化配置并与平台调度机制保持一致。
- 【system-test】评审 MD 模板化、正向看护、按模型 token 预算、not_contains 修复。

#### 问题修复 Bug Fix
- 【init.sh】修复 global 模式安装时静默覆盖用户 `AGENTS.md/CLAUDE.md` 的问题。

### 【2026-06-02】
#### 新特性 New Features
- 【ops】合入 Ascend C SIMT 算子知识层。
- 【elementwise】添加 UB 常驻复用与 Bank 冲突规避优化设计文档。
- 【dfx】新增 dfx triage runner 和 performance diagnosis。

#### 特性增强 Feature Enhancement
- 【gitcode-issue-gen】通过分支路由集成 gitcode-issue-report。

#### 问题修复 Bug Fix
- 【ops-registry】增加 blackbox / whitebox / cp3 gates。
- 【ascendc-env-check】重构 npu-smi 解析方案，废弃主表格解析，采用结构化子命令 + 分层降级策略，增强测试与 eval 用例。

### 【2026-06-01】
#### 新特性 New Features
- 【ops-simulator】添加算子仿真的流水线空泡分析。
- 【plugins-community】将 EasyASC DSL 移至 plugins-community。

#### 特性增强 Feature Enhancement
- 【pypto-precision-compare】更新精度对比 Skill。
- 【pypto-precision-debug】更新精度调试 Skill。
- 【code-reviewer】Code Reviewer Agent 大版本更新。
- 【registry-invoke-template】修复文档幻觉。

### 【2026-06-02】
#### 新特性 New Features
- 【CAKE2 cake】新增 `plugins-community/collaborative-agent-kernel-evolution`：进化式 AscendC 算子生成 Team（仅 Vector 算子）。含 3 个 Agent（cake / cake-evo / cake-partial）与 19 个内置 Skill，覆盖「描述 → PyTorch 参考 → functional → AscendDSL baseline → DSL lowering 到 AscendC → 编译评估 →（可选）进化式多变体性能优化」全流程。自包含单插件，复用 cannbot `ascendc-docs-search`（改编为 `cake-docs-search` 并标明来源）。

### 【2026-05-29】
- 【代码检视】新增大型 PR 检视（>10 文件自动切换，按文件组并行，负载感知波次）和快速检视（零子 Agent，定向问题 inline 输出）两种工作流，配套 7 个 step 文件。方法论增强：负向证据分值降低防 AI 偷懒、SEC-11.3 新增逐位类型交叉验证检视方法。子 Agent 改为 Grep 定位 + Read offset/limit 按需读取条例章节，不再整篇加载。`check_bounds.py` 新增能力边界提示与手动推演指引。四仓历史数值安全数据集（33 例）。

### 【2026-05-28】
#### 新特性 New Features
- 【Skill 治理】新增 `cannbot-skill-reviewer` Skill，用于审查新增或修改的 `SKILL.md` 是否符合 CANNBot 入库要求，输出成文法门禁、判例法九维评分、阻塞项和整改建议。

#### 测试框架 Test Framework
- 【本地 fast 测试】修复 `tests/run-tests.sh --fast` 在无 AI CLI 环境下仍检查默认 `opencode` 的问题，使 fast 测试可自动降级为 `Platform: none`。

### 【2026-05-27】
#### 新特性 New Features
- 【ACLNN】支持 ACLNN 直接注册 PyTorch 进行自测试，在 ST 测试目录新增 torch/ 标准化测试结构（CMakeLists.txt、test.py、golden.py、compare.py、torch_adapter.cpp），同步更新 st-test-guide 与 ops-tester Agent。

#### 特性增强 Feature Enhancement
- 【ops-registry-invoke】新增模板穿刺验证与重试机制（v1.2.0），支持 A1-P 并行穿刺、A1-P-Retry 自动重试及算子迭代整合，重构验收阶段为精度/性能两级验收。同步更新 workflow 全部文档与 agents。

### 【2026-05-26】
#### 重构 Refactor
- 【算子直调，ops-direct-invoke】Step 1 环境检查从 `verify_environment.sh` + `environment.json` 改为 `/ascendc-env-check` skill 采集 + `workflows/templates/environment-template.md` 填空生成 `environment.md`；删除 `verify_environment.sh`（438 行）。Architect/Developer/Reviewer 同步切到 `environment.md`，`--npu-arch` 编译参数显式收敛到 `/npu-arch` skill。
- **本次重构未完成项**：目前 Step 2 门禁仅依赖 agent 自填的 Markdown 状态行（正则校验），失去了脚本时代 `validation.all_passed` 的确定性。后续需让 `/ascendc-env-check` 输出机器可读 sidecar（如 `docs/.env-check.json`），由 agent 在生成 `environment.md` 时双写，AGENTS.md 同时校验 sidecar 与 Markdown 一致，以闭环防止幻觉。

#### 新特性 New Features
- 【Ascend C】新增 `cuda2ascend-simt` 技能（实验版，位于 `ops-lab/`），支持将 CUDA 算子迁移到 Ascend C SIMT。**仅支持 Ascend 950 PR平台**。覆盖 `standalone sample`、`torch_npu`、`pybind` 三类交付形态，按原始工程形态选择对应产物。
- 产物固定输出在 `ported-ops/<operator_name>/`，附中文 `plan.md` 与 `README.md`，并在 Ascend 950 PR 上做硬件验证后才报 `success`。
- **当前不支持**：native JIT 路径（`nvrtc`、运行时编译、extension JIT 加载）、torch 复数 dtype 分支、device 侧 `double`（FP64）执行路径、CUDA 生态库依赖（cuBLAS / cuDNN / cuFFT / cuSPARSE / Thrust / CUB / NCCL 等）、协作组、Ascend C SIMD API、矢量编程 API。
- 【PyPTO】更新 Golden 生成 Skill（pypto-golden-generate），优化算子精度验证流程。

#### 测试框架 Test Framework
- 【PR 模板】统一测试报告命名规范，修正 PR 模板指引。

### 【2026-05-25】
#### 新特性 New Features
- 【性能调试】新增 Scalar 编码与诊断类目（ascendc-perf），扩展性能分析覆盖场景。
- 【代码检视】新增 MC²（Micro/Macro/Compute/Communication）领域规则，增强 ascendc-code-review 在通信与计算重叠场景的检视能力。
- 【PyPTO】更新需求理解 Skill（pypto-intent-understand），优化意图解析与规格生成准确性。
- 【文档测试】补充 ST 系统测试框架文档与本地开发调试指南，降低社区开发者学习门槛。

#### 测试框架 Test Framework
- 【跨平台稳定性】修复测试框架跨平台兼容性问题，新增自包含 HTML 报告，优化终端失败摘要与紧凑输出。
- 【失败修复指南】内嵌 UT 失败修复指南及一键复制提示词，提升测试失败排查效率。
- 【License 统一】统一全部测试脚本 License 声明为 CANN Open Software License 2.0。

#### 问题修复 Bug Fix
- 【环境检查】修复 npu-smi 表解析在特定设备布局下误判 NPU-occupied 的问题。
- 【工作流集成】修复 catlass-op-dev-workflow 错误引用为 fake skill 目录的问题，改为集成到 workflows/references/。

### 【2026-05-22】
#### 新特性 New Features
- 【UT 开发】新增 def dtype 二进制匹配能力，支持默认数据类型自动推断。
- 【Matmul】基于 Blaze 框架新增 matmul 单算子直调支持。
- 【治理模型】新增 CANNBot Skills 社区治理模型（GOVERNANCE.md），定义贡献者/Committer/SIG 角色与协作规则。

#### 特性增强 Feature Enhancement
- 【测试框架】优化 CI 门禁看护能力及系统测试（ST）框架，增强增量检测与变更驱动评测。

#### 问题修复 Bug Fix
- 【CI 门禁】修复 ST 任务检测遗漏及安装脚本路径问题。
- 【Trae IDE】修复 TRAE 初始化与 UT 测试的兼容性问题。

### 【2026-05-21】
#### 新特性 New Features
- 【Tiling】新增 tiling-solver Skill，提供自动化 Tiling 方案求解能力。

#### 特性增强 Feature Enhancement
- 【示例提示词】补充算子直调工程调用与验证的示例提示词。

### 【2026-05-20】
#### 新特性 New Features
- 【GitCode 协作】新增 infra/ 域 4 个协作 Skills：gitcode-pr-handler（PR 标题/描述自动生成）、gitcode-issue-gen（PR→Issue 关联生成）、gitcode-issue-handler（Issue 端到端处置）、gitcode-toolkit（API/Token/日志共享参考）。
- 【CI 入口】新增 skill 能力看护 CI 入口及脚本（gate_check.sh），支持变更驱动 ST 评测。

#### 特性增强 Feature Enhancement
- 【Triton】README 补充 triton-op-generator plugin 安装信息。
- 【diff 工具】升级 diff 工具以支持 cannbot 仓库的 diff 获取。

#### 问题修复 Bug Fix
- 【模型推理】修复 torch_npu_list 中算子接口链接错误。
- 【路径链接】修复多处路径引用错误。

### 【2026-05-19】
#### 新特性 New Features
- 新增支持 Triton 算子生成，适用于通过 Triton 开发高性能 Ascend NPU 算子。
- 采用 6 阶段工作流驱动，覆盖从任务构建到性能优化的完整生成流程，支持迭代修复与自动优化。

### 【2026-05-19】
#### 特性增强 Feature Enhancement
- 【环境检查】check_env.sh 新增 CANN 版本自动检测（从 version.info 读取版本号与运行时依赖基线），新增「CANN 版本兼容性」文档章节，补充版本信息来源、官方配套关系查询和常见版本问题 FAQ。
- 【快速入门】ops-direct-invoke / ops-registry-invoke 前置条件升级至 CANN Toolkit ≥ 9.0.0，ops-direct-invoke 补充芯片支持（Atlas A2/A3、910B/950PR）和 CLI 工具清单。

### 【2026-05-16】
#### 新特性 New Features
- 【官方插件】全面适配 Trae 全局安装，`ops-code-reviewer`、`ops-direct-invoke`、`ops-registry-invoke`、`pypto-op-orchestrator` 4 个插件统一支持 `global trae` 和 `project trae` 两种级别。

#### 重构 Refactor
- 【模型推理】将 `model-infer-optimize` 从 `model/teams/` 迁移到 `plugins-official/` 主线结构，新增完整 plugin 结构（AGENTS.md、hooks、workflows、init.sh、quickstart.md）。

#### 代码清理
- 删除旧版 OpenCode 插件安装机制的遗留产物 `package.json`。

### 【2026-05-15】
#### 特性增强 Feature Enhancement
- 【算子直调】将 RegBase 最佳实践指导集成到 `ops-direct-invoke` 工作流，Architect 负责 SIMD/MemBase vs RegBase 方案决策，Reviewer 承接 RegBase 路线审查。
- 【安装部署】init.sh 支持从任意目录执行安装，可通过 `install_path` 参数指定目标项目路径，配置文件统一安装到项目根目录或 `CONFIG_ROOT`。

#### 问题修复 Bug Fix
- 【UT 开发】修复 UT skill ascend950 SOC 问题，新增交互模式。

### 【2026-05-14】
#### 重构 Refactor
- 【Ascend C】Skill `ascendc-npu-arch` 重命名为 `npu-arch`，作为通用 NPU 架构知识技能供各领域共用。同步更新全量引用（ops、plugins、tests、docs）。

### 【2026-05-12】
#### 新特性 New Features
- 【图模式】torch-compile 加入 plugin-official，提供 PyTorch torch.compile 图模式编排入口。
- 【安装部署】将 TileLang 从 `ops-lab/tilelang` 转移至 `plugins-community/tilelang-op-orchestrator`。

#### 特性增强 Feature Enhancement
- 【测试框架】优化行尾扫描排除临时目录，PR 模板增加 UT_Test 故障排查指引。

### 【2026-05-11】
#### 新特性 New Features
- 新增 ascendc-crash-debug 技能，剥离 runtime-debug 中卡死/崩溃/挂起内容，与 precision-debug 形成三分类调试体系。
- 调整技能分类：env-check 归入调试与测试类，ops-profiling 归入工具辅助类。
- 【安装部署】CANNBot 官方插件新增支持 Cursor IDE 和 Trae IDE 安装，总体已支持 OpenCode / Claude / Trae / Cursor。

#### 文档更新
- 修正 README、quickstart 中仓库路径（skills/ → cannbot-skills/），清理废弃的 `.opencode/` 目录。

#### 测试
- 新增 CRLF 行尾检测测试和 `.gitattributes`。

### 【2026-05-09】
#### 新特性 New Features
- 新增 Simt C Style Skills。

#### 配置变更
- 添加 `.gitignore` 文件。

### 【2026-05-07】
#### 新特性 New Features
- 新增 kernel 直调工程接入 ACLNN / GEIR 接口的 skill（ascendc-direct-invoke-to-registry-invoke），支持 kernel 直调形式改造成自定义算子工程。

#### 问题修复 Bug Fix
- 【测试框架】修复 tests 测试框架无效/冗余测试项，补齐 test-version.sh 看护盲区，统一为 git-diff 变更检测模式。
- 【skill 迁移】配合主仓进行 skill 位置迁移。

### 【2026-05-06】
#### 新特性 New Features
- 新增算子注册调用的开发工作流（ops-registry-invoke），支持 ACLNN 和 GEIR 两种接入方式，覆盖需求分析到代码检视全流程。

### 【2026-04-30】
#### 配置 Configuration
- 【代码仓库更名】https://gitcode.com/cann/skills 更名为 https://gitcode.com/cann/cannbot-skills ，原名称和路径可继续访问，建议使用新名称和路径。

### 【2026-04-29】
#### 新特性 New Features
- 【工程模板，registry-invoke-template】新增注册调用自定义算子工程模板 Skill，提供标准工程结构、代码模板、UT / ST 样例和多芯片架构参考。

### 【2026-04-28】
#### 新特性 New Features
- 【安装部署】新增支持TRAE安装。

### 【2026-04-25】
#### 特性增强 Feature Enhancement
- 增强 ascendc-precision-debug 和ascendc-runtime-debug 的调试能力。

### 【2026-04-24】 
#### 新特性 New Features
- 新增 Ascend C 性能调优知识货架。

### 【2026-04-23】
#### 文档 Documentation
- 【使用样例】在 Readme.md 新增 Skills 的使用样例。

### 【2026-04-21】
#### 问题修复 Bug Fix
- 【测试框架】修复测试框架及识别到的多项校验问题，包括版本漂移自动恢复、文件内容质量检查等。

### 【2026-04-20】
#### 新特性 New Features
- 【最佳实践，regbase】新增 regbase 配置最佳实践知识。
- 【工程实践】新增 add / div 算子 fp16 / bf16 → fp32 类型转换实践。
#### 问题修复 Bug Fix
- 【环境检查，env-check】修复 verify_environment.sh 设备计数返回的bug。
- 【目录重构】统一算子目录命名（ops → operators）。

### 【2026-04-18】
#### 问题修复 Bug Fix
- 【性能分析，ops-profiling】修复 ops-profiling 技能名称不一致的问题。

### 【2026-04-17】
#### 新特性 New Features
- 【测试开发，st-design】新增精度模式自动生成。
- 【Tiling 设计，tiling-design】完善 broadcast tiling 设计文档。
- 【代码检视，ops-direct-invoke】新增初始化脚本和快速入门指南，完善 CANNBot 代码检视环境的搭建与使用说明。

### 【2026-04-16】
#### 新特性 New Features
- 【插件安装，Plugin】新增 Plugin 化安装体系，支持 Claude Code 和 OpenCode 两种插件安装方式：
  - Claude Code 用户：通过 `/plugin marketplace add` 注册，`/plugin install` 按 Team 安装。
  - OpenCode 用户：通过 `opencode plugin` 命令安装，支持项目级和全局安装。
  - 支持按 Team 精简安装（ops-direct-invoke / pypto-op-orchestrator），避免加载不需要的 Agents 和 Skills。
- 【Session Hook】ops-direct-invoke 和 pypto-op-orchestrator 两个 Team 新增 session-start Hook，会话启动时自动注入 CANNBot 上下文，无需用户手动执行初始化命令。
- 【模型推理优化】新增 NPU 模型推理端到端优化 Skill 体系（model-infer-*），覆盖框架适配、并行策略、KVCache/FA、融合算子、图模式适配等完整优化链路。
- 【模型推理优化】新增 3 个 SubAgent（model-infer-analyzer / model-infer-implementer / model-infer-reviewer），支持多角色协同的阶段化优化工作流。
- 【模型推理优化】新增 infer-model-optimize-team，通过 init.sh 一键安装推理优化环境。
- 【TileLang】新增 TileLang 算子设计和开发技能（ops-easyasc-dsl）。
#### 特性增强 Feature Enhancement
- 【安装方式，README】README 快速开始新增"方式一：Plugin 安装（推荐）"和"方式二：脚本安装"，按 Claude Code / OpenCode 分类说明安装步骤。
- 【算子直调，ops-direct-invoke】init.sh 和 quickstart.md 适配 Plugin 安装方式，支持按 Team 隔离配置。
- 【PyPTO，pypto-op-orchestrator】init.sh 和 quickstart.md 适配 Plugin 安装方式，支持按 Team 隔离配置。
- 【版本维护，tests】新增 Plugin 版本维护测试框架（tests/unit/teams/test-version.sh），支持版本一致性校验和自动化测试。
#### 问题修复 Bug Fix
- 【Plugin 安装】修复 OpenCode Plugin 安装问题。

### 【2026-04-14】
#### 新特性 New Features
- 【仿真，ops-simulator】新增Ascend 950 仿真的Skill：支持 Ascend 950 仿真，并且输出性能分析报告和流水线图。
#### 特性增强 Feature Enhancement
- 【UT单元测试，ascendc-ut-develop】支持针对 ops-transformer 算子仓的 UT、CSV 代码重构，分离数据与测试代码，提高调试效率。
- 【aclnn 接口测试用例设计，ascendc-st-design】新增支持aclIntArray / aclFloatArray / aclBoolArray / aclScalarList / aclIntArray类型接口生成ST用例。
#### 问题修复 Bug Fix
- 【算子直调，ops-direct-invoke】修改verify_environment.sh脚本，返回environment.json 固定为1的bug，应该按实际设备的npu count返回。
    
### 【2026-04-13】
#### 新特性 New Features
- 【Team调度】支持team级代码条例全量检视，review team 派发条例给代码检视。支持子agent 并行检视、验证，提升检视效果，降低上下文的压力。

### 【2026-04-10】
#### 新特性 New Features
- 【PyPTO】新增 Skill：pypto-api-explore，PyPTO API 探索与文档查阅。
- 【PyPTO】新增 Skill：pypto-golden-generate，Golden 数据生成与验证。
- 【PyPTO】新增 Skill：pypto-intent-understand，用户需求解析与规格生成。
- 【PyPTO】新增 Skill：pypto-op-design 算子方案设计，含快速参考和设计模板。
- 【PyPTO】新增 Skill：pypto-op-develop 算子开发实现，含错误排查、约束参考、测试模板和环境脚本。
- 【PyPTO】新增 Skill：pypto-op-perf-tune 性能分析与调优（frontend / incore / swimlane 三个子模块）。
- 【PyPTO】新增 Skill：pypto-precision-debug 精度问题定位与调试。
- 【PyPTO】新增 Skill：pypto-precision-compare 精度对比验证（含二分查找和自动化脚本）。
- 【PyPTO】新增 Agent：pypto-op-analyst: 算子分析 Agent。
- 【PyPTO】新增 Agent：pypto-op-developer: 算子开发 Agent。
- 【PyPTO】新增 Agent：pypto-op-perf-tuner: 性能调优 Agent。
- 【PyPTO】新增 Team：pypto-op-orchestrator 算子开发编排 （含初始化脚本和快速入门）。

### 【2026-04-09】
#### 特性增强 Feature Enhancement
 - 【Ascend C】【代码检视，ascendc-ops-reviewer】优化了 ascendc-ops-reviewer Agent 的检视流程，增加了多维度检视表格、代码侧别识别和置信度评定机制。新增 Ascend C 的 API 最佳实践、性能编码规范和 TopK 编码问题清单。

### 【2026-04-07】
#### 新特性 New Features
- 【Ascend C】【代码检视，ascendc-ops-reviewer】ascendc-ops-reviewer Agent支持GitCode PR的代码检视。
- 【Ascend C】【Kernel 架构】新增 Agent：ascendc-kernel-architect，<<<>>>直调支持多agent协同。
- 【Ascend C】【Kernel 架构】新增 Skill：ascendc-direct-invoke-template。
 
### 【2026-04-02】
#### 文档 Documentation
- 【开发规范】新增  CANNBot 开发规范，包含：Skill、Agents、Teams。
 #### 配置 Configuration
- 【Issue模板】新增 Issue 模板。

### 【2026-04-01】
#### 特性增强 Feature Enhancement
- 【Ascend C】【代码检视，ascendc-ops-reviewer】ascendc-ops-reviewer Agent 支持GitCode PR的代码检视。

### 【2026-03-26】
#### 新特性 New Features
- 【Ascend C】【白盒测试用例，ascendc-whitebox-design】新建 Skill：ascendc-whitebox-design，白盒测试用例。

### 【2026-03-25】
#### 特性增强 Feature Enhancement
- 【Ascend C】【代码检视，ascendc-ops-reviewer】搭建 ascendc-code-reviewer Agent的基础框架和工作流、支持检视条款的扩充和修改。

### 【2026-03-20】
#### 新特性 New Features
- 【Ascend C】新增 Skill：ascendc-api-best-practices，Ascend C 的 API 使用最佳实践。
- 【Ascend C】新增 Skill：ascendc-code-review，Ascend C 代码检视。
- 【Ascend C】新增 Skill：ascendc-docs-search，文档搜索。
- 【Ascend C】新增 Skill：ascendc-env-check，NPU 设备查询、CANN 环境验证。
- 【Ascend C】新增 Skill：ascendc-kernel-develop-workflow，七阶段工作流。
- 【Ascend C】新增 Skill：npu-arch，NPU 架构知识、芯片型号映射。
- 【Ascend C】新增 Skill：ascendc-precision-debug，算子精度调试。
- 【Ascend C】新增 Skill：ascendc-runtime-debug，算子运行时错误调试。
- 【Ascend C】新增 Skill：ascendc-st-design，接口测试用例设计。
- 【Ascend C】新增 Skill：ascendc-task-focus，任务聚焦，解决长任务“迷失在中间”的问题。
- 【Ascend C】新增 Skill：ascendc-tiling-design，Tiling 设计方法论。
- 【Ascend C】新增 Skill：ascendc-ut-develop。UT单元测试用例开发与覆盖率增强。
