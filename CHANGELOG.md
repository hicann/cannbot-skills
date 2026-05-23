## 🔥 更新日志
### 【2026-05-23】
#### 新特性 New Features
- 【Issue 模板】新增 `request-for-comments` RFC 提案模板，对齐 Rust/React RFC 格式，`gitcode-issue-gen` 同步支持。

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