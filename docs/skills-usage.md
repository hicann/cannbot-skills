# CANNBot Skills 使用样例

本文档汇总各 Skill 的典型使用样例。每个 Skill 给一段可直接复制、按需替换占位符的 prompt。

📖 [安装指南](installation-guide.md) · [功能清单](feature-list.md) · [架构设计](architecture-design.md) · [README](../README.md)

## 前置条件速查

| Skill 类别 | 前置条件 | 说明 |
|-----------|---------|------|
| 代码编译/运行类 | CANN 开发环境 | 仅影响编译运行，知识检索不受影响 |
| GitCode 协作类 | `GITCODE_TOKEN` 环境变量 | 见下方 [!IMPORTANT] 提示 |
| 性能采集类 | NPU 设备 + CANN 环境 | 需真实设备 |
| 其他知识类 | 无 | 开箱即用 |

---

## Ascend C 算子开发

### ascendc-registry-invoke-to-direct-invoke

注册算子转 `<<<>>>` kernel 直调（算子迁移，不是从零开发）。

```
请使用 ascendc-registry-invoke-to-direct-invoke 技能，完成如下算子迁移：

【任务】将 rms_norm 算子从注册算子工程迁移到当前代码仓的 `<<<>>>` kernel 直调形式。

【源码路径】
- 算子原型与 tiling（host 侧）：<源工程 op_host 绝对路径>
- kernel 入口函数（device 侧）：<源工程 op_kernel 绝对路径>
- torch 接口定义（可选）：<torch_adapter 绝对路径>
- 原始测试脚本（用于精度对齐）：<test 脚本绝对路径>

【目标】
- 目标代码仓：当前工作目录
- 目标平台版本：dav-2201
- 交付标准：kernel + tiling + host 独立可编译运行，精度与原始测试脚本对齐
```

**使用建议**：

- 路径写**绝对路径**，让 skill 不必猜测源码位置。
- 明确**平台版本**（如 `dav-2201` / `dav-3510`），影响 cmake 配置与目标仓约定对齐。
- 没有 torch / 测试脚本时对应行可删，但"精度对齐"需至少保留一份可跑的原始用例作为参考系。
- 三原则（kernel 零修改 / tiling 数学零修改 / 只改框架胶水）、全量迁移、先确认交付边界等行为已内置在 SKILL.md，prompt 里不必重复。

**预期输出**：Agent 调用 `ascendc-registry-invoke-to-direct-invoke` skill，产出 kernel + tiling + host 独立可编译工程，附迁移说明与精度对齐报告。


### ascendc-direct-invoke-to-registry-invoke

 `<<<>>>` kernel 直调转注册算子（算子迁移，不是从零开发）。

```
请使用 ascendc-direct-invoke-to-registry-invoke 技能，完成如下算子迁移：

【任务】将/path/rms_norm.asc <源kernel直调文件绝对路径> kernel直调工程接入ACLNN/GEIR接口，生成在/path/rms_norm_single_op <迁移后的绝对路径>，目标芯片ascend910b <目标芯片版本，建议和当前运行环境一致，否则无法进行结果验证>

```

**使用建议**：

- 路径写**绝对路径**，让 skill 不必猜测源码位置。

**预期输出**：Agent 调用 `ascendc-direct-invoke-to-registry-invoke` skill，生成 ACLNN/GEIR 接口注册算子工程，含 tiling + host + kernel 完整结构。

### cuda2ascend-simt

将CUDA算子迁移到 Ascend C SIMT，根据原始工程形态选择 `standalone sample` / `torch_npu` / `pybind` 三类交付形态。**仅支持 Ascend 950 PR**。产物落在 `ported-ops/<operator_name>/`，附中文迁移说明文档 `plan.md` 与 `README.md`。

```
请使用 cuda2ascend-simt 技能，完成如下 CUDA → Ascend C SIMT 迁移：

【任务】将 <算子名> 从 CUDA 实现迁移到 Ascend C SIMT 实现。

【源码路径】（二选一）
- CUDA 源工程根目录：<源工程绝对路径>
- torch 算子根目录：<源工程绝对路径>

【目标】
- 输出目录：ported-ops/<算子名>/
```

**使用建议**：

- 路径写**绝对路径**，避免 skill 猜测源码位置。
- **不要**主动要求降级到 `standalone sample`：torch 扩展请保留 `torch_npu`，pybind 工程请保留 `pybind`，只有当依赖链或注册路径无法保留时才允许降级，且需在 `plan.md` 记录原因。
- **当前不支持迁移**的特性：native JIT（`nvrtc`、运行时编译、extension JIT 加载）、torch复数dtype分支、device 侧 `double`执行路径、CUDA 生态库依赖（cuBLAS / cuDNN / cuFFT / cuSPARSE / Thrust / CUB / NCCL 等）、协作组、Ascend C SIMD API、矢量编程 API。如源码包含上述特性，会以 `remove_and_record` 排除或上报 `blocked`，不会隐式替换或自实现生态库 / 协作组 / SIMD 等价物。
- 重大降级（抽象层 flatten、kernel 多分支合并为单一通路、device 路径降级为 host fallback 等）会触发硬停审批门，需用户显式选择后才会动手实现。
- 仅当在Ascend 950 PR硬件完成精度验证后才会报 `success`，否则按 `incomplete` / `blocked` / `failed` 处理。

**预期输出**：Agent 调用 `cuda2ascend-simt` skill，在 `ported-ops/<算子名>/` 下产出 Ascend C SIMT 实现，附 `plan.md` 迁移说明与 `README.md`。

### ascendc-code-review

算子代码检视。支持文件检视、PR 检视、快速定向排查，>10 文件自动切换大型 PR 模式。

**Plugin 模式[推荐]**（先 `cd plugins-official/ops-code-reviewer && bash init.sh project opencode`，详见 [quickstart](../plugins-official/ops-code-reviewer/quickstart.md)）：
```
帮我检视 split_core.cpp
全量检视 PR https://gitcode.com/cann/ops-transformer/pull/3604
检查 split_core.cpp 是否有数值溢出问题
```

**Skill 驱动模式：**
```
/ascendc-code-review 帮我检视 split_core.cpp
/ascendc-code-review 全量检视 PR https://gitcode.com/cann/ops-transformer/pull/3604
/ascendc-code-review 检查 split_core.cpp 是否有数值溢出问题
```

---

## Skill 治理工具

### cannbot-skill-reviewer

审查新增或修改的 `SKILL.md` 是否符合 CANNBot 入库要求，输出结构门禁、九维评分、阻塞问题和整改建议。

**使用示例：**

```
/cannbot-skill-reviewer 请审查这个新 skill 是否可以提 PR：ops/my-new-skill/SKILL.md
```

也可以直接运行本地审查脚本：

```bash
python infra/cannbot-skill-reviewer/scripts/review_skill.py ops/my-new-skill
```

**使用建议：**

- PR 场景下只审查新增或修改的 `SKILL.md` 及其随附 `references/`、`scripts/`、`assets/`。
- 自动门禁 `error` 是阻塞项；即使九维总分较高，也必须先修复。
- 涉及 NPU、CANN、torch_npu、性能或精度结论但无法实测时，报告中必须标记 `dry_run` 或 `partial`。

---

## GitCode 协作工具

> [!IMPORTANT]
> **前置条件**：所有 GitCode 协作 skill 都需要 `GITCODE_TOKEN` 环境变量（首次未设会在 Step 0 询问）。
>
> ```bash
> export GITCODE_TOKEN=********************
> ```
>
> 获取方式：登录 GitCode → 右上角头像 → **个人设置** → **访问令牌** → **新建访问令牌** → 勾选 `pull_requests`、`issues` 权限 → 生成并复制。

### gitcode-pr-handler

根据 GitCode PR 的代码变更，重新生成 PR 标题（约定式提交）与描述（沿用仓库 PR 模板），并通过 API 写回 PR。**只**处理 PR 标题与正文，不创建 Issue。

**使用示例：**

```
/gitcode-pr-handler https://gitcode.com/cann/ops-math/pull/1668
```

仓库无 PR 模板时降级到默认描述格式；交互节奏为「环境预检 + 终局确认」，中间无打断。

### gitcode-issue-gen

根据 GitCode PR 的代码变更，按变更类型自动选用 Issue 模板（feature-request / bug-report / documentation 等），生成关联 Issue 并完成 PR ↔ Issue 双向关联，**可选**自助 Assign 给当前 token 用户。

**使用示例：**

```
/gitcode-issue-gen https://gitcode.com/cann/ops-math/pull/1668
```

PR 描述中已识别到 `#issue_number` 时会询问"是否仍创建新 Issue"；Issue 创建成功后弹一次"是否 assign 给我"。

> [!TIP]
> **如需同时更新 PR 文案 + 创建 Issue**，顺序调用两个 skill：
>
> ```
> /gitcode-pr-handler   https://gitcode.com/cann/ops-math/pull/1668
> /gitcode-issue-gen https://gitcode.com/cann/ops-math/pull/1668
> ```

### gitcode-toolkit

GitCode 协作类 skill 的共享基础库（内部共享库，**不直接响应用户触发**）。为 `gitcode-pr-handler`、`gitcode-issue-gen`、`gitcode-issue-handler` 提供：

- **references/** — API 文档、环境预检、URL 解析、Git 操作等共享知识
- **scripts/** — 确定性操作脚本（无需 LLM 手工拼装命令）

**开发者使用脚本**：

```bash
# 解析 GitCode URL
python infra/gitcode-toolkit/scripts/parse_gitcode_url.py "https://gitcode.com/cann/ops-math/pulls/123"
# → {"owner":"cann","repo":"ops-math","type":"pr","number":123}

# 环境预检
bash infra/gitcode-toolkit/scripts/preflight.sh
# → JSON 结构化报告（token/git/curl/python3/tmp/git-author）

# PR 上下文一键获取
python infra/gitcode-toolkit/scripts/fetch_pr_context.py --repo cann/ops-math --pr 123
# → JSON: work_dir, base_branch, merge_base, changed_files, commits
```

**Skill 开发者引用**：其他 GitCode skill 通过相对路径引用 toolkit 的 references 和本文档章节，无需在自己的 SKILL.md 中重复实现。

### gitcode-issue-handler

使用统一状态机处理 GitCode Issue，公开名称和单 Issue 入口保持不变，同时支持：

- **显式单 Issue**：从 URL 推导目标仓库，不受批量时间窗或 `no_attention` 过滤。已有回复、责任人或 PR 作为诊断证据，用于避免重复动作。
- **当前仓库批量处理**：默认只对 `need_attention` 做获取、分类、诊断并生成动作预览；用户批准当前仓库、Issue 清单和动作范围后，才执行评论、指派或代码交付。
- **只回复**：用户明确说“只回复 / 答疑 / 不改代码”时，仅做有证据的文字诊断、回评与 GET 回查，不创建分支、commit、push 或 PR。

代码修复在受管 worktree 内执行，且必须通过环境一致性、稳定复现、最终根因、最小方案和测试门禁。单 Issue 的 PR 路径有三类业务确认点：修改前确认根因与方案；每条外部评论展示目标和完整正文后确认；验证完成后，用一次聚合确认覆盖精确暂存、commit、功能分支 push、创建 PR 和首次触发 CI。直接推送上游不包含在这三类授权中，commit 形成后还要单独确认 exact remote、目标分支和 commit SHA。PR 只创建、不自动合并。每次运行最后生成可审计的精简 Markdown 报告。

仓库特定配置、分类数据、缓存、复现证据和处理报告统一放在目标仓库的
`.cannbot/gitcode-issue-handler/`；最新报告为 `reports/latest.md`。新配置优先，仓根旧配置
只作只读兼容回退，新产物不再散落在仓根。

**首次安装（每个工具、每个安装级别只需一次）**：通过 CANNBot 现有安装机制把
`gitcode-issue-handler` 和 `gitcode-toolkit` 安装到同一工具的 `skills/` 目录。Python
已经可以导入 `requests` 和 `yaml` 时，可以跳过依赖安装命令。例如：

```bash
python3 -m pip install -r /path/to/cannbot-skills/infra/gitcode-issue-handler/requirements.txt
install-helper install gitcode-issue-handler gitcode-toolkit --tool opencode --level project
```

install-helper 将两者安装为独立 Skill。支持自动暴露 Skill slash 入口的 OpenCode 版本
可继续使用 `/gitcode-issue-handler`；其他版本以及通过
`npx skills` 安装的工具均可使用自然语言调用：`使用 gitcode-issue-handler 处理
<Issue URL>`。

Claude Marketplace 的 `infra-skills` 已同时包含两者；Codex 等其他工具也可用
`npx skills` 同时选择两项。完整安装、批量配置、更新和卸载说明见
[安装与配置指南](../infra/gitcode-issue-handler/docs/installation-guide.md)。安装完成后不需要
为每个 Issue 重复执行上述命令，直接在目标仓库目录中触发 Skill 即可。

以下 slash 示例适用于 Claude Marketplace，以及支持自动暴露 Skill slash 入口的 OpenCode。
其他客户端将示例首行替换为 `使用 gitcode-issue-handler` 即可。

**处理单个 Issue：**

```text
/gitcode-issue-handler
issue_url=https://gitcode.com/cann/ops-math/issues/1511
```

**只分析和回复，不修改代码：**

```text
/gitcode-issue-handler
只回复 https://gitcode.com/cann/ops-math/issues/456，不改代码
```

**批量预览，默认 dry-run：**

```text
/gitcode-issue-handler
处理当前仓库需要关注的 Issue
```

**明确授权批量执行：**

```text
/gitcode-issue-handler
对当前仓库最近 7 天的 need_attention Issue 批量自动执行，
按预览范围评论、指派、修复并创建 PR，不允许直接推送上游。
```

也可以先运行 dry-run，检查仓库、Issue 清单、动作范围和交付模式，再明确批准该次预览。
批准只覆盖预览中的范围；直接推送上游始终需要另行确认。

**预览长期未响应咨询 Issue：**

```text
/gitcode-issue-handler
预览当前仓库已答复且长期无响应的咨询 Issue，不要实际关闭。
```

自动闭环默认只预览。实际执行必须在检查结果后明确要求执行 `auto-close-stale` 并使用
`--apply`；未明确要求时不得评论或关闭 Issue。

单 Issue 模式不要求 YAML 或预建运行目录。批量模式可从当前仓库 remote 推导目标，也可
按需复制 `assets/` 中的配置模板；运行数据目录由流程创建。
