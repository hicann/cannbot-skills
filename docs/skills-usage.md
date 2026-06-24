# CANNBot Skills 使用样例

本文档汇总各 Skill 的典型使用样例。每个 Skill 给一段可直接复制、按需替换占位符的 prompt。

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


### ascendc-direct-invoke-to-registry-invoke

 `<<<>>>` kernel 直调转注册算子（算子迁移，不是从零开发）。

```
请使用 ascendc-direct-invoke-to-registry-invoke 技能，完成如下算子迁移：

【任务】将/path/rms_norm.asc <源kernel直掉文件绝对路径> kernel直调工程接入ACLNN/GEIR接口，生成在/path/rms_norm_single_op <迁移后的绝对路径>，目标芯片ascend910b <目标芯片版本，建议和当前运行环境一致，否则无法进行结果验证>

```

**使用建议**：

- 路径写**绝对路径**，让 skill 不必猜测源码位置。

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

> **如需同时更新 PR 文案 + 创建 Issue**，顺序调用两个 skill：
>
> ```
> /gitcode-pr-handler   https://gitcode.com/cann/ops-math/pull/1668
> /gitcode-issue-gen https://gitcode.com/cann/ops-math/pull/1668
> ```

### gitcode-issue-handler

读取 Issue → **自动判断要不要改代码** → 分两条路径：

- **PR 路径**：克隆 fork → 改代码 → 测试 → 提交 → 上游 PR（覆盖 bug 修复 / 功能增强 / 文档补全等任意代码变更场景）
- **Comment 路径**：只读克隆主仓 → 分析 → 答复评论（覆盖答疑 / 设计澄清 / 用法说明等不需改代码场景）

模式由 Step 1.5 决定，主要看 Issue 内容是否需要改代码；用户消息有"修复 / 提 PR"或"只回复 / 答疑 / 不改代码"会直接锁定。

**使用示例（PR 路径，典型代码变更）：**

```
/gitcode-issue-handler
issue_url=https://gitcode.com/cann/ops-math/issues/1511
fork_url=https://gitcode.com/your-name/ops-math.git
```

PR 路径下仅给 Issue 链接时会弹窗询问「自动 fork / 手动粘贴 fork 链接 / 取消」。

**使用示例（Comment 路径，典型答疑）：**

```
/gitcode-issue-handler
帮我答复一下 https://gitcode.com/cann/ops-math/issues/456 这个咨询问题
```

Comment 路径全程只读，不 fork、不 commit、不开 PR；只克隆上游主仓做分析后发评论。

**可选参数：**

| 参数 | 说明 |
|------|------|
| `issue_url` | GitCode Issue 链接（必填） |
| `fork_url` | 你 fork 出来的仓库链接（仅 PR 路径需要；缺省时交互询问） |
| `base_branch` | 上游目标分支（仅 PR 路径用），默认 `master` |
