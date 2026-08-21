# CANNBot · Catlass DSL 算子开发快速入门

## 概述

`catlass-dsl-generator` 为 Ascend NPU CATLASS DSL 算子开发提供完整的工具链，覆盖设计、编码、基准测试、性能优化与知识管理五大阶段。内置**离线 OKF 知识库**，支持 API 查询、样例检索、问题排查与性能调优，无需网络连接。

---

## 一、安装

> 将 `<plugin-source>` 替换为本仓库的本地绝对路径、`owner/repo` 或 Git URL。安装前请确认来源可信。

### Codex

```bash
codex plugin marketplace add <plugin-source>
codex plugin add catlass-dsl-generator@catlass-dsl-generator-dev
```

运行 `codex plugin list` 确认插件存在，然后新建会话以加载 Skill。

### Claude Code

```bash
claude plugin marketplace add <plugin-source>
claude plugin install catlass-dsl-generator@catlass-dsl-generator-dev
```

重启 Claude Code，或在会话中运行 `/reload-plugins`。本地临时试用：

```bash
claude --plugin-dir <local-plugin-path>
```

### Cursor

将完整仓库放到 Cursor 的本地插件目录：

```bash
mkdir -p ~/.cursor/plugins/local
git clone <git-repository-url> ~/.cursor/plugins/local/catlass-dsl-generator
```

已有本地 checkout 时也可直接链接到 `~/.cursor/plugins/local/catlass-dsl-generator`。随后执行 `Developer: Reload Window`，在 `Settings > Plugins` 中确认插件已加载。

> GitHub 仓库也可在 Cursor Agent 中使用 `/add-plugin <git-repository-url>` 导入。

### OpenCode

在目标项目的 `opencode.json` 中添加：

```json
{
  "plugin": ["catlass-dsl-generator@git+<git-repository-url>"]
}
```

用户级安装则修改 `~/.config/opencode/opencode.json`。重启 OpenCode 后使用原生 `skill` 工具确认五个 Skill 均可发现。更多说明见 [`.opencode/INSTALL.md`](.opencode/INSTALL.md)。

---

## 二、快速上手

开发新算子只需两步：

> **第一步** — 使用 `catlass-dsl-design` 生成算子规格、初步设计、Torch
> `reference.py` 精度标杆、`definition.json` 和 `workload.jsonl` 用例：

```text
使用 catlass-dsl-design 设计一个 <算子名称> 算子。
输入、输出、dtype、shape、数值语义和性能目标是：...
```

> **第二步** — 检查生成的可读 `DESIGN.md`，确认接口、语义、允许路径和测试方法后批准。
> 严格的命令 argv 和状态机字段直接冻结在 develop run 根目录的 `state.json`：

```text
我批准 DESIGN.md，请使用 catlass-dsl-develop 按设计完成开发。
```

`develop` 会读取与 `DESIGN.md` 摘要绑定的 state config，自动完成**任务拆分 → 实现 → 调试 → Review → 测试 → Benchmark/Profiling**全流程。最终交付始终包含当前实现的性能现状和 candidate profiler 原始数据；仅当设计声明了性能门槛且初始实现未达标时，才会进入 optimize 阶段。

---

## 三、可用技能

| Skill | 用途 | 示例 |
|:------|:-----|:-----|
| [`catlass-dsl-design`](skills/catlass-dsl-design/SKILL.md) | 生成规格、Torch 精度标杆和 workload 用例 | `设计一个 float16 masked softmax 算子` |
| [`catlass-dsl-develop`](skills/catlass-dsl-develop/SKILL.md) | 按已批准的设计完成开发 | `按照 DESIGN.md 实现并完成测试` |
| [`catlass-dsl-bench`](skills/catlass-dsl-bench/SKILL.md) | 独立验证正确性与测量性能 | `用 solution、workload 和 definition 执行 benchmark` |
| [`catlass-dsl-optimize`](skills/catlass-dsl-optimize/SKILL.md) | 独立优化已有算子 | `优化 matmul，目标 mean_ms < 0.5` |
| [`catlass-dsl-knowledge`](skills/catlass-dsl-knowledge/SKILL.md) | 查询或维护本地 OKF 知识 | `查询 layout 不一致的排查方法` |

> **提示：** `debug` 和 `review` 已内置在 `develop` 流程中，无需单独调用。批准后的 `DESIGN.md` 不应直接修改；需求变化时请建立新的 develop run 并重新批准 Resolved Plan。

---

## 四、独立使用

### Benchmark

三个输入文件的接口定义见 [`catlass-dsl-bench/SKILL.md`](skills/catlass-dsl-bench/SKILL.md)。
NPU candidate 必须通过单融合 kernel anti-hack：每次 `run` 只 launch 一个源码声明的
`@tla.kernel`，并禁止用 Torch/Tensor 计算算子替代 DSL；失败结果不计性能。
candidate 只在逐 trial profiler 窗口内执行，并复用其中一次输出做正确性检查；
`anti_hack_manifest.json` 以 SHA-256 绑定每个 iteration 的单行 kernel CSV，供工作流复检。

```bash
python3 skills/catlass-dsl-bench/scripts/bench.py \
  --solution skills/catlass-dsl-bench/templates/solution.json \
  --workload skills/catlass-dsl-bench/templates/workload.jsonl \
  --definition skills/catlass-dsl-bench/templates/definition.json \
  --output .catlass-dsl/runs/example/benchmark
```

### Optimize

调用 optimize 时需提供算子路径、测试命令、benchmark 指标、性能目标和迭代上限：

```text
使用 catlass-dsl-optimize 优化 src/matmul.py。
完整测试命令为 ...，benchmark 指标为 performance.candidate.mean_ms，
目标小于 0.5 ms，最多迭代 8 轮。
```

### Knowledge

首次使用先初始化，之后可按关键词、标签或算子族查询：

```bash
# 初始化知识库
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py initialize \
  --project-root <project>

# 按关键词查询
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py query \
  --project-root <project> --text "layout"

# 按算子族 + 标签查询
python3 skills/catlass-dsl-knowledge/scripts/record_knowledge.py query \
  --project-root <project> --operator-family matmul --tag optimization
```

> 知识涵盖 DSL API、算子样例、调试方法、优化技巧及项目实测经验。查询仅读取本地文件；来源链接仅用于归因和可选审计。

---

## 五、使用要求

- 直接在当前 **Git workspace** 中运行，并明确允许修改的路径
- 测试、benchmark 和 profiling 命令应能从目标项目直接执行
- 融合算子每次入口调用只 launch 一个 CATLASS kernel，NPU 结果包含
  `anti_hack.status=passed` 和 `kernel_details.csv` 证据
- 缺少 CANN、NPU 或 CATLASS DSL 环境时，相关结果为 `not_run`，不会被视作通过

---

## 六、插件自检

```bash
python3 -m pytest
python3 -m compileall -q skills
```
