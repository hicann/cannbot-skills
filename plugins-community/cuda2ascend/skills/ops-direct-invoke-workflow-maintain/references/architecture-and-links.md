# 工作流架构与链接结构

> **首次维护本工作流前，先通读本文**。理解「源文件在哪、软链接怎么来、谁覆写谁」，才能判断改哪个源文件会影响哪个链接，避免改错层、破坏契约。

## 两层仓库与文件组织

工作流的源文件分布在**两个仓**，通过 init 软链接汇合到算子仓的运行时目录。维护时**永远改源文件，不改软链接目标**。

### 基类仓（CANNBot）

```
cannbot-skills/plugins-community/cuda2ascend/   # 基类插件根 = PLUGIN_ROOT
├── AGENTS.md                     # PM 主 Agent 配置（final，仅入口）：定位 PM 身份，
│                                 #   引导加载 ops-direct-invoke-workflow skill，本身不承载流程
├── init.sh                       # 基类构造函数：搭建工作区 + 绑定 virtual 组件
├── hooks/opencode/permission-guard.js   # 动态权限插件（opencode 侧，按角色限写）
├── hooks/claude/permission-guard.js     # 动态权限 hook（claude 侧，同一套规则语义）
├── agents/                       # 三类子 Agent 源文件（final，init 扁平化链接）
│   ├── architect.md              # 方案设计
│   ├── developer*.md             # 开发（含 -code/-test/-doc 按权限分的变体）
│   └── qa.md                     # 验收，加载 workflow-cp* Skill 完成各 CP 点验收
├── skills/                       # 基类默认 skill 实现
│   ├── ops-direct-invoke-workflow/           # 工作流编排 skill（final，承载整个流程）
│   ├── repo-*/                               # 仓库领域知识 skill（virtual，可被子仓 override）
│   ├── workflow-doc-templates/               # 交付件模板 skill（virtual，可被子仓 override）
│   ├── workflow-cp*/                         # 各 CP 点验收标准 skill（virtual，可被子仓 override）
│   ├── workflow-agent-permissions/           # 各 agent 权限规格（virtual，可被子仓 override）
│   └── ops-direct-invoke-workflow-maintain/  # 本维护 skill
└── README.md                     # 设计思想 + 设计约束章节
```

共享 skill 还可能来自插件外的 `../../ops/`、`../../infra/`（见下方解析顺序）。

> 设计约束原件仍在本维护 skill 的 `references/design-constraints.md`（勿改）。原 `docs/`（需求/约束/方案）已整合进 README（设计约束章节）与备份文件，不再随工作流保留。
>
> 使用指南 QUICKSTART.md 与接入指南 PLUGIN.md 已生成但第一版暂不随工作流发布，暂存于 `.cannbot/`。

## 工作流以 skill 承载（编排架构）

整个工作流编排**作为一个 skill 运作**（`ops-direct-invoke-workflow`），而非写进 AGENTS.md。分工：

- **AGENTS.md（PM）**：只做入口——定位 PM「只调度不执行」的身份边界，引导「面对请求先查是否有对应 skill，按 skill 指导的流程操作」。不写具体阶段/CP/回退逻辑。
- **`ops-direct-invoke-workflow` skill**：承载整个流程编排——阶段划分、各环节的角色/输入/输出/交付件、CP 验收与回退关系、状态机。PM 加载它后按其指导调度子 Agent。
- **该 skill 属 final**：是所有接入仓共享的稳定编排契约，子仓**不 override 它**；子仓的定制通过 override virtual skill（`repo-*` 领域知识、`workflow-cp*` 验收标准）实现，编排流程本身不变（对应开闭 O）。

这样，「做什么/按什么流程」集中在 workflow skill，「怎么做/依据什么领域知识」下放到 virtual skill，「谁来做」由 agents 承载，三者解耦。

#### `ops-direct-invoke-workflow` skill 内部组织

```
skills/ops-direct-invoke-workflow/
├── SKILL.md                      # 角色总览 + CP 标记说明 + 【统一流程表】+ 通用约定 + 参考资源
│                                 #   统一流程表内联于此（8 阶段粗体分组、15 步骤 + 8 CP 点、回退备注），
│                                 #   一眼见全貌；列：编号|流程|角色|输入|输出|说明|备注
└── references/                   # 明细按【关注点】切分（非按阶段）
    ├── task-prompts.md           # 各步 Task 调用契约：角色/输入/输出/验收标准/引用的逻辑名
    ├── data-flow.md              # 各步文件 I/O + .cannbot 目录布局 + 真值源约定
    ├── error-handling.md         # 回退关系表 + 过程有界最大轮次 + 错误类型判定
    └── state-schema.md           # .cannbot/state.json 字段定义与更新时机
```

维护要点：
- **改流程 = 改 workflow skill，不是改 AGENTS.md**。阶段/步骤/CP 的增删改在此。
- SKILL.md 守 F1：只放统一流程表（编排总览）与路由；每步明细在 references，按关注点组织——改回退阈值只动 error-handling，改调用契约只动 task-prompts。
- 交付件模板一律引用 `workflow-doc-templates` 逻辑名，**不在 workflow skill 内联模板**。
- 四个 references 相互引用（task-prompts↔data-flow↔error-handling↔state-schema）；改动其一时检查交叉引用一致性。

### 子类仓（算子仓，如 ops-blas）

```
<算子仓>/
├── agent/                        # 子仓 override 源 + 子类 init
│   ├── init.sh                   # 子类构造函数：拉基类仓 → 调基类 init → 透传 --override
│   └── skills/                   # 覆写基类同名 skill（按目录名匹配）：
│       ├── repo-*/               #   仓库领域知识 skill
│       ├── workflow-doc-templates/ # 交付件模板 skill
│       ├── workflow-cp*/         #   各 CP 点验收标准 skill
│       └── workflow-agent-permissions/ # 各 agent 权限规格
└── .cannbot/                     # 中间目录（gitignore），init 生成
    └── cannbot-skills/           # clone 下来的基类仓
```

### 运行时目录（init 生成，勿手改，勿提交）

init 把源文件软链接到算子仓根的运行时目录（OpenCode 的 `.opencode/` 或 Claude Code 的 `.claude/`，由 init 的 tool 参数决定）：

```
<算子仓>/
├── AGENTS.md            ->  基类 PLUGIN_ROOT/AGENTS.md
├── CLAUDE.md            ->  基类 PLUGIN_ROOT/AGENTS.md（仅 claude）
├── .opencode/agents/*.md  ->  源 agent 文件（基类 agents/ 扁平化，全 final）
│   （claude 为 .claude/agents/*.md）
│   （codex 为 .codex/agents/*.toml，从 hooks/codex/*.toml 生成，
│     __CANNBOT_AGENT_SOURCE__ 占位符替换为 agents/*.md 绝对路径）
├── .opencode/skills/*/    ->  源 skill 目录（被 override 的指向子仓 agent/skills/）
│   （claude 为 .claude/skills/*/）
│   （codex 为 .agents/skills/*/，非 .codex/skills/）
├── .opencode/plugin/      ->  权限插件 permission-guard.js
│   （claude 为 .claude/hooks/permission-guard.js + .claude/settings.json 注册 PreToolUse）
│   （codex 无权限 hook，init 时输出 WARNING 告知降级）
├── .cannbot/              ->  中间文件 + asc-devkit + cann-samples
└── .cannbot/permissions/*.js  ->  init Step 4.5 复制自 workflow-agent-permissions skill（角色配置文件）
```

## init 链接机制（软链接怎么来的）

理解软链接的来源，才能判断改哪个源文件会影响哪个链接。

### 基类 init（`plugins-community/.../init.sh`）

| 步骤 | 动作 | 链接关系 |
|------|------|---------|
| 1 | 建 `.cannbot/` 中间目录 | — |
| 2 | 链接配置文件 | `<install>/AGENTS.md` → `PLUGIN_ROOT/AGENTS.md`；claude 时额外 `<install>/CLAUDE.md` → 同源 |
| 3 | **扁平化**链接 agents | 递归 `agents/`，按 basename 链接到运行时 `agents/*.md`（opencode `.opencode/agents/`、claude `.claude/agents/`）；**codex** 从 `hooks/codex/*.toml` 生成 `.codex/agents/*.toml`（`__CANNBOT_AGENT_SOURCE__` 替换为 `agents/*.md` 绝对路径） |
| 4 | 链接 skills | 收集到的每个 skill 名 → 运行时 `skills/<name>`（opencode `.opencode/skills/`、claude `.claude/skills/`、**codex `.agents/skills/`**）；带 `--override <dir>` 时用 `<dir>/skills/` 同名替换、基类没有的新增 |
| 4.5 | 生成权限配置 | 从运行时 `skills/workflow-agent-permissions/hooks/` **复制**（非软链接）到 `.cannbot/permissions/`，**缺失才生成、已存在保留**（工作区配置优先） |
| 4+ | 权限 hook | opencode：`hooks/opencode/permission-guard.js` → `<install>/.opencode/plugin/`；claude：`hooks/claude/permission-guard.js` → `<install>/.claude/hooks/`，并在 `.claude/settings.json` 幂等注册 PreToolUse hook（matcher `Write\|Edit\|MultiEdit\|NotebookEdit`）；**codex：无权限 hook**，init 输出 WARNING 告知降级（角色隔离靠 prompt，非机制保证） |
| 5/6 | clone asc-devkit / cann-samples 到 `.cannbot/` | — |

**skill 收集来源（两步取并集去重）**：① 枚举本地 `skills/` 下所有目录；② 解析每个 agent（`agents/*.md`）frontmatter 的 `skills:` 列表。

**skill 源解析顺序**：本地 `skills/` → 共享 `../../ops/` → `../../infra/`。同名时靠前者优先。

### 动态权限 hook（opencode / claude）

按角色限制写权限——`.cannbot` 所有角色可写（任意类型）、项目外拒绝、code/test/doc 按角色限权：

- **实现**：opencode 用 `hooks/opencode/permission-guard.js`，在 `tool.execute.before` 里 throw 阻断违规写入；claude 用 `hooks/claude/permission-guard.js`，作为 PreToolUse hook 读 stdin 事件 JSON，违规时 exit 2（stderr 回传模型）。两者同一套规则语义。
- **加载**：opencode 由 init Step 4+ 链接到 `<install>/.opencode/plugin/` 自动加载；claude 链接到 `<install>/.claude/hooks/` 并在 `.claude/settings.json` 注册。
- **角色来源**：opencode 用 sessionID → `client.session.get` 反查当前 agent 名（实测坐实）；claude 用 hook 输入的 `agent_type` 字段（主线程无该字段，即 PM）。
- **规格真值源**：`skills/workflow-agent-permissions/hooks/*.js`，每角色一文件（ESM `export default { categories, exts }`），init Step 4.5 复制到 `.cannbot/permissions/`；hook 扫描该目录逐个 import、按文件名即角色名 per-agent 合并到内置默认值（默认值为纯防御兜底，与 skill 文件保持同步——L8 约束，**两侧 hook 的内置默认值都要同步**）。
- **覆写方式**：子仓整体 override `workflow-agent-permissions` skill，或仓内直接编辑 `.cannbot/permissions/<Role>.js`。
- **PM 启动闸口**：PM 每次会话开始检查 `.cannbot/permissions/` 是否齐全（7 个角色文件），异常则拒绝执行任务并提示用户退出当前 CLI 会话重跑 init.sh。
- **热更新**：opencode hook 在 plugin 启动时加载一次，修改配置后需重启 opencode 生效；claude hook 每次调用都是独立进程，配置即改即生效。
- **codex 降级**：codex 无 PreToolUse / `tool.execute.before` 等拦截点，不部署权限 hook。角色写权限隔离由 AGENTS.md prompt 约束，非机制保证（违反 F2，已显式接受）。init 时输出 WARNING 告知用户。

### 子类 init（`<算子仓>/agent/init.sh`）

通用脚本（不含仓名硬编码），放到任何「`<repo>/agent/init.sh` 且 `agent/` 下有 `skills/`」的仓即可用（agents 为 final、子仓不 override，故 `agent/` 下无 `agents/`）。两步：

1. 在仓根建 `.cannbot/`，clone/更新基类仓到 `.cannbot/cannbot-skills`。
2. **一次**调基类 init，install_path 固定为仓根，透传 `--override <repo>/agent`（展开为 `<repo>/agent/skills`）。

即：基类 init 在**同一次运行内**先建基础工作区，再用子仓 override 目录替换同名组件。banner 只出现一次（子仓 init 自身不打 banner）。

### override 匹配规则（虚实现绑定）

| 类型 | 匹配依据 | 行为 |
|------|---------|------|
| skill | override 目录下的**子目录名** | 同名替换基类默认；基类没有的**新增** |

**关键**：工作流编排始终通过**逻辑名**引用组件（如 `repo-coding-rules`、`workflow-cp5`）。子仓覆写的是「逻辑名 → 实现」这层绑定，编排层不感知底层实现被替换（依赖倒置 + 里氏替换）。所以子仓 override 时**逻辑名必须与基类一致**（skill 目录名、SKILL.md 的 `name:` 字段都要对齐），否则匹配不上，退化为「新增」。
