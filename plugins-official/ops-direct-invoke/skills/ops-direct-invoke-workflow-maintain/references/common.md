# 通用修改收尾

> 适用于所有文件类型的修改。每次修改完成后逐项检查。

## 修改后必做

### 1. 检查本 skill 是否需要同步更新

若修改改变了工作流结构（新增/删除阶段或角色、变更 init 部署方式、调整目录结构或链接关系），检查本 skill 的 SKILL.md 描述是否仍准确：
- 两层仓库与文件组织树
- init 链接机制表
- override 匹配规则表

若修改动了流程（阶段 / CP 点 / 回退关系），额外核对插件根 `README.md` 的「开发流程概览」表是否仍与 `ops-direct-invoke-workflow/SKILL.md` 的统一流程表一致——SKILL.md 是权威真值源，README 概览为派生，不一致则同步更新 README。

### 2. 运行 init 使配置生效

若改动涉及新增/删除 skill、agent，或改了 init 脚本，**必须自行**运行对应 init 使软链接生效，**禁止**让用户退出会话重新运行：
- 基类侧改动：`bash <PLUGIN_ROOT>/init.sh <level> <tool> <install>`
- 子仓侧改动：`bash <repo>/agent/init.sh <level> <tool>`

仅修改已存在 skill/agent 的**内容**（不增删目录/文件）时，软链接已指向源文件，无需重跑 init。

### 3. Agent 正文 skill 引用与 frontmatter 一致性（改 agent 时必查）

改动 agent 文件后，**扫描正文**中所有 skill 依赖声明——包括但不限于：
- "使用 `xxx` skill" / "加载 `xxx` skill"
- "引用 `xxx`" / "依据 `xxx`"
- "按 `xxx` 的指导" / "`xxx` 作为…"

对每一条，核对 frontmatter `skills:` 列表中是否已包含该 skill。**缺失即补，多余即删**（对应 review-checklist L7）。原因：init 只扫描 frontmatter 收集 skill，正文中引用但未声明的 skill 运行时加载会失败。

### 4. 按检视条款自查

对照 [review-checklist.md](review-checklist.md) 逐条检查，重点是契约兼容（改基类时）与链接一致性。

### 5. 询问用户是否需要独立检视

修改完成后，使用问卷询问用户是否需要由另一个 Agent 实例做一致性检视（对应「职责分离」——修改方与检视方不应为同一实例）。

## 契约变更专项（改基类时必查）

若本次修改动了基类对外契约（virtual 组件逻辑名、角色输入输出格式、调用约定、init 参数），必须：
1. 确认是**新增**而非破坏性变更；
2. 若确为破坏性变更，列出受影响的已接入仓，提供迁移路径，并显式通知；
3. 检查已接入仓（如 ops-blas）的 override 是否仍能按逻辑名匹配生效。

## `example/init.sh` 兼容性（改 `init.sh` 时必查）

`example/init.sh` 已分发到各子仓，基类无法自动更新它们。改动 `init.sh` 的 CLI 契约后**必须执行以下流程**：

1. **对比检测**：用 `diff` 或人工逐行比对基类 `init.sh` 的参数解析段与 `example/init.sh` 的调用方式，判断是否仍兼容。重点比对：
   - 参数名 / 参数值格式（如 `--override` / `--override-skills` 已删除等）
   - 位置参数顺序与数量
   - override 展开逻辑（目录结构要求）

2. **不兼容 → 发问卷知会用户**：以结构化问卷向用户确认：
   - 受影响的子仓清单
   - 新旧用法对比（旧：`--override-skills <path>/skills` → 新：`--override <path>`）
   - 用户是否选择「本次同步更新子仓 init」还是「暂不迁移，保留旧兼容层」

3. **用户选择「暂不迁移」时**：基类 `init.sh` 必须保留对旧参数的兼容（接受并 warn，不报错退出），至少维持一个发布版本后再移除。

