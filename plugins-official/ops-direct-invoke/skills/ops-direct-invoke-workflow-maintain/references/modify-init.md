# 修改 init 脚本

> 适用于：基类 `init.sh`（构造函数）、子仓 `<repo>/agent/init.sh`（派生构造）。
> init 决定「源文件 → 软链接」的绑定，是多态的落地点，改动直接影响所有接入仓能否正确初始化。

## 两个 init 的分工

- **基类 init**（`plugins-official/.../init.sh`）：搭建基础工作区——链接主 Agent、扁平化链接 agents、收集并链接 skills、链接权限 hook（opencode 插件 / claude settings.json 注册）、clone 依赖仓；`--override` 在同一次运行内用 `<dir>/skills/` 完成子仓 skill 覆盖。
- **子类 init**（`<repo>/agent/init.sh`）：通用派生脚本——拉基类仓 → 调一次基类 init（install_path 固定为仓根）→ 透传 `--override <repo>/agent`。不含仓名硬编码。

## 修改基类 init 的红线

1. **位置参数契约**：`[level] [tool] [install_path]` 的顺序与语义须保持兼容（对齐旧版 ops-direct-invoke）。目标工具支持列表由基类 `SUPPORTED_TOOLS` 单点维护与校验，子仓 init 只把首参原样透传、不校验工具名；新增工具 = 基类追加 `SUPPORTED_TOOLS` + 增加对应安装分支，**已分发的子仓 init 零改动**。多余位置参数报错（防未知工具名被静默误当 install_path）。
2. **override 参数契约**：`--override <dir>` 的名称与语义是子仓 init 依赖的接口，属对外契约，只增不破坏（S6）。展开时使用 `<dir>/skills/` 与 `<dir>/AGENTS.md`（存在时，见红线 8）。
3. **skill 收集顺序**（本地 skills/（含 `plugin-*/` 嵌套子 skill）→ 共享 ops/ → infra/）与 **override 匹配规则**（skill 同名替换+新增；`AGENTS.md` 存在即替换）若变更，须同步更新本 skill 的 SKILL.md 描述与 review-checklist。
4. 遵守 F2（机制优于自然语言）：能在 init 里用脚本保证的初始化行为，不要下放成 agent 的 prompt 要求。
5. **`example/init.sh` 兼容性**（review-checklist C1–C4）：`example/init.sh` 已分发到各子仓，改基类 init 的 CLI 契约后必须：
   - 对比基类参数解析与 `example/init.sh` 的调用方式是否仍兼容。
   - 不兼容 → 向用户发结构化问卷（受影响子仓 + 新旧用法对比），由用户决策是否本次迁移。
   - 用户选「暂不迁移」→ 基类保留旧参数兼容层（接受并 warn），至少维持一个版本后再移除。
6. **Step 4.5 权限配置生成契约**：从已链接的运行时 `skills/workflow-agent-permissions/hooks/`（opencode `.opencode/skills/`、claude `.claude/skills/`）整体复制到 `.cannbot/permissions/`。**缺失才生成、已存在保留**（工作区配置优先）。模板路径经软链接解析，自动获得子仓 override 版本。模板目录缺失仅 warn、不 fail（hook 走内置默认值兜底）。
7. **AGENTS.md 覆写**：`--override` 目录含 `AGENTS.md` 时，Step 2 链接子仓版为 PM 入口，且 skill 收集第二步同步改读子仓版 frontmatter——**两处读源必须同源**，否则子仓登记的 skill 收集不到。
8. **Step 5.5 工作流配置生成契约（唯一配置，聚合插件注册）**：`.cannbot/settings.json` 是工作流配置唯一文件（version / mode / surveyed / plugins / updated_at），由 Step 5.5 一次生成。**每次重扫重写**：扫描 `skills/plugin-*/`（基类 + override，override 同名优先）frontmatter 的 `workflow-hook` / `workflow-stages` / `standalone` 生成 `plugins`，保留各插件 `enabled`、并入新增（新增时顶层 `surveyed` 复位）、剔除失效；`workflow-hook` 须通过格式与挂载点存在性（基类流程表）校验、`workflow-stages` 必填，不合法仅 warn 不注册。`--mode interactive|silent`（非法值报错退出）写入 `mode`（未传保留现有值，首先生成默认 `interactive`）；`--plugin-enable <name> on|off` 直接改对应插件 `enabled`（未注册仅 warn）。旧版 `.cannbot/plugin-registry.json` 一次性迁移并入后删除，不再生成。两个参数均为可选，不改既有位置参数与 `--override` 语义；生成失败仅 warn 不 fail（工作流按默认交互模式运行）。


## 修改子类 init 的注意

1. 保持通用性：不写死仓名，路径全部基于 `SCRIPT_DIR` 推导（`REPO_ROOT` = 上一级）；这样任何满足 `<repo>/agent/{init.sh,skills/}` 结构的仓都能直接用。
2. 子仓 init 自身不打 banner（banner 由基类打，全流程仅一次）。
3. 分支/仓库地址等配置用脚本顶部常量，不轻易暴露为命令行参数（保持接口简洁）。

## 脚本编码

- 保留版权头。
- 处理含中文/空格的路径时用 `-z`/数组等安全方式，避免 word-splitting。
- 软链接一律用绝对路径（`realpath`），避免相对链接在不同 CWD 下失效。

## 收尾

改完自行跑一遍 init 验证（建议用隔离的临时安装目录，勿污染源码目录；可用 `--cannbot-skills`/本地依赖仓避免联网），确认软链接指向正确后执行 [common.md](common.md)。
