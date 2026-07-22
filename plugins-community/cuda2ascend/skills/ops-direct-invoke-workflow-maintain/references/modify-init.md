# 修改 init 脚本

> 适用于：基类 `init.sh`（构造函数）、子仓 `<repo>/agent/init.sh`（派生构造）。
> init 决定「源文件 → 软链接」的绑定，是多态的落地点，改动直接影响所有接入仓能否正确初始化。

## 两个 init 的分工

- **基类 init**（`plugins-community/.../init.sh`）：搭建基础工作区——链接主 Agent、扁平化链接 agents、收集并链接 skills、链接 opencode 权限插件、clone 依赖仓；`--override` 在同一次运行内用 `<dir>/skills/` 完成子仓 skill 覆盖。
- **子类 init**（`<repo>/agent/init.sh`）：通用派生脚本——拉基类仓 → 调一次基类 init（install_path 固定为仓根）→ 透传 `--override <repo>/agent`。不含仓名硬编码。

## 修改基类 init 的红线

1. **位置参数契约**：`[level] [install_path]` 的顺序与语义须保持兼容（对齐旧版 ops-direct-invoke）。目标工具仅支持 opencode（`opencode` token 仍被兼容消费）。
2. **override 参数契约**：`--override <dir>` 的名称与语义是子仓 init 依赖的接口，属对外契约，只增不破坏（S6）。展开时只使用 `<dir>/skills/`。
3. **skill 收集顺序**（本地 skills/ → 共享 ops/ → infra/）与 **override 匹配规则**（skill 同名替换+新增）若变更，须同步更新本 skill 的 SKILL.md 描述与 review-checklist。
4. 遵守 F2（机制优于自然语言）：能在 init 里用脚本保证的初始化行为，不要下放成 agent 的 prompt 要求。
5. **`example/init.sh` 兼容性**（review-checklist C1–C4）：`example/init.sh` 已分发到各子仓，改基类 init 的 CLI 契约后必须：
   - 对比基类参数解析与 `example/init.sh` 的调用方式是否仍兼容。
   - 不兼容 → 向用户发结构化问卷（受影响子仓 + 新旧用法对比），由用户决策是否本次迁移。
   - 用户选「暂不迁移」→ 基类保留旧参数兼容层（接受并 warn），至少维持一个版本后再移除。
6. **Step 4.5 权限配置生成契约**：从已链接的 `.opencode/skills/workflow-agent-permissions/hooks/` 整体复制到 `.cannbot/permissions/`。**缺失才生成、已存在保留**（工作区配置优先）。模板路径经软链接解析，自动获得子仓 override 版本。模板目录缺失仅 warn、不 fail（hook 走内置默认值兜底）。


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
