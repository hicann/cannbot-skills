# CANNBot-Insight

LLM 编码 Agent 的 Session 级可观测工具。辅助长上下文分析、模型幻觉问题治理，以及 Agent Session 中上下文窗口增长的监控与优化。

**[English Documentation](README.md)**

## 功能介绍

导入 opencode sessions.db 或 Claude Code JSONL 日志，逐轮分析 Agent Session：

- **Token 与费用** — 每轮 token 五项拆解柱状图，按模型上下文窗口显示占比；根据使用量和模型定价估算费用
- **上下文增长** — 按 subagent session 展示上下文增长曲线；动画回放 context window 变化过程，含 subagent 生成/消亡标记；`/compact` 压缩标记与上下文下降标注，支持多次压缩
- **上下文治理** — 查看 LLM 输入上下文组成：可见消息 + 稳定的 "System (hidden)" 开销；输入窗口在每个 `/compact` 边界正确截断
- **Subagent 追踪** — 识别 subagent session，构建 dispatch→response 链路；SVG 连线图展示主↔子关系
- **Skill 事件** — 跟踪每轮 skill load/invoke/use 事件
- **Workflow** — 按 workflow skill 自身的进程标记（非 thinking 输出里的 阶段一/二/三/四；"后续可继续执行阶段二"等未来时态忽略）划分阶段；skill 按其 dispatch root turn 落入对应阶段，所以只完成到 阶段一 的 session 就显示为单个 阶段一——并非每个 session 都跑完整 workflow。无标记时退化为家族间隔启发式
- **概念追踪** — 跨轮次关键词搜索，查看传播链路和 DAG 图
- **文件读取分析** — 分析文件读取冗余，检测重复和不必要读取
- **Session 对比** — 对比两个 session 的 token、费用、耗时、工具调用和 subagent

## 方式一：Web UI

**需要 Node.js >= 20.x**（v18.19.x 无法安装 better-sqlite3 / Prisma 6）。如果有 nvm，`start.sh` 会自动切换到 Node 20 LTS。

日志文件位置：
- opencode: `~/.local/share/opencode/sessions.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`，也可指定目录自动扫描

```bash
./start.sh              # 自动安装 + 迁移 + 启动 Web UI，端口 21025
./start.sh -u           # 更新依赖 + 迁移 + 启动 Web UI
./start.sh -f           # 清除 .next 缓存，重新编译
```

浏览器打开 `http://localhost:21025`。导入日志文件后，点击 session 进入 9 个分析 Tab。

Web UI 还支持：导出 session 为独立 SQLite 或层级 Markdown；上传 session 到 CANNBay（带提交信息对话框）。

## 方式二：CLI 上传 + Web 分析

适用于 SSH 远程服务器、Web IDE 等无浏览器环境。CLI 一步完成导入和上传，之后在 Web UI 上分析。

日志文件位置：
- opencode: `~/.local/share/opencode/sessions.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`，也可指定目录自动扫描

```bash
# 从源文件一步上传（源类型根据文件自动识别）
npx tsx src/cli/index.ts upload --file ./sessions.db           # 多个 session 时交互式选择
npx tsx src/cli/index.ts upload --file ./logs/                 # Claude JSONL（目录）
```

上传后会交互式填写提交信息。后端自动启动，上传完成后自动关闭。

上传后在 Web UI 上查看分析：导入时点击 **CANNBay** 按钮，直接从仓库选择 DB 文件导入，无需手动下载。
