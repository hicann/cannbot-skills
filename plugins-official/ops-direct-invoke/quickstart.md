# 算子直调开发工作流快速入门指南

三步跑通：安装 → 启动 → 说需求。详细工作流介绍见 [README.md](README.md)。

## 快速使用

### Step 1 运行 init 脚本

```bash
bash init.sh   # 默认：项目级 + opencode，安装到当前目录
```

其他用法：

```bash
bash init.sh global claude                              # 全局级 + 其他工具（可选 codex / dsh / trae）
bash init.sh project opencode /path/to/your_project     # 装到指定目录（也可省略 = 当前目录）
```

### Step 2 启动 Agent 工具

```bash
opencode    # 或 claude / codex
```

- **trae**：打开 TraeCode IDE
- **dsh**：启动 DeepSeek Harness

### Step 3 执行工作流

在工具对话中输入：

```
帮我开发一个 xxx 算子
```

一句需求即可启动完整工作流（需求分析 → 方案设计 → 代码开发 → 验收 → 上库）。复杂算子建议补全公式、输入输出规格。

## 输出

工作流运行时产物统一落在 `.cannbot/`：

```
.cannbot/
├── <算子名>/                   # 每个算子一个目录：状态、文档、验收报告、问卷
├── permissions/                # 角色权限配置
├── settings.json               # 运行时配置（模式 / 插件）
├── 环境信息.md                  # 环境统计
└── asc-devkit/ cann-samples/ ops-tensor/   # 依赖仓（首次安装克隆）
```

算子代码、测试、文档等最终交付物落在工作区各自专业目录，不进 `.cannbot`。

## 高级配置

**静默模式**（无人值守，自动推进到完成或阻断）：

```bash
bash init.sh project opencode --mode silent
```

或在会话中对 PM 说「开启静默模式」。

**流程插件**（PR 上库 / 性能迭代 / 经验总结，按需启用）：

```bash
bash init.sh --plugin-enable plugin-pr-submit on
```

## 常见问题

**安装后怎么验证成功？**

```bash
ls .opencode/skills/ .opencode/agents/ .cannbot/permissions/
```

（其他工具把 `.opencode` 换成对应的运行时目录。）

**如何更新？** 重新 `git pull` 后重跑 `bash init.sh`（幂等，已存在配置保留）。

**更多帮助？** `bash init.sh --help` 查看全部安装选项；详细流程与定制方式见 [README.md](README.md)。
