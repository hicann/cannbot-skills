# CANNBot-Insight Windows 原生环境搭建指南

> 面向零基础用户，从一台干净的 Windows 10/11 电脑开始，一步步把 CANNBot-Insight 跑起来。
>
> 预计耗时：20–30 分钟（含下载安装）。
>
> 适合人群：不想装 WSL、希望在原生 Windows cmd/PowerShell 里直接使用的用户。
>
> 如果你更习惯 Linux 命令行，或需要与 opencode 在同一 Linux 文件系统里协作，请改读 [WSL 搭建指南](setup-wsl.md)。

---

## 目录

1. [它是什么](#1-它是什么)
2. [环境要求](#2-环境要求)
3. [第一步：安装 Git](#第一步安装-git)
4. [第二步：安装 Node.js 20+](#第二步安装-nodejs-20)
5. [第三步（可选）：安装 Python 3](#第三步可选安装-python-3)
6. [第四步：获取源码](#第四步获取源码)
7. [第五步：一键启动](#第五步一键启动)
8. [第六步：首次使用](#第六步首次使用)
9. [常用操作](#常用操作)
10. [故障排查](#故障排查)
11. [下一步](#下一步)

---

## 1. 它是什么

CANNBot-Insight 是 LLM 编码 Agent（如 opencode、Claude Code）的 Session 级可观测工具。导入 Agent 的运行日志后，可以逐轮查看 Token 消耗、上下文增长、子 Agent 调度、Skill 事件、文件读写、工作流审计等 9 个维度的分析。

它由两部分组成：

- **Web UI 主程序**（必装）：Next.js 应用，提供浏览器可视化的全部 9 个分析 Tab。
- **smart-agent**（可选）：Python 写的 AI 审计后端（v2）。不装也不影响核心功能，只是 Audit Tab 的 v2 Python 分析用不了，v1 TypeScript 版仍可用。

本指南会把两部分都装好。

---

## 2. 环境要求

| 项目 | 要求 | 说明 |
|------|------|------|
| 操作系统 | Windows 10 1809+ / Windows 11 | 64 位 |
| 磁盘 | 约 1.5 GB | Node.js + 依赖 + 数据库 |
| 网络 | 能访问 gitcode.com 与 npm 官方源 | 公司代理需额外配置 |
| 浏览器 | Edge / Chrome / Firefox | 现代浏览器即可 |
| Git | 2.30+ | 拉取源码 |
| Node.js | **20.0.0+**（硬性要求） | v18.x 装不上 better-sqlite3 和 Prisma 6 |
| Python | 3.8+（可选） | 仅 smart-agent 需要 |

> ⚠️ Node 版本是硬门槛。v18.19.x 会在安装 `better-sqlite3` 原生模块时编译失败。启动脚本 `start.bat` 会自动检查并在版本过低时退出。

---

## 第一步：安装 Git

### 1.1 下载

打开浏览器访问：<https://git-scm.com/download/win>

页面会自动推荐 64-bit Git for Windows 安装包（如 `Git-2.45.x-64-bit.exe`），点击下载。

### 1.2 安装

双击运行安装包，关键选项如下（其余保持默认即可）：

- **Select Components**：保持默认，确保勾选了 `Git Bash Here`。
- **Adjusting PATH environment**：选 **`Git from the command line and also from 3rd-party software`**（默认项，这样 cmd 里能用 `git`）。
- **Configuring line ending**：选 **`Checkout Windows-style, commit Unix-style`**（默认 `core.autocrlf=true`）。
- 其余一路 `Next` → `Install` → `Finish`。

### 1.3 验证

新开一个 **cmd 窗口**（`Win+R` 输入 `cmd` 回车），执行：

```bat
git --version
```

看到类似输出即成功：

```
git version 2.45.0.windows.1
```

> 💡 如果提示 `'git' 不是内部或外部命令`：关闭所有 cmd 窗口重开（PATH 刷新），或检查安装时 PATH 选项是否选了第二项。

### 1.4（首次使用 Git 必做）配置身份

```bat
git config --global user.name "你的名字"
git config --global user.email "you@example.com"
```

---

## 第二步：安装 Node.js 20+

### 2.1 下载

访问 Node.js 官网：<https://nodejs.org/zh-cn/download>

选择 **LTS（长期支持版）** 的 **Windows Installer (.msi)** 64-bit 版本。确保版本号 ≥ 20.x（例如 `v20.x.x LTS`）。

> 💡 如果你之后需要在多个 Node 版本间切换，可改用 [nvm-windows](https://github.com/coreybutler/nvm-windows/releases) 安装。本指南用 MSI 一键装更简单。

### 2.2 安装

双击 `.msi`，一路 `Next`：

- **Custom Setup**：保持默认，会自动把 `node`、`npm`、`npx` 加入 PATH。
- 是否勾选 "Automatically install necessary tools"（含 Python 和 VS Build Tools）：**建议勾选**。这会顺带装好 Python 和 C++ 编译工具链，后面 `better-sqlite3` 编译原生模块时需要。如果没勾，后面遇到编译报错时再补装也行。

### 2.3 验证

**关闭并重开**一个 cmd 窗口（让 PATH 生效），执行：

```bat
node -v
npm -v
npx -v
```

预期输出（版本号以你装的为准）：

```
v20.x.x
10.x.x
10.x.x
```

> ⚠️ 如果 `node -v` 显示 `v18.x`，说明系统里残留了旧版 Node。请先在"设置 → 应用"里卸载旧版，再重开 cmd 验证。

---

## 第三步（可选）：安装 Python 3

> 仅当你在上一步 Node 安装里**没有**勾选 "Automatically install necessary tools" 时才需要做这步。用于 smart-agent（AI 审计 v2 后端）。

### 3.1 下载

访问 <https://www.python.org/downloads/windows/>，下载 **Python 3.11+** 的 64-bit 安装包。

### 3.2 安装（关键勾选）

双击运行，**安装界面底部务必勾选 `Add Python to PATH`**，然后点 `Install Now`。

### 3.3 验证

重开 cmd 窗口：

```bat
python --version
```

预期：

```
Python 3.11.x
```

> 💡 如果提示 Microsoft Store 的 python 别名拦截：到 `设置 → 应用 → 应用执行别名`，关掉 `python.exe` 和 `python3.exe` 的开关，再重试。

> 💡 不装 Python 也能用 Insight 的全部核心功能，`start.bat` 检测不到 Python 时会打印一行跳过提示，AI 审计 Tab 的 v1（TypeScript）路径照常工作。

---

## 第四步：获取源码

CANNBot-Insight 是 `cannbot-skills` 仓库里的一个子目录。我们克隆整个仓库再进去即可。

### 4.1 选择存放位置

建议放在一个**路径不含中文和空格**的目录，例如 `C:\dev`。先在 cmd 里创建并进入：

```bat
mkdir C:\dev
cd /d C:\dev
```

> ⚠️ 路径里含中文或空格（如 `C:\Users\张三\我的文档`）后续偶尔会导致 npm 脚本出错，请尽量避免。

### 4.2 克隆仓库

```bat
git clone https://gitcode.com/cann/cannbot-skills.git
```

首次克隆会下载较全量代码，耐心等待。看到 `Updating objects: 100% done` 即完成。

### 4.3 进入 cannbot-insight 目录

```bat
cd cannbot-skills\plugins-community\cannbot-insight
```

### 4.4 验证

```bat
dir start.bat
```

能看到 `start.bat` 文件即路径正确。

> 💡 想省磁盘只拉这一个子目录？可用 sparse-checkout：
> ```bat
> git clone --filter=blob:none --sparse https://gitcode.com/cann/cannbot-skills.git
> cd cannbot-skills
> git sparse-checkout set plugins-community/cannbot-insight
> cd plugins-community\cannbot-insight
> ```
> 新手建议用上面的全量克隆，更省心。

---

## 第五步：一键启动

`start.bat` 把所有脏活累活都做了：检查 Node 版本、安装 npm 依赖、按需重建 `better-sqlite3` 原生模块、创建 `.env`、运行 Prisma 数据库迁移、启动 smart-agent（如有 Python）、启动 Web 服务、打开浏览器。**你只需要双击它。**

### 5.1 首次启动（会自动安装）

在当前 cmd 窗口里执行（推荐，能看到完整日志）：

```bat
start.bat
```

或者直接在文件资源管理器里**双击 `start.bat`**。

首次运行会看到类似进度：

```
[setup] Node.js v20.x.x detected
[setup] Installing dependencies...
  ... (npm install 输出，首次约 2–5 分钟) ...
[setup] better-sqlite3 native module ... OK
[setup] Creating .env with DATABASE_URL...
[setup] Advanced tabs: 0 (use -a flag to enable)
[setup] Generating Prisma client...
[setup] Running Prisma migration...
  ... Applied migration ...
[start] Launching smart-agent (Python) on port 21026...
[start] smart-agent ready at http://localhost:21026
[start] Launching CANNBot-Insight on port 21025...
[start] Starting Next.js dev server...
[start] Waiting for server at http://localhost:21025...
[start] Server ready - opening http://localhost:21025
[start] Server running at http://localhost:21025
[start] Close this window to stop the server and smart-agent.
```

浏览器会自动打开 <http://localhost:21025>，看到 Insight 首页（一个空的会话列表）即**启动成功**。

> 💡 如果窗口一闪而过：说明脚本报错退出了。改在 cmd 里手动 `cd` 到目录后运行 `start.bat`，就能看到完整错误信息对照[故障排查](#故障排查)解决。

### 5.2 默认端口

- **Web UI**：`21025`（被占用时自动顺延到 21026、21027…）
- **smart-agent**：`21026`（仅当装了 Python）

### 5.3 何时算"装好了"

浏览器能打开 `http://localhost:21025` 并看到 Insight 首页，就说明环境完整。第一次进入是空列表，下一步导入数据。

---

## 第六步：首次使用

### 6.1 准备一份 Agent 日志

Insight 能导入两种日志：

| 来源 | 文件位置（在产生日志的机器上） |
|------|-------------------------------|
| opencode | `~/.local/share/opencode/sessions.db`（SQLite） |
| Claude Code | `~/.claude/projects/<hash>/sessions/<id>.jsonl`，也可指定整个目录批量扫描 |

> 💡 在 Windows 上，opencode 一般装在 WSL 里，`~/.local/share/opencode/sessions.db` 是 WSL 路径。Claude Code 的 `.jsonl` 文件可以通过 `\\wsl$\<发行版>\...` 或拷贝到 Windows 任意目录再导入。

把日志文件放到任意位置，例如 `C:\dev\sessions.db` 或 `C:\dev\claude-logs\`。

### 6.2 导入

1. 浏览器里在 Insight 首页点击 **"导入"** 按钮。
2. 选择源类型（opencode .db / Claude .jsonl / 目录）。
3. 选好文件后点导入。导入完成后会跳到会话列表。
4. 点击任意会话进入详情，依次浏览 **Overview → Turns → Workflow → Trace → Subagents → Skills → Interactions → AI Workflow → Context** 9 个 Tab。

### 6.3 也支持 CLI 上传（适合远程机器）

如果日志在一台 SSH 远程机器上、不方便拷贝，可在那台机器上用 CLI 一步上传：

```bat
npx tsx src\cli\index.ts upload --file .\sessions.db
```

上传后会交互式填写描述，后端自动起停。之后在 Web UI 导入时点 **CANNBay** 按钮直接选库导入。

---

## 常用操作

### 启动（日常）

每次开发机重启后，只需双击 `start.bat`。因为依赖已装好，启动会很快（10–20 秒）。

### 更新到最新版

```bat
git pull
start.bat -u
```

`-u` 表示先 `npm install` 拉新依赖，再迁移启动。

### 清缓存重编译

遇到编译异常时用：

```bat
start.bat -f
```

清掉 `.next` 缓存从头编译。

### 开启高级 Tab

```bat
start.bat -a
```

显示 Subagents / Interactions / AI Workflow 三个高级 Tab。

### 端口被占用想强制复用 21025

```bat
start.bat -k
```

杀掉占用 21025 端口的进程，固定用该端口。

### CLI 模式（无浏览器）

```bat
start.bat -c sessions
start.bat -c stats
start.bat -c session <taskId>
```

支持的子命令：`tui sessions session turn search compare stats import delete config`。完整参考见 [docs/cli-commands.md](cli-commands.md)。

### 停止服务

关闭 cmd 窗口，或在窗口里按 `Ctrl+C` 然后按任意键。Web 服务和 smart-agent 会随窗口一起关闭。

---

## 故障排查

### Q1：`'node' 不是内部或外部命令` / `'npm' 不是内部或外部命令`

Node 没加入 PATH。最简单办法：**关闭所有 cmd 窗口，重新打开**。若仍不行，重启电脑让系统刷新 PATH。还不行就重装 Node 并确保勾选了 "Add to PATH"。

### Q2：`start.bat` 一闪就关

脚本报错退出。在 cmd 里手动进入目录运行 `start.bat` 看完整错误。常见：

- Node 版本 < 20 → 卸载旧版重装 Node 20 LTS。
- npm install 超时 → 公司代理，配置：
  ```bat
  npm config set proxy http://你的代理:端口
  npm config set https-proxy http://你的代理:端口
  ```
  或换镜像源：
  ```bat
  npm config set registry https://registry.npmmirror.com
  ```

### Q3：`better-sqlite3` 编译失败 / `node-gyp` 报错

缺 C++ 编译工具链。装 [Visual Studio Build Tools](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)，勾选 "使用 C++ 的桌面开发"，或在装 Node 时勾选 "Automatically install necessary tools"。装完重开 cmd 重试 `start.bat -u`。

### Q4：端口 21025 已被占用

`start.bat` 默认会自动顺延到下一个空闲端口（看启动日志里 `Launching ... on port XXXXX`）。想强制杀占用进程用 `start.bat -k`。

### Q5：浏览器没自动打开

手动访问日志里打印的地址，通常是 <http://localhost:21025>。

### Q6：导入提示文件不存在 / 路径错

- 路径含中文或空格时用双引号包起来。
- WSL 路径在 Windows 里访问：`\\wsl$\<发行版名>\home\<用户>\.local\share\opencode\sessions.db`。

### Q7：Prisma 报 `DATABASE_URL` 找不到

`start.bat` 会自动创建 `.env`。若仍报错，手动在 `cannbot-insight` 目录下新建文件 `.env`，内容一行：

```
DATABASE_URL="file:./dev.db"
```

再重试。

### Q8：smart-agent 起不来 / Audit Tab 的 v2 报错

确认装了 Python 3 并在 PATH 里（`python --version`）。`start.bat` 启动日志里应能看到 `smart-agent ready at http://localhost:21026`。没看到就是 Python 缺失或端口被占。即使没起，Audit Tab 的 v1 路径仍可用。

### Q9：克隆仓库很慢 / 超时

gitcode 偶尔慢，可设代理：

```bat
git config --global http.proxy http://你的代理:端口
```

或用前面提过的 sparse-checkout 只拉子目录。

### Q10：升级 Node 版本后启动报原生模块 ABI 不匹配

`start.bat` 内置了自动检测 + `npm rebuild better-sqlite3`，正常会自愈。若仍失败，手动：

```bat
npm rebuild better-sqlite3
start.bat
```

---

## 下一步

- 读懂功能全貌：[README-zh.md](../README-zh.md)
- 看整体架构：[docs/architecture.md](architecture.md)
- 学 CLI 高级用法：[docs/cli-commands.md](cli-commands.md)
- 在多机器间分享 session：用 Web UI 里的 **CANNBay** 上传 / 导入功能

---

> 反馈与问题：请在仓库提 Issue。
