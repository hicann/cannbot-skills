# CANNBot-Insight WSL 环境搭建指南

> 面向零基础用户，从一台干净的 Windows 10/11 电脑开始，在 WSL2（Linux 子系统）里把 CANNBot-Insight 跑起来。
>
> 预计耗时：25–40 分钟（含 WSL 与 Node 安装）。
>
> 适合人群：习惯 Linux 命令行、或 opencode 装在 WSL 里、想和 Insight 在同一文件系统协作的用户。
>
> 如果你想直接在 Windows 原生 cmd 里跑、不装 WSL，请改读 [Windows 原生搭建指南](setup-windows.md)。

---

## 目录

1. [为什么用 WSL](#1-为什么用-wsl)
2. [环境要求](#2-环境要求)
3. [第一步：安装 WSL2 与 Ubuntu](#第一步安装-wsl2-与-ubuntu)
4. [第二步：WSL 基础配置](#第二步wsl-基础配置)
5. [第三步：安装 Git](#第三步安装-git)
6. [第四步：安装 Node.js 20+（用 nvm）](#第四步安装-nodejs-20用-nvm)
7. [第五步（可选）：安装 Python 3](#第五步可选安装-python-3)
8. [第六步：获取源码](#第六步获取源码)
9. [第七步：一键启动](#第七步一键启动)
10. [第八步：首次使用](#第八步首次使用)
11. [常用操作](#常用操作)
12. [故障排查](#故障排查)
13. [下一步](#下一步)

---

## 1. 为什么用 WSL

CANNBot-Insight 的 Web 服务本身跨平台，但用 WSL 有几个好处：

- **与 opencode 同文件系统**：opencode 的 `~/.local/share/opencode/sessions.db` 在 WSL 里是原生路径，导入不用跨系统拷贝。
- **与 start.sh 完全对齐**：项目主启动脚本是 `start.sh`，在 Linux 下行为最完整（自动起 smart-agent、自动用 Windows cmd 开浏览器）。
- **Linux 工具链齐全**：`curl`、`lsof`、`python3` 开箱即用，不用单独装。

WSL2 是微软官方的 Linux 子系统，跑在真实 Linux 内核上，性能接近原生。本指南全程使用 **WSL2 + Ubuntu**。

---

## 2. 环境要求

| 项目 | 要求 | 说明 |
|------|------|------|
| Windows | Windows 10 2004+（内部版本 19041+）/ Windows 11 | 需支持 WSL2 |
| 虚拟化 | BIOS 里开启 VT-x/AMD-V | WSL2 依赖 Hyper-V 虚拟化 |
| 磁盘 | C 盘剩余 ≥ 5 GB | WSL 镜像 + Node + 依赖 |
| 内存 | ≥ 8 GB | 建议 16 GB |
| 网络 | 能访问 gitcode.com、npm、Ubuntu 源 | 公司代理需额外配置 |
| Windows 浏览器 | Edge / Chrome / Firefox | WSL 里启动后会自动调用 Windows 浏览器打开 |

> ⚠️ 公司电脑若禁用虚拟化或装了第三方杀毒拦了 Hyper-V，WSL2 起不来，请改用 [Windows 原生指南](setup-windows.md)。

---

## 第一步：安装 WSL2 与 Ubuntu

### 1.1 一键安装（推荐）

以**管理员身份**打开 **PowerShell**（开始菜单搜 "PowerShell" → 右键 → "以管理员身份运行"），执行：

```powershell
wsl --install
```

这条命令会自动完成：

- 启用 WSL2 与虚拟机平台功能
- 下载 WSL2 Linux 内核
- 安装默认发行版 **Ubuntu**（LTS 版）

完成后会提示重启电脑。**重启。**

> 💡 Windows 11 较新版本里 `wsl --install` 已是默认 WSL2。Windows 10 旧版可能需要手动指定 `wsl --set-default-version 2`。

### 1.2 首次进入 Ubuntu

重启后会自动弹出（或在开始菜单点 "Ubuntu"）一个 Linux 终端窗口，要求设置：

- **用户名**：小写字母，如 `dbz`（不要用中文）。
- **密码**：输入时屏幕不显示任何字符是正常的，敲完回车即可。

> ⚠️ 别忘了这个密码，后面 `sudo` 要用。

设置完看到提示符 `用户名@电脑名:~$` 即进入 Ubuntu。

### 1.3 验证 WSL 版本

回到 PowerShell 执行：

```powershell
wsl -l -v
```

应看到 `Ubuntu` 状态 `Running`、`VERSION` 列是 `2`。若 VERSION 是 1，转换：

```powershell
wsl --set-version Ubuntu 2
```

### 1.4（可选）更新到最新 Ubuntu 包

在 Ubuntu 终端里：

```bash
sudo apt update && sudo apt upgrade -y
```

输入你刚设的密码。这一步会让后续装包更顺。

---

## 第二步：WSL 基础配置

### 2.1 解决时间可能不同步

WSL 偶尔出现时钟偏移（导致 git/npm 报 SSL 时间错误）。修一次：

```bash
sudo hwclock -s
date
```

日期时间正确即可。

### 2.2 配置 npm / git 代理（公司网络才需要）

如果你在公司内网需要代理：

```bash
git config --global http.proxy http://用户名:密码@代理:端口
npm config set proxy http://代理:端口
npm config set https-proxy http://代理:端口
```

家用网络跳过这步。

### 2.3 换 npm 国内镜像（国内家用网络推荐）

加速依赖安装：

```bash
npm config set registry https://registry.npmmirror.com
```

### 2.4 配置 Windows 防火墙放行

首次启动 Insight 时，Windows 会弹"允许防火墙通过"提示，勾选**专用网络**允许即可。如果没弹、浏览器访问不了，在 PowerShell（管理员）临时关闭防火墙测试：

```powershell
Set-NetFirewallProfile -Profile Private -Enabled False
```

确认能访问后再开回来。

---

## 第三步：安装 Git

Ubuntu 默认带 Git，但版本可能较老。先验证：

```bash
git --version
```

有版本号（如 `git version 2.42.0`）就行。若提示没装：

```bash
sudo apt install -y git
```

配置身份（首次使用必做）：

```bash
git config --global user.name "你的名字"
git config --global user.email "you@example.com"
```

---

## 第四步：安装 Node.js 20+（用 nvm）

> ⚠️ Node ≥ 20.0.0 是硬要求。Ubuntu `apt` 仓库里的 Node 通常是 18 甚至更老，**不要用 apt 直装**。用 nvm 多版本管理最省心。

### 4.1 安装 nvm

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
```

装完**重新加载 shell**（关闭终端重开，或执行）：

```bash
source ~/.bashrc
```

验证：

```bash
nvm --version
```

有版本号即成功。

> 💡 如果 `curl` 报拉取失败：公司代理或网络问题。手动从 GitHub 下载脚本再 `bash install.sh`，或先用 Windows 浏览器下载到 `C:\Users\你\Downloads`，再在 WSL 里 `bash /mnt/c/Users/你/Downloads/install.sh`。

### 4.2 安装 Node 20 LTS

```bash
nvm install 20
nvm use 20
```

### 4.3 验证

```bash
node -v
npm -v
npx -v
```

预期（版本号以你装的为准）：

```
v20.x.x
10.x.x
10.x.x
```

> 💡 `start.sh` 检测到 Node < 20 会尝试用 nvm 自动切到 20，所以装好 nvm + Node 20 后基本不会出问题。

### 4.4 准备 C++ 编译工具链（装 better-sqlite3 需要）

`better-sqlite3` 是原生模块，安装时要编译。Ubuntu 装好 build-essential 即可：

```bash
sudo apt install -y build-essential python3
```

装一次就够，以后切 Node 版本只 `npm rebuild better-sqlite3` 即可（`start.sh` 会自动做）。

---

## 第五步（可选）：安装 Python 3

smart-agent（AI 审计 v2 后端）需要 Python 3。Ubuntu 默认就有，验证：

```bash
python3 --version
```

看到 `Python 3.x.x` 即可。若没有：

```bash
sudo apt install -y python3 python3-pip
```

装 smart-agent 的 Python 依赖（只 `requests`）：

```bash
cd ~/cannbot-skills/plugins-community/cannbot-insight/smart-agent
pip3 install -r requirements.txt
```

> 💡 没装 Python 也不影响核心功能，`start.sh` 会自动跳过 smart-agent，Audit Tab 走 v1 TypeScript 版。

---

## 第六步：获取源码

CANNBot-Insight 是 `cannbot-skills` 仓库里的一个子目录。建议放在 WSL 的 Linux 文件系统里（如 `~/`），**不要**放在 `/mnt/c/` 下（跨文件系统读写会慢很多）。

### 6.1 克隆

```bash
cd ~
git clone https://gitcode.com/cann/cannbot-skills.git
```

> 💡 国内网络慢可用浅克隆：`git clone --depth 1 https://gitcode.com/cann/cannbot-skills.git`。
>
> 只想要这一个子目录，用 sparse-checkout：
> ```bash
> git clone --filter=blob:none --sparse https://gitcode.com/cann/cannbot-skills.git
> cd cannbot-skills
> git sparse-checkout set plugins-community/cannbot-insight
> cd plugins-community/cannbot-insight
> ```

### 6.2 进入目录

```bash
cd ~/cannbot-skills/plugins-community/cannbot-insight
```

### 6.3 验证

```bash
ls start.sh
```

看到 `start.sh` 即路径正确。赋予执行权限（git 一般会保留，保险起见）：

```bash
chmod +x start.sh
```

---

## 第七步：一键启动

`start.sh` 把所有脏活累活都做了：检查 Node 版本（过低时尝试 nvm 自动切）、安装 npm 依赖、按需重建 `better-sqlite3` 原生模块、创建 `.env`、运行 Prisma 数据库迁移、启动 smart-agent（如有 Python）、找空闲端口、启动 Web 服务、自动打开 Windows 浏览器。**你只需要一条命令。**

### 7.1 首次启动（会自动安装）

```bash
./start.sh
```

首次运行会看到类似进度：

```
[setup] Installing dependencies...
  ... (npm install 输出，首次约 2–5 分钟) ...
[setup] better-sqlite3 native module ... OK
[setup] Creating .env with DATABASE_URL...
[setup] Advanced tabs: false (use -a flag to enable)
[setup] Running Prisma migration...
  ... Applied migration ...
[start] Launching smart-agent (Python) on port 21026...
[start] smart-agent ready at http://localhost:21026 (PID 12345)
[start] Launching CANNBot-Insight on port 21025...
[start] Waiting for server at http://localhost:21025...
[start] Server ready — opening http://localhost:21025
```

WSL 模式下 `start.sh` 会自动调用 Windows 的 `cmd.exe start` 打开默认浏览器，跳到 <http://localhost:21025>，看到 Insight 首页（一个空的会话列表）即**启动成功**。

> 💡 如果浏览器没自动打开，手动访问日志里打印的地址（默认 <http://localhost:21025>）。

### 7.2 默认端口

- **Web UI**：`21025`（被占用时自动顺延到 21026、21027…）
- **smart-agent**：`21026`（仅当装了 Python）

### 7.3 何时算"装好了"

浏览器能打开 `http://localhost:21025` 并看到 Insight 首页，就说明环境完整。第一次进入是空列表，下一步导入数据。

### 7.4 退出

在终端里按 `Ctrl+C`。`start.sh` 会自动清理 smart-agent 子进程。Next.js dev server 会被一并终止。

---

## 第八步：首次使用

### 8.1 准备一份 Agent 日志

Insight 能导入两种日志：

| 来源 | WSL 里的默认路径 |
|------|------------------|
| opencode | `~/.local/share/opencode/sessions.db` |
| Claude Code | `~/.claude/projects/<hash>/sessions/<id>.jsonl`，也可指定整个目录批量扫描 |

> 💡 **WSL 的优势**：如果 opencode 就装在这个 WSL 里，日志默认路径就在你的家目录，导入界面直接填 `~/.local/share/opencode/sessions.db` 即可，零拷贝。

### 8.2 导入

1. 浏览器里在 Insight 首页点 **"导入"**。
2. 选择源类型（opencode .db / Claude .jsonl / 目录）。
3. 填日志路径或选择文件后导入。导入完成跳到会话列表。
4. 点任意会话进入详情，依次浏览 **Overview → Turns → Workflow → Trace → Subagents → Skills → Interactions → AI Workflow → Context** 9 个 Tab。

### 8.3 也支持 CLI 上传（适合远程机器）

如果日志在另一台 SSH 远程机器上，可在那台机器上用 CLI 一步上传到本机 Insight：

```bash
npx tsx src/cli/index.ts upload --file ./sessions.db
```

或在 TUI 里按 `u` 键上传。上传后交互式填描述，后端自动起停。之后在 Web UI 导入时点 **CANNBay** 按钮直接选库导入。

---

## 常用操作

### 启动（日常）

每次开机后只需打开 Ubuntu 终端：

```bash
cd ~/cannbot-skills/plugins-community/cannbot-insight
./start.sh
```

因为依赖已装好，启动 10–20 秒。

### 更新到最新版

```bash
git pull
./start.sh -u
```

`-u` 表示先 `npm install` 拉新依赖，再迁移启动。

### 清缓存重编译

```bash
./start.sh -f
```

### 开启高级 Tab

```bash
./start.sh -a
```

显示 Subagents / Interactions / AI Workflow 三个高级 Tab。

### 端口被占用想强制复用 21025

```bash
./start.sh -k
```

### CLI / TUI 模式

```bash
./start.sh -c tui
./start.sh -c sessions
./start.sh -c stats
```

支持的子命令：`tui sessions session turn search compare stats import delete config`。完整参考见 [docs/cli-commands.md](cli-commands.md)。

### 让服务后台常驻

`start.sh` 默认前台跑、关终端即停。想让它后台常驻：

```bash
nohup ./start.sh > ~/insight.log 2>&1 &
```

之后 `tail -f ~/insight.log` 看日志，`pkill -f "next dev"` 停服务。

### 停止服务

终端里 `Ctrl+C`，或后台模式 `pkill -f "next dev"`、`pkill -f "smart-agent"`。

---

## 故障排查

### Q1：`wsl --install` 失败 / WSL2 起不来

- 确认 Windows 版本够新（`winver` 看内部版本 ≥ 19041）。
- BIOS 里开虚拟化（VT-x / AMD-V）。
- 管理员 PowerShell 跑 `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart` 和 `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart`，重启。
- 第三方杀毒拦了 Hyper-V，临时关闭测试。
- 实在不行改用 [Windows 原生指南](setup-windows.md)。

### Q2：Ubuntu 终端里中文乱码 / 退格键失效

编辑 `~/.bashrc` 末尾加：

```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
```

`source ~/.bashrc` 生效。推荐用 Windows Terminal 而非老的 conhost。

### Q3：`curl: (60) SSL certificate problem: certificate has expired` / 时间不对

WSL 时钟偏移。修：

```bash
sudo hwclock -s
```

### Q4：浏览器访问 `http://localhost:21025` 打不开

- 看 `start.sh` 日志里实际端口（可能顺延到 21026）。
- Windows 防火墙：管理员 PowerShell `Set-NetFirewallProfile -Profile Private -Enabled False` 测试。
- WSL2 端口转发问题（旧版 Windows 偶发）：重启 WSL `wsl --shutdown` 后重进。
- Windows 11 22H2+ 已支持 localhost 自动转发，多数情况无需额外配置。

### Q5：`npm install` 卡住 / 超时

- 公司代理：`npm config set proxy http://代理:端口`、`npm config set https-proxy http://代理:端口`。
- 国内家用：`npm config set registry https://registry.npmmirror.com`。
- 仍慢：加 `--fetch-timeout=600000 --fetch-retries=5`。

### Q6：`better-sqlite3` 编译失败 / `node-gyp` 报错

缺编译工具链：

```bash
sudo apt install -y build-essential python3
```

装完 `npm rebuild better-sqlite3`。`start.sh` 会自动检测重建，正常无需手动。

### Q7：切了 Node 版本后启动报 ABI 不匹配

`start.sh` 内置自动 `npm rebuild better-sqlite3`，会自愈。若仍失败手动：

```bash
nvm use 20
npm rebuild better-sqlite3
./start.sh
```

### Q8：Prisma 报 `DATABASE_URL` 找不到

`start.sh` 会自动创建 `.env`。若仍报错，手动在项目目录新建 `.env`，一行：

```
DATABASE_URL="file:./dev.db"
```

### Q9：导入提示文件不存在

- opencode 默认在 `~/.local/share/opencode/sessions.db`（WSL 家目录）。导入界面里 `~` 可能不被展开，改成绝对路径 `/home/你的用户名/.local/share/opencode/sessions.db`。
- 从 Windows 拷进来的文件放 WSL 家目录更稳，别放 `/mnt/c/`（慢且偶有权限问题）。

### Q10：Windows 行尾(CRLF) 导致脚本报 `bad interpreter`

git 默认在 Windows 下 clone 时会把 `\r\n` 带进来。修：

```bash
cd ~/cannbot-skills
git config --global core.autocrlf input
git rm --cached -r .
git reset --hard
cd plugins-community/cannbot-insight
chmod +x start.sh
```

### Q11：端口 21025/21026 一直被占

```bash
lsof -i :21025
kill -9 <PID>
```

或直接 `./start.sh -k`。

### Q12：smart-agent 起不来 / Audit Tab v2 报错

- `python3 --version` 确认装了。
- `cd smart-agent && pip3 install -r requirements.txt` 装依赖。
- 看 `start.sh` 日志是否有 `smart-agent ready at http://localhost:21026`。
- 端口被占：`lsof -i :21026` 杀掉。
- 没起也不影响，Audit Tab 的 v1 路径照常工作。

---

## 下一步

- 读懂功能全貌：[README-zh.md](../README-zh.md)
- 看整体架构：[docs/architecture.md](architecture.md)
- 学 CLI 高级用法：[docs/cli-commands.md](cli-commands.md)
- 在多机器间分享 session：用 Web UI 里的 **CANNBay** 上传 / 导入功能

---

> 反馈与问题：请在仓库提 Issue。
