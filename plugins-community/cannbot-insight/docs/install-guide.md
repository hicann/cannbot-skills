# CANNBot-Insight 安装教程（tgz 分发）

> 适用于 v1.83.0+。无需克隆源码、无需 npm registry、无需认证配置，从 `.tgz` 包文件安装。

## 一、发布方（每次发版）

### 1. 确认 export-view bundle 已构建（HTML 导出功能依赖）

```bash
cd cannbot-insight
npm run build:export-view     # 生成 public/export-view.js（若已存在可跳过）
```

### 2. 打包

```bash
npm pack                      # 生成 cannbot-insight-1.83.0.tgz（约 900KB）
```

产物 `cannbot-insight-<version>.tgz` 在当前目录。

### 3. 分发

把 tgz 上传到 **GitCode Release**：仓库页面 → Releases → 新建 → 添加 `cannbot-insight-1.83.0.tgz` 作为附件 → 发布。用户从 Release 下载。

> **发新版本**：改 `package.json` 的 `version`（如 `1.83.1`）→ 重复步骤 1-3。

---

## 二、安装方（用户）

### 前置要求

- **Node.js >= 20**（v18.19.x 装不动 better-sqlite3 / Prisma 6；用 nvm 可 `nvm install 20 && nvm use 20`）
- 首次安装会编译 `better-sqlite3` 原生模块，需 `python3` + `make` + `g++`（多数 Linux 自带；prebuild 命中则免编译秒装）

### 安装

```bash
# 从 GitCode Release 下载 cannbot-insight-1.83.0.tgz 后
npm install -g ./cannbot-insight-1.83.0.tgz
```

- 首次约 1-2 分钟（拉 deps + 编译 better-sqlite3）
- 运行时依赖（next / prisma / better-sqlite3 等）自动从**公共 npmjs** 拉取，**包本体私有在 tgz 里，无需任何 registry 或认证配置**

### 启动

```bash
cannbot-insight
```

- **首次启动**：自动 `prisma migrate` + `next build`（约 30s-2min，仅首次），随后启动 + 开浏览器
- **后续启动**：秒级
- 地址 `http://localhost:21025`（被占自动找空闲端口）
- 数据在 `~/.cannbot-insight/`（`insight.db` + `.next` 构建缓存），安装包目录保持只读

### CLI 子命令（直接透传）

```bash
cannbot-insight upload --file ~/.local/share/opencode/opencode.db --list
cannbot-insight sessions
cannbot-insight --help
```

### 启动标志 / 环境变量

| 标志 / env | 作用 |
|------------|------|
| `-a`, `--advanced` | 显示高级 tab（wireRounds/replay） |
| `-k`, `--kill` | 杀占用 21025 端口的进程，复用该端口 |
| `-f`, `--fresh` | 清空 `.next` 构建缓存重新构建 |
| `CANNBOT_INSIGHT_PORT=<N>` | 自定义端口 |
| `CANNBOT_INSIGHT_HOME=<path>` | 自定义数据目录（默认 `~/.cannbot-insight/`） |

### smart-agent（可选，自动探测）

若机器有 `python3`，启动器自动拉起 smart-agent 后端（端口 21026，breather / v2 分析功能）；无则静默降级，主功能不受影响。

### 升级 / 卸载

```bash
npm install -g ./cannbot-insight-1.84.0.tgz    # 升级（下载新版 tgz 后）
npm uninstall -g cannbot-insight              # 卸载
```

---

## 三、故障排查

| 现象 | 处理 |
|------|------|
| `better-sqlite3` 编译失败 | 装 `python3 make g++`，或换 Node 20 LTS（prebuild 命中率高） |
| `npm ERR! EACCES` 权限错误 | 不要用 `sudo`；改用 nvm 管 Node，或配 npm prefix 到用户目录 |
| 端口被占 | `cannbot-insight -k`，或 `CANNBOT_INSIGHT_PORT=21030 cannbot-insight` |
| 远程无头 VM 打不开浏览器 | 正常降级为打印 URL，`ssh -L 21025:localhost:21025` 端口转发后本地访问 |
| 首次 build 卡住 / 报错 | 确认 `~/.cannbot-insight/` 可写；`cannbot-insight -f` 清缓存重建 |
| 找不到 `cannbot-insight` 命令 | 确认 npm 全局 bin 在 PATH（`npm bin -g` 或 `npm config get prefix`） |

---

## 附：数据与目录布局

```
~/.cannbot-insight/
├── insight.db              # SQLite 库（Prisma 8 models）
└── .next/                  # next build 产物（首次构建生成）

<npm-global-prefix>/lib/node_modules/cannbot-insight/   # 包安装目录（只读）
├── bin/cannbot-insight.mjs    # 启动器
├── public/export-view.js      # HTML 导出 bundle（随包预构建）
├── prisma/                    # schema + migrations
├── smart-agent/               # Python 后端（可选）
└── src/                       # 源码（现场 build 用）
```
