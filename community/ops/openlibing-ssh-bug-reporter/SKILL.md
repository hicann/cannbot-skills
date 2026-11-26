---
name: openlibing-ssh-bug-reporter
description: 为 openLiBing ResourceManager (VS Code / CodeBuddy 远程开发扩展) 的 SSH 连接问题生成标准化的 Bug 报告、Issue 与 PR 描述。当用户提到该扩展 SSH 连接失败、要提 issue、要提 PR、要写 bug 报告时使用。
---

# openLiBing ResourceManager SSH Bug 报告生成器

为 openLiBing ResourceManager 扩展（`openlibing.resourcemanager` 插件）的 SSH 连接问题，生成可直接提交到仓库的标准化 GitHub/GitLab Issue 与 PR 文档。

## 何时使用本 Skill

- 用户说"SSH 连不上 / 报错 / 提 issue / 提 PR / 写 bug 报告"
- 涉及该扩展在 Windows / macOS / Linux 下通过跳板机连接远程开发环境的失败场景

## 已知问题知识库（诊断时对照）

### 错误码 → 根因映射

| 错误信息 | 根因 | 修复方向 |
|---------|------|---------|
| `posix_spawn: No such file or directory` | 使用裸 `ssh` 命中 Windows 原生 OpenSSH，ProxyCommand 场景 spawn 失败 | 改用扩展自带 Cygwin `ssh.exe` 完整路径（`PlatformHelper.getToolPaths`） |
| `Bad port '178.127.55.228'` | cmd.exe 嵌套引号使 ProxyCommand 参数错位 | 改用 `bash.exe -c` + 参数数组，避免双重引号 |
| `Permission denied (publickey)` | Cygwin ssh home 为 `/home/<user>/`，找不到 Windows `C:\Users\<user>\.ssh\id_rsa` | 连 ProxyCommand 和主连接都加 `-i <key> -o IdentitiesOnly=yes` |
| bash 启动失败 / UTF-8 脚本报错 | 宿主注入的 `BASH_ENV` 指向损坏脚本 | spawn bash 时清除 `env.BASH_ENV=''` |
| `Connection timed out` / 目标 IP 不可达 | `targetIp` 来自平台 API 下发，非硬编码；可能平台数据过期或网络策略限制 | 连接前做 TCP 预检，不可达快速失败；降低重试 |
| 跳板机端口不对 | `jumpPort` 默认 `22`，实际可能是 `2222` | 允许配置/平台覆盖跳板端口 |

### 关键代码位置

- `out/utils/SSHHelper.js`
  - `constructor`：初始化 `sshPath` / `bashPath`
  - `buildSshCommand(host, port, username, fingerprint, command, wrapWithPID1Env, jumpHost)`：拼 SSH 命令与 ProxyCommand
  - `executeCommand(command, timeoutMs, onStdout, onStderr)`：spawn 执行
  - `testConnectionSimple(...)`：连接测试，默认重试 30 次 × 15s/次（5 分钟）
- `out/utils/PlatformHelper.js`：`getToolPaths` / `findAvailableSshKey` / `getDefaultCygwinKeyPath` / `isWindows`
- `out/register/ResourceEnvironmentCommands.js`：`updateDeviceProxyInfo` 从 `getMachineInfo` 取 `machineInfo.targetIp`、`jumpIp`、`jumpPort` 等

### 重要事实

- `178.127.55.228` 这类"目标 IP"**不是代码硬编码**，而是 HidevLab 平台 API 返回的 `machineInfo.targetIp`。扩展仅写入 SSH config 并尝试连接。
- Windows 下必须使用扩展自带 `out/tools/bin/{ssh,bash}.exe`，且传给 bash 的路径需转换为 Cygwin 格式（`/cygdrive/c/...`）。

## 工作流（SOP）

1. **收集信息**：让用户提供——操作系统、扩展版本、完整报错文本、是否用跳板机、跳板机 IP:端口、目标 IP:端口、是否已有公私钥。
2. **本地复现诊断**（可选，若用户愿意）：
   - 用 Node 脚本通过扩展自带 `bash.exe -c "<ssh.exe> ..."` 复现，清除 `BASH_ENV`。
   - `Test-NetConnection <ip> -Port <port>` 判断是代码问题还是网络/平台数据问题。
3. **分类**：对照上表判断属于"代码 bug"还是"平台数据/网络问题"。
4. **生成文档**：
   - `issue.md`：标题、环境、复现步骤、期望/实际、日志、根因、建议修复。
   - `pr.md`：针对代码 bug，给出最小改动 diff（参考"已知问题知识库"修复方向），含测试说明。
   - 若是平台数据问题，生成说明文档请用户联系平台侧，并附"扩展侧健壮性改进建议"作为可选 PR。
5. **输出**：把文档写入工作区（如 `ISSUE.md` / `PR.md`），并告知用户可直接复制提交。

## 输出模板

### issue.md 模板

```markdown
# [Bug] <一句话现象，含关键错误>

## 环境
- 扩展版本: openlibing.resourcemanager-<version>
- 宿主: VS Code / CodeBuddy <version>
- 操作系统: Windows 10/11 / macOS / Linux
- 是否跳板机: 是(<jumpIp>:<jumpPort>) / 否
- 目标: <targetIp>:<port>

## 现象
<完整报错，粘贴日志>

## 复现步骤
1. ...
2. ...

## 期望
连接成功并建立远程环境。

## 实际
<错误>

## 根因（如已定位）
<对照知识库说明>

## 建议修复
<最小改动方案>
```

### pr.md 模板

```markdown
# Fix: <一句话>

## 问题
<现象 + 根因>

## 修改
文件: out/utils/SSHHelper.js
- <改动点 1>
- <改动点 2>

## 验证
- [ ] Windows 跳板机场景可连接
- [ ] 密钥认证通过
- [ ] 不可达 IP 不再长时间阻塞
```
