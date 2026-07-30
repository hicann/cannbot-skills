---
name: ascendc-cross-gen-port
description: >
  跨代际 AscendC 算子移植入口。把一个 AscendC 算子从来源架构移植到用户指定的目标架构/产品
  （当前 arch22→arch35，如 910C/V220→950PR/V300；规划更多目标与反向跨代）。用户用自然语言
  指定目标（arch35 / 950PR / A5 / SoC编号 / 代际皆可）；来源架构由代码分析自动识别。
  触发：当用户需要把某 AscendC 算子跨代际移植到指定目标架构时使用。
argument-hint: >
  输入：source（来源算子的 ops-nn 目录，如 .../cann/ops-nn/activation/gelu）、
  target（自然语言目标，如 "arch35"/"950PR"/"A5"）。来源架构无需指定（自动识别）。
---

# ascendc-cross-gen-port — 跨代际移植入口

**薄 NL 前端**，底层调用插件自带的编排器引擎（`engine/`，`python -m orchestrator`）的 **`--port-a3` mode**。
引擎已随插件打包，**无需外部 a5_ops**。本 skill 只负责：把用户的话翻成对应 mode 调进引擎，再回报结果。

## 职责（三步）

1. **解析目标 + NPU 连接**：把用户自然语言目标归一为 canonical target（arch35/950PR/A5/V300→`a5`；
   arch22/910C/A3/V220→`a3`）。本步只归一目标，不猜测引擎路径；**来源架构由代码分析自动识别**，
   无需用户指定。
2. **确定引擎根目录（禁止依赖 cwd 或全盘搜索）**：Skill loader 会回显
   `Base directory for this skill: <绝对路径>`。将该**原样绝对路径**替换到下方
   `<skill-base>`；插件级唯一 resolver 先读安装 manifest 中的 `engine_root`（兼容旧 manifest 的
   `hooks_settings_engine`），校验 `src/scripts/orchestrator/__main__.py`；未经过
   `init.sh` 的源树调用才回退到 `realpath(<skill-base>)/../../engine`。resolver 成功后必须检查
   `$ENGINE_DIR/workspace/.ascendc_env` 已存在并配置来源 A3 与目标 A5；下方启动示例把解析、检查和
   启动放在同一次 Bash tool call 中，shell 变量不跨调用复用。
   **不得**把 `<skill-base>` 本身当插件根、不得从当前目录猜测，也不得用 `find /`
   搜索引擎。
3. **调用引擎 `--port-a3` mode**（op 名自动取来源目录 basename）——**必须用流式后台启动，用户 console 才有实时日志**：
   ```python
   # agent 侧用 Bash 工具起，NPU lane 默认 0
   Bash(
       command="SKILL_BASE=\"<skill-base>\"; RESOLVER=\"$(realpath \"$SKILL_BASE\")/../../scripts/resolve_engine.py\"; ENGINE_DIR=\"$(python3 \"$RESOLVER\" --base-dir \"$SKILL_BASE\")\" || exit $?; test -f \"$ENGINE_DIR/src/scripts/orchestrator/__main__.py\" || exit 2; test -f \"$ENGINE_DIR/workspace/.ascendc_env\" || { echo 'configure $ENGINE_DIR/workspace/.ascendc_env first' >&2; exit 2; }; cd \"$ENGINE_DIR\" && CLAUDE_CONFIG_DIR=\"${CLAUDE_CONFIG_DIR:-$HOME/.claude}\" PYTHONPATH=src/scripts python3 -m orchestrator --port-a3 <ops-nn-op-dir> --lane 0",
       run_in_background=True,
   )
   ```
   **CLAUDE_CONFIG_DIR 必须显式带上（自包含铁律）**：引擎 spawn 的子 agent（`claude --agent` worker）从 `CLAUDE_CONFIG_DIR` 解析 skill/agent/KB。若不显式带、且上游某层把它丢了，worker 会回退到 `~/.claude`（可能装着别的算子生成套件）→ 用错知识库、破坏自包含。所以启动命令**必须** `CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"` 前缀，把入口的 config-dir 显式贯穿到 orch → worker（不靠隐式继承）。
   **启动纪律（违反则 console 全黑、录屏/交互体验差）**：
   - 用 `run_in_background=True`；**禁** trailing `&`、**禁** `nohup`（都脱离 CC task tracker → UI 看不到、`TaskOutput`/`TaskStop` 失效）。
   - **禁** shell 重定向 `> foo.log 2>&1` —— Bash 工具已给 task 自己的输出文件并经 `TaskOutput` 实时暴露；重定向把日志吞进文件、console 变黑箱。
   - **禁** 截断/缓冲 pipe（`| tail`/`| head`/`| grep`）—— 会 buffer 到进程 EOF 才吐，实时流没了、`TaskOutput` 也只剩截断输出。
   - 起 **bare `python3 -m orchestrator …`**，**不要经 `orch` 的 `tee` wrapper**（tee 到非终端 pipe 会 block-buffer → console 不实时）。看进度用 `TaskOutput(task_id)` / Read task 输出文件，别把 viewer pipe 焊进启动命令。
   `--port-a3` 隐含目标 a5、参考基线 = A3-CANN 实测真值（非 CPU-PyTorch）、归档 = ops-nn 镜像布局。
   **真值不可豁免**：每次调用都必须当次在 arch22 来源 NPU 执行并生成与本次调用绑定的
   capture provenance。schema gap 最多只能复用带来源记录的输入配方，不能复用缓存、归档或已提交的输出。
   引擎走确定性流水线（O0→O6：解析→分类→A3参考采集→移植→构建→精度验证→[性能优化]→报告），
   经安全网校验，按双层 KB 反馈环（c>b>a）注入/沉淀。
4. **回报结果**：读 `engine/workspace/<op>/verification.json` 的 customer-view 判据（PASS + 逐 case 计数、
   bit-exact vs A3 证据），连同复现指引一并回报。**只有精度验证 PASS 才报成功**，否则如实报。

> 编排细节、FSM、安全网、双层 KB 见插件 `ARCHITECTURE.md`。当前依赖 Claude Code 运行时（见 README
> 底座依赖与适配路线）。**移植的所有流水线逻辑都在引擎里、不在本 skill——本 skill 只是薄入口。**
