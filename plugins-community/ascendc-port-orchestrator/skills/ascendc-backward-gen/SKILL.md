---
name: ascendc-backward-gen
description: >
  正向→反向 AscendC 算子生成入口。由一个可微 PyTorch 正向算子，自动生成其反向（梯度）AscendC
  算子并在 NPU 上验证精度。可用自然语言指定目标芯片（a3/a5 或 arch22/arch35）。
  触发：当用户需要为某正向算子生成对应反向算子时使用。
argument-hint: >
  输入：forward（正向规格 .py，定义可微 `forward` 函数 + `BACKWARD_SPEC` dict）；
  target（自然语言目标芯片，可选，如 "a3"/"a5"/"arch35"）。
---

# ascendc-backward-gen — 正向→反向生成入口

**薄 NL 前端**，底层调用插件自带的编排器引擎（`engine/`，`python -m orchestrator`）的 **`--backward` mode**。
引擎已随插件打包，**无需外部 a5_ops**。本 skill 只负责：把用户的话翻成对应 mode 调进引擎，再回报结果。

## 职责（三步）

1. **解析目标**：若用户指定目标芯片，归一为 canonical target（a3=910C/V220/arch22；
   a5=950PR/V300/arch35）。本步只归一目标，不猜测引擎路径。
2. **确定引擎根目录（禁止依赖 cwd 或全盘搜索）**：Skill loader 会回显
   `Base directory for this skill: <绝对路径>`。将该**原样绝对路径**替换到下方
   `<skill-base>`；插件级唯一 resolver 先读安装 manifest 中的 `engine_root`（兼容旧 manifest 的
   `hooks_settings_engine`），校验 `src/scripts/orchestrator/__main__.py`；未经过
   `init.sh` 的源树调用才回退到 `realpath(<skill-base>)/../../engine`。resolver 成功后必须检查
   `$ENGINE_DIR/workspace/.ascendc_env` 已存在，并在其中写入归一后的 `TARGET=`；下方启动示例把
   解析、检查和启动放在同一次 Bash tool call 中，shell 变量不跨调用复用。
   **不得**把 `<skill-base>` 本身当插件根、不得从当前目录猜测，也不得用 `find /`
   搜索引擎。
3. **调用引擎 `--backward` mode**（op 名自动取 `<spec 文件名>_grad`）——**必须用流式后台启动，用户 console 才有实时日志**：
   ```python
   # agent 侧用 Bash 工具起，NPU lane 默认 0
   Bash(
       command="SKILL_BASE=\"<skill-base>\"; RESOLVER=\"$(realpath \"$SKILL_BASE\")/../../scripts/resolve_engine.py\"; ENGINE_DIR=\"$(python3 \"$RESOLVER\" --base-dir \"$SKILL_BASE\")\" || exit $?; test -f \"$ENGINE_DIR/src/scripts/orchestrator/__main__.py\" || exit 2; test -f \"$ENGINE_DIR/workspace/.ascendc_env\" || { echo 'configure $ENGINE_DIR/workspace/.ascendc_env first' >&2; exit 2; }; cd \"$ENGINE_DIR\" && CLAUDE_CONFIG_DIR=\"${CLAUDE_CONFIG_DIR:-$HOME/.claude}\" PYTHONPATH=src/scripts python3 -m orchestrator --backward <forward_spec.py> --lane 0",
       run_in_background=True,
   )
   ```
   **CLAUDE_CONFIG_DIR 必须显式带上（自包含铁律）**：引擎 spawn 的子 agent 从 `CLAUDE_CONFIG_DIR` 解析 skill/agent/KB；不显式带、上游某层丢了它 → worker 回退 `~/.claude`（可能装着别的套件）→ 破坏自包含。启动命令**必须** `CLAUDE_CONFIG_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"` 前缀，显式贯穿入口 config-dir 到 orch → worker。
   **启动纪律（违反则 console 全黑、录屏/交互体验差）**：
   - 用 `run_in_background=True`；**禁** trailing `&`、**禁** `nohup`（都脱离 CC task tracker → UI 看不到、`TaskOutput`/`TaskStop` 失效）。
   - **禁** shell 重定向 `> foo.log 2>&1`（Bash 工具已给 task 自己的输出文件并经 `TaskOutput` 实时暴露；重定向把日志吞进文件、console 变黑箱）。
   - **禁** 截断/缓冲 pipe（`| tail`/`| head`/`| grep`，会 buffer 到 EOF 才吐）。
   - 起 **bare `python3 -m orchestrator …`**，不要经 `orch` 的 `tee` wrapper（tee 到非终端 pipe 会 block-buffer → console 不实时）。看进度用 `TaskOutput(task_id)` / Read task 输出文件。
   引擎自动：由正向 spec 用 `torch.autograd.grad`（CPU/fp64）生成**精确梯度真值**（参考）→ 生成反向
   AscendC kernel → 构建 → 精度验证（与真值对齐，T1 严格 CPU-truth 为默认社区口径）→[性能优化]→报告；
   目标 kernel agent 启动前必须已有可评分的 CPU/fp64 真值；不得以目标 kernel 输出反向构造真值。
   全程经安全网（防作弊/防退化）+ 双层 KB 反馈环（c>b>a）。
4. **回报结果**：读 `engine/workspace/<op>_grad/verification.json` 的 customer-view 判据（PASS + 逐 dtype 计数），
   连同复现指引一并回报。**只有 NPU 上精度验证 PASS 才报成功**，否则 incomplete/blocked/failed 如实报。

## 正向 spec 格式（示例见 `scripts/reference_provider/examples/gelu_spec.py`）
一个 `.py` 定义：可微 `forward(**inputs)` + `BACKWARD_SPEC = {"wrt": [...], "inputs": {name: {"shape": [...]}},
"cases": [...], "dtypes": [...], "seed": N}`。

> 编排细节、FSM、安全网、双层 KB 见插件 `ARCHITECTURE.md`。当前依赖 Claude Code 运行时（见 README
> 底座依赖与适配路线）。**backward 的所有流水线逻辑都在引擎里、不在本 skill——本 skill 只是薄入口。**
