---
name: ascendc-cross-gen-port
description: >
  跨代际 AscendC 算子移植入口。把一个 AscendC 算子从来源架构移植到用户指定的目标架构/产品
  （当前 arch22→arch35，如 910C/V220→950PR/V300）。用户用自然语言指定目标架构（arch35 / 950PR /
  A5 / SoC编号 / 代际皆可）；来源架构由代码分析自动识别。参数包括：1） 必选。待port的arch22算子实现目录
  2) 推荐。KernelBench风格 (model.py 和test_case.json) 的算子golden与测试集合。
argument-hint: >
  必选：待移植的 arch22 算子实现目录（ops-nn 源算子目录，如 .../cann/ops-nn/activation/gelu；或
  TileLang2AscendC 工程目录 model_new_ascendc.py + kernel/）。
  推荐：KernelBench 风格 golden（算子实现 .py，如 model.py + 测试集合 test_case.json，同 stem
  JSON/JSONL sidecar）。
  目标架构可用自然语言给出（arch35 / 950PR / A5），来源架构自动识别，无需指定。
---

# ascendc-cross-gen-port — 跨代际移植入口

**薄 NL 前端**，底层调用插件自带的编排器引擎（`engine/`，`python -m orchestrator`）的 **`--port-a3-ops` mode**。
引擎已随插件打包，**无需外部 a5_ops**。本 skill 只负责：把用户的话翻成对应 mode 调进引擎，再回报结果。

## 参考输入选择

跨代移植的输入统一为两部分，两个 source 模式共用同一形态。**实现目录必选；参考模式必须且只能二选一**
（npubench 或 a3_live，见下），裸调用 fail-closed。两个 `--mode` 名称按来源格式分类：
`port-a3-ops`（CANN ops 仓通用格式）/ `port-a3-tilelang2ascendc`（TileLang2AscendC 插件输出格式）：

1. **必选：待移植的算子实现目录**
   - 常规移植：arch22 `ops-nn` 源算子目录（来源架构自动识别）。
   - TileLang2AscendC 工程：`model_new_ascendc.py + kernel/` 目录（`--mode port-a3-tilelang2ascendc`）。
2. **KernelBench 风格 golden**（task `.py` + 同 stem `.json` / `.jsonl` sidecar 文件对，格式示例见
   `examples/npukernelbench-native/`）：`--reference-source npubench --npubench-task <task.py>`，必要时再加
   `--npubench-root <root>`。原样冻结 task/root，不读取 A3 runtime/capture。两种模式的 golden 相同。
   常规 ops-nn 来源下 golden 为**推荐**；TileLang2AscendC 工程来源下 golden 为**必需**（唯一真值，无替代）。
   输入尚非该格式时，请用户先由输入提供方准备为该格式并复核语义，插件不自动生成 task。
   **perf 对比基准**：此模式下报告的加速比 = 目标实现 vs golden 参考实现（同一 A5 环境 W3/R5 msprof 实测）。
3. **实时 A3 真值（显式可选，仅常规 ops-nn 来源）**：仅当用户明确要求 fresh A3 CANN truth 时选择
   `a3_live`——命令行传 `--reference-source a3_live`，或 `.ascendc_env` 配置 `PORT_A3_REFERENCE_SOURCE=a3_live`
   （两者等价，均为显式选择），并确认 A3 与 A5 都已配置；它不能与 npubench 路径混用。TileLang2AscendC 工程来源
   **只支持 npubench golden**（引擎拒绝该组合，以冻结 task 为唯一真值）。**perf 对比基准**：此模式下精度对照当次
   A3 实测输出，加速比 = 目标实现 vs A3 实现实测。

**加速比不可横向比较**：npubench 的加速比基准是 golden 参考实现，a3_live 的基准是 A3 实现——两种模式测出的
加速比含义不同，数值不可直接对比。

## 职责（三步）

1. **解析目标 + NPU 连接**：把用户自然语言目标归一为 canonical target（arch35/950PR/A5/V300→`a5`；
   arch22/910C/A3/V220→`a3`）。本步只归一目标，不猜测引擎路径；**来源架构由代码分析自动识别**，
   无需用户指定。
2. **确定引擎根目录（禁止依赖 cwd 或全盘搜索）**：Skill loader 会回显
   `Base directory for this skill: <绝对路径>`。将该**原样绝对路径**替换到下方
   `<skill-base>`；插件级唯一 resolver 先读安装 manifest 中的 `engine_root`（兼容旧 manifest 的
   `hooks_settings_engine`），校验 `src/scripts/orchestrator/__main__.py`；未经过
   `init.sh` 的源树调用才回退到 `realpath(<skill-base>)/../../engine`。resolver 成功后必须检查
   `$ENGINE_DIR/workspace/.ascendc_env` 已存在并配置目标 A5；仅 `a3_live` 还要求来源 A3。下方启动示例把
   解析、检查和启动放在同一次 Bash tool call 中，shell 变量不跨调用复用。
   **不得**把 `<skill-base>` 本身当插件根、不得从当前目录猜测，也不得用 `find /`
   搜索引擎。
3. **调用引擎 `--port-a3-ops` mode**（op 名自动取来源目录 basename）——**必须用流式后台启动，用户 console 才有实时日志**：
   ```python
   # agent 侧用 Bash 工具起，NPU lane 默认 0
   Bash(
       command="bash \"$(realpath \"<skill-base>\")/../../scripts/launch_orchestrator.sh\" --skill-base \"<skill-base>\" --mode port-a3-ops --source <ops-nn-op-dir> --lane 0 --reference-source npubench --npubench-task <task.py>",
       run_in_background=True,
   )
   ```
   用户明确要求实时 A3 时，**替换**上例的 npubench reference 参数（不混用）为：
   ```python
   --reference-source a3_live
   ```
   **启动逻辑集中在 `scripts/launch_orchestrator.sh`（两个 harness 共用）**，本 skill 不再自带一份。
   该脚本负责：解析引擎根（经 `resolve_engine.py`，不猜 cwd、不全盘搜）、校验
   `.ascendc_env`、探测当前 harness 并导出 `AOG_HARNESS_BACKEND`、按 harness 设置正确的
   config-dir 变量，然后 exec 编排器。
   之所以必须集中：这段逻辑里最容易漂移的两处**失败都是静默的** ——
   `AOG_HARNESS_BACKEND` 漏设不会报错，只会回落 Claude Code（于是在 opencode 会话里
   spawn `claude`）；`CLAUDE_CONFIG_DIR` 漏设也不会报错，worker 会回退到 `~/.claude`
   （可能装着别的算子生成套件）→ 用错知识库、破坏自包含。启动器把两者都显式贯穿到
   orch → worker，不靠隐式继承。
   **启动纪律（违反则 console 全黑、录屏/交互体验差）**：
   - 用 `run_in_background=True`；**禁** trailing `&`、**禁** `nohup`（都脱离 CC task tracker → UI 看不到、`TaskOutput`/`TaskStop` 失效）。
   - **禁** shell 重定向 `> foo.log 2>&1` —— Bash 工具已给 task 自己的输出文件并经 `TaskOutput` 实时暴露；重定向把日志吞进文件、console 变黑箱。
   - **禁** 截断/缓冲 pipe（`| tail`/`| head`/`| grep`）—— 会 buffer 到进程 EOF 才吐，实时流没了、`TaskOutput` 也只剩截断输出。
   - 起 **bare `python3 -m orchestrator …`**，**不要经 `orch` 的 `tee` wrapper**（tee 到非终端 pipe 会 block-buffer → console 不实时）。看进度用 `TaskOutput(task_id)` / Read task 输出文件，别把 viewer pipe 焊进启动命令。
   `--port-a3-ops` 隐含目标 a5、归档 = ops-nn 镜像布局。`a3_live` 的参考基线是当次 A3-CANN 实测真值，
   每次都要生成与本次调用绑定的 capture provenance，不能复用缓存、归档或已提交的输出；
   `npubench` 则以冻结的 task/sidecar 为唯一功能真值，O5 用 provider-owned 精度与 W3/R5
   msprof 评测。引擎走确定性流水线（O0→O6：解析→分类→参考采集→移植→构建→精度验证→性能→报告），
   经安全网校验，按双层 KB 反馈环（c>b>a）注入/沉淀。
4. **回报结果**：读 `engine/workspace/<op>/verification.json` 的 customer-view 判据（PASS + 逐 case 计数、
   与 `reference.source` 一致的证据），连同复现指引一并回报。

> 编排细节、FSM、安全网、双层 KB 见插件 `ARCHITECTURE.md`。Claude Code 与 OpenCode 均通过同一启动器使用
> 该入口；**移植的所有流水线逻辑都在引擎里、不在本 skill——本 skill 只是薄入口。**
