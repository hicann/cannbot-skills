---
name: autoresearch
description: AutoResearch 算子内核自动优化团队与开发流程。适用于在 Ascend NPU 上迭代优化 Triton Ascend、Ascend C 或 Ascend C CATLASS 内核时。
mode: primary
skills: []
permission:
  external_directory: allow
---

# Op AutoResearch

本仓库是独立的算子内核自动优化工作区。Agent 通过 plan → edit → eval → keep/discard 循环优化可度量指标；状态机、hooks、slash command、verifier 和 worker 均在本仓库内。

## 快速开始

```bash
python -m pip install -e .
# HTTP worker 需要：python -m pip install -e ".[worker]"
```

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices 0
/autoresearch --resume
```

完整操作说明见 `.claude/commands/autoresearch.md` 与 `AUTORESEARCH.md`。

## 不变量

1. `.ar_state/plan.md` 是计划的唯一事实来源，只能由 `create_plan.py` 和 `pipeline.py` 的结算事务写入。
2. plan ID 全局单调递增：`p1, p2, ...`，不可复用或跳号。
3. 每个 plan item 必须结算为 KEEP / DISCARD / FAIL，或在 REPLAN/DIAGNOSE 边界静默丢弃；计数器仍需前进。
4. baseline、plan commit 和 round settlement 各自拥有事件事务，在一次原子 `state.json` 保存中写入结果、目标 phase 和 replay sentinel。hooks 只观察提交后的状态并输出 guidance。
5. 可编辑文件严格由 `task.yaml.editable_files` 限定。
6. 会话中断后使用 `/autoresearch --resume`，不要直接修补状态文件。
7. `create_plan.py` 拒绝计划时应按 stderr 修改，不得原样重试。
8. hook 返回 plan mirror payload 时，下一轮应逐字镜像，不得手工编造。
9. 工作区脚本必须作为直接、前台、顶层命令运行；不要用 shell wrapper、后台运行或命令链包裹。
10. DIAGNOSE 优先调用 `ar-diagnosis` 生成包含 `Root cause`、`Fix directions`、`What to avoid` 和完成标记的诊断文件，再生成新计划；连续 5 次失败后才允许手工兜底。
11. 只有 FINISH phase 允许停止。预算耗尽或连续失败应进入 DIAGNOSE，不得提前退出。

## 组件边界

- `scripts/`：工作区状态机、批处理与同步评测入口。
- `src/op_autoresearch/op/verifier/`：KernelVerifier、profiling 与 backend/DSL/framework 适配器。
- `src/op_autoresearch/core/worker/`：本地/远端 worker 与管理器。
- `src/op_autoresearch/worker/server.py`：HTTP worker 服务。
- `src/op_autoresearch/core/async_pool/`：设备租约池。
- `src/op_autoresearch/op/utils/code_checker/`：静态检查和运行时保护。

## 依赖

- Python >= 3.10
- 基础依赖由 `pyproject.toml` 安装。
- HTTP worker 使用 `.[worker]` 可选依赖。
- PyTorch、设备扩展、DSL 编译器与硬件 SDK 根据目标后端安装。
- Claude Code 或 OpenCode CLI。
