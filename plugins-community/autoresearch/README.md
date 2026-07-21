# Op AutoResearch

一个独立的、由代码 Agent 驱动的算子内核迭代优化项目。它把可度量的内核验证与性能评测组织为：

```text
BASELINE -> PLAN -> EDIT -> EVAL -> KEEP / DISCARD / FAIL
                                      -> REPLAN / DIAGNOSE / FINISH
```

本仓库自带完整工作区、KernelVerifier、Ascend worker、设备租约池、代码检查器、Claude Code/OpenCode 接入和批量任务工具。当前支持 `triton_ascend`、`ascendc` 和 `ascendc_catlass`。

## 安装

```bash
python -m pip install -e .
# 需要启动 HTTP worker 时：
python -m pip install -e ".[worker]"
```

Ascend 运行时需另行安装，包括 PyTorch/MindSpore、torch_npu、CANN，以及任务所需的 Triton Ascend 或 CATLASS；这些大型运行时不由本项目的通用依赖自动安装。

## 快速开始

把 reference 与 seed kernel 放入 `workspace/`，在仓库根目录启动 Claude Code 或 OpenCode，然后执行：

```text
/autoresearch --ref workspace/<op>_ref.py --kernel workspace/<op>_kernel.py \
  --op-name <op> --devices <device-id>
```

监控任务：

```bash
python scripts/dashboard.py <task_dir> --watch
```

详细用法见 [AUTORESEARCH.md](AUTORESEARCH.md)，Agent 运行约束见 [AGENTS.md](AGENTS.md)。

## Worker

本地启动、查询和停止：

```bash
op-autoresearch worker --start --backend ascend --arch ascend910b3 --devices 0
op-autoresearch worker --status
op-autoresearch worker --stop
```

远端 worker 在 `config.yaml` 的 `remote_worker.hosts` 中配置；`repo_path` 指向远端本仓库根目录：

```bash
op-autoresearch worker --remote-host eval-host --start --backend ascend --devices 0
op-autoresearch worker --remote-host eval-host --status
op-autoresearch worker --remote-host eval-host --stop
```

## 目录

```text
scripts/                  状态机、评测桥、批处理与工作流
src/op_autoresearch/      verifier、worker、设备池及共享运行时
.claude/                  Claude Code 命令、hooks 与诊断 Agent
.opencode/                OpenCode 命令、插件与外层循环
ar_examples/              三种受支持 DSL 的最小示例
tests/                    工作区契约测试与移植的组件单测
```

复制来源的精确提交记录在 `SOURCE_REVISION`，开源组件元数据记录在
`README.OpenSource`。AKG 衍生运行时代码保留 Apache-2.0 许可和 NOTICE；
CANN 示例代码适用 CANN-2.0，完整文本分别见 `LICENSE-APACHE` 和 `LICENSE`。
