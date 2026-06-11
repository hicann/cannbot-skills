# Benchmark 评测框架

批量评测框架：通过 opencode 执行算子生成任务，对 cann-bench 中的算子进行定向评测。

```
提示词模板 → opencode run -f prompt.txt → 算子生成 → 收集评测结果
```

## 目录结构

```
tests/benchmark/
├── config/
│   └── eval_config_mini.yaml    # Mini子集配置 (16个算子)
├── prompts/
│   └── op_dev_prompt.txt        # 提示词模板
├── runner/
│   ├── run_eval.py              # 批量评测执行器
│   ├── setup_cann_bench.py      # cann-bench 下载/初始化
│   ├── isolation_check.py       # 隔离检查
│   ├── cleanup.py               # 清理脚本
│   └── archive_run.py           # 归档脚本
└── results/                     # 评测结果输出 (gitignore)
```

## 依赖

- Python 3.8+, `pyyaml`
- `opencode` CLI (需在 PATH 中)
- `git` (用于 clone cann-bench)

## 快速开始

```bash
# 全量评测 (自动扫描 cann-bench 所有算子)
python tests/benchmark/runner/run_eval.py --all

# Mini子集评测 (16个核心算子)
python tests/benchmark/runner/run_eval.py -c tests/benchmark/config/eval_config_mini.yaml

# 只评测单个算子
OPS_FILTER="level2/softmax" python tests/benchmark/runner/run_eval.py -c tests/benchmark/config/eval_config_mini.yaml
OPS_FILTER="level1/exp" python tests/benchmark/runner/run_eval.py --all

# 使用指定分支的 cann-bench
python tests/benchmark/runner/run_eval.py --all --cann-bench-branch dev

# 强制更新 cann-bench
python tests/benchmark/runner/run_eval.py --all --update-cann-bench
```

## cann-bench 管理

cann-bench 在首次评测时自动 clone，无需手动管理。也可独立操作：

```bash
# 手动下载/确认
python tests/benchmark/runner/setup_cann_bench.py

# 更新到最新
python tests/benchmark/runner/setup_cann_bench.py --update

# 重置 (删除后重新 clone)
python tests/benchmark/runner/setup_cann_bench.py --reset
```

## 运维命令

```bash
# 隔离检查 (评测前自动运行)
python tests/benchmark/runner/isolation_check.py

# 清理 cann-bench 和 results
python tests/benchmark/runner/cleanup.py --dry-run
python tests/benchmark/runner/cleanup.py --force

# 归档评测产物到 git 远程分支
python tests/benchmark/runner/archive_run.py --name run-001
python tests/benchmark/runner/archive_run.py --dry-run
```

## 关键概念

- **算子目录**: `cann-bench/tasks/level{N}/{op_name}/`，含 cases, desc, golden, proto
- **参考工程**: `direct_launch_example` (算子直调, 产出 .whl)
- **交付件**: `cann_bench-xxx.whl`，需放到算子目录的 `dist/` 下
- **算子类别**: Elementwise / FusedComposite / Normalization / Reduction / ScatterUpdate / IndexGather / Contraction / SortSelect / LayoutTransform
