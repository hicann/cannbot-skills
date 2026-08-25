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
│   ├── progress.py              # 进度显示 (tqdm 可选, 无强制依赖)
│   ├── report.py                # 汇总报告生成 (md/html)
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

# 指定评测模型 (格式 provider/model, 用 opencode models 查看可用模型; 不提供时交互问询)
python tests/benchmark/runner/run_eval.py -c tests/benchmark/config/eval_config_mini.yaml \
    --model zhipuai-coding-plan/glm-5.2

# 模型白名单: config allowed_models 生效, 白名单外的 --model/问询输入会被拒绝
# --all 模式或需覆盖 config 白名单时用 --allowed-models (逗号分隔)
python tests/benchmark/runner/run_eval.py --all \
    --allowed-models kimi-for-coding/k3,zhipuai-coding-plan/glm-5.2 \
    --model kimi-for-coding/k3

# 使用指定分支的 cann-bench
python tests/benchmark/runner/run_eval.py --all --cann-bench-branch dev

# 强制更新 cann-bench
python tests/benchmark/runner/run_eval.py --all --update-cann-bench
```

## 超时与重试

```bash
# 超时优先级: op.timeout > config category_timeouts[category]
#   > config default_timeout > OP_TIMEOUT 环境变量 (默认 21600s)
OP_TIMEOUT=7200 python tests/benchmark/runner/run_eval.py -c ...

# 连接层重试: SERVE_RETRY 次 (默认 3), 覆盖连接错误与超时
SERVE_RETRY=3 python tests/benchmark/runner/run_eval.py -c ...

# 算子级重跑: 状态非 success 或交付不完整 (缺 .whl) 时整算子重跑,
# OP_RETRY 为每算子最大尝试次数 (默认 1 = 不重跑);
# 重跑会恢复上次持久化的算子产物 (含 dist/*.whl), agent 在此基础上继续
OP_RETRY=2 python tests/benchmark/runner/run_eval.py -c ...

# 注意: 单算子最坏墙钟 = OP_RETRY × SERVE_RETRY × 算子超时
# (每次 serve 尝试都可能用满超时, 如 OP_RETRY=2/SERVE_RETRY=3/超时7h → 42h),
# 调参时留意乘法关系
```

## 评测产物

- `results/summary.yaml` — 每算子 status/duration_s/tokens/cost/model_actual/attempts/op_attempt/delivery_ok
- `results/report_{run_id}.md` / `.html` — 汇总报告 (通过率/交付完整/耗时/token/cost 统计), 评测结束自动生成
- `results/{op}/sse_events.jsonl` — SSE 过程事件持久化 (subagent 分发/session 状态)
- `operators/{op}/` — 算子产物持久化 (csrc/ops 源码 + cann_bench/__init__.py + tests/ + dist/*.whl + docs/), 重跑时自动恢复, 也是归档数据源

退出码: 全部算子 success 且交付完整 (delivery_ok) 为 0, 否则为 1。

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
# 隔离检查 (评测前自动运行: example 目录洁净 + tasks/*/dist 无遗留 whl + 评测端口无残留 serve)
python tests/benchmark/runner/isolation_check.py

# 清理 cann-bench 和 results
# (含 tasks/*/dist/ 跨轮次遗留交付 whl, 防止泄露参考实现/绕过交付门禁)
python tests/benchmark/runner/cleanup.py --dry-run
python tests/benchmark/runner/cleanup.py --force

# 归档评测产物到 git 远程分支
# 数据源为 operators/ 持久化产物, 按 direct_launch_example 格式组装交付工程
# (构建文件 + csrc/ops/{op} + cann_bench/__init__.py + tests/{op} + dist/*.whl),
# 校验后提交到 cann-bench 归档分支的 eval_delivery/{op}/ 下
python tests/benchmark/runner/archive_run.py --name run-001
python tests/benchmark/runner/archive_run.py --dry-run
```

## 关键概念

- **算子目录**: `cann-bench/tasks/level{N}/{op_name}/`，含 cases, desc, golden, proto
- **参考工程**: `direct_launch_example` (算子直调, 产出 .whl)
- **交付件**: `cann_bench-xxx.whl`，需放到算子目录的 `dist/` 下
- **算子类别**: Elementwise / FusedComposite / Normalization / Reduction / ScatterUpdate / IndexGather / Contraction / SortSelect / LayoutTransform
- **残留 serve 检查**: 旧评测遗留的 `opencode serve` 进程会持续监听评测端口 (默认 4096)，劫持新 runner 的会话请求 (session 落到已删除的旧工作目录，message POST 返回 500 ServeError)。评测前框架自动检测并清理：先 `/shutdown` 优雅关闭，超时后按 PID 强杀 (仅当监听进程确为 opencode)。可手动 `ps aux | grep "opencode serve"` 排查。
- **遗留交付 whl 清理**: `tasks/{op}/dist/` 是算子交付位置，跨轮次评测会残留先前模型的交付 whl——后续 agent 可直接解包获取参考实现信息 (接口签名/kernel 符号)，且 `delivery_complete` 会把残留 whl 误判为本轮交付。评测前框架自动清空该目录 (每轮评测含重跑前必做)，隔离检查也会拦截残留 (可用 `--skip-isolation-check` 跳过，不推荐)。
