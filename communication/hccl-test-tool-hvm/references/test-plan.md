# 提前固定测试覆盖计划

在 spec 完成、候选构建之前生成 JSON 计划：一次确定覆盖目的、拓扑、布局、配置和证据断言。基线与回归使用同一矩阵与环境；先逐 executor 冒烟，再展开覆盖。计划检查不代替源码推导或验证结果。

## 文件格式

顶层 `schema_version: 1`，必填 `install_dir`、`cann`（set_env.sh 文件）、`cluster` 和非空 `batches`；可选 `test_bin_dir`。每个 batch：

| 字段 | 含义 |
|---|---|
| `id` | 安全且唯一的批次标识 |
| `role` | ordinary / baseline / directed / regression |
| `covers` | 非空字符串列表，写清布局、拓扑或边界覆盖目的 |
| `modes`, `ops`, `dtypes`, `comms` | 字符串列表 |
| `sizes` | 正整数列表，单位为测试程序的数据量字节 |
| `env` | HCCL_* 环境变量到字符串的映射；运行前清除继承的 HCCL_*，所需配置必须写入各批次；禁止覆盖工具管理的安装路径与引擎变量 |
| `artifact` | baseline/regression 必填；相对计划文件或绝对的 artifact.json 路径 |
| `expect_algo` | directed 必填，且 env 必须设置 HCCL_ALGO；配置 DSL 与注册名不同 |
| `timeout` | 单用例秒数，默认 180 |
| `assertions` | 可选日志断言列表；每项 name、basis、pattern、minimum |

断言的 `basis` 写明源码公式和参数含义；`pattern` 恰有一个数字捕获组，每个用例的匹配最大值须达到 `minimum`。表达式应限定目标 executor/template 上下文，避免任意数字或其他层的 loop 满足断言。executor 多 loop 与 template repeat 分别覆盖。

例如同时接入 AllGather Parallel/Sole Ring 的定向冒烟计划（路径必须替换；DSL、拓扑和注册全名须适用目标仓）：

```json
{
  "schema_version": 1,
  "install_dir": "/实际路径/hccl_vm_install",
  "cann": "/实际路径/cann/set_env.sh",
  "cluster": "ascend950_cluster_32_server_normal.yaml",
  "batches": [
    {
      "id": "parallel-smoke",
      "role": "directed",
      "covers": ["Parallel 两层接入、显式启用与 Checker 冒烟"],
      "modes": ["AI_CPU"],
      "ops": ["allgather"],
      "dtypes": ["int32"],
      "sizes": [64],
      "comms": ["122"],
      "env": {"HCCL_ALGO": "allgather:parallel{ring,ring}"},
      "expect_algo": "AicpuAllGatherParallelRingRing"
    },
    {
      "id": "sole-smoke",
      "role": "directed",
      "covers": ["Sole 单级两卡定向冒烟"],
      "modes": ["AI_CPU"],
      "ops": ["allgather"],
      "dtypes": ["int32"],
      "sizes": [64],
      "comms": ["112"],
      "env": {"HCCL_ALGO": "allgather:sole{ring}"},
      "expect_algo": "AicpuAllGatherSoleRing"
    }
  ]
}
```

注册全名须对照 [显式选路入口](../../hccl-aicpu-best-practice/references/executor-selector.md#4-hccl_algo-显式选路保留默认选择) 和目标源码核对。
选择器开关按目标版本消费点决定：部分版本需要 `HCCL_USE_NEW_SELECTOR=1`，已直接进入该选择链的版本无需照搬。
上述顺序适合新改 Parallel 的任务；只有 Sole 时删除 Parallel 批次，其他任务按实际接入风险排序。
每个冒烟批次的列表均为单元素，避免命中失败后继续跑同 executor 的多档数据。
完整计划在这些冒烟后补布局/类型/边界覆盖，并配置同矩阵的 baseline/regression；此示例仅说明格式。

## 检查与执行

```bash
python3 <hccl-test-tool-hvm路径>/plan.py check <plan.json>
python3 <hccl-test-tool-hvm路径>/plan.py run <plan.json> --role directed --output <新的计划运行目录> --dry-run
# 审查命令后去掉 --dry-run；同一 output 不重复使用。
```

执行只选择指定角色，串行调用 run.sh；首个失败批次后停止，不自动重试或安装。一个批次可能包含多个用例；需要冒烟立即停止时保持冒烟批次小而独立。`HWLOC_COMPONENTS` 固定为 `-gl,-opencl`。

`summary.json` 保存批次时间、命令、覆盖目的、结果计数、失败及未执行批次数。直接从记录生成耗时与用例统计，不手工相加推算；dry-run 不能记为通过。计划执行成功不代替 `evidence.py compare`、两轨全通过或覆盖审查。
顶层 `totals` 区分尝试批次、带用例结果的执行批次、dry-run 批次、例次、通过/失败与用例时间；
`wall_seconds` 是该次计划调用的墙钟时间，`batch_wall_seconds` 是各批次调用之和。
两者与用例时间之间的差额含启动、收集等开销，不应自动归为权限等待。
跨计划汇总按唯一 run 目录去重；修复前失败记录保留在历史统计中，不能算作最终候选通过证据。

## 多 loop 参数推导

先根据目标 executor 的实际缓冲区、scratch 倍数、对齐和 rank 分层预算计算 `maxCountPerLoop`；再核实目标测试程序的字节数与 executor count 的换算。ReduceScatter 的总输入量与每 rank count 不能凭 AllGather 经验等同。

```bash
python3 <hccl-test-tool-hvm路径>/plan.py loop-size \
  --max-count-per-loop <已推导阈值> --test-bytes-per-count <已核实换算系数> \
  --alignment-count <合法count对齐单位>
```

工具只计算刚超过阈值且对齐的 count 和测试字节数，不证明输入公式、缓冲区参数合法。优先采用源码允许的小 HCCL_BUFFSIZE，将推导依据与 `loopTimes > 1` 日志断言写入计划；模板 `repeatNum > 1` 另设断言。
在 `basis` 写清 executor 方法、缓冲区容量、scratch 倍数、对齐、测试程序 bytes→count 公式及来源。
对称内存/单轮旁路可能取消分块限制，须选实际走分块的合法配置。Parallel 预算示例见
[AllGather 四阶段契约](../../hccl-aicpu-best-practice/references/12-allgather-parallel.md#4-多-loop-的预算入口)。
若该版本没有直接输出 loopTimes，使用可区分两轮的目标 executor 日志或先增加明确诊断；
总 KernelRun 次数须扣除 rank、阶段、迭代等因素，不能直接当作 loop 次数。

通信域能力记录绑定 Checker 版本、集群配置和 rootinfo，变化后重验。遇失败按完整日志定位阶段，下一实验记录假设和唯一变化；不持续扩大数据量或重复相同矩阵以碰到 PASS。

失败或缺失用例的计划生成、证据补齐及切包前门禁见 [定点补测与切包前检查](recovery.md)。
