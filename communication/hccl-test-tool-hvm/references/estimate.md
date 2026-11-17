# estimate 性能估算

`estimate` 是 HCCL-VM 的性能估算模式，用于在功能验证通过后估算本次开发算法的执行耗时，并将结果纳入最终报告。使用方法依据用户提供的 `/home/hccl/vibe/estimate.md`（2026-10-12），以下已保留必要步骤，无需依赖该本地文件再次读取。

## 前置条件与组网

- 当前候选算法已取得有效的 HCCL-VM 功能验证 PASS 证据；继承已核验的 CANN、HCCL-VM、测试程序、引擎配置和候选包身份。
- 功能验证会话结束后，在同一共享环境中串行启动独立的 estimate 会话。为本次估算建立唯一证据目录，保存命令、完整 stdout/stderr 和各命令退出码。
- 默认组网为 `112`。用户指定组网时，将 `hccl-vm mock-comm 112` 的参数改为对应通信域，核对当前集群配置支持该拓扑。此处通信域参数与 `mpirun -np` 是两个独立参数。
- 确认目标算法支持所选组网。若默认 112 无法命中或不被目标算法支持，记录限制并交由调用方确认适用组网，不能把其他算法的估算记为此次开发算法的结果。
- 沿用功能验证中已核验的显式选路配置（如 `HCCL_ALGO`，需要时），并从本次原始日志核对实际算法注册名。1G 数据量或组网变化可能改变选路，功能验证的命中证据不能代替本次命中证据。
- 核对当前安装提供 `@estimate` 及其 Python/simpy 依赖；缺失时记录 NOT_READY 和具体证据，不自行安装。估算单独记录环境，结束后恢复本次覆盖的配置，不将其带入 Checker 会话。

### estimate 环境与 CCU 模式

目标 HCOMM 的 `test/hccl_vm/README.md` §4.9.4 要求估算业务启动前关闭链路心跳和一致性校验（幻影对端无法参与），在独立估算 shell/会话准备：

```bash
export HCCL_DFS_CONFIG=cluster_heartbeat:off,inconsistent_check:off
```

有其他已确认 DFS 配置时保留其余项，仅合并上述两个键。此设置仅用于 estimate，不代替功能验证的冻结配置。

CCU 还须继承本次功能验证的模式和依赖身份，在启动 estimate 会话前设置 `HCCL_OP_EXPANSION_MODE` 为冻结的 `CCU_SCHED` 或 `CCU_MS`，并沿用已核验的 `HCCL_ENABLE_OPEN_CCU=1` 与显式选路配置；不切换为 AI_CPU。核对本次实际加载的 HCCL/HCOMM 库和候选包，记录两仓身份及模式。当前工具不能估算该模式/算法时记录失败与限制，不用其他模式或 fallback 的耗时代替。固定单进程估算也不证明单 rank 算法分支或全 rank 功能覆盖。

## 执行方法

在已加载上述环境的 shell 中，从确认过的 HCCL-VM 安装目录启动：

```bash
cd <install_dir>/bin
./hccl-vm start ascend950_cluster_32_server_normal.yaml --estimate
```

`ascend950_cluster_32_server_normal.yaml` 是用户提供的示例集群配置；使用当前安装中已确认、支持所选通信域的配置，并记录实际路径和哈希。

启动后，在该 HCCL-VM 会话内依次执行。以下以 ReduceScatter 为例，`<estimate_dir>` 替换为本次唯一证据目录的绝对路径：

```bash
hccl-vm mock-comm 112
mpirun --allow-run-as-root --oversubscribe -np 1 \
  "${ASCEND_HOME_PATH}/tools/hccl_test/bin/reduce_scatter_test" \
  -b 1G -e 1G -d int32 -o sum -w 0 -n 1 -c 0 \
  > <estimate_dir>/estimate-test.log 2>&1
hccl-vm plugin run @estimate
```

每条命令成功退出后才执行下一条；失败时保存证据并停止本次估算。保留插件完整输出及其实际输出目录，不把管道或日志收集命令的成功当作测试程序/插件成功。

参数约束：

| 参数 | 要求 |
|---|---|
| `mpirun -np` | 固定为 `1`，即使 `mock-comm` 使用更大通信域也不能改为目标 rank 数 |
| `-b`、`-e` | 均固定为 `1G` |
| `-w`、`-n`、`-c` | 固定为 `-w 0 -n 1 -c 0` |
| 测试程序 | 将 `reduce_scatter_test` 替换为此次开发算法对应的算子测试程序 |
| `-d`、归约参数（示例 `-o sum`） | 按目标算法和实际测试程序支持范围选择，归约参数仅在适用时使用 |
| `mock-comm` | 默认 `112`，可根据用户组网需求修改 |

现有 `run.sh` 按通信域 rank 数启动 mpirun，并使用 `-c 1` 与 Checker；不要直接用它执行 estimate，也不要将 estimate 结果交给 Checker 的 PASS 判定或 baseline/regression 比较器。性能估算与功能验证分别保存证据。

## 结果读取与报告

读取 `hccl-vm plugin run @estimate` 本次输出的“任务执行耗时”或 `makespan`，保留原始单位；这是算法性能估算的主要耗时。输出中的“软件运行耗时”是仿真软件自身的运行时间，不能作为算法执行耗时。

从插件打印的实际输出目录保存 `perf_profiling.json` 和 `trace_comm*_iter*.json` 等本次生成文件，并记录路径和哈希。用户示例中的输出位于 `<install_dir>/plugin/light_performance/data/<时间戳>/`，具体目录以本次输出为准，不复制示例中的 CheckerL2 路径或时间戳。

提供以下内容给调用方，作为最终报告中的“算法性能估算”部分呈现给用户：

- 估算状态：成功、失败或未执行；失败/未执行时附原因与证据路径。
- 估算对象：算子、预期及实际命中的算法注册名、候选源码/包身份和对应功能验证证据。
- 估算条件：集群配置、实际 `mock-comm` 参数及拓扑、数据类型、归约操作（适用时）、固定 `1G` 数据量与完整命令。
- 算法估算耗时：本次 `makespan`/任务执行耗时；如存在多个算子或迭代，分别列出，不能混为单个算法结果。
- 主要观察：结合实际输出说明 stream 完成时间、任务数量、网卡带宽/端口统计等；记录未配对 WAIT、依赖丢弃、未解析 rank 等插件实际提示，以及它们对结论的限制。
- 原始证据：测试日志、插件输出、退出码、结果目录、profiling/trace 文件和哈希。
- 性能结论：明确这是所记录组网和参数下的仿真估算，不能作为实机测量或其他规模的性能结论。只有同条件、同估算方法且来源可追溯的基线存在时才报告性能改善比例；否则仅报告本次估算和观察。

例如，用户文档的示例输出中“任务执行耗时”为 `2560.48 us`，可写为“在所记录条件下估算耗时约 2.56048 ms”。该值仅用于说明读取方式，不能作为本次开发算法的结果。估算没有成功完成、缺失结果文件或无法确认目标算法命中时，不输出成功的性能结论；既有功能验证结论仍保留其独立证据。

## 结构化估算证据

在本轮独立证据目录中保存 `estimate.json`，用于索引本次估算的条件、源码/包身份、耗时和原始文件；该文件由调用方整理，不是 HCCL-VM 插件原生产物。失败或未执行时记录状态、原因及已有证据，不填入推测耗时。

成功证据的字段如下，数值必须来自本次执行，不能复制上面的示例耗时：

| JSON 字段 | 内容 |
|---|---|
| `schema_version`、`status` | 整数 `1`、字符串 `SUCCESS` |
| `source_identity` | 与功能验证所用候选一致的源码身份 |
| `operator`、`dtype` | 本次算子与数据类型字符串 |
| `comm_domain` | 实际通信域字符串，如 `"112"`；与本次实际命令一致 |
| `expected_algorithm`、`observed_algorithm` | 已核验的目标注册名和本次日志中的实际注册名，成功时一致 |
| `makespan_us` | 本次任务执行耗时，统一换算为 us 的有限正数 |
| `parameters` | `{"np": 1, "begin": "1G", "end": "1G", "warmup": 0, "iterations": 1, "check": 0}` |
| `exit_codes` | `start`、`mock_comm`、`mpirun`、`estimate` 四个整数退出码，成功时全部为 `0`；start 为会话结束后取得的真实退出码 |
| `files` | `candidate_package`、`cluster`、`test_log`、`plugin_log`、`profiling`、`trace` 六项，每项为 `{"path": "绝对路径", "sha256": "64位小写十六进制"}` |
| `engine`（CCU 必填） | 字符串 `CCU`；AICPU 原有证据无需补该字段 |
| `ccu_mode`（CCU 必填） | 本次实际模式 `CCU_SCHED` 或 `CCU_MS`，与冻结设计/测试计划一致 |
| `hcomm_source_identity`（CCU 必填） | 与功能验证及调用方登记一致的 HCOMM 依赖身份；`source_identity` 仍为 HCCL 单仓身份，不拼接两串 |

`candidate_package` 指功能验证和估算共同使用的归档候选包；`cluster` 指本次实际集群配置。`profiling` 指本次 `perf_profiling.json`，`trace` 指对应目标算子/迭代的 trace 文件。存在多个 trace 时，分别在报告中列出，`files.trace` 记录与该条估算耗时对应的文件。

将插件生成的 profiling/trace 复制归档至调用方明确指定的本轮证据目录内，测试/插件日志、候选归档包及 `estimate.json` 也应位于该目录范围；集群配置可以引用已核验的原文件。保存全部命令及实际参数至报告和原始日志，报告正文继续记录归约操作、包/库身份、拓扑、观察和限制。核对索引结构、固定参数、身份、算法命中与文件哈希；索引完整不能代替真实执行或人工核对原始日志。
