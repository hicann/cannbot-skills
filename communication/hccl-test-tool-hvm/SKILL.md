---
name: hccl-test-tool-hvm
description: 触发：需要用 HCCL-VM 对 AICPU 集合通信算子执行 Checker、生成结构化证据、比较基线与回归，或为失败/缺失用例生成定点补测计划时使用。环境安装、Agent 调度和任务完成判定不属于本 Skill。
---

# HCCL-VM 测试能力

HCCL-VM 是无需实物 NPU 的集合通信仿真工具。本 Skill 提供环境就绪检查、覆盖计划校验、用例执行、证据固化、基线比较和定点补测能力。Skill 名为 `hccl-test-tool-hvm`；实际工具的 `hccl-vm` 命令及 HCOMM 中的 `test/hccl_vm` 路径保持原名。

## 能力边界

- 本 Skill 定义怎么检查和执行 HCCL-VM，以及什么证据能证明某个用例或测试批次通过。
- 调用方提供测试目标、冻结矩阵、环境路径、源码/包身份和执行授权；已提供的信息不得重复询问。
- 本 Skill 不选择 Agent，不推进 Plugin 状态，不决定构建、安装或 Review 的调度顺序，也不宣布整个开发任务完成。
- 本 Skill 不修改 HCCL 生产源码或测试期望。缺少环境、权限或可信包时输出 `NOT_READY` 及证据，不自行转入环境安装或源码修复。
- 环境准备由 `hccl-env-setup` 提供，环境静态检查由 `hccl-env-check` 提供；是否调用它们由上层工作流决定。
- HCCL-VM 工具目录固定复用 `${HCOMM_CODE_HOME}/test/hccl_vm`；该目录已提供现成工具和构建/安装入口，不再下载或克隆 CheckerL2。目录或产物缺失时输出 `NOT_READY`，不自行补充下载。

## 输入契约

执行前至少取得：

- HCCL-VM 安装目录、`${HCOMM_CODE_HOME}/test/hccl_vm` 工具目录和 CANN `set_env.sh` 路径；
- 算子、模式、数据类型、数据量、通信域和集群拓扑；
- 测试目的：普通检查、基线、定向命中或回归；
- baseline/regression 使用的 `artifact.json`，以及调用方要求的源码和环境身份；
- 定向测试的 `HCCL_ALGO` 配置与预期注册名；
- 已授权的共享环境操作范围。

真实缺项不能安全推导时，返回缺项列表。算子、目标 rank、测试矩阵或授权已由上层冻结时直接继承，不重新定义。

## 按任务加载资源

| 任务 | 入口 |
|---|---|
| 检查二进制、CANN、测试程序、设备库和 rootinfo | [环境就绪检查](references/environment.md) |
| 固定多批次覆盖、检查参数、计算多 loop 边界 | [测试覆盖计划](references/test-plan.md) 与 `plan.py` |
| 执行一个测试矩阵并保存原始日志 | `run.sh` |
| 检查证据、比较 baseline/regression | [基线证据与回归比较](references/evidence.md) 与 `evidence.py` |
| 生成失败/缺失用例补测计划、合并证据 | [定点补测与切包前检查](references/recovery.md) 与 `repair.py` |

只读取当前任务所需的参考。工具帮助是参数真源：

```bash
bash <hccl-test-tool-hvm>/run.sh --help
python3 <hccl-test-tool-hvm>/plan.py --help
python3 <hccl-test-tool-hvm>/evidence.py --help
python3 <hccl-test-tool-hvm>/repair.py --help
```

`<hccl-test-tool-hvm>` 必须解析为当前已加载 Skill 的目录；不要从家目录或历史副本猜测编排脚本位置。HCCL-VM 的实际工具/构建目录使用 `${HCOMM_CODE_HOME}/test/hccl_vm`，不要将 CheckerL2 目录作为替代路径或下载目标。

## 参数技术约束

### 模式与覆盖对象

- `CCU_SCHED`：CCU 调度模式。
- `CCU_MS`：CCU 多流模式。
- `AI_CPU`：AICPU 展开模式，需要对应设备侧符号和 QEMU 环境。
- 前两项是 HCCL-VM 工具参数，不表示本 Plugin 已对外支持 CCU 算法开发；CCU 请求由路由门禁阻断。
- 支持算子：`allgather, allreduce, alltoall, alltoallv, reduce, reduce_scatter, scatter, broadcast`。
- 支持数据类型以 `run.sh --help` 和指定测试程序相邻源码为准；不能因拼写合法推断目标算法支持该类型。
- Runner 默认关闭。只有测试计划明确要求时才传 `--runner`；`--no-checker` 的结果不能作为 Checker PASS。

### 通信域

| 目标 rank | 主通信域 | 可选跨 server 覆盖 | 路由冒烟 |
|---:|---|---|---|
| 2 | 112 | 121 | 112 |
| 4 | 114 | 122 | 112 |
| 8 | 118 | 124、141 | 112 |
| 16 | 128 | 144 | 112 |
| 6（非对称） | 12_2_4 | — | 112 |

8-rank 算法不能只用 112/114 作为主验证。该表只提供已知映射；冻结测试计划或当前 Checker/集群能力记录有更具体约束时以后者为准。

### 定向与回归

- `--role` 是证据类别 `ordinary|baseline|directed|regression`，不是 Agent 角色。
- `directed` 必须同时提供非空 `HCCL_ALGO` 和 `--expect-algo`，证明新算法实际命中。
- `baseline` 与 `regression` 必须提供各自包的 `artifact.json`，使用相同矩阵和可比较环境。
- baseline/regression 不设置 `HCCL_ALGO`；默认选路和显式选路是不同证据轨道。
- 配置 DSL 与注册全名不是同一字段。AICPU 场景按 [显式选路说明](../hccl-aicpu-best-practice/references/executor-selector.md#4-hccl_algo-显式选路保留默认选择) 核对。

## 执行

单批次示例：

```bash
bash <hccl-test-tool-hvm>/run.sh \
  --install-dir <hccl_vm_install> \
  --cann <CANN_HOME>/set_env.sh \
  -c <cluster.yaml> \
  -m <CCU_SCHED|CCU_MS|AI_CPU> \
  -o <operator> -d <dtype> -s <size> -t <comm-domain> \
  --timeout <seconds> \
  --log-dir <evidence-parent>
```

按证据类别追加 `--role`、`--artifact`、`--env`、`--expect-algo` 和 `--phase`。先用 `--dry-run` 检查新组装的命令；dry-run 永远不能记为测试通过。

执行约束：

- `run.sh` 为每轮建立唯一 run 目录并管理该轮共享内存清理，不重复使用输出目录。
- 保留完整 stdout、stderr、原始日志和退出码；不要通过 `tail` 截断证据。
- 多批次默认使用已检查的 `plan.py run`；目标 executor 各用独立单用例定向冒烟，优先检查新增/变更的接入链，通过后展开覆盖。首个失败批次后停止，不自动重试或安装。
- 多 loop 使用已核验的 executor 阈值与测试程序换算调用 `plan.py loop-size`，并保留路径日志断言；不连续增大数据量试探。
- 同一共享环境串行执行计划；汇总读取 `summary.json`，dry-run 不计入已执行用例，未分类时间不归因成权限等待。
- 同条件偶发性诊断最多重跑一次；后续实验必须有明确假设、唯一变化和区分标准。
- 环境、包、测试程序或矩阵变化会影响可比较性，必须记录新身份。

## 结果语义

单个用例只有同时满足以下条件才是 PASS：

- mpirun 仅有一次成功退出；
- 存在完整且成功的 `CHECKER_RUN_SUMMARY`；
- 没有失败汇总或 PASS 中夹带 error；
- 要求算法命中时，完整算法集合和预期注册名一致；
- 原始日志、测试程序、包和配置身份可追溯。

批次 PASS 表示该矩阵的技术验证通过，不等同于 Review PASS、Plugin 状态推进或整个任务完成。输出至少包含 run 目录、总数/通过数/失败数、失败用例与原因、实际算法名、配置和证据哈希。

## 已知限制

- 共享内存、CANN 安装、rootinfo 和 HCCL-VM 安装目录可能被多个调用方共享，执行前必须由上层保证串行和授权。
- `--dry-run`、静态检查、文件存在或任意一条 `Checker Success` 文本都不能替代完整证据。
- 工具离线测试只验证脚本行为，不执行真实 HCCL/CANN 构建、安装或 Checker。
