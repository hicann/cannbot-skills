<!--
范例：逆向自 src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_one_shot.cc
可与源码逐行对照，用来校准记法。写新 template 时复制根目录 dataflow-spec-template.md，不要复制本文件。
-->

# 数据流规格：InsTempAllReduceMesh1DOneShot

## 1. 元信息

| 字段 | 值 |
|---|---|
| spec 类型 | 逆向存量（照 ins_temp_all_reduce_mesh_1D_one_shot.cc 反写，可与源码逐行对照） |
| 算子 | `all_reduce` |
| 类名 | `InsTempAllReduceMesh1DOneShot` |
| 文件名 | `ins_temp_all_reduce_mesh_1D_one_shot` |
| 引擎 | `aicpu` |
| 拓扑 | `Mesh1D`（level0 全连接） |
| 变体 | `one-shot` |
| 父类 | `InsAlgTemplateBase` |
| 所属层级 | 单层（只做 level0） |
| 兄弟 template | 无 |
| 算法名 | `AicpuAllReduceSoleMeshOneShot` |
| 绑定 executor | `InsV2AllReduceSoleExecutor` + `TopoMatch1D` |
| 启用方式 | `自动选路` —— 逆向描述已有算法，实际选择链路须核对目标仓 |
| 默认选路影响 | `不变` —— 逆向存量 spec，不修改选路 |

> Mesh 全连接下一步把整份数据推给所有对端的 scratch，各自再把收到的 N-1 份本地归约进 output。
> 只有 1 跳，通信量 O(N)，小数据量占优。

## 2. 适用条件

| 维度 | 约束 | 理由 |
|---|---|---|
| rankSize | `2..8` | 通信量随 rankSize 线性增长，超过 8 卡 `CalcCostCoeff` 返回 `{}` |
| 拓扑 | level0 全连接（Mesh1D） | 需要任意两 rank 直连 |
| 数据类型 | 全部 | 64-bit 走软归约兜底 |
| reduceOp | 全部 | PROD 走软归约兜底 |
| 数据量 | 小数据量占优 | scratch 占用 = N × 数据量，大数据量会把 loop 切碎 |
| 模式 | 单算子 OPBASE | |
| selector | 以目标仓 selector 与配置解析入口为准 | template 本身不决定选择链路 |
| 平台 | template 侧不约束 | 平台限制需核对 selector |
| inplace | 由 selector 拦截重叠场景 | 与第 7 章一致 |

## 3. 资源

| 项 | 表达式 |
|---|---|
| thread 总数 | `N > 1 ? N : 1` |
| slaveThreadNum | `thread 总数 - 1` |
| notifyNumPerThread | 每个从流 1 个 |
| notifyNumOnMainThread | `= slaveThreadNum` |
| notifyIdx main→sub | 全 0 |
| notifyIdx sub→main | `0, 1, ..., slaveThreadNum-1` |
| channel 申请 | `CalcChannelRequestMesh1D`，push 到 `channels[0]`（level0 一层） |
| 每远端 channel 数 NCH | 1（取 `channels.at(p)[0]`） |
| 逻辑 peer | `全部 N-1 个对端` |
| 申请链路 | `全部 N-1 个对端` |
| 过度申请理由 | 无 |
| 多 channel 切分 | 不切分（NCH = 1） |

不覆写 `GetThreadNum()` / `GetRes()`：`KernelRun` 里直接用 `templateResource.threads.size()`，
并校验它等于 `N`，不等就报错退出。

## 4. Buffer 布局

```
SCR (hcclBuff)  一轮 loop 占用 = N × S
┌──────────┬──────────┬─────┬──────────┐
│ rank 0   │ rank 1   │ ... │ rank N-1 │      每段 S 字节
└──────────┴──────────┴─────┴──────────┘
 0          S          2S          (N-1)S
             ↑ 第 r 段专收 rank r 推过来的整份数据

IN  ： 本 rank 完整输入，S 字节
OUT ： 本 rank 完整输出，S 字节（先被 IN 覆盖，再逐份累加）
```

| 项 | 值 |
|---|---|
| **scratch 倍数** | `N`（`templateRankSize_`） |
| 理由 | 每个 rank 在我 scratch 里独占一段，共 N 段 |

## 5. 切片

```
SS = S          # one-shot 不切片，每段就是整份数据
seg(r) = SCR[r * S, S]     # 第 r 段
```

均分成 N 段，第 r 段偏移 `r*S`、长度 `S`，无尾块问题（每段等长）。
校验：`seg(N-1).offset + seg(N-1).len == N * S`。

> **已知前提**：`seg` 用 `R`（通信域 userRank）直接作下标，
> 因此本 template 只在 `subCommRanks_[0] == [0..N-1]`（userRank 与算法内序号一致）的单层场景成立。
> 分层复用时必须先 `GetAlgRank(myRank_, rankList_, algRank)` 换算，否则会写错段。

**repeat 与 stride**：

| 符号 | 值 | 说明 |
|---|---|---|
| `RPT` | `1` | 绑定的是单层 `InsV2AllReduceSoleExecutor`，它固定传 `repeatNum = 1` |
| `IRS` / `ORS` | `0` / `0` | `RPT = 1`，没有第二块 |
| `ISS` / `OSS` | `S` / `S` | one-shot 不切片，整份就是一片 |
| scratch 的 repeat 跨度 | 不适用 | 只有一次 repeat |

> 与源码对照：`ins_temp_all_reduce_mesh_1D_one_shot.cc` 里**没有** `for (rpt ...)` 循环，
> 全文一次都没引用 `repeatNum`。这是上面「只在单层场景成立」的另一个面：
> 它一旦被 2/3 级 executor 当 intra 构件复用，只会处理第 0 块。
> 新写 template 时建议无条件套上 repeat 循环（`RPT = 1` 时零代价退化），见 `references/07` §1.5。

| 字段 | 值 |
|---|---|
| 参数赋值来源 | AllReduce Sole executor OrchestrateLoop；此例逆向描述已有 one-shot，使用前核对目标仓分支 |
| 算法 rank 映射 | 仅 subCommRanks[0]=[0..N-1]，此例 userRank 等于算法 rank |

该存量 one-shot 只处理一次，input/output 整份拷贝不按 rank 分片；不把此消费方式推广到 AllGather。
数值例：本轮 S=256B，输入/输出本轮基址偏移为64B；scratch 每 rank 一段，rank2 段从512B起。

```layout-check
{"cases": [
  {"name": "local input", "formula": "b", "vars": {"b": 64}, "expected": 64, "length": 256, "capacity": 1024, "element_size": 4},
  {"name": "local output", "formula": "b", "vars": {"b": 64}, "expected": 64, "length": 256, "capacity": 1024, "element_size": 4},
  {"name": "scratch rank2", "formula": "b+i*S", "vars": {"b": 0, "i": 2, "S": 256}, "expected": 512, "length": 256, "capacity": 1024, "element_size": 4}
]}
```

## 6. 数据流


```
# 校验：thread 数必须恰好等于 N
assert threadNum == N

# 阶段一：主流本地拷贝，把自己那份先落到 output
T0: LocalCopy  IN[0,S] -> OUT[0,S]

if N == 1: return                       # 单 rank，拷贝完就结束

sync main -> sub

# 阶段二：每个从流对一个远端，把整份数据推进对方 scratch 的第 R 段
for i in 1..N-1:
    p = subCommRanks[0][(R + i) % N]    # 错峰，避免全员同时打同一目标
    T[i]: SendRecvBatchWrite  IN[0,S] -> SCR@p[R*S, S]

sync sub -> main

# 阶段三：主流把 scratch 里其余 N-1 段累加进 output
if needAicpuReduce: barrier(all threads)
for r in 0..N-1 where r != R:
    T0: LocalReduce  SCR[r*S, S] into OUT[0,S]
```

阶段二只写了"我推给谁"这一个方向：写模式下 `rxSlicesList_` 不参与搬运，
对端推给我的那一份由框架的 notify 握手保证，见 `references/07` §2.1。

## 7. 边界条件

- [x] `C == 0`：**当前实现未做早退**，靠上层 executor 保证 count 非 0。新写 template 建议补上。
- [x] `N == 1`：只做 `LocalCopy`，`subCommRanks_[0].size() == 1` 时直接返回
- [x] 尾块 / 非均衡切分：不涉及，N 段等长
- [x] `needAicpuReduce`：INT64 / UINT64 / FP64 / PROD 时，归约前 `barrier(all threads)`
      （`BatchModeEnd → BatchModeStart → ThreadJoin`），确保 scratch 已写完再软归约
- [x] `isPcie`：本 template 不区分，固定走写模式
- [x] `graphMode`：不支持，仅 OPBASE
- [x] `symMem`：不支持
- [x] 多 jetty：不支持，固定取 `channels.at(p)[0]`
- [x] `RPT > 1`（被分层 executor 复用）：不支持，本 template 只用于单层；复用前要先补 repeat 循环 + algRank 换算
- [x] inplace / 重叠：由 selector 层拦截，本 template 不处理

## 8. Cost model

| 项 | 值 |
|---|---|
| 不适用时 | `rankSize > 8` → `return {}` |
| 策略 | `必须实现` —— 保留已有 CalcCostCoeff |
| 自动阈值证据 | 无阈值变更（逆向存量 spec） |
| netType | `param.netType`（`CLOS` 时 portNum=8，否则 1） |
| taskNum | 1 |
| 传输项 A | `CalcMeshParam(n, netType, portNum, rankSize, A)` |
| n 的取法 | `n = param.dataRatio * param.rankSize`（executor 统一传 two-shot 的单片大小，one-shot 乘回去） |
| 本地计算项 B | `B1(localCopy，仅当 inputBuffer != scratchBuffer) + (rankSize-1) × B2(localReduce)`；另有 D=下发项 |
| 延迟项 C | `CalcLatencyParams(taskNum, EngineType::AICPU, C)` |

## 9. 验证

| 项 | 值 |
|---|---|
| hccl-vm 参数（轨道 A 定向） | `AI_CPU` 引擎 + 仓内确认的 HCCL_ALGO 配置 + `--expect-algo AicpuAllReduceSoleMeshOneShot` + allreduce / 目标数据类型 / 目标数据量 + 主验证通信域（Checker 门禁） |
| hccl-vm 参数（轨道 B 回归） | 不设 `HCCL_ALGO` 默认路径 + 与基线同覆盖；选中算法须与基线一致 |
| 覆盖规格 | 单 Server 2/4/8 卡 |
| 重点场景 | count 边界（0 / 1 / 非对齐）、INT64、PROD、大 count 触发多轮 loop |

## 10. 待确认

无（本文件是对既有实现的逆向描述）。
