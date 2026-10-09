# 05 · Ring 族评审（R01–R09）

## 1. 先核对目标版本与选路登记

Ring 的实现与登记随版本变化，以目标仓源码为准。历史观察：`6ebb413b`（2026-09-04）
没有 AICPU Ring；`b54a1a28` 有未完成选路登记的 AllGather Ring。2026-09-22 用户提供的
Broadcast Ring 开发报告描述了链式全量广播，不能继续把所有 Ring 都视为 RS/AG 轮转分片。
这些历史案例不证明当前目标仓的实现正确。

目标是继承 `InsAlgTemplateBase` 的 AICPU template；V1 `scatter_ring.*` 不作为接口蓝本。
X01 按目标仓逐项核对，不维护 API 镜像：

| 登记点 | 缺失后果 |
|---|---|
| `src/common/alg_parse.h` 的 `AlgoType` | `props.algoType` 无有效枚举 |
| `alg_parse.cc` 的 `ALGO_TYPES` | DSL 无法解析算法 token |
| `GetAlgoTypeToNameMap()` | 枚举无法转换为注册名 |
| `GetAlgoNameToTypeMap()` | 注册名无法转换回枚举 |

还需人工确认按族分流、属性解析与 executor 注册；只补实际缺项。

**判据来源**：轮转分片规则沿用 [RS + AG 算法定义](https://www.cs.fsu.edu/~xyuan/paper/09jpdc.pdf)，
与本仓 API 的映射仍属实验判据；链式广播清单来自用户报告和该任务的数据流约定。
使用时对照目标版本的源码与 Spec，不把案例当作官方通用接口规范。

## 2. 邻接通信：R01（MAJOR，需确认语法匹配）

核对对端来自实际环序的相邻 rank，channel key 是通信域 userRank。常见写法：
`nextIdx = (myRankIdx_ + 1) % N`、`prevIdx = (myRankIdx_ + N - 1) % N`。
遍历全部 rank 取 channel，或完全看不到邻居推导时，脚本提示 R01；辅助函数、等价表达式
须人工核实，正则缺少特征不能单独证明算法错误。

根据 wrapper 的读/写契约确认需要哪些 channel。**代码同时出现读、写原语可能只是互斥模式分支，
不能据此断言是双向环。** 链式广播的 root 与末端可能各只使用一侧数据边。

## 3. 先分形态，再检查 R02 / R03 / R07

| 形态 | 算子语义 | 判据 |
|---|---|---|
| 轮转分片，单段 | AllGather / ReduceScatter | 每 rank 每段通常 N−1 步；片索引随 step 变化 |
| 轮转分片，双段 | AllReduce 的 RS + AG | 两段各 N−1 步；两段共用分片布局 |
| 链式全量转发 | 本次 Broadcast | 全局 N−1 跳；每 rank 每次分块/repeat 至多一收一发，见 §7 |
| 其他／未知 | 如尚未确认的 Scatter 变体 | 先读 Spec 与实现，不套用上述循环和分片公式 |

脚本优先按类名/文件名推算子语义，避免把 AllReduce 目录中的 Broadcast/AllGather 构件当作
AllReduce。Broadcast 与未知算子保留 X01、R01、W/C、S 检查，R02–R07 不自动应用，输出 R08
说明人工待查范围。**算子名只用于选择清单，不证明形态正确；轮转算子的特殊变体也需确认适用性。**

仅对轮转分片：

- R02（MAJOR）：检查 step 循环及上界与 rankSize 的关系；人工确认每段确为 N−1，脚本不是数值证明器。
- R03（MAJOR）：检查随 step 变化的片索引。常见 RS 发 `(idx-step+N)%N`、收前一片；
  AG 索引整体后移一位。等价推导人工确认，不能强制变量名或表达式文本完全一致。
- R07（INFO）：检查尾片长度、偏移的共同来源；RS/AG 两段不得各自生成不一致的布局。

正反例见 `assets/fixtures/ins_temp_all_reduce_ring_{good,bad}.cc`，它们是不参与编译的规则夹具。

## 4. 数据依赖：R04（轮转分片 MAJOR）

轮转分片中，第 k 步收到的数据用于第 k+1 步时，必须证明先收完再使用。
脚本只检查明显缺少同步特征的情况；出现 Notify 或 Pre/PostSync 名称不等于同步正确。
人工沿 wrapper 核对完成语义、线程与 notify 配对、循环及分支位置。
链式广播也有收后转发依赖，但由 R08 清单核对，不强制每 rank 存在 step 循环。

## 5. 资源模型：R05 / R06（INFO）

资源倍数与线程数量是本仓映射提示，不是纯拓扑能决定的常量。
轮转分片 AllReduce 常见 scratch 为一份、图模式可为零，AllGather/ReduceScatter 的布局可能需要 N 份；
实际大小须由模式、切片和 executor 消费方式证明。R06 提醒检查按 rankSize 申请从流是否必要。

Broadcast/未知形态不套用这些数值预期：逐模式检查 scratch、线程、notify 申请与实际访问，
特别注意 OFFLOAD 与其他 OpMode 的资源条件是否与执行分支一致。

## 6. 评审 checklist

- [ ] X01 登记、executor 接入与显式配置命中证据完整。
- [ ] 按算子语义与 Spec 确认算法形态；记录各 R 规则适用／不适用及依据。
- [ ] 邻居索引与 userRank 映射正确，端点只申请实际需要的资源。
- [ ] 轮转分片核对 R02–R07；链式广播按 §7；未知形态写明未覆盖并人工分析。
- [ ] C01–C16、模式枚举、支持性状态时序及 Spec 同步作用域已核对。
- [ ] 功能证据覆盖相关 rank/count/root 边界；未实测范围单列，不能用静态通过替代。
- [ ] Scatter 链式 relay 的“收下即留”按 R09 核对每段输出落位、回搬分流和在位/错位依据。

## 7. 链式广播人工清单：R08（INFO，必需人工核对）

先由 Spec 确认是全量沿单向环传播；若是分片流水等其他实现，按实际数据流另作证明。
对每次分块/repeat，检查：

1. root 映射到算法环序正确，非零 root 同样成立；root 从本地输入出发，不等待前驱数据。
2. 末端为环上 root 的前驱，不再把数据送回 root；N=1 单独处理。
3. 中间 rank 从前驱接收后向后继转发；读/写两种模式分别沿 wrapper 证明接收完成先于使用。
4. 沿 root 出发的有效数据边遍历：所有 rank 恰好覆盖一次，共 N−1 条边，无断链或闭环等待。
   N−1 是全局跳数，不是每个 rank 的本地循环次数。
5. 收发的长度、缓冲区、offset 和 repeat/尾块对应一致；不要求存在轮转 chunk 索引或 RS/AG 表。
6. scratch、thread、notify 按实际模式验证；声明不支持的路径在有效状态上拦截。

R08 提示本身不是缺陷；评审人完成清单后写具体证据。发现断链、错误终点或提前转发时，
按实际后果提出人工 finding。当前脚本不会自动证明这些控制流性质，退出码 0 不能代替此步骤。

## 8. Scatter 链式 relay 落位：R09（INFO，必需人工核对）

对“收下即留”的本 rank 切片，先从绑定的每个 executor 记录 `outBuffType`、`outputPtr`、
`outBuffBaseOff`、`hcclBuffBaseOff` 和 repeat stride，再沿上一跳收包地址核对目标槽位。
`HCCL_BUFFER` 段可能已经由 `RecvWrite` 直写到目标槽位，此时再做 scratch→out
`LocalCopy` 会形成自拷贝；若源、目标错位，该段仍须搬移。要求实现按落位分流，
在分支旁注释“已在位”或“需错位搬移”的地址依据，不以 `outBuffType == HCCL_BUFFER` 单独决定早退。
`ins_temp_scatter_ring.cc:233–253` 是已在位而跳过的实例。脚本只提示 R09，评审人需核对事实并写出结论。
