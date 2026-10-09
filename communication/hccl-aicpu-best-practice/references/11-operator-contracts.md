# 算子接线卡：executor 输入与布局

用于核对已确认算法设计如何接入目标版本 executor，不绑定 Mesh/NHR/Ring。只读目标算子条目；算子整体语义和算法阶段由设计规格提供。
来源：用户提供的 HCCL `88a2a65cdd46a6b16091282fdeac81a391dcfd22`，
`include/hccl.h`、下表 executor 与公共 `template_utils.h`；文件指纹见 [source-baseline.json](source-baseline.json)。
这些是指定版本/指定入口的摘要，不是所有 executor、运行模式和设备的通用布局。

## 1. 所有算子共享的工程契约

- 构造、`KernelRun`、`CalcRes`、`CalcScratchMultiple` 等接口见 [SKILL §3](../SKILL.md#3-接口契约)。
  生成代码用 `barebone`，算法逻辑自行实现。
- `count`、`sliceSize`、stride 的生产者是绑定 executor；用户 API 的 count 不能直接替代 template 的 count。
- 指针、base offset、rank/repeat stride 共同决定地址。规则切片公式和量化例子只维护在
  [五参数真源](09-template-data-params.md)，不为每个算子复制一套公式。
- `IN/OUT` 表示当前阶段的输入/输出；中间阶段可能指向 CCL buffer，Broadcast 还可能原地输出。
  远端地址从相应 channel 的有效远端内存获取，不能把本端指针当远端地址。
- 线程、peer、scratch、同步顺序由新算法推导。是否实现 `GetRes/GetThreadNum` 由 executor 的实际调用点决定。
- 算法身份、注册和显式选路的实现接线见 [executor/selector 契约](executor-selector.md)。

## 2. 定长算子：分别选择对应条目

以下仅记录已核对 executor 的字段赋值，不能代替算子整体语义或算法正确性证明。多级 template 只承担整体语义的一段，不能把全局通信域大小直接替代本层 rankSize。

### 2.1 已核对的单层 executor 布局

文件均在 `src/ops/<op>/algorithm/executor/`，函数为 `OrchestrateLoop`。
记 `D` 为该 executor 的完整 `dataSize_`，`S` 为本轮 `sliceSize`，`b` 为已处理字节数。
下表路径中 `repeatNum=1`、两个 RepeatStride 为 0、`hcclBuffBaseOff=0`。
其余运行模式分支、对称内存和分层组合另行核对。
AllGather Parallel 已单列[四阶段契约](12-allgather-parallel.md)，含 repeat/stride、资源消费和成本透传；
选用该 executor 时加载，不能套用下面的 Sole 行。

| op / 文件 | 当前轮 count / S | 输入/输出类型 | ISS / OSS | in/out base offset |
|---|---|---|---|---|
| all_reduce / `ins_v2_all_reduce_sole_executor.cc` | 本轮完整向量 / count×dtypeSize | INPUT → OUTPUT | 0 / 0 | b / b |
| all_gather / `ins_v2_all_gather_sole_executor.cc` | 本轮单 rank 贡献 / count×dtypeSize | INPUT → OUTPUT | 0 / D | b / b |
| reduce_scatter / `ins_v2_reduce_scatter_sole_executor.cc` | 本轮单 rank 输出片 / count×dtypeSize | INPUT → OUTPUT | D / 0 | b / b |
| broadcast / `ins_v2_broadcast_sole_executor.cc` | currSize/dtypeSize / currSize | INPUT → INPUT，两个指针均为 param.inputPtr | 0 / 0 | currloopOffset / currloopOffset |
| reduce / `reduce_sole_executor.cc` | 本轮归约向量 / count×dtypeSize | INPUT → OUTPUT | 0 / D | b / b |
| scatter / `ins_v2_scatter_sole_executor.cc` | 本轮单 rank 输出片 / count×dtypeSize | INPUT → OUTPUT | D / 0 | b / b |

AllGather 的 OSS、ReduceScatter/Scatter 的 ISS 跨的是完整块 `D`，不能在分块后替换为 `S`。
Reduce 中存在 OSS 不代表非 root 也需要产出结果；字段含义要结合实际消费操作。
Broadcast 不应强行使用 `param.outputPtr`。所有 root 算子都须核对当前层的 root 映射。

### 2.2 AllReduce 分阶段接入

若用 `ins_v2_all_reduce_two_shot_sole_executor.cc`，它绑定两个 template，
并由 `GenBaseTempAlgParams` 设置 ReduceScatter 为 INPUT → HCCL_BUFFER、AllGather 为 HCCL_BUFFER → OUTPUT。
`GenTempAlgParamsReduceScatter` 的 `count=currDataCount`，但
`sliceSize=(currDataCount/rankSize)*dtypeSize`，尾片另由 `tailSize` 描述。
这里不能套用单 template AllReduce 的 `sliceSize=count*dtypeSize`。

用户的 Ring 可以自行选择单 template 多阶段，或复用已有分段 executor；
选择依据是接口和布局是否匹配，不要求采用库上已有的算法结构。
其它 sequence/parallel/OmniPipe 组合超出此单层卡片；只核对实际绑定的分段赋值和资源消费函数。

## 3. 变长算子：不套用单一 rank stride

| 算子 / 来源文件 | 已核对的区别 | 开发要点 |
|---|---|---|
| all_gather_v / `ins_v2_all_gather_v_sole_executor.cc::OrchestrateLoop` | 从 varData 读取各 rank 的 counts/displs；`allRankSliceSize` 为各 rank 本轮字节数，`allRankProcessedDataCount` 为已处理元素数 | 分别跟踪各来源进度及输出位置；不能以本 rank 的空数据提前退出整个协议 |
| reduce_scatter_v / `ins_v2_reduce_scatter_v_sole_executor.cc::OrchestrateLoop` | `allRankDispls` 在此函数中乘 dataTypeSize 转为字节；`allRankSliceSize` 为各目标本轮字节数 | 不要再次给已转换的 displs 乘 dtypeSize；按目标片完成所有贡献归约 |
| all_to_all_v / `ins_v2_all_to_all_v_sole_executor.cc::OrchestrateLoop` | 每轮填写 sendCounts/recvCounts/sdispls/rdispls，并提供 processedDataCount、peerRdispls | counts/displs 保持元素单位，收发地址按各自 dtype 转字节；分别处理每个 peer 的进度和零长度 |

上述文件路径前缀同 §2.1。AllGatherV 的 displs 转换发生在哪一层须跟踪实际生产者/消费者，
不能照搬 ReduceScatterV 的转换位置。AlltoAllV 的本地 rdispls 不能替代对端的 peerRdispls。
AlltoAll 定长与 AlltoAllVC 矩阵入口以 `include/hccl.h` 为起点核对其实际转换路径，
尚未在此卡中固化对应 executor 布局，不自动当作 AlltoAllV 的别名。

## 4. Barrier 与点对点

- Barrier：`src/ops/barrier/algorithm/executor/ins_v2_barrier_sole_executor.cc::Orchestrate`
  用零初始化的 `TemplateDataParams` 调用 KernelRun；其任务是同步，不能套用 `count==0` 成功早退。
- Send/Recv、BatchSendRecv：API 语义和 executor 与集合通信不同；从目标算子目录定向核对匹配 peer、
  tag、各项长度及资源请求。本卡未覆盖其布局，不从 AllReduce 骨架继承归约或全 rank 循环。

## 5. 自定义实现的接线核对

对照已确认 Spec 核对目标 executor/分支的输入输出 buffer、字段赋值、资源消费与注册入口；发现差异时将具体函数和字段返回设计方。卡片解决框架接线问题，不替代新算法的数据流证明。
