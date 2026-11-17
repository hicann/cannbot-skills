# 按算子测试输入预检

冻结矩阵前用`preview_cases.py --op <算子目录名>`；支持AllGather、AllReduce、ReduceScatter、Broadcast、Reduce、Scatter，只加载本算子来源。只预测已核验hccl_test中`-b S -e S`单点输入的count/bytes和基本设备分配，不运行二进制、不证明算法支持、内核切片或分支覆盖。范围扫描的步进规则需另核对。

```bash
python3 "$SKILL_DIR/scripts/preview_cases.py" --op reduce_scatter \
  --hccl-test-root /path/to/cann/tools/hccl_test \
  --rank-count 8 --dtype int32 --sizes 64 1048576 1048579 \
  --reduce-op sum --max-total-bytes 67108864 --max-aggregate-bytes 1073741824
```

`--rank-count`是实际MPI rank数，不是HVM通信域编号。映射取已安装`hccl-test-tool-hvm/run.sh`的COMM_RANKS或帮助；当前118→8、112→2、114→4。算子目录、DSL和HVM名字按 [索引](operator-index.md) 区分；例如AllReduce的Skill参数是`--op all_reduce`，HVM参数是`-o allreduce`。

## 已核对模型

[来源配置](hccl-test-collectives-profile.json)逐字节绑定公共`common/src/hccl_test_common.{cc,h}`（参数/对齐/分配）和`hccl_opbase_rootinfo_base.cc::init_data_count`（ceil字节转元素），再绑定本算子`opbase_test`文件。任何所选文件CHANGED/MISSING返回非零且不发出猜测预测；其他算子文件变化不使本算子失效。二进制哈希另外记录，`model_verified`不证明来源与二进制对应，须有可信构建清单。

设S为请求字节，N为rank数，w为dtype宽度，C为实际传API的count，B=C×w：

| --op / 专属来源文件 | 对齐与count | API语义 | 工具每rank send/recv 分配 | 全rank基本分配 |
|---|---|---|---|---|
| all_reduce / hccl_allreduce_rootinfo_test.cc | A=S，C=ceil(A/w) | 各rank输入/输出B | B / B | 2NB |
| all_gather / hccl_allgather_rootinfo_test.cc | A=rank规整(S)，C=ceil(ceil(A/w)/N) | 各rank输入B、输出NB | B / NB | N(N+1)B |
| reduce_scatter / hccl_reducescatter_rootinfo_test.cc | 同AllGather规整与两次ceil | 各rank输入NB、输出B | NB / B | N(N+1)B |
| broadcast / hccl_brocast_rootinfo_test.cc（源码拼写如此） | A=S，C=ceil(A/w) | root源B、各rank在同一buf接收B | B / 0，输入输出别名 | NB |
| reduce / hccl_reduce_rootinfo_test.cc | A=S，C=ceil(A/w) | 各rank输入B、只有root结果B | B / B，所有rank均分配 | 2NB |
| scatter / hccl_scatter_rootinfo_test.cc | 同AllGather规整与两次ceil | root输入NB、各rank输出B | NB / B，所有rank均分配 | N(N+1)B |

rank规整：先S=max(S,N)，S≤16MiB按N向下对齐；S>16MiB按N×512向下对齐。只有AllGather/ReduceScatter/Scatter设置need_ranksize_alignment；AllReduce/Broadcast/Reduce不套这个规则。参数非正值被工具拒绝，不能借`--sizes 0`覆盖零数据kernel。极大rank造成规整为0时预检阻止其作为正常输入。

例如N=8/int32、S=1048579：AllGather/ReduceScatter/Scatter规整为1048576，C=32768，与S=1048576等效；AllReduce/Broadcast/Reduce不按rank规整，C=262145，实际B=1048580，两输入不同。S=1：前三种分rank算子C=1、总span=32；另外三种C=1、B=4。Reduce的非root输出无结果语义，但工具确实分配；Scatter非root输入同理，预算不可只算有效数据。

输出的`effective_total_bytes`用于保持单点CLI数据量尺度：分rank三算子为NB，其余为B；不是全rank总内存。`input/output_bytes_per_rank`描述工具数据跨度（Broadcast输出与输入别名）；`allocated_send/recv_bytes_per_rank`描述实际基本分配；`semantic_bytes`明确root/nonroot数据角色。`aggregate_input_output_bytes`才是表中全rank基本设备分配。

`--max-total-bytes`同时限制请求S和实际单点span，`--max-aggregate-bytes`限制全rank基本send/recv总和；未提供预算时相应结果为null。二者均不含host临时副本、Checker/VM放大、scratch、内存分配对齐和常驻开销，完整预算应在既有TEST_PLAN叠加现场测量及余量，不能把基本估算当宿主机峰值。

## root、归约类型与实际执行

Broadcast/Reduce/Scatter用`--root-rank`（默认0，范围[0,N)）；AllReduce/ReduceScatter/Reduce用`--reduce-op sum|min|max|prod`（默认sum）。不适用的参数、非法root拒绝。此处只验证工具语法；当前Sched Mesh归约样本禁PROD及部分dtype，必须按自己的 [接入卡](operator-index.md) 和注册属性筛选；预检成功不代表CCU支持。

当前HVM run.sh固定归约`-o sum`且未暴露root透传，默认root0。`execution_requirements`会给出hccl_test的`-r/-o`参数；非零root或非sum需先核对实际runner能力，使用经验证的定向runner或直接MPI命令并保存同等日志、Checker、库身份及证据。预检不修改共享runner，也不把这些计划自动塞进当前checker-plan.json。HVM参数`-o`是算子名，而底层hccl_test的`-o`是归约类型，勿混用。非默认矩阵单独保留含root/reduceType的case-id和命令元数据；当前HVM原生case key与compare不自动区分这两个维度，不能用默认用例的汇总结果代替其覆盖证据。

API span始终是整数count×w，不能凭非整除CLI字节声称内部非dtype对齐尾片覆盖；例如Broadcast模板会再按字节分片，内部对齐另证。等效输入列出而不自动删除，独立冒烟或复现可保留并在计划写明目的。

实际executor分支已核对maxCountPerLoop时，追加`--max-count-per-loop <count> --loop-basis '文件、方法、分支与推导依据'`才预测loop；不默认256MiB。scratch、对称内存或模式变化要重新计算。`-n 3`只表示重复调用；换地址FastLaunch须地址/缓存/重放证据，Scatter当前样本的缓存原地累加风险见专属卡。

原`preview_allgather_cases.py`及 [旧来源配置](hccl-test-allgather-profile.json)保持兼容，计算委托共用实现。预检结果引用到既有TEST_PLAN并记录来源/工具哈希、真实量、预算和缺口；返回0不是Checker PASS。
