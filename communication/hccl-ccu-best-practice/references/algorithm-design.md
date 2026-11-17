# 算法设计：数据流先于实现

独立完整方案使用 [设计模板](../assets/design-template.md)。调用方已有DESIGN/PLAN/TEST_PLAN时，每template的数据流证明使用 [专用Spec模板](../assets/dataflow-spec-template.md)，引用已有规格与JSON契约，不重复文件计划或测试矩阵。已有设计只补受影响的决策，不强制重写文档。

## 算子语义和算法选择

先固定输入输出语义、数据总量/每 rank 计数、root、dtype/reduce、in-place、rank/子通信域与物理拓扑。AllGather 的每 rank 输出偏移、ReduceScatter 的最终片归属、AllReduce 的贡献集合分别推导，不能沿用同一公式。

依据问题选择 Ring、Tree、Mesh、分层或自定义调度，写出每一阶段结束时每 rank 持有的数据及证明。CCU_SCHED/CCU_MS 是需要核对的执行模式；样本是否可用要由所需原语、目标 CCU 版本和路径条件证明，不能由算法名称推断。

先确定 template 内可以完成的变化，再判断是否需要新 executor、selector 或 TopoMatch。保持默认路由和新增默认候选是不同任务，分别声明支持矩阵与回归责任。成本模型只在目标选路确实消费它时维护，不虚构带宽/延迟系数。

## 地址和布局

对每条传输/归约写明：

| 字段 | 必须解释的问题 |
|---|---|
| src/dst owner | 哪个 userRank、哪个子通信域索引，如何换算 |
| buffer | 用户输入/输出、CCL buffer、scratch 或片上 buffer |
| base 与偏移 | 基址是否已包含 `inBuffBaseOff/outBuffBaseOff`，避免重复加偏移 |
| slice | 起始字节、长度、尾片、dtype 对齐；`count` 与 bytes 的换算 |
| repeat | `input/outputRepeatStride`、repeat 数与执行器外层 loop 的关系 |
| token | 与实际地址注册范围、所属 rank、生命周期对应的 token 来源 |
| 归约 | 每个输入贡献的次数、目标初始化、跨阶段累加次序 |

证明每次访问在已注册范围内、应有片段完整覆盖、写入冲突受控且没有遗漏/重复归约。对于不整除、最后 rank、零片、单 rank、in-place，按实际算子契约单独推演。

AllGather 样本通过 executor 给的 stride/repeat 处理布局，不能用 `rank * count` 替代。某 rank 本轮无数据也可能仍负责地址交换或完成通知；早退必须证明所有对端不会等待它。

## 资源和同步协议

列出各 kernel 的 die、资源组、channel→peer 映射、XN/Notify 索引、Event mask、片上 buffer、scratch 和并行实例数。区分硬件版本容量、已分配资源与当前可用连续资源；不要把常量样本当所有芯片的容量。

每个同步点记录生产者、消费者、阶段、条件及复用方式：

1. 地址/token 交换完成后才能发起依赖这些地址的传输。
2. 本地搬运完成事件与远端可读/已消费通知分别建模。
3. 等待掩码的每一位都必须在同一实际执行路径有生产者；多个 producer 可组成聚合等待。
4. 对循环/流水线记录 mask 复用和轮次隔离，检查上一轮信号是否可能满足下一轮等待。
5. 将阻塞等待画成跨 rank/kernel 依赖图，确认不存在全部参与者先等待后生产的环。

禁止统一套用“一次前同步、一次后同步”或每 rank 一个线程。跨 die 需要实际 channel 或受支持的同步接口；字符串标签形式的本地 Event 不作为跨 die/rank 通知。

## 启动与重放契约

建立每个 kernel 的独立参数表：槽索引、字段语义/单位、host 填充值、kernel 加载目标、缓存位置、复用时是否变化。

FastLaunch 需覆盖可变地址及其派生值。例如输入输出地址变化会影响 alias/in-place 判定；不仅要刷新指针。依赖固定 count/dtype/topology 的缓存字段须有缓存 key/失效路径的源码依据。未知时查清后实现，不能宣称任意参数重放安全。

脚本 [check_launch_contract.py](../scripts/check_launch_contract.py) 检查声明的槽映射；字节范围、数学正确性、原语语义、协议活性和缓存失效仍须按源码检视与运行验证。
