# AICPU 通信算法设计

本文只定义算法语义、阶段和需要证明的决策，不提供可直接复制的 C++ 实现。目标版本的算子 API、绑定 executor 和实际拓扑始终优先于历史示例；不能凭算法名称推断资源、性能或支持范围。

## 从算子语义开始

| 算子 | 整体输出语义 | 设计时必须证明 |
|---|---|---|
| AllReduce | 每个 rank 获得所有输入的逐元素归约 | 每份贡献恰好参与一次归约，完整结果到达所有 rank |
| AllGather | 每个 rank 按 rank 顺序获得所有输入片 | 来源 rank 与输出槽位一一对应，不误做归约 |
| ReduceScatter | 逐元素归约后，每个 rank 保留自己的目标片 | 归约完整性、片归属和输出落位 |
| Broadcast | root 的输入传播到所有 rank | root 映射、转发前的数据可见性与原地输出 |
| Reduce | root 获得全部输入的归约结果 | 非 root 仍完成协议，但不假设其输出有效 |
| Scatter | root 输入分片后各 rank 获得自己的片 | 源片号、目标 rank 和落位一致 |

变长算子须分别跟踪各 rank/peer 的 count、displacement 和进度；不能把定长算法的单一 rank stride 直接外推。Barrier 与点对点协议也不能套用集合通信的空数据早退或全 rank 循环。多级算法中，一个 template 可能只承担整体语义的一段。

## 选择形态，不复制实现

- Mesh one-shot：可用一次全连接交换后本地归约来实现 AllReduce。设计时逐一证明目标 rank 的 scratch 槽位、通信并发、同步以及资源容量；不能把某份历史实现的线程数或 scratch 倍数当通用常量。
- Mesh two-shot：可拆成 ReduceScatter 与 AllGather。明确两阶段的片归属、交接 buffer、阶段间可见性及每个 rank 最终获得的完整结果。
- NHR：按目标版本实际步数和 peer 生成逻辑推导每步的数据归属与归约顺序；NHR 的步数不能直接套用 Ring 公式。
- Gather/Scatter 类：分别证明 root、非 root 的输入输出语义，以及 repeat/stride 和尾片地址；不因两者名称相近而复用同一落位公式。
- Ring、Tree 或用户自定义调度：无需先找到同名模板。先确定阶段、逻辑边、每轮持有的数据、最终输出和可推进性，再对照可用接口落地。

这些形态是可选设计参考，不是性能排序。性能结论必须来自目标拓扑与数据量上的测量。

## 自定义调度的设计检查

1. 继承已确认的算子、通信域、拓扑、运行模式、dtype/reduceOp 和默认选路范围；缺失会改变语义的输入作为待确认项，不猜测。
2. 为每个阶段列出输入片、输出片、逻辑 peer、收发方向、归约位置及状态转移。Ring 的前驱/后继先由本层逻辑 rank 推导，再映射到通信域 userRank；双 rank 时注意边重合。
3. 区分 executor 分块、template repeat 和通信 step。独立推导每轮地址、非整除尾片、scratch 生命周期和覆盖关系；不把本轮长度当作完整块 stride。
4. 从并发与依赖推导 channel、thread、notify 和同步点。每个等待必须有可达的发送方；证明读前写完成，scratch 不会被提前复用。
5. 确定算法身份、注册/选路方式、支持矩阵以及默认选择的影响。只显式启用与调整默认选择是不同决策；未标定成本不能当作性能证据。
6. 用[数据流记法](07-dataflow-spec.md)和[规格模板](../dataflow-spec-template.md)记录每个 template 的阶段与地址，运行 `scripts/check_spec.py`、`scripts/check_layout.py`。静态检查不证明死锁自由或功能通过。

一个 Spec 对应一个 template；多阶段、多层组合在 Spec 中标明所属层级及兄弟 template。实现时必须依据目标源码核对 executor 提供的 count、buffer 类型、base offset、repeat/stride、资源消费和注册入口。设计完成后，具体 C++ 接口、骨架、编译与 Checker 交由相应实现和验证能力处理。
