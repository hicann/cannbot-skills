---
type: operator
title: block_sparse_attention Arch22 执行与完整调用测量
description: 核对真实装载、异步缓冲和设备任务覆盖，保证正确性与性能结果绑定当前实现。
tags:
- catlass-cpp
- sparse-attention
- block_sparse_attention
- BlockSparseAttention
- 块稀疏注意力
- arch22
operator_families:
- sparse-attention
architectures:
- DAV_2201
status: stable
generated: {"by": "process:sparse-attention-knowledge-port"}
verified: [{"by": "process:cross-source-check"}]
sources: [{"id": "sparse-attention-execution-and-measurement-source", "resource": "audit:custody/sparse-attention#execution-and-measurement", "title": "execution-and-measurement 固定来源（提取侧独立保管）", "kind": "audit-record"}]
---

# 接口与概念

## 算子算法

测量对象是满足同一输入输出约定的完整 block_sparse_attention 调用。
每次内部必需的 metadata 搬运、mask 压缩、主计算和后处理都按所选口径纳入，
不能缓存本次必需工作以制造收益。

## 分核策略与基本块切分

按实际调用、stream、输入和实现版本识别任务；不固定 kernel 数量或物理卡号。
融合、分核或 metadata 路径变化后，任务与 copy 的预期也必须重新验证。
任务表中的某个类型不永久对应某个计算阶段。

## 数据路径与存储层级

host 源缓冲活到异步读取完成，device 目的缓冲活到最后消费者完成。
使用缓存 allocator 时，按实际运行库登记 stream 生命周期。
函数返回、局部对象析构或一次 enqueue 都不表示设备已完成读取。

## 流水排布、同步关系与数值精度

运行身份包括入口、adapter、kernel 及依赖解析结果，不只是磁盘文件哈希。
同名动态库、RUNPATH 或已加载的同 SONAME 库可能使比较执行了错误版本，
此时即使输出正确，也不能形成该候选的性能结论。

# 用法

## 构建与实载版本

检查实际 device 编译命令、依赖、架构选项和日志，不以某一种 CMake 拼写作为唯一依据。
资源或事件协议改变后，同步核对设计中的总预算、容量表、类型及生命周期说明。
持久事件 ID 占用与当前在途通知数分开记录。

静态检查依赖路径后，在实际运行进程中核对加载映射（例如 `/proc/<pid>/maps`）及其哈希。
无法排除动态库缓存时，可用独立进程隔离被比较版本；每侧入口、依赖和运行前后版本都要一致。
复制构建目录或更改 Python 模块名不足以证明 kernel 已切换。

## 完整调用计时

先明确采用 host 调用墙钟、设备事件区间，还是设备任务时长之和；这些指标不能混加。
若采用设备任务求和，按以下步骤核验：

1. 区分测试准备阶段的输入传输与算子每次调用内部必需的 copy、预处理、计算、后处理及事件。
2. 同时查看设备任务表与 host API trace；仅 kernel 列表可能漏掉 copy。观察到 memcpy API
   而缺少对应设备时长时，标记计时覆盖未验证，不能填零或用 host API 时长代替。
3. 按 stream 与调用关联识别外部计时标记；不能把所有 EVENT_RECORD 两两配对或全部排除，
   allocator 的内部事件可能具有同样类型。
4. 逐调用核对任务类型、数量、归属和完整时长；未归属任务、缺失 copy 或边界错配先修复。
5. 用已知 copy+计算的正例及故意漏 copy 的负例验证解析器，再验证真实候选。

搬运预期取决于实现版本。消除搬运后，按实际数据路径更新预期并逐调用核对；
记录实际发生的传输，不以固定数量代替任务覆盖检查。

# 代码模式

```text
actual entry + loaded dependencies + build configuration
    -> input / reference / comparator identities
    -> version-specific task and copy expectations
    -> per-call coverage check
    -> same-device paired measurement and uncertainty
```

复用测量配置时，更新入口与依赖的路径和版本标识，并在采样前验证真实加载结果。

# 约束

单次调用后的立即同步不能证明异步生命周期安全：排队不同输入，延后同步后逐一验证
O/LSE；支持多 stream 时再覆盖交错执行。排队测试与真并发压力测试的结论范围分开。

执行或 profiling 失败时，用最小设备探针区分环境与候选问题。记录未执行、采集未完成或
数值失败的真实状态。变更设备或会话后保存差异并重采完整比较集，不拼接不同设备的片段。

# 失败表现

错误动态库被加载、host 缓冲过早释放、内部事件被当成测量标记过滤、copy 未进入设备计时，
都可能形成看似合理的错误结论。只有源码相同或 kernel 哈希相同，也不能证明整个调用相同。

# 验证方法

先用同一产物的 A/A 对照检查版本、输入、O/LSE 和计时链，再执行真实配对。
保持设备、输入、预热和采样规则一致，保存正反顺序、逐例数据、离散度与退化样本。
构建成功、精度通过、计时覆盖完整、性能达标分别报告；前几项均不替代最终性能判断。

[^sparse-attention-execution-and-measurement-source]: 审计编号 sparse-attention-execution-and-measurement-source；固定来源与版本记录由提取侧独立保管，不随生成侧资料分发。
