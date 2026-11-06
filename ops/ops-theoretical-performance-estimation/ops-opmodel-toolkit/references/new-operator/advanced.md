# 非均匀工作量与分组执行

用于 per-AIV 分摊、攒批、稀疏/空任务和驻留状态；基础编写流程见[模型编写](authoring.md)。

## 每 AIV 工作量

从实际硬件的 `aiv_per_aic` 和算法切分轴推导各 AIV 的有效元素数，不固定为 2。
非整除时保留最重 lane 与总量两个口径；五行分给两个 lane 的完整量仍是五行，
不能用最重三行乘二表示总量。Cube M/K/N 使用其真实聚合形状。

原生聚合模型可按 `vec_split_dim` 分摊；已预切分的 VF 声明 `aiv_presplit: true`。
搬运的 `pipeline.aiv_share_exempt` 仅用于已由模型结构表达分摊的 stage，且名称必须
存在于 phases。物理执行已逐 owner 展开，不再套聚合模型的重复缩放。
完整流量、单 lane 计费量和共享服务总量分别核对，见[性能清单](../framework/performance-manifest.md)。

## 攒批与尾批

将多个生产 task 聚合为一次消费时，先定义 batch_size、实际非空数及输出行数。
典型构建条件为 `(ptid + 1) % batch_size == 0 or is_last_in_group`，但只有算法
确实按组攒批时才能采用；不要把组边界跨过去拼成另一份语义。

| 路径 | 必须保留的行为 |
| --- | --- |
| 非触发 task | 保存实际已产生数据，推进必要 token；不释放尚待消费的 workspace |
| 完整批触发 | 等待该批全部生产者，计算实际工作量，释放实际取得的槽 |
| 尾批触发 | 处理不足 batch_size 的有效数据，不能遗漏或补算无效输出 |
| 全空批 | 保留控制连续性，不伪造计算、输出或 buffer 取得 |

不要把触发周期当作输入固定事实；参数变化后重新计算触发和容量需求。

## 稀疏窗口与 pad-only task

若语义要求只输出落在有效区间内的完整窗口，单段 `[start, start + length)` 的
完整窗口数可独立推导为：

```text
max(0, floor((start + length) / window) - ceil(start / window))
```

该式仅适用于按全局窗口网格取完整窗口的语义。不同 batch/段独立计算，再按语义汇总；
不能先拼接长度后整除。尾 padding、边界 carry 或 partial 输出需另按算法定义。

可在 workload 推导中保存逐 group/task 的有效量，由 `ptid` 与 group 选择。
空 task 不执行不存在的搬运与输出；有攒批时同时保存非空生产者数，以对应池释放。
测试覆盖开始偏移、跨窗口、短段、空段及多 batch，防止只在整齐 shape 上成立。

## 最重核与位置相关成本

分配相同任务数不代表成本相同。mask、有效区间、gather 或复用使成本随逻辑位置变化时，
必须用实际任务 ID 和逐核投影评分。不能固定取任务 0..n-1，也不能只取平均工作量。

共享全核结果取实际 makespan；代表核路径明确选择依据与缺失的跨核争用。
任务数、非空核数、逐组尾块及共享服务量应能从 assignment 复核。
详见[分核模型](../framework/split-model.md)。

## 跨 task 状态

递推累加器、原地转换、控制布尔值分别采用对应契约，不用额外自环或池名模拟全部状态：

- 物理分配上多逻辑版本：[存储、缓冲与版本接口](../framework/ub-storage.md)。
- L1/UB 固定槽、读后覆盖：[固定存储](../framework/ub-storage.md)。
- 分片交接及环形槽：[存储、缓冲与版本接口](../framework/ub-storage.md)。
- 生产一次、全部读者后覆盖：[控制状态](../framework/control-state.md)。

跨组引用和槽位回绕须符合版本声明；提前就绪不等于源可重用。先核对所选后端和搜索器的
[支持范围](../framework/backend-capabilities.md)，无法表达的组合保留明确限制。
