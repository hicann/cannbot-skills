# Broadcast 范式 CANNBotDSL 能力映射

> **层级**：CANNBotDSL 语言适配层。对 [../../generic/capability-contract.md](../../generic/capability-contract.md) 中全部能力谓词给出 CANNBotDSL（下文简称 DSL）真值与 API 实体。
>
> **真值状态声明**：本表首期基于已有 DSL API 职责分类与实现资料填写。材料只能证实到"职责类目存在"层面的谓词填类目事实并标注来源；无法证实的标**待验证**。待验证谓词的影响与收敛规则见 §3。

## 1 谓词真值表

| 谓词 | DSL 真值 | 依据 / 验证义务 |
|------|------|------|
| `CAP_BROADCAST_IN_TRANSFER` | **待验证** | DSL 存在数据搬运类目（`ops.memcpy`：DMA、copy engine、分块搬运、padding、`mem_copy`），但搬运是否支持广播轴 stride=0 随路展开未证实。验证义务：查 CANNBotDSL 搬运 API 的 stride/广播参数。若为 false，通用层数据流退化为独立广播步骤（P 值 +1，见 [../../generic/liveness-fusion.md](../../generic/liveness-fusion.md) 冲突处理） |
| `CAP_REG_CHAIN_FUSION` | **待验证** | DSL 存在寄存器算术类目（`ops.reg`：寄存器算术、比较、mask、load/store），链式中间值是否驻留寄存器不落片上 buffer 未证实。验证义务：查 `ops.reg` 运算的数据落点语义 |
| `CAP_FUSED_ARITH` | **待验证** | 寄存器算术类目存在，但"一条接口完成两步运算"的融合指令集合未证实。验证义务：查 `ops.reg` 融合类接口清单 |
| `ALIGN_GRANULARITY` | **待验证** | Buffer/Channel 构造的对齐粒度未证实。验证义务：查 Buffer/Channel 构造 API 的对齐约束。影响：通用层 `perBufBytes` 对齐公式（[../../generic/splitting.md](../../generic/splitting.md)）暂以假设值代入并声明 |
| `TRANSFER_MAX_DIMS` | **待验证** | 分块搬运的维数上限未证实。影响：通用层搬运策略分叉（全维单次 vs 逐段）暂不定型 |
| `FUSION_CHAIN_MAX_LEN` | **待验证** | 依赖 `CAP_REG_CHAIN_FUSION` 的验证结果 |
| `CONVERT_BREAKS_CHAIN` | **待验证** | 类型转换与寄存器链的关系未证实（标量 cast 已有类目，张量级 cast 落点未证实） |
| `CONVERT_INPLACE` | **待验证** | 精度转换可否原地完成未证实。影响：转换中转 buffer 是否需要（P 值推导） |
| `COMPUTE_PRECISION` | **待验证** | DSL 范式统一计算精度策略未证实，不排除按 dtype 原位计算。影响：`perBufElems` 公式的字节宽度取值 |
| `BUFFER_MODEL` | **Buffer / Channel 双原语**：阶段内临时量用 Buffer 显式管理；跨阶段/跨角色生产消费用 Channel（acquire/commit/wait/release 数据流接口） | 已有 DSL Buffer/Channel 用法资料与 API 数据流类目；具体行为仍需核对目标 DSL 能力 |
| `SYNC_PRIMITIVE` | 类目存在：pipe、barrier、busy wait、事件、buffer 同步、跨核同步（`ops.sync`）；Channel 自带 acquire/commit/wait/release 交接语义。**具体调用规则待验证** | 调用方提供的 DSL 同步 API 资料 |
| `EXEC_UNITS` | **待验证** | 架构/平台信息类目（`ops.arch` / `ops.info`：block/core/subblock 索引、内存容量）存在，但搬入/计算/搬出执行单元划分及事件配对规则未证实。影响：通用层 sync-model 的事件对推导暂以 Channel 交接语义保守替代 |

## 2 通用术语 → DSL 实体对照

| 通用层术语 | DSL 实体 |
|------|------|
| 片上 buffer（阶段内临时量） | Buffer（指定 MemLoc / dtype / shape / 生命周期，当前阶段可反复读写） |
| 片上 buffer（生产/消费交接） | Channel（produce/consume 成对，acquire/commit/wait/release） |
| 设备侧计算 | `@kernel` / `@jit` 装饰的 DSL kernel 及其阶段方法 |
| Host 侧调度 | 公开入口 + Host launcher（torch 张量作公开接口） |
| 搬入 / 搬出 | 数据搬运类目（`ops.memcpy`：DMA、copy engine、layout/format transform、分块搬运、`mem_copy`） |
| tile / 切分 | tiler（`tensor-types-and-layout` 类目） |
| 寄存器计算 | `ops.reg` 类目（寄存器算术、比较、mask、load/store、规约） |
| 多核 / 架构参数 | `ops.arch` / `ops.info` 类目（block/core/subblock 索引、内存容量） |
| 编译期特化（RANK 分档） | constexpr / JIT / kernel 装饰器（`language-and-kernel` 类目）；具体特化机制待验证 |

## 3 待验证谓词的使用与收敛规则

1. **产出声明**：设计/代码产出引用任一待验证谓词支撑的通用层公式时，必须显式标注"该结论依赖待验证假设"并写出验证义务
2. **禁止虚假确定**：不得把"职责类目存在"写成"能力已验证"——API 分类只描述代码职责，不证明硬件支持、算法适用性或运行时行为（API 分类的证据边界）
3. **真值收敛**：CANNBotDSL 权威材料（wheel API 提取索引、官方文档、实测结论）就位后，逐项替换待验证真值并删除对应验证义务；某谓词长期填不出真值时，按 [../../generic/capability-contract.md](../../generic/capability-contract.md) §1 规则 3 修订通用层抽象而非绕过

## 4 关键语言级附加约束

1. **设备计算归属**：spec / DESIGN 规定在设备上完成的值计算、类型转换、布局搬运、掩码/尾块处理、最终输出合成，必须落在 CANNBotDSL kernel 内；禁止在公开入口、Host launcher 或其可达辅助函数中用 `torch` / `torch.nn.functional` 小算子（会触发设备算子或复制的操作，如 `torch.add`、`.to(dtype)`、`F.*`、隐式 copy 的视图变换）补全实现。Host 仅限：读取 shape/dtype/device 元数据、分配输出/workspace、建立零拷贝视图、发起 DSL kernel
2. **Channel/Buffer 选择**：数据需要生产/消费同步才用 Channel；仅为当前阶段可反复读写的临时量用 Buffer；produce/consume 必须成对、不越生命周期，临时量没有交接语义时不硬建 Channel
3. **Channel 深度非规范**：depth=1/2/3 是样例选择，按在飞任务数、资源预算与复用时序核定，不固定取值
4. **尾块纪律**：不把无效尾行写入输出；布局表面相同不等于 Channel 兼容（NZ/ND 物理布局需逐项核对）
5. **平台参数获取**：核数、内存容量等经架构/平台信息类目 API 获取，严禁写死
