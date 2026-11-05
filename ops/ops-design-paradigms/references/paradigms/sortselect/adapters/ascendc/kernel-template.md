# SortSelect AscendC Kernel 骨架

公共计算流见 [compute-flow](../../generic/compute-flow.md)。这里落地 API 与阶段对应关系。

设计、编码与验收均须遵守 [实现复用与替换约束](design-spec.md#实现复用与替换约束)，记录逐路线排序主体、前后处理并行方式及同步协议。输出区间、布局适配和索引宽化不自动授权重写排序主体；构键和恢复按 [辅助处理并行度](kernel-helpers.md#辅助处理并行度) 复用，保留 Sort32/MrgSort 但改成逐行单线程前后处理仍属于实现差异。

## 按模板读取

| 已选算法 | 计算文档 | 本文档直接说明的内容 |
|---|---|---|
| 插入 | [Kernel](templates/insertion/kernel-template.md) | 独立行 pitch、严格比较、值/索引同步移动、合批写回 |
| 两阶段 | [Kernel](templates/two-stage/kernel-template.md) | Stage1 原位置、inverse/packed 推导、Stage2 方向、恢复与别名 |
| 归并 | [Kernel](templates/merge/kernel-template.md) | Sort32 run、四路尾组、proposal 偏移、合批与逐行模式 |
| 单核 Radix | [Kernel](templates/radix-one-core/kernel-template.md) | 隐式索引重载、halfIndex、零符号恢复及 Queue 生命周期 |
| 多核 Radix | [Kernel](templates/radix-more-core/kernel-template.md) | 行/核组映射、统计/排列/回溯/散射、状态发布与跨批复用 |

各模板中的伪码完整描述数据通路、执行机制与生成时的适配位置；其中阶段名是计算角色，不是新的 SDK API，也不要求生成同名函数。生成代码可改变组织方式，但批处理粒度、Vector/SIMT 分工、DMA/原子操作的用途和生产消费关系属于实现内容。具体 API 签名按目标 SDK 核对，机制替换按 `design-spec.md` 的依据与代价检查执行。只展开命中模板；公共比较/类型/VF 辅助约束仍在 [kernel-helpers.md](kernel-helpers.md)。

## 实现落点

- 插入：value/index 行 pitch 独立对齐；BF16 比较转 fp32，转换 buffer 计入预算。数据搬入完成后再执行 SIMT，移值时索引一起移动。
- 两阶段：先复现 Stage1 与两条恢复路线。uint16 packed key 校验发生在 Host；输入 buffer 在 Stage1 最后读后别名为最终 values，final index 可暂存零标志/Stage2 keys，生命周期不重叠才能复用。Stage2 紧凑键采用 uint16，按类型与 Host 容量校验核对。
  - Stage1 必须满足当前稳定性契约；`Sort<T, false, config>` 的第二模板参数不可猜成 stable 开关，核对目标安装版本签名。恢复段归属不会修复 Stage1 的不稳定性。
  - 最终写回保留整批 B×L 工作域，排序和必要恢复仍处理完整 N。核对 `StoreBatch` 及其 VF 调用者，不将原本合批的搬出默认改成标量逐行调用 VF、小块 DMA 和逐行等待；声明多线程或保持同一 TilingKey 不能证明粒度未变。批量写回及粒度调整依据见 [批次写回粒度](output-mapping.md#保持批次写回粒度)。
- proposal 归并：使用 proposal offset，不以普通 tensor 元素偏移替代 `GetSortOffset`。四路尾组传实际 counts 与 validBit；每层更新 selector，抽取最终有效 N。升序转换、NaN 和稳定 tie 行为检查对应 wrapper。该布局要求属于 proposal API。独立 SIMT 索引归并须先满足 [按证据限定替换范围](design-spec.md#按证据限定替换范围) 中的要求，再按自身数组布局核算；此布局例外不能用来跳过原语适用性验证或全面替换排序实现。
- radix one-core：使用 implicit-index Sort overload；sourceOrder 在有零符号保存需求时用于按输出索引恢复。查询 scratch 后才决定是否容纳双缓冲。
- radix more-core：基类和 helper 共用 CounterT 模板，按 N 选择 uint32/int64 计数与前缀。32 位全局桶偏移使用 DMA 原子加，64 位使用 SIMT 原子加；两者共用 lookback 状态机。中间原轴索引宽度不超过输出宽度，末轮转换为 T2；Host/Kernel 与 workspace 的 4/8 字节布局必须一致。digit 是 8 bits，256 桶，轮数按 value 字节宽，不因计数宽度改变。首轮建立原索引，以后跟随前轮索引；hist、prefix/status、scatter 与每轮 `SyncAll` 一起核对。各核都须走过合法栅栏路径。

## 统一输出适配

按 [公共输出映射](../../generic/output-mapping.md) 配置 `first_rank` 和 L。插入、两阶段、单核 Radix 在最终 UB 恢复/写回应用区间；两阶段先恢复行归属。归并在最终 proposal 抽取时应用区间；多核 Radix 在末轮全局名次确定后应用区间。所有路线的中间排序与同步仍按 N 保留。

API 层按 [布局判定](../../generic/shape-preprocessing.md#物理转置的判定与验收) 直接使用、reshape 或物理转置，主体只处理连续尾轴行并输出长度 L。实际写回表达、资源查询及批次复用见 [AscendC 输出适配](output-mapping.md)，不为各算子复制排序模板。

## 示例代码组织

[Sort 示例](example/sort/README.md) 的文件职责如下：

- `op_kernel/sort.cpp`：框架 Kernel 入口与 schedule 模板分派。
- `op_kernel/arch35/sort_tiling_data.h`、`sort_tiling_key.h`：Host/Kernel 参数契约与模板键。
- `op_kernel/arch35/` 及 `common/`：短轴、两阶段、常规核内归并与单核/多核 Radix。
- `op_host/arch35/`：选路、资源查询、TilingData 和 workspace 预算。
