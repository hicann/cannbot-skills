# Convolution DSL 性能调优经验

> **层级**：CANNBotDSL 语言适配层专有文档（[paradigm-dev-rules.md](../../../../paradigm-dev-rules.md) §4.3）。实现结构见 [implementation-guide.md](implementation-guide.md)，能力真值与缺口见 [capability-mapping.md](capability-mapping.md)。
>
> 本文只沉淀**可复用的调参经验**：怎么测、什么有效、什么有坑、什么时候到达结构上限、代价模型怎么校准。
> 原始 msprof 数据、对比表、采集/调参脚本与固定环境快照留在**库外验证报告**，不入库（与 `sortselect/adapters/dsl` 的约定一致）。

## 1 测量口径（必须先固定）

- 采集：`msprof --task-time=l1 --ai-core=on --ascendcl=off --runtime-api=off`，取 `op_summary` 中 `Task Type ∈ {AI_CORE, MIX_AIC}` 的 kernel 时长。
- 采样：每例 3 warmup + 20 测量，报告 **p50** 与 **min**；共享 NPU 上 median 会漂移，关键结论至少复测一次（同一配置的中位数可能因共享设备负载漂移，需复测确认）。
- 归属：按 launch 顺序切 task，逐例用 `Input Shapes` 核对规格；要求每例每调用 1 个 kernel task、Block Num 与发射核数一致。
- 口径边界：**只比 kernel 时间**，不含 Host 分发、tiling 决策、编译与内存分配；DSL 公开调用的 Host 开销是独立话题（数百 µs 量级，端到端对比必须单列）。
- 对比基线：native 用 `torch.nn.functional.conv*`（生产实现参照）；AscendC 用**已调优**的可运行候选（不是范式示例），所以 DSL 的差距是保守上界。注意 native 的 conv1d 会被升维成 4D conv、个别形状走 MIX_AIC，口径需在报告里标注。

## 2 有效的结构手段（按收益量级排序）

| 手段 | 做法 | 适用前提 | 收益量级（抽样经验） |
|------|------|---------|--------------|
| 整行带 + 单条多行 DMA + 硬件 pad | 行带按输入整行宽 `W` 存放；`mem_copy` 一次搬多行；左右 pad 交给 `Load3DFeatureMap(pad_left,pad_right)` | `W ≤ 32767`，窗口不越过 `W+pad_left+pad_right` | 小/中 kernel 1.1~1.6×；小空间深通道（Q 类）约 2× |
| M 超块（`nsubs`） | 同一 feature/weight 分块复用给多个 M 子块，每子块独立 L0C 累加器 | `4·MT·NT·nsubs ≤ L0C`、`nsubs ≤ 8` | 任务数下降 2~4×，C/M 类 20~60% |
| 多行输出展平（`mrows`） | 一个任务覆盖 `mrows` 个输出行，A 侧行重叠从 `KH×` 降到 `(PH+(mrows−1)·Sh)/mrows` | `Ow == W`、`mrows ≤ Oh` | 行重叠敏感的大空间 1.2~1.5×；Q 0.23×→0.97×（与 CT 合并计） |
| L1/L0 双缓冲 | `RingBuffer(depth=2)`，**不手写预取** | `2·MT·K ≤ L0A`、`2·NT·K ≤ L0B` | 归约步 ≥4 时 1.1~1.2×；steps=1 无收益 |
| 定向零填 | 只在上下/深度越界时清行带；权重分形尾部只在启动时清一次 | — | 1~1.2×（大行带/多步用例更明显） |
| 扩大 CT（大 Ci） | 减少归约步数 | `CT·KH·KW ≤ 4095`，且给双缓冲留 L0 空间 | Q 类 1.6×；但会挤压 MT/NT，需联合搜索 |

## 3 无效或有害的尝试（不要重复踩坑）

1. **手写预取 + `range_constexpr` 展开 + depth=2**：`steps≥8` 时编译爆炸（≥300 s 不返回）；`steps=4` 虽可编译，但运行慢 1e3~1e4 倍。正确做法是只声明 `RingBuffer(depth=2)`，让编译器按 Buffer 依赖排布重叠。
2. **整行带下继续传 `m_extent=ceil16(row_count)`**：尾块窗口越过 `W+pad_right`，运行时 AICore error；应传 `m_extent=row_count` 并保证窗口在界内（允许非 16 对齐）。
3. **用 L0C 切片或 `from_tensor(offset=...)` 作 matmul 目标**：`MatMulOp` 创建失败；每个 M 子块必须声明独立 `Buffer`。
4. **子块循环里重复 `next_read()` 读同一个 B 分块**：`depth=2` 时读写游标错位，表现为整块数值偏差（不是尾块）；B 每个归约步只读一次。
5. **非驻留权重 + 深归约 + 多 channel tile**：权重流量被 `tasks·steps·NT·K` 放大，通常显著劣于驻留；代价模型已含惩罚，不要只按 A 侧流量选切块。
6. **只看总字节不看每核执行量**：任务数少于核数时（小输出），选“任务更少但每核更慢”的切块会亏；需要按 `sqrt(cores/tasks)` 之类的空转核惩罚。
7. **流水折扣给所有 depth=2 配置**：`steps=1`（点卷积）没有可重叠窗口，给折扣会误选更差的 tile；折扣必须随 `steps` 生效。

## 4 结构上限判断（什么时候该停）

- **大空间 + 窄 N**（如 `Co=64`、`512×512`）：A 侧每字节约换 `N` 次乘加，L1→L0A 流量 `≈ M·K·2` 与切块无关，受 L0A 带宽约束；进一步只能靠「整平面 A 驻留」（每输入行只读一次）或换 GEMM 分解（A=权重、B=im2col）。
- **小 Ci + 大核**（如 `Ci=3`、`k=7`）：K 方向被 `C0=16` 与 `rs→16` 填充放大（有效利用率可低至 ~15%），MT 又被 `MT·K ≤ L0A` 限制；这是分形打包的固有损失，优先确认是否是算法选择问题（如先用 1×1 降维）。
- **宽 `Ow` + 小 `MT` 受限**：`mrows`/`nsubs` 能缓解，但受 `L0C = 4·MT·NT·nsubs` 上限；进一步要减少 N（`NT`）以换取更大 M 覆盖，需按目标形状实测权衡。

## 5 代价模型系数怎么校准

1. 固定一套 20+ 形状（覆盖 1D/2D/3D、尾块、stride/dilation、大 Ci、大空间），对少量代表形状**实测每个候选切块**的 kernel 时间；
2. 按“字节项 → 固定开销项 → 尾块浪费项 → 负载/空转项 → 流水折扣”的顺序拟合系数，每步只调一层，避免互相掩盖；
3. 每次结构改动（整行带、超块、行展平、双缓冲）后必须重新校准，旧系数会系统性偏向旧结构；
4. 系数只用于**候选间排序**，禁止当绝对耗时预测；文档只保留相对量级，原始数据留在库外；
5. 校准结果必须用**未参与拟合**的形状抽查，确认没有明显反转。

## 6 交接清单（把调参交给下一个人时）

- 测量口径、环境版本、SoC、采样次数与归属校验方式；
- 当前生效的结构开关（depth/`nsubs`/`mrows`/`resident`）与它们在哪些形状上被选中；
- 已知的“无效尝试”清单（本文 §3），避免重复投入；
- 结构上限的判断依据与下一步候选（如 fullLoad、整平面 A 驻留）；
- 原始数据位置（库外报告）与复现命令。
