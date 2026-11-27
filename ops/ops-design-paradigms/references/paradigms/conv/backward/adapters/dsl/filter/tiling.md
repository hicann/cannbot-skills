# dW：把 MN_StreamK 计划绑定到 DSL

求解顺序和公式沿用 [generic tiling](../../../generic/filter/tiling/tiling.md)；本页只解释哪些信息进入编译实例、哪些随调用传入。当前普通窗口和拆核窗口各是一类实现，`splitWo` 是运行时宽片长度。

## 1 三个尺度分别保留

| 尺度 | generic 字段 | DSL 中的用途 |
|---|---|---|
| 一次 Cube 计算 | `baseM/baseN/baseK` | 申请 L0 槽容量；实际 `mUse/nUse/kUse` 构造本轮视图 |
| 一次 L1 搬入 | `stepKa/stepKb`、`al1Bound/bl1Bound` | 两路父段覆盖范围和单槽容量；各自独立换驻 |
| 一个输出任务 | `singleCoreM/N/...` | group、kd、M/N 输出范围；任务内部完成全部归约 |

M 是 co；N 的顺序是 `(ci1,kh,kw,ci0)`；K 是输出 H/W。batch×dout 也参与归约，但不与 Load3D 的 `k` 混为一谈：Load3D 的 `m` 对应这里的 K，Load3D 的 `k` 对应这里的 N。

## 2 编译实例与运行时参数

| 信息 | 绑定方式 |
|---|---|
| FP16/BF16、x/dedy 格式、dedw 格式 | 编译期类型/布局分支；两路输入保持同类型 |
| `isSplitKernelHW` | 静态窗口实现分支，例如构造对象的布尔属性并由 `cbd.const_expr` 选择 |
| `groupEnlarge=false、streamkType=0、hf32Flag=0` | 当前范围的固定条件；不是待传入后忽略的参数 |
| 片上容量、物理槽位置、槽数 | 预编译实例的资源上限；Host 选择能够容纳实际计划的实例 |
| batch、通道、空间、kernel、stride/dilation/pad、groups | 运行时标量或动态 Tensor 规格；真实 GM 步幅由其计算 |
| 有效 `base*`、`step*`、父段长、任务数、尾长 | 可在实例容量内运行时变化，构造有效视图并检查不超出静态存储 |
| `usedCoreNum` | 实际 launch 与任务分发共同使用；不能写死 32/36 |

例如声明 `Buffer(L0C, float32, make_layout((MCAP, NCAP)))` 只预留容量。若此次实际 L0C 使用紧凑 `mAligned×nAligned` 排布，显式构造相应 `physical_layout`，再由相同布局派生 MMAD 和 Fixpipe 的视图；若沿用容量的物理 pitch，则输出也保留该 pitch。不能只缩小 shape 后默认二者相同。

不要把 Host 的任意 Python shape 属性捕获成 `self.h/self.w` 编译常量后，声称只凭 `TensorSpec` 中有一个 `Dim` 就完成了动态 shape。更换空间或通道后，应能复用同一已编译模板，且生成指令消费新参数。

## 3 单位、范围与发布

- `cin1G/cout1G` 在既有共享字段中是**通道数**，不是通道分形数；GM 的组起点用实际 CinG/CoutG。
- `al1Bound/bl1Bound` 是单份输入元素数；FP16/BF16 乘 2 后才是字节。`from_tensor(offset=...)` 用字节，`make_layout(..., stride=...)` 用元素。
- Host 的宽整数乘积与设备端地址计算都使用足够位宽。负空间坐标先做有符号裁剪，再构造合法 GM 视图。
- A/B 槽数、L0C 槽数及 bank 起点沿用最终 tiling 计划。`RingBuffer(depth=2)` 申请两槽，不证明它们在目标硬件不同 bank；需要固定地址时按公共 buffer 文档绑定。
- 使用 `task = get_block_idx(); task < totalTasks; task += get_block_num()` 等价的任务步进。需要跨核初始化时，参与同步的核数也来自同一 launch 计划。

Python Host 不需要照抄 Ascend C 的二进制 TilingKey 编码；须保留 key 所代表的语义。保存“实例身份、容量、实际字段、launch 核数”即可追溯分派。若通过其他工程封装注册，则由该工程定义 ABI，不能把任意 Python 对象当作设备可读的 TilingData。

## 4 交付前复算

检查全部实际槽的地址区间、L0A/B/C 峰值及两路 L1 父段峰值，而不只检查有效 tensor 元素数。保留 generic 的普通整窗判定 `(baseN/16) % (hk*wk) == 0`，尾块仍走同一输出所有权。

若预算依赖整窗条件，按编译实例容量裁剪 `baseN` 后必须重新核验；非整窗块可能跨入额外的通道分形，不能继续沿用整窗的父段通道跨度。行带预算也要覆盖首段、内部段和尾段，不能只检查被 padding 裁短的首段。把最终选块生成的实际 N 块与 H 段代回搬运视图，确认每个父段都落在单槽内。

至少复算一个非整窗 N 块和一个切宽尾块：核位置连续性、通道尾数、A/B 父段覆盖、首个实际 MMAD、段间写回范围。修改资源份数或有效块后，连同 [kernel](kernel-template.md) 与 [同步](sync-implementation.md) 复核，不在 adapter 中另造选块策略。
