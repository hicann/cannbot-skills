# 搬运声明、工作量与结果

编写 `move` 时分别确定逻辑形状、真实读写量、物理布局容量、依赖和释放边界。
成本与机制来自本次选定的[硬件配置](../hardware.md)，不能由一条 API 名称推断
实际指令数或设备服务能力。基础 DAG 字段见[DAG 参考](dag-reference.md)。

## 三种数量与观察范围

| 项目 | 核对方式 |
| --- | --- |
| 逻辑输入／输出 | 根据 shape、dtype、窗口、padding 和转换语义独立计算 |
| 实际源读取／目的写入 | 根据显式读取策略计算；重复读取与补零分别记录 |
| 物理容量 | 按布局、stride、NZ 尾部空位和实际槽大小计算，不能以逻辑字节替代 |
| 命令数 | 按实际 move 或调用声明计数；内部服务单元不增加命令数 |

支持的搬运报告以 `accounting` 分列命令数、服务单元数、逻辑源量、实际源读量、
目的量及观察区间。逐资源的工作、速率、服务、延迟和等待用于诊断，资源区间可能
重叠；完整耗时取对应命令的 `accept → end`。`queue_wait` 与它已包含的
`channel_wait/initiation_wait` 不能重复相加。DAG 接收前的依赖／存储等待另行核对。
不以完整指令跨度反算一个资源的带宽；相关分析见[空泡分析](../dag-bubble-opt.md)。

## 有限布局与窗口变换

`move.transform` 描述有限变换，`move.size` 为目的二维逻辑形状，`source_shape`
为源形状。源目的 dtype 相同，`element_bytes` 与实际 dtype 一致；不能同时使用
覆盖字节量的 `move.transfer`。下列路径需要实际配置支持，未接入时明确失败。

| `kind` | 输入输出语义 | 当前可用路径 |
| --- | --- | --- |
| `dn2nz` | 同一矩阵从 ND 到显式 block 的 NZ | 显式 GM 读取、L1→L0 或公共本地服务 |
| `load3d` | channels-last 二维／三维卷积窗口，越界补零 | L1→L0 或公共本地服务 |
| `nz2dn` | 同一矩阵从 NZ 到 ND | NoQuant L0C→GM FIX 或公共本地服务 |

端点使用 `source_layout/destination_layout`，缺省 ND。NZ 指定
`source_block/destination_block`，不同时使用 strides；ND 可用两个正的字节步距
`source_strides/destination_strides`，须对齐元素且不能造成别名。
这些声明不证明某芯片支持任意形状、布局或模式，也不提供任意 scatter/gather。

下面是放入完整模型的窗口展开片段：输入为 2×2 单通道矩阵，右侧和下侧补零。

```yaml
move:
  id: expand
  src: {tensor: x, mem: L1}
  dst: {tensor: a, mem: L0A}
  size: [4, 4]
  transform:
    kind: load3d
    source_shape: [4, 1]
    destination_shape: [4, 4]
    element_bytes: 2
    input_spatial: [2, 2]
    kernel: [2, 2]
    stride: [1, 1]
    dilation: [1, 1]
    pad_before: [0, 0]
    pad_after: [1, 1]
    read_policy: full_source_once
```

`source_shape` 第一维为输入空间各维之积，第二维为通道数。stride/dilation 为正，
padding 非负；`row_offset/column_offset` 可选择合法输出子矩阵，尾块不得越界。
地址展开有元素预算，数值回放另有预算；大输入可预测不意味着数值已验收。

| `read_policy` | 真实读取含义 | 上例 BF16 读取／输出 |
| --- | --- | --- |
| `full_source_once`（默认） | 整个声明输入 tile 读取一次 | 8 B / 32 B |
| `per_reference` | 非补零引用逐项读取，重复引用重复计量 | 18 B / 32 B |

每个空间轴产生 `(2 + 0 + 1 - 2) / 1 + 1 = 2` 个窗口位置，共 4 个窗口。
这里输出是窗口展开矩阵，每个窗口保留 4 个元素（含补零），因此为 `[4, 4]`、
`4 × 4 × 2 = 32 B`，不是卷积归约后的输出。完整源为 `2 × 2 × 2 = 8 B`；
四个窗口的有效引用数分别为 4、2、2、1，逐引用读取为 `(4 + 2 + 2 + 1) × 2 = 18 B`。

两种策略都是条件模型，不能当作真实 Load3D 缓存行为。前者要求源粒度及对应容量
覆盖完整源 tile；后者包含纯 padding 单元时，所经阶段须显式按目的载荷
`buffer_basis: destination` 预留有限空间。零源读取不填成正字节。

报告的 `logical_input_bytes/source_read_bytes/output_bytes` 分别说明逻辑输入、
实际读取与逻辑输出；`source_capacity_bytes/destination_capacity_bytes` 表示
地址跨度或完整 NZ 分块容量。例如 3×3 BF16、NZ block 2×2 的逻辑量为 18 B，
物理容量为 32 B。固定分配及每次写入都须能容纳实际跨度。

## 跨步 GM 读取

支持显式动态 GM→L1／UB 的读取声明；字段放在 `move` 内：

```yaml
gm_strides: {rows: 3, row_bytes: 20, stride_bytes: 256}
gm_read_address: 120
```

有效量为 `rows * row_bytes`，与源有效量一致；行不重叠，行数受接口预算限制。
空洞不是有效流量。普通路径目的 payload 连续；DN2NZ 可由 `source_strides`
描述源行几何，同时提供 `gm_strides` 时两者须一致。数值回放需要真实源地址，
不能用“假定对齐”代替地址绑定。此入口不推断任意 DMA 描述符或设备缓存效率。

## 源复用、目的就绪与容量

支持的本地 `transfer_service` 可配置 `source_release: source_service_done`；
首阶段须为 `basis: source`，且其来源证明代表完整源读取。默认 `command_done`
保持整命令完成边界。源的全部真实读者完成读取后才能复用，目的消费者仍等目的
完整就绪；目的槽覆写仍须等旧消费者结束。阶段名字、信用归还或更大槽数不能替代这些约束。

循环 pool 的每次写入按 NZ／stride 的实际跨度检查。显式 `physical_bytes` 是
每个物理实例的最终容量，UB 每 AIV 不再重复除以 AIV 数；还须覆盖原逻辑 `bytes`
需求。逐组核对各自写入与容量；槽复用不把循环次数乘入固定分配。零次循环和未执行
分支没有写入需求，已声明固定分配仍占容量。静态 `needs_replay` 须通过真实输入复核，
不能当作通过；超容候选保留失败，不删除容量约束继续评分。

## 原子与数值边界

NZ2DN 可与完整连续 GM 区域的 NoQuant 同 dtype 原子 add 组合，需要显式
`GM_ATOMIC` 服务和合法初始化／贡献声明；不同时添加 `cycles` 覆盖。
转换加原子、任意部分重叠区域、逐块提前可见和逐块数值原子顺序仍不支持。
声明成本不等于实机原子争用校准。

数值回放由用户提供真实读写视图。普通变换使用 `operation: transfer_transform`，
`attrs.indices` 对应输出到输入的映射，补零用 null；逐项核对 dtype、layout、
block、偏移和地址。原子使用适用的 `atomic_add`。配置及周期一致不替代
[执行数值检查](numeric-execution-validation.md)或设备验证。
