# DSL 多核 Radix Buffer

## 1 每核 UB

T 是 tile 长度，D 是输入字节数，H 是 CounterT 字节数（4/8）。

| 区域 | 字节数 | 生命周期 |
|---|---:|---|
| digits | T | 当前字节构键至本 tile 散射结束 |
| permutation | 4T | 多核中保存 `局部原位置 + (桶号<<16)`，局部排序至散射结束 |
| byte_order | T | 字节内四轮 2-bit 稳定排列 |
| local offsets | 512 | uint16 桶起点，统计/排列分别生成 |
| scatter bases | 256H | lookback 完成至本 tile 散射结束 |
| raw_tile | DT | 各轮原值 DMA 输入，保留至 Scatter 完成 |

另有 `index_tile`，占 HT 字节，后续轮 DMA 加载原轴索引并存活至散射结束。静态声明为 `T×(6+D+H)+512+256H`；Host 使用 `T×(6+D+H)+4096` 覆盖固定区及对齐余量。SIMT 保留 32768 字节另扣。低层 strided 变体使用相同静态分配。

## 2 GM workspace

P=P_act，K=ceil(N/T)。计数按单行 N<2^30 选择 H=4，否则 H=8；地址和最终索引始终 int64。

| 区域 | 字节数 | 用途 |
|---|---:|---|
| ping/pong values | 2PND | 与输入同 dtype，跨轮保存原值位模式 |
| ping/pong ids | 2PNH | 原轴索引随原值传递 |
| hist | PK×256H | 本轮频数，统计后供所有 lookback 回退读取 |
| states | PD(K+1)×256H | 每 digit 的 K 份前缀状态和一份全局桶起点 |

当前公开包装总量为：

```text
P×N×(2D+2H) + P×256H×[K+D(K+1)]
```

六个工作区依次为 ping values、ping ids、pong values、pong ids、hist、states。hist、states 的 TensorSpec 使用独立动态维度，原值 TensorSpec 与输入同 dtype。两侧原值/id 按 P 行完整分配，输出、布局区及运行时内部占用另计。

## 3 状态寻址与复用

本批有效行数 v≤P，状态元素基址为 `(byte×v+row)×(K+1)×256`；tile t 的桶 d 使用 `base+t×256+d`，全局桶起点使用 `base+K×256+d`。尾批按 v 建立紧凑视图，其范围不超过按 P 分配的容量；初始化、生产和消费必须采用相同 v。

每批先清零全部 D 轮状态与全局桶起点，再全核汇合。hist 每轮全部有效 tile/256 桶完整覆盖，无需 Host 清零。前缀状态为 0 或 `−inclusive−1`；不能在消费结束前复用状态。非末轮 Scatter 后全核屏障保护 hist 复用与 ping/pong 交换；批间屏障保护下一批清零。

UB permutation 从局部排序存活至本 tile Scatter，不落 GM；不同 tile/核的共享元数据只通过 hist/states 交换。当前原值随原轴索引逐轮搬移，末轮直接写 values；不再按原索引回读 x。

## 4 输出与贯穿算例

最终字段按 M×L 分配，中间仍按完整 N；非末轴布局区单独核算。

取 M=P=2、N=16385、T=2048、K=9、float32（D=H=4）：原值/id 两侧共 524320 字节，hist 为 18432 字节，states 为 81920 字节，总算法 workspace 为 **624672 字节**。每核静态声明为 `14T+1536=30208` 字节，Host 预算 `14T+4096=32768`，编译对齐仍需核对。

来源：[PreparedSort](../../example/sort/entry.py)、[RadixMoreCore](../../example/sort/sort_radix_more_core.py)。
