# Transpose 的 bias / BT 通路

仅用于 Conv3DTranspose：bias 为 FP32、长度为总 Cin，沿输出通道广播。普通 dX 不分配 BT。本页承接既有的首次 MMAD 融合、全量/分块 bias 和无贡献输出流程。

## 1 资源与搬运

DSL 用 `MemLoc.BIAS` 表示 BT。准备顺序是 GM→L1→BT；对应接口都是 `mem_copy`，首次乘加传 `bias=bt_view`。

```python
# 接口片段：bias_capacity = alignUp(bias_use * 4, 64) // 4，单位是 FP32 元素。
# 分配字节数为 bias_capacity * 4，且不超过 BT/L1 容量。
bias_l1 = cbd.Buffer(cbd.MemLoc.L1, cbd.dtypes.float32,
                     cbd.make_layout((bias_capacity,)), data_format="nd")
bt = cbd.Buffer(cbd.MemLoc.BIAS, cbd.dtypes.float32,
                cbd.make_layout((bias_capacity,)), data_format="nd")
# bias_l1 的有效区与补齐区准备完成后：
cbd.mem_copy(bt, bias_l1)
# bias_for_n 是当前输出 N 块对应的 BT 子视图。
cbd.matmul(acc, a0, b0, init=True, bias=bias_for_n)
# 同一输出块的后续 K/kd：
cbd.matmul(acc, a0, b0, init=False)
```

`init=True, bias=...` 在 DSL 中表示首次从 BT 建立累加值；不要机械照抄 Ascend C 的 `cmatrixInitVal=false` 到这里。无 bias 的首次调用仍使用 `init=True`，但不传 bias。

`isBiasFullLoad=1` 时装入任务通道段，当前 BT 偏移为 `cin_start-channel_start`；分块模式装入当前 N 块，BT 偏移为零。两种模式的 GM 元素起点分别是 `group_idx*CinG+channel_start` 和 `group_idx*CinG+cin_start`，再乘 4 转字节偏移。

BT 读取按 64B 覆盖。普通 GM→L1 尾补齐可能只到 32B，所以需要保证**整个 BT 读取范围**已定义。例如 17 个 FP32 有效值为 68B，DMA 到 96B，BT 会覆盖 128B；额外 32B 不能遗留旧数据。可先用 `cube_fill_l1_zero` 清理这个对齐区，再保留 MTE2 同流水顺序并搬入有效 bias；普通 ND copy 的 `copy_options(pad_val=0)` 可指定 DMA 尾部填零。所有清理都在分配范围内完成。

全量 bias 的身份由基址、group、通道起点和有效范围共同决定；M/深度切换不改变它。资源预算包含 bias L1 和 BT，不能只统计 A/B/C。

## 2 每个输出块只加一次

首次标志绑定到实际发出的第一条 MMAD，可能不是 kd=0。后续 K 和 kd 不再传 bias，从同一 L0C 继续累加。最终一次 Fixpipe 转成 FP16/BF16，避免先转换输出再做 FP32 bias 相加造成不同舍入。

当一个输出块完全没有乘积贡献时，沿用既有 ComputeForBias 路径：用全 padding 的 Load3D 准备零 A/B，执行一次带 BT 的 MMAD，再正常 Fixpipe。该路径仍写全部有效 M/N，不靠输出预清零替代 bias。

原零窗口配置的 DSL 字段如下；创建的 L0 视图须满足对应容量及方向，尤其 Load3D 的 L0B 路径会转置展开矩阵：

| 字段 | 零 A → L0A | 零 B → L0B |
|---|---|---|
| `feature.height / width` | 1 / 32 | 1 / 32 |
| `feature.channels` | 16 | `alignUp(baseN,16)` |
| `feature.pad_top`，其余 pad=0 | 255 | 255 |
| `filter / stride` | `(1,1)` / `(1,1)` | `(1,1)` / `(1,1)` |
| `window.m_extent / k_extent` | `alignUp(baseM,16)` / 16 | 16 / `alignUp(baseN,16)` |
| `window.m_start / k_start` | 0 / 0 | 0 / 0 |

此处 K=16 是纯 bias 乘加的局部长度，不改写正常归约的 baseK。生成后核对所有窗口落在 padding 区，操作数为零，并检查 Load3D、MMAD 与 Fixpipe 的生命周期。

## 3 验证

至少覆盖无 bias、非零 bias、多轮 K、首个有效 kd 非零、纯 bias 输出、多 M/N 块、分组及通道尾部。用有效 bias 长度 1/8/9/16/17 检查 32B 与 64B 对齐边界；每次输出只加一次 bias，重复执行不读取上次调用遗留的 BT/L0C。
