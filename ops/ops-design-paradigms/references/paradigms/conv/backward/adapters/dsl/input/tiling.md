# input Tiling 的 DSL 绑定

切分算法沿用 [通用 Tiling](../../../generic/input/tiling/tiling.md)。Python Host 根据运行时形状和当前可用资源求解，再选择对应 DSL 编译实例；不会因为换了语言而改用固定块表。

## 1 三模板与实现职责

| 模板与选择顺序 | DSL 中保留的行为 |
|---|---|
| 先检查 [small-shape](../../../generic/input/tiling/templates/small-shape/tiling.md) | 按原条件联合搜索 M/N 与任务。输出任务不足时调整基本块，不把核数固化到代码。 |
| 再检查 [l1-resident](../../../generic/input/tiling/templates/l1-resident/tiling.md) | 在规定的复用域内保留完整 K 的 weight L1 视图；M 块切换仍重建对应 L0 子窗口。 |
| 其余合法规格使用 [standard](../../../generic/input/tiling/templates/standard/tiling.md) | 经验起点、AdjustBaseMNK、CalStepK 与容量回调按原流程执行。 |

三者共享 `M=Hi*Wi`、`N=CinG`、`K=alignUp(CoutG,16)*Hk*Wk`；Dk 是外层归约，不能在每个 Dk 都重置同一输出的 L0C。`baseM/N/K` 是单次计算容量，`singleCore*` 是任务范围，`baseUse*` 是本轮有效长度。

## 2 编译配置与运行时字段

| 归属 | 内容 | DSL 承接 |
|---|---|---|
| 编译配置 | dtype、三个外部 format、weight 变换分支、是否有 bias、片上最大容量与槽数 | 构造程序对象时固定；设备静态分支用 `cbd.const_expr`。 |
| 运行时几何 | batch、Cin/Cout、group、各 D/H/W、kernel、stride、dilation、pad、目标形状 | Host 归一化后通过 TensorSpec 的动态维度和标量参数传入。 |
| 运行时执行字段 | `singleCore*`、有效基本块、stepKa/stepKb、初始化标志、当前任务起点与有效范围 | 与同一 Host 求解结果绑定；设备按字段解码，不重新猜测模板。 |
| 有符号派生值 | 反向 pad、裁剪起点、负偏移 | 保留有符号类型，合法区间确认后才形成 GM 地址。 |

片上分配可使用静态容量，实际视图必须容纳当前有效块及其对齐跨度。选择容量变体时重算五路 buffer、bias 和所有同时存活槽；不能仅把超限参数截短后继续。对相同模板，换通道或空间尺寸复用同一编译产物，检查真实动态性。

`coreNum` 来自本次流的可用 AIC 配额，launch 与任务分配共用该值；设备从 `cbd.get_block_num()` 获取实际启动数。32/36 核只代表验证配置，不是模板常量。

## 3 weight 变换分支

先形成最终 baseN，再按现有规则选择 `loadB2Condition`。下列是 **Host Python**，`cin` 为总输入通道数，`weight_bytes=2`：

```python
kernel_hw = hk * wk
kernel_dhw = dk * kernel_hw
if kernel_dhw == 1:
    weight_mode = "transpose_only"
elif weight_format == "NDHWC" and base_n >= kernel_hw:
    weight_mode = "reverse_only"
elif weight_format == "NCDHW" and kernel_dhw <= 64 // weight_bytes // cin:
    weight_mode = "reverse_only"
else:
    weight_mode = "transpose_and_reverse"
```

编译实例保存这个语义分支即可；DSL 不需要复制 Ascend C 的整数 TilingKey 编码。三种分支对应的 GM→L1 父布局也不同，见 [指令映射](instruction-mapping.md)。容量检查失败时报告受影响的布局或指令范围，不能偷偷改选另一种分支。

当前配置保留普通通道 C0=16、原始 group 寻址、GM→L1 weight、完整 Dk 归约；standard/small-shape 的 `enableFullLoad=0`，l1-resident 为 1。没有实现的 C04、kernelSplitMode、groupEnlarge、split-K 等标志不得从原结构中直接透传为已支持功能。

## 4 发布前核对

每个执行字段都应能指出一个生产者和一个消费者；可从几何推导的值选择在 Host 或设备一处计算。特别检查 L1A/L1B 的父 K 起点不同、尾 K 不缩短父布局 pitch、最终任务与 bias 驻留预算一致。正式调用形式见 [公共入口](../kernel-entrance.md)，任务与生命周期见 [Kernel 组织](kernel-template.md)。
