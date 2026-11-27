# input Kernel 组织

本页把 [通用计算流程](../../../generic/input/compute-flow.md) 的阶段绑定到 DSL；任务解码、空间窗口和 K 配对仍使用该流程的公式。standard、small-shape 和 l1-resident 共享计算主体，变化的是任务划分和父段保留时间。

## 1 阶段与状态

| 阶段 | DSL 中实现的内容 | 需要保持的状态 |
|---|---|---|
| Init | 建立 L1A/L1B、L0A/L0B、L0C；按需建立 bias L1/BT | 最大容量、物理 stride、槽数与 Host 一致。 |
| Task | 按本核任务序列解码 batch/group/D/M/N | 原张量基址、有效范围、weight 身份。 |
| PrepareParent | 根据各自 stepKa/stepKb 更新父窗口，用 `mem_copy` 搬入 | A/B 父段的 K 起点与末端分别维护。 |
| PrepareL0 | `load3d` 取 fmap 子窗口，`mem_copy` 取 weight 子窗口 | 同一段有效 K、各自局部起点、尾块物理跨度。 |
| Compute | `matmul` 累加到同一 L0C | 首次有效贡献标志；跨所有有效 kd 保留累加。 |
| CopyOut | `mem_copy` 按输出 stride 转换并写回 | 只写有效 M/N；驻留 weight 仍可供下一 M 块使用。 |

阶段函数可写成 `@cbd.jit` helper，设备入口使用 `@cbd.kernel`。helper 名称由生成工程定义；阶段边界本身不产生同步。Host 和动态编译方式见 [公共入口](../kernel-entrance.md)。

## 2 任务与归约骨架

以下是**流程伪代码**，`decode_task`、`prepare_parent`、`intersect_k_windows` 等是待实现的阶段函数，不是 DSL API：

```text
为本核分配片上资源
for task in 本核的任务序列:
    解码 batch / group / depth / 空间范围 / Cin 范围
    对每个 M/N 输出基本块:
        first_contribution = true
        保持本输出的同一份 L0C
        for kd in 该输出深度的有效卷积核深度:
            求对应 dY 深度；不满足 stride 同余或范围时跳过
            while 本 kd 的归约 K 尚未完成:
                按 stepKa、stepKb 各自准备或复用 L1 父段
                取两个父段与 baseK 的交集，得到本轮 K
                Load3D(A)，Load2D(B)
                MMAD：首次有效贡献初始化，后续累加
                first_contribution = false
        若有贡献：Fixpipe 到输出
        若无贡献：按初始化协议保留零；有 bias 则执行纯 bias 通路
结束当前复用域；不再覆盖仍有消费者的槽
```

外层任务分发沿用选中模板的定义。若该模板采用跨步分发，则设备写作 `range(cbd.get_block_idx(), task_count, cbd.get_block_num())`；不要在另一个地方使用固定 32/36 计算下一任务。

`first_contribution` 只在实际发出 MMAD 后更新。边界可能令前几个 kd 没有数据，不能用 `kd == 0` 代替它。每个 M/N 输出块重新初始化一次，每个 K 子块不能重新初始化。所有 K 都无效时，不得把未初始化的 L0C 直接搬到输出。

## 3 让布局和累加配对

L0A、L0B 的本轮有效 K 必须相同；A 的 `k_start` 相对 A 父段，B 的起点相对 B 父段。将绝对 K 起点同时传给两者，只在父段都从零开始时偶然正确。

容量 buffer 与参与指令的 view 分开使用。尤其 M 尾块采用紧凑的 `m_compute` 时，要同时更新 Load3D 的目的分形步幅、L0A 的 MMAD view、L0C 的实际布局和 Fixpipe 源 view；若保留容量跨度，则整条链都保留这一跨度。普通切片应保留父 pitch，不要在写回前把有间隙的数据重新标成紧凑矩阵。

weight 变换只发生在 [三种既有分支](instruction-mapping.md) 中；同一物理 L0B 可以给搬入和计算建立不同语义视图，但二者的元素地址必须一致。不要额外引入 GM 中间转置或逐元素重排来掩盖布局错误。

## 4 最小验收覆盖

除 [通用验证](../../../generic/input/verification.md) 外，DSL 生成物应覆盖下面几个能区分实现错误的边界：

- NCDHW/NDHWC 的 dY、dX 与两种 weight format；三种 weight 变换各有命中记录。
- 两个 L1 父段边界不重合、通道尾块、M 尾块、D stride 同余与前几个 kd 为空。
- M/N 至少各覆盖两个输出块，其中包含尾块；特别用 `CinG > baseN` 检查第二个 N 块的 weight 搬入偏移和 dX 写回偏移是否都计入对应 N 块起点（任务起点加任务内块偏移）。单块通道用例通过，不能证明跨 N 块的地址正确。
- stride>1 的批量插零搬入；检查生成指令的矩阵数和 stride，确认没有多出逐 H 循环。
- l1-resident 的首次加载、跨 M 复用、group/N/weight 身份变化后的重载。
- Transpose 的非零 bias、多轮 K 与完全无乘积贡献区域；每块只加一次 bias。

记录编译配置、实际动态参数、生成的 Load3D/Load2D/Fixpipe 和同步，再独立比对 Golden。接口片段能编译不等于三模板的完整覆盖已经通过。
