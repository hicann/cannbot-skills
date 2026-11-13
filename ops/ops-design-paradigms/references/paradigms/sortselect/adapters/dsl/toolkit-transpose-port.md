# Transpose 的 DSL 映射

Ascend C Sort example 在 `op_api/aclnn_sort.cpp` 调用公共 `l0op::Transpose`：输入一次、values 与 indices 各一次。example 没有自写转置 kernel。公共 Transpose 的 arch35 实现还区分 NDDMA、Gather 等路线，不能将“调用同一接口”理解为所有形状使用同一种算法。

本页对应 CANNBotDSL 0.8.0、DAV_3510 上的 [SortTranspose](example/sort/sort_transpose.py)。该版本没有与 `l0op::Transpose` 等价的自动选路入口，但可直接使用已有搬运原语，不必全部逐元素手写。

## 原语与限制

| 需求 | DSL 表达与限制 |
|---|---|
| 交换逻辑轴 | `Tensor.permute` 只构建视图，不搬数据 |
| 将转置布局物化到 UB | `mem_copy(ub, source.permute(...), copy_options(engine="nddma"))`；默认 `auto` 不选择 NDDMA |
| 写回 GM | `mem_copy(destination, ub)`；目标使用可写的 `tile_slice`，动态 tile 通过 `make_tiler(..., alignment=...)` 声明 |
| UB 内转置 | `mem_copy` 不提供通用 UB→UB 转置，仍需 raw VF；本例优先在 GM→UB 时完成布局变化 |
| int64/uint64 | NDDMA 直接输入仅支持 8/16/32 位整数及 float16/bfloat16/float32；64 位拆为相邻 uint32 字，保持末维大小 2，一起搬运 |

64 位必须在 Host 将 Tensor **别名绑定**为 uint32 根参数。只在 kernel 内 `view(uint32)` 仍会被 NDDMA 的 GM/UB 物理类型检查拒绝。`WordPairTranspose` 使用 PyTorch 的 dtype view，不做数值转换、不复制设备数据；索引高半字、浮点 payload 等都不能用数值转换代替位模式搬运。

`build_transpose_program` 的 Python 调用仍接收原 dtype 的扁平 Tensor。Native 导出的 64 位配置实际为 uint32 TensorSpec、`words=2`，Tensor 长度加倍，`outer/height/width` 仍为逻辑元素数。int64/uint64 共用该配置；直接 Native 调用须遵守物理 ABI。

Python 的 `WordPairTranspose` 在建立别名前检查输入和输出 dtype 均等于构建时的逻辑 dtype；错误类型在启动前拒绝。它转发运行参数的关键字调用，并代理 `close` 和上下文管理。复用输出后一次调用会覆盖前一次结果，关闭前仍需等待设备完成。

## 搬运布局

完整短行采用紧凑 UB 布局，展平后连续写出，减少短 DMA burst；普通二维 tile 使用规则行距表达 GM 行间隔。64 位小容量短行使用 SIMT，其余路径使用 NDDMA。输入与恢复方向分别根据形状选路。

## 当前切块

以下 H/W 是本次转置的输入高/宽，O 为外层批数；输入与恢复分别应用此表。分支从上到下匹配。

| 条件 | 实现 |
|---|---|
| 64 位、H≤32、O×H×W≤131072 | SIMT；W<4096 用 32×32；否则 H≤8 用 8×128，8<H≤16 用 16×128，16<H≤32 用 64×128 |
| H≤64，且 H≤16、W≥65536 | NDDMA，完整 H×512，紧凑写回 |
| H≤64，且 W≥2048 | NDDMA，完整 H×128，紧凑写回 |
| 其他 H≤64 | NDDMA，完整 H×32，紧凑写回 |
| H≥65536、W≤16 | NDDMA，512×16 |
| H≥4096、W≤64 | NDDMA，128×64 |
| 其余 | NDDMA，短边≥128 用 64×64，否则 32×32 |

尾块通过 `tile_slice` 的实际范围与 UB 的有效域处理。每个核循环处理互不重叠的 tile，没有跨核依赖。NDDMA 路径的 UB 复用依赖由编译器按 buffer 访问插入；SIMT 回退保留两次线程屏障。UB 最大声明仍为 `max(T×(T+1),128×65)×D`，其中 D 为逻辑 dtype 字节数；最大 64 位配置为 66560 字节，Host 容量检查不变。

当前是单 UB buffer；公共 Transpose 的队列双缓冲、针对布局的 Gather 等属于后续可验证方向，不能宣称已经完全复现公共算子的实现或性能。比较时仍累计三次转置和排序 kernel，不能只比较 NDDMA 的某一个有利方向。

## 地址、stride 与尾块

下列公式对应 `SortTranspose.tiled/packed_output`。令 Q=words：普通 dtype 为 1，64 位逻辑元素为 2 个 uint32 字。H/W 和 tile 高宽 R/C 都按逻辑元素计数；下面的扁平地址、stride、count 则按物理 dtype 元素计数，不是字节数。原输入视图为 `[O,H,W,Q]`，`permute(0,2,1,3)` 后按 `[O,W,H,Q]` 遍历，输出也为 `[O,W,H,Q]`。

普通二维 tile 以 `nR=ceil(H/R)`、`nC=ceil(W/C)` 划分。核 c 处理 `task=c+k×blocks<O×nR×nC`：

```text
b  = task // (nR*nC)
ri = (task // nC) % nR; ci = task % nC
r0 = ri*R; c0 = ci*C
vr = min(R,H-r0); vc = min(C,W-c0)
```

对 `0≤i<vr, 0≤j<vc, 0≤q<Q`，源、UB 与目的的元素下标分别是：

```text
src = ((b*H+r0+i)*W+c0+j)*Q+q
ub  = (j*R+i)*Q+q
dst = ((b*W+c0+j)*H+r0+i)*Q+q
```

因此 UB 的有效 shape 为 `(1,vc,vr,Q)`，stride 为 `(R*C*Q,R*Q,Q,1)`。GM 两端用名义 tiler `(1,C,R,Q)` 及坐标 `(b,ci,ri,0)`；`tile_slice` 根据原域裁出实际尾块，UB 绑定同一有效范围。尾块仍用名义 R 作为 UB 行距，不把有效 vr 误作下一行的步长；未写入的容量余量不属于本次 copy 的有效域。全局地址和任务计算保持 int64。

紧凑写回分支要求 H≤64，tile 覆盖完整 H，只沿 W 分块。此时 r0=0，`ub=(j*H+i)*Q+q`，连续有效长度为 `count=vc*H*Q`，输出起点为 `(b*W+ci*C)*H*Q`。输入名义 tiler `(1,C,64,Q)` 的 H 维会裁成实际 H；UB 不保留 64 或 R 的行间 padding。写回将输出视作 `[O,W*H*Q]`，动态 tiler 为 `(1,C*H*Q)`、`alignment=(1,1)`，再从 UB 的 `[1,count]` 视图连续复制。尾块定位始终使用名义 C，不能改成 vc。

例如 O=5、b=0、H=7、W=4097、C=128，最后一块 ci=32，vc=1。32 位路径紧凑 UB 仅含 7 项，目的起点为 `4096*7=28672`；64 位路径对应 14 个 uint32 字，目的起点为 57344。此例 O×H×W=143395，超过小容量回退门槛，因此两种类型均进入紧凑 NDDMA。Q 只扩展物理表示，不改变 H/W 或逻辑元素顺序。

## 代码分支与验证对应

以下方法位于 [test_device.py](example/sort/tests/test_device.py)，是可执行的回归入口；方法通过不等于覆盖了所有形状或全部门槛的相邻点。

| 实现约束 | 对应验证 |
|---|---|
| 普通/紧凑 NDDMA、SIMT 回退、尾块与长块切换 | `test_rectangular_transpose_boundaries`：11 dtype、两个方向、outer=2，包含 2047/2048/2049、4095/4096/4097、65535/65536/65537 长边及多种短边 |
| 64 位高低字、浮点 NaN payload/正负零、同址输入更新 | `test_transpose_bit_patterns_rebinding`：按原始字节比较输出与输入，覆盖小容量回退和长块 NDDMA |
| Python 64 位别名的逻辑类型合同及关键字参数 | `test_transpose_wide_binding_contract`：正确输入运行，错误输入/输出 dtype 在启动前拒绝 |
| Sort 包装、输入不改写和四 kernel 合同 | `test_non_tail_layout`；完整性能比较还需从 profiler 核对三次转置与一次排序 |

Native 导出编译合同与 Native 部署运行是不同验收项；Python 设备测试不能替代后者。性能仍需累计两个方向及 int64 索引恢复，并保留软件版本、SoC、预热/采样口径和原始结果。

## 本地源码核验入口

- DSL 安装包：`cannbotdsl/ops/memcpy.py` 的 `copy_options` / `mem_copy`，`cannbotdsl/tensor.py` 的 `tile_slice` / `make_tiler`。
- CANN 公共算子头：`include/aclnn_kernels/transpose.h`。
- ops-math：`conversion/transpose/op_kernel/transpose_apt.cpp`、`arch35/transpose_cut_two_axis.h`、`arch35/transpose_with_gather.h`。这些文件证明公共算子存在相应实现，不代表某个实测 case 一定命中该路线；判定实际路线需要运行时 tiling 信息。

正确性覆盖见 [设备测试](example/sort/tests/test_device.py)：11 dtype 双向尾块、长块切换、64 位高低字、NaN payload/正负零与同址输入更新。性能数据及固定环境留在库外验证报告，不将单个平台上的门槛写成通用最优参数。
