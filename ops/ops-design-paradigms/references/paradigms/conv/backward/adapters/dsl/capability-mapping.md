# 卷积反向 DSL 能力与指令映射

本页说明公共操作的接口落点。具体窗口、矩阵朝向和输出地址分别在 [dX 指令映射](input/instruction-mapping.md)、[dW 指令映射](filter/instruction-mapping.md) 中展开。接口版本依据见 [索引](index.md)。

## 1 公共操作

| 既有流程需要的能力 | DSL 表达 | 实现时核对什么 |
|---|---|---|
| ND2NZ / DN2NZ，GM→L1 | `mem_copy(dst, src)`，由两端 layout、源 `permute` 和目的物理格式选转换 | 逻辑矩阵方向、批数量、源/目的 pitch；批搬运是否仍是一条指令 |
| L1 fmap→L0A/L0B 的 Load3D | `load3d(dst, fmap, feature=Load3DFeatureMap(...), window=Load3DWindow(...), ...)` | fmap 的本地 H/W、pad、stride、dilation、通道及 M/K 起点；见 [Load3D 使用说明](load3d.md) 和分支参数表 |
| L1 权重/梯度→L0 的 Load2D | `mem_copy(dst, src)`，需要转置时显式建立源 `permute` 和目的布局 | 卷积的 L0B 排列可能不同于普通 MatMul 的默认推导；不能只交换逻辑 shape |
| L1 未写区域补零 | `cannbotdsl.ops.memcpy.cube_fill_l1_zero` | 按物理存储范围清零；仅在存在插零洞或未覆盖区域时使用 |
| MMAD，B16→FP32 | `matmul(dst, lhs, rhs, init=..., bias=...)` | 第一段初始化，后续段累加同一 L0C 输出块；bias 仅在约定首段使用 |
| Fixpipe NZ2ND / NZ2DN | `mem_copy(gm_dst, l0c_src)` | 从两端实际布局生成写回方式；非紧凑输出保留真实步长 |
| dX 写回转换 | `mem_copy(gm_dst, l0c_src.convert(Tout))` | FP32 累加在最后写回时转换，尾部有效尺寸正确 |
| dW FP32 原子写回 | `mem_copy(gm_dst, l0c_src, copy_options(atomic="add"))` | 输出清零先于任何原子写入；不同核重叠的地址确实属于同一归约目标 |
| Transpose bias table | GM→L1→`MemLoc.BIAS`，`matmul(..., bias=bt)` | BT 容量、补齐、当前 N 窗口及只加一次；见 input 分支 |
| 片上槽的自动依赖 | `Buffer`/`RingBuffer` 及其 Tensor 访问 | ready 与允许覆盖是两种依赖；选槽动作本身不是同步 |
| 显式本核流水事件 | `cube_sync_notify/wait`、`vec_sync_notify/wait` | 两端均以 `(生产流水, 消费流水, event_id)` 传参；只补尚未表达的依赖 |
| 跨核初始化与消费 | 融合 AIV 清零；MTE3 mode 0 汇合，再以 mode 2 通知 FIXPIPE | GM 共享地址不能默认获得跨核同步；采用分支规定的初始化协议 |
| standard 的 `ApplyBalancedMSplit` 能力条件 | 按实际 M 起点派生 fmap 窗口及输出 view，不是一个 DSL 开关 | 改变 singleCoreM 后，非整行起点、Load3D 与写回都须正确；未验证时不把该能力标为可用 |

`mem_copy` 的第三个参数是 `copy_options(...)`，不是任意 ASC 参数结构；本版本不使用 `transpose=True` 关键字。`matmul` 是一次 Cube 计算的 DSL 表达，不等于调用外部 MatMul 算子。

## 2 同步与执行域

Kernel 内的 Cube 操作发到 AIC，Vector/VF 操作发到 AIV，混合程序由编译器生成对应执行部分。不能用一条普通 Python `if` 的位置推断某段只在一个 AIV 上执行；需要区分 AIV 子核时显式使用子核编号并核对生成的分支。

保留默认自动同步，由 Tensor 的存储身份、布局和访问决定片上依赖。单份 Buffer 也不能靠函数调用先后推定异步完成。改用别名视图时保留同一存储身份；对完全经 GM 交接的跨核阶段，按分支另行建立依赖。

`cube_sync_all()` / `vec_sync_all()` 是对应核内的流水屏障；`global_sync_all()` 是整个 launch 的集体屏障。不要把局部的 Load→MMAD 或 MMAD→Fixpipe 依赖升级成全核同步。使用集体屏障时，空任务核也必须到达相同轮数，并保证 launch 的参与 block 能同时驻留。

## 3 核对生成指令

用实际实现入口导出代码。下面的 `your_compile_entry.py` 指调用方已有的编译脚本，不是本目录附带的程序：

```bash
CANNBOTDSL_NPU_ARCH=dav-3510 \
CANNBOTDSL_CACHE_DISK_DISABLE=1 \
CANNBOTDSL_DUMP_ASCENDC=1 \
CANNBOTDSL_DUMP_DIR=./dsl_dump \
python your_compile_entry.py
```

按一轮基本块追踪 GM→L1→L0→MMAD→Fixpipe，比较实际布局、参数、搬运次数和依赖。关注批量 copy 是否被拆成 H/通道循环，Load3D 是否使用本地父段几何，L0B 是否多做了不需要的转置，Fixpipe 是否使用预期的输出步长。Load3D 配置写入是否消减以所用编译器产物为准，不根据源码调用次数承诺最终指令数。

区别设备执行次数与反汇编的静态指令数：分支克隆、展开和运行时循环都会改变两者关系。比较性能时再用 profiler 查看整算子耗时以及 MTE1、MTE2、Cube、Fixpipe 等流水；各流水可以重叠，不能把它们相加当作总耗时。

## 4 能力结论如何填写

表中的接口依据是 Python 源码与既有开发实现；完整算子仍以当前编译实例和设备验证为准。若没有自动推导但能用显式 `physical_layout` 正确表达，属于适配工作。若所需指令确实不能表达，或一条批搬运被拆为多条指令，记录具体输入布局、参数、版本与生成结果。

数值通过不证明指令及性能等价。保留 generic 的原有支持边界和验证要求；不要因为某个示例只测了整块，就把通道尾块、宽片、stride 放大或不同输出格式当作已覆盖。
