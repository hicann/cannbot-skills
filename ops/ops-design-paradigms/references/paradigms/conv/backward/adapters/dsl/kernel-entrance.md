# DSL 入口、Tiling 与验证

本页将 generic 的参数发布和启动契约映射到 Python/DSL。dX 的三模板和 dW 的 MN_StreamK 仍分别使用原有选块规则。

## 1 三层入口

| 位置 | 负责什么 |
|---|---|
| 普通 Python 函数 | 校验规格、按 generic 归一化为 3D、查询资源、运行 tiler、分配输出及 workspace、选择编译实例 |
| `@cbd.host` | 接收实际 Tensor 和标量，按顺序执行 `kernel[blocks](...)`；输出通过形参传入 |
| `@cbd.kernel` / 设备 `@cbd.jit` | 解码任务，建立视图，执行搬运、计算、写回和同步 |

`@host` 不承担 Torch 数值计算。需要清零、转置、补洞或归约时，按既有流程使用设备实现。调用方负责在异步执行结束前保持 Tensor、workspace 和编译程序有效。

需要预清零时，在主 kernel 中接入 [融合 AIV 清零与 FIXPIPE 等待](initialization.md)；是否初始化仍由当前分支的计划决定。

## 2 TilingData 如何交给 DSL

DSL 不需要照搬 C++ `TilingData` 结构或数字 TilingKey，但必须保留字段来源、单位和消费者。

| 原设计信息 | DSL 放置位置 |
|---|---|
| input/filter、模板、dtype、format、hasBias 及改变数据通路的 flag | 类构造配置或构建函数的编译期参数 |
| baseM/baseN/baseK 的容量、物理槽数与布局 | 编译期配置；形成有限的模板实例 |
| 本次 N/C/D/H/W、卷积核尺寸、stride/dilation/pad、group、有效尾长 | 运行时 Tensor 元信息或明确的标量形参 |
| 父段尺寸、任务区间、实际归约段数 | 当前 tiling 结果；必须被 Kernel 地址和循环实际消费 |
| 核数与初始化参与者 | 本次 launch 参数及对应同步协议 |

编译期容量可以是固定的，真实几何不能被悄悄捕获为类常量。若与动态 ASC 基线比较，应让这些几何以运行时值进入相同镜像。用 `TensorSpec` 的 `Dim` 描述动态维度，标量使用 `cbd.dtypes.int32/int64` 等规格；仅在参数表写“动态”不足以证明取址仍是动态的。

`cbd.compile(host_entry, *specs)` 的规格顺序与形参一致。输入、输出和 workspace 都传入自己的 `TensorSpec`；非紧凑 Tensor 还需描述真实 stride。只将影响生成代码和静态分配的配置用于编译缓存，换 shape 后重新计算任务、窗口、尾长和核数。

下面是**入口绑定片段**：`dw_kernel` 是调用方已经实现的 `@cbd.kernel`，使用 NCDHW、groups=1，接收三个 Tensor 和本次 strideH，其余已校验的属性按同样方式传入。示例省略这些属性，只说明动态 shape、标量规格和 launch 的对应关系，不是完整卷积程序。

```python
import cannbotdsl as cbd
from cannbotdsl import Tensor, TensorSpec, Dim, dtypes

@cbd.host
def launch(x: Tensor, dy: Tensor, dw: Tensor, stride_h, blocks):
    dw_kernel[blocks](x, dy, dw, stride_h)

n, ci, co = Dim("n"), Dim("ci"), Dim("co")
di, hi, wi = Dim("di"), Dim("hi"), Dim("wi")
do, ho, wo = Dim("do"), Dim("ho"), Dim("wo")
dk, hk, wk = Dim("dk"), Dim("hk"), Dim("wk")
program = cbd.compile(
    launch,
    TensorSpec((n, ci, di, hi, wi), dtypes.float16),
    TensorSpec((n, co, do, ho, wo), dtypes.float16),
    TensorSpec((co, ci, dk, hk, wk), dtypes.float32),
    dtypes.int32, dtypes.int32,
)
# x、dy、dw 是真实设备 Tensor，sh、blocks 是本次 Host 选出的整数。
program(x, dy, dw, sh, blocks)
# 最后一次异步使用完成后，由持有方调用 program.close()。
```

同一编译入口中，同名 `Dim` 表示运行时长度必须相等；共享逻辑维度可复用同一 `Dim` 对象。这不是“具有相同类型或上限”的声明。例如 `(n, co, do, hw, hw)` 会强制 H=W；Ci/Co、Di/Do、Kd/Kh/Kw 也不能仅因同属通道或空间轴而共用一个 `Dim`。卷积形状关系和片上容量仍由入口校验。

dX 的 NCDHW 入口若按 `(dy, weight, dx)` 接收 Tensor，可复用上面的维度对象这样绑定；普通 group 下 `ci_g` 与 `ci` 独立，Host 另查 `ci == groups * ci_g`：

```python
ci_g = Dim("ci_g")
dx_specs = (
    TensorSpec((n, co, do, ho, wo), dtypes.float16),  # dy
    TensorSpec((co, ci_g, dk, hk, wk), dtypes.float16),  # weight
    TensorSpec((n, ci, di, hi, wi), dtypes.float16),  # dx
)
```

切换静态 format 时，`TensorSpec` 的轴序、设备端 `shape` 解码和 GM 视图必须一起对应到真实存储。例如 NDHWC weight 按 `(Co,Dk,Hk,Wk,CiG)` 排列，Dk/Hk/Wk 来自 `shape[1:4]`；NCDHW 按 `(Co,CiG,Dk,Hk,Wk)` 排列，核尺寸来自 `shape[2:5]`（格式范围见 [input scene](../../generic/input/scene.md)）。只改 TensorSpec 和搬运 stride、仍按旧格式读取几何，可能把通道数当成卷积核宽度，进而算错地址。

运行时标量类型由 compile 规格绑定；前面的片段没有给 `stride_h/blocks` 加 Python `int` 注解，避免与显式 `int32` 规格发生类型冲突。

## 3 资源与启动核数

以下为普通 Python 的资源查询片段；调用前已初始化目标设备和实际执行流：

```python
import cannbotdsl as cbd

info = cbd.get_platform_info()
if not info.available or info.cube_core_num <= 0:
    raise RuntimeError("无法取得当前执行流的 AIC 核配额")
available_aic = info.cube_core_num
available_aiv = info.vector_core_num
capacity = {name: cbd.get_mem_size(name)
            for name in ("l1", "l0a", "l0b", "l0c", "ub", "bt")}
```

本版本的 `get_platform_info(stream=...)` 可按指定流查询有效核数；不传时使用当前流。`get_mem_size` 返回编译目标的容量表，单位字节。先确认实际设备与编译目标一致，资源为零时报告具体缺项。

Tiler 根据有效核数选块、分核，`@host` 用选定的 `blocks` 启动，Kernel 用 `cbd.get_block_idx()` / `cbd.get_block_num()` 解码同一任务计划。32/36 核均由运行参数承接，不把 32、36 或物理总核数写死在分工和屏障里。MIX 清零还要核对有效 AIV 数和所用 AIC/AIV 配对关系。

### 3.1 Launch 与任务分配使用同一核数

区分平台可用核配额、请求的 launch 上限和最终启动的 `blocks`。如果 tiling/meta 中保存 `used_core_num` 并用于任务轮转、清零分区或同步参与者计数，该字段必须与对应 kernel 的实际启动核数一致；不能把平台配额直接当作实际启动数。Host 从同一份计划生成 meta 和 launch 参数，并在启动前检查一致性。

轮转任务推荐使用实际启动数：

```python
for task in range(cbd.get_block_idx(), task_count, cbd.get_block_num()):
    # 处理 task 对应的完整任务。
    ...
```

`get_block_num()` 只解决这段轮转循环的步长来源；其他按核数划分的任务区间、清零区域、workspace 和同步参数仍须与本次 launch 匹配。多 kernel 的计算与合并阶段可使用不同核数，各自保持计划与启动参数一致。

核数扫描时，每个档位重新生成匹配的计划/meta，再调用对应 launch；不能只把 `program(..., blocks)` 的最后一个参数改小。例如总任务数为32、meta 步长仍为32，却只启动1核时，仅任务0被执行，其余31个任务漏算。正确的1核完整算子测试应执行全部32个任务。各档位先检查完整输出与 Golden 一致，再比较性能；若只测单任务微基准，应单独标注工作量，不能与完整算子延迟直接比较。

## 4 验证与交付

先按 [input verification](../../generic/input/verification.md) 或 [filter verification](../../generic/filter/verification.md) 执行已有 Golden 与覆盖要求，再补齐 DSL 特有的检查：

1. 保存实际 DSL 构建、CANN/编译器、目标、静态配置及运行时参数。编译通过和设备数值通过分别记录。
2. 检查生成的矩阵方向、父 pitch、有效窗口、批搬运数量和同步；方法见 [能力映射](capability-mapping.md)。
3. 同一模板改变通道或空间几何，复用同一镜像；覆盖尾块、空窗口、槽回绕、单段/多段归约和本次实际核数。950 上验证 32/36 核配额时分别保存结果；没有测到的配额不能宣称已覆盖。
4. dW 只测权重梯度，dX 只测输入梯度；将初始化、cast、归约等必要设备阶段计入完整设备耗时。kernel 性能基线排除 Python、编译、Host tiling、绑定及启动提交开销，按 [DSL 性能基线](performance-baseline.md) 采集设备计时；公开入口和直调 program 的墙钟耗时不得冒充 kernel 耗时。
5. 原子累加不自动保证确定性。若交付要求确定性，明确其归约协议并逐位复跑；不得把一次 Golden 通过当作确定性证明。

本次适配不改 generic 的数值契约。精度、确定性和性能结论都针对实际编译及执行的组合；不能由接口存在或另一模板通过推及全部组合。
