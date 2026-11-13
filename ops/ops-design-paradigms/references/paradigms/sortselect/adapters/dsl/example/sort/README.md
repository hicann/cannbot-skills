# CANNBotDSL Sort 示例

本例以 CANNBotDSL 实现稳定 Sort，返回排序值和原轴内索引。包含插入、单核 Radix、多核 Radix 和两阶段排序；公开入口负责非末轴布局转换。

本目录演示统一范式的 Sort 配置（first_rank=0、L=N、输出 values/indices）。同一主体用于生成 Argsort、TopK 和 KthValue：按 [公共映射](../../../../generic/output-mapping.md) 配置输出字段与名次区间，在最后写回应用区间，非尾轴使用共用布局规范化，不按长轴/短轴设置专用非末轴模板；连续尾轴及等价布局不启动转置。语言表达见 [输出适配](../../implementation-guide.md#统一输出适配)。公开 Python 入口仍仅提供 Sort，不是其他接口的现成运行库。

## 算法模板

| 模板文件 | Kernel | Host 路由 | 适用场景 |
|---|---|---|---|
| [sort_small_axis_insertion.py](sort_small_axis_insertion.py) | InsertionSort | `ROUTE_INSERTION`（1） | 批量短轴插入排序 |
| [sort_radix_one_core.py](sort_radix_one_core.py) | RadixOneCore / NarrowRadixOneCore | `ROUTE_RADIX_ONE_CORE`（2） | 一个核处理可放入 UB 的单行 |
| [sort_small_axis_two_stage.py](sort_small_axis_two_stage.py) | SmallAxisTwoStage | `ROUTE_TWO_STAGE_INVERSE`（4）/ `ROUTE_TWO_STAGE_GROUPED`（5） | 多行合批排序，再恢复各行；4 为逆映射，5 为稳定行号分组 |
| [sort_radix_more_core.py](sort_radix_more_core.py) | RadixMoreCore | `ROUTE_RADIX_MORE_CORE`（3） | 单 kernel 内完成长轴分块、按单行轴长选择 int32/int64 的中间索引与桶计数/前缀，并逐字节稳定散射 |

路由编号由 `sort_host.py` 的 `ROUTE_*` 常量统一维护；`build_program` 与编译期分派均拒绝未知编号。多核路线即使回退到默认 1024 元素块，也会核对 UB 预算，容量不足时在编译前报错。

每条排序路线均在一次 kernel launch 内处理本次调用的全部行。连续末轴（I=1）直接排序；非末轴 N>1 且 I>1 时，先用 DSL 转置 `[O,N,I]→[O,I,N]`，沿连续末轴排序，再分别恢复 values 和 indices，共四个 kernel。转置优先使用显式 NDDMA 将 GM 的 permute 视图物化到 UB，再 DMA 写回；短行输出在 UB 紧凑排列。64 位通过两个 uint32 字保持位模式，小容量 64 位短行保留 SIMT。分块与 SDK 限制见 [转置映射](../../toolkit-transpose-port.md)。N=1 或轴后仅 singleton 维度时不做物理转置；表中的 Host 路由号不是 kernel 数量。

这里四个 kernel 的计数只对应本例 L=N 且确需转置的配置。生成 L≠N 的接口时按 [布局判定](../../../../generic/shape-preprocessing.md#物理转置的判定与验收) 分别判断输入与输出；L=1 时输出恢复只需 reshape，不能直接套用本例的 `needs_transpose` 标志给两份输出各启动一次转置。

归并路线尚未适配。非末轴通过共用布局包装复用表中主体。two-stage 的两种恢复属于同一模板，通过编译期 `rank_inverse` 选择：N≤32 使用逆排名，较长轴稳定地按行号分组，当前公开 B≤256，第二阶段仅一个字节。局部单行 Radix 在同一路由内按输入字节数选择 1/2 字节局部排列或 32/64 位键/索引双缓冲。连续插入批次采用 DMA 输入、线程独立行内排序和连续 SIMT 写回。选路门槛与恢复阈值见 [选路与恢复策略](../../tiling.md#两阶段恢复硬约束与实现策略)。

## 入口与公共原语

| 文件 | 职责 |
|---|---|
| [entry.py](entry.py) | 公开入口、输入校验、Tiling 选型、workspace、编译及 Native 导出 |
| [sort_host.py](sort_host.py) | `@host` 按路由启动一个计算 kernel；多核 Radix 的逐字节阶段在 Kernel 内完成 |
| [sort_common.py](sort_common.py) | 容量常量、dtype 宽度、单调排序键和字节提取 |
| [sort_transpose.py](sort_transpose.py) | 非末轴输入转入及 values/indices 转回的共用 DSL 分块转置 |
| [radix.py](radix.py) | 共用的 SIMD 字节提取、permutation 与 key/index Gather；[toolkit 源码映射](../../toolkit-radix-port.md) |

读取单个算法时，从对应模板开始；路由条件见 `entry.py`，实际启动见 `sort_host.py`，多核阶段依赖见 `RadixMoreCore.__call__`。

## 使用

```python
from entry import sort
values, indices = sort(x, dim=-1, descending=False, stable=True)
```

支持 rank≥1 的连续 NPU 张量、合法正负轴、升降序和 11 dtype：float16/bfloat16/float32、int8/16/32/64、uint8/16/32/64。输出 shape 与输入相同，indices 为 int64 原轴内坐标；`stable=False` 不要求等值元素顺序，本例仍复用稳定排序实现并返回稳定结果，不另设非稳定路线。空输入直接返回空输出。浮点正负零保留原位模式，NaN 按方向放在端点并保持输入次序。

UB 采用 `cb.Buffer(mem_loc, dtype, cb.make_layout(shape))` 接口，VF 内无符号比较使用显式同类型常量；64 位键在无符号域提取高低字，不对有符号商额外修正。完整的数据表示和作用域约束见 [实现指南](../../implementation-guide.md#数据表示)。

公开入口对排序主体始终传 `inner=1`，非末轴经过 `PreparedSort` 的共用转置包装。Kernel 保留 `inner!=1` 的低层调用能力，独立验证见 `tests/test_device.py::DeviceTests.test_strided_schedules`；该直接调用不改变公开入口的统一布局规范。

复用编译与缓冲时使用 `PreparedSort.run()` / `close()`；下一次运行覆盖前次输出，关闭前须等待设备完成。公开 `sort()` 等待完成后关闭程序。

容量取设备 UB 与核数，two-stage 批次同时核对计数和行号字节上限。浮点批处理候选域为 N≤256；全部 buffer 预算和同步见 [implementation-guide](../../implementation-guide.md)。MergeSort 尚未提供模板；当前多核路径以全核阶段屏障和无等待 lookback 组织流水，不按字节拆分 kernel。

浮点插入门槛沿用参考基线，性能拐点尚未确定；低并发已有 Radix 更快的样例，不应将该门槛当作已验证的性能推荐。选路策略、证据边界与待调优状态见 [选路策略](../../tiling.md#dtype-路由表)。

`@export("sort_example")` 声明六个配置：float32 插入的两个方向、float32 局部 SIMD、float32 two-stage 逆映射、int32 two-stage 行号分组、float32 多核 Radix（int32 工作区）。`@export("sort_transpose")` 另外声明覆盖 11 个逻辑 dtype 的默认转置配置（方块基准 32×32，包含运行形状分支；int64/uint64 共用 uint32 成对搬运配置）；64×64 配置通过运行时编译使用，若部署为 Native require 则须另行导出相应配置。非末轴 Native 使用时须同时发布并注册这两个导出组。64 位转置的 Native 根参数为 uint32，Tensor 元素数为逻辑元素数的两倍，`outer/height/width` 仍按 64 位逻辑元素计数；Python 入口自动绑定别名，不能将该物理 ABI 误作 int64 TensorSpec。具体 width/batch/direction 见 `entry.py`；其他排序配置通过运行时编译使用。

## 多核流水与工作区

从 `RadixMoreCore.__call__` 开始阅读：每批初始化状态，每轮执行 `Histogram → 局部排列/lookback/Scatter`。首轮读取 x，后续轮切换原值/id ping-pong，末轮直接输出。连续输入采用 DMA/SIMD，低层跨步输入使用 SIMT L2。

Histogram 发布频数并原子累加全局桶起点；Scatter 生成稳定局部排列，lookback 命中完整前缀时直接使用，否则累加已完成的 hist，不等待前驱核。每批初始化后、每轮统计后、非末轮散射后和批次复用前均有全核屏障，空闲核也须参加。状态编码及地址算例见 [多核 Kernel](../../templates/radix-more-core/kernel-template.md)。

工作区依次为两侧原值/id、hist 和 states，共六个 Tensor。原值使用输入 dtype；当 N<2^30 时中间索引与计数使用 int32，否则使用 int64，最终输出索引始终为 int64。`build_program` 默认 int64 工作区，窄计数配置传 `narrow_counts=True`。

`PreparedSort` 按活跃行容量 P 在单次 launch 内分批。令 D 为输入字节数、H 为中间索引/计数字节数、K=ceil(N/T)，算法工作区字节数为：

```text
PN(2D+2H) + P×256H×[K+D(K+1)]
```

`workspace_bytes` 统计实际分配的六个 Tensor，不含输出、布局区及运行时内部占用；空输入为 0。多核的正 groups 表示活跃行容量，0 表示全部 M 行。容量规划见 [Tiling](../../tiling.md#多核活跃行容量)，各缓冲容量见 [多核 Buffer](../../templates/radix-more-core/dag-buffers.md)。

## 验证范围与复现

设备测试需要可用 NPU、已加载的 CANN 环境，以及包含 CANNBotDSL、PyTorch 和 torch_npu 的 Python 环境。**NPU 测试不在仓库普通 CI 中执行**；仓库的 `test_language_route.py` 检查资料入口与语言路由，不代表算子编译或上板验收。

在本目录执行，自包含设备用例不依赖库外输入文件：

```bash
python tests/test_device.py -v
```

| 验证资产 | 范围与边界 |
|---|---|
| `test_device.py` | 直接编译各路由、两种恢复、升降序、原轴索引、输入未修改、更新输入重跑、非末轴与低层 `inner=3` |
| 转置边界与位模式 | 11 dtype 双向尾块、2048/4096/65536 门槛；64 位高低字、浮点 NaN payload/正负零、同址更新输入重跑 |
| 公开入口边界 | 空输入、length=1（尾轴与非末轴）、`stable=False` 仍使用稳定排序 |
| Host 拒绝与容量边界 | 未知路由（包括直接编译）、注入 UB 不足时明确拒绝、默认块容量恰好满足时放行 |
| 局部 SIMD LSD | two-stage 1/2 字节输入的奇偶轮切换、32 位 DMA 构键的有符号/无符号极值、浮点 NaN payload 与正负零按位保留 |
| 64 位 Radix | 跨 2^32、符号高半部、零/非零低半部与完整排序 |
| 多核工作区与提交粒度 | 单次排序 launch、有界 P_act 及尾批复用、更新输入重跑、按实际 Tensor 核算容量；超过复用门槛后算法 scratch 可复用，输出与布局区仍随 M 增长 |
| 选路边界 | 整数插入门槛的下方/命中、two-stage 的首个匹配档位；稳定行分组的 255/256/257 行字节边界与尾批，均检查 Host 路线与精度 |

Argsort、TopK、KthValue 仍属于生成适配指导，本例未提供其独立接口或设备验收；后续须单独覆盖 first_rank/L 边界和输入/输出布局条件。全量性能矩阵不在上述测试范围；改变分核、批次或提交粒度时，性能须累计全部排序与布局搬运，并与可运行基线对比。软件版本、环境路径、个案结果与性能数据留在库外验证报告。

单核 `RadixOneCore` / `NarrowRadixOneCore` 每次仅排序一行，低层构建参数 `groups` 必须为 1；其他值在编译前拒绝。每核仍可顺序处理多行，合批排序使用两阶段模板。
