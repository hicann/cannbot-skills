# CANNBotDSL Sort 示例

本例以 CANNBotDSL 实现稳定 Sort，返回排序值和原轴内索引。包含插入、单核 Radix、多核 Radix 和两阶段排序；公开入口负责非末轴布局转换。

本目录演示统一范式的 Sort 配置（first_rank=0、L=N、输出 values/indices）。同一主体用于生成 Argsort、TopK 和 KthValue：按 [公共映射](../../../../generic/output-mapping.md) 配置输出字段与名次区间，在最后写回应用区间，非尾轴使用共用布局规范化，不按长轴/短轴设置专用非末轴模板；连续尾轴及等价布局不启动转置。语言表达见 [输出适配](../../implementation-guide.md#统一输出适配)。公开 Python 入口仍仅提供 Sort，不是其他接口的现成运行库。

## 算法模板

| 模板文件 | Kernel | Host 路由 | 适用场景 |
|---|---|---|---|
| [sort_small_axis_insertion.py](sort_small_axis_insertion.py) | InsertionSort | `ROUTE_INSERTION`（1） | 批量短轴插入排序 |
| [sort_radix_one_core.py](sort_radix_one_core.py) | RadixOneCore | `ROUTE_RADIX_ONE_CORE`（2） | 一个核处理可放入 UB 的单行 |
| [sort_small_axis_two_stage.py](sort_small_axis_two_stage.py) | SmallAxisTwoStage | `ROUTE_TWO_STAGE_INVERSE`（4）/ `ROUTE_TWO_STAGE_PACKED`（5） | 多行合批排序，再恢复各行；4 为逆映射，5 为紧凑键二次排序 |
| [sort_radix_more_core.py](sort_radix_more_core.py) | RadixMoreCore | `ROUTE_RADIX_MORE_CORE`（3） | 单 kernel 内完成长轴分块、int64 全局桶计数/前缀及逐字节稳定散射，不设 32 位计数变体 |

路由编号由 `sort_host.py` 的 `ROUTE_*` 常量统一维护；`build_program` 与编译期分派均拒绝未知编号。多核路线即使回退到默认 1024 元素块，也会核对 UB 预算，容量不足时在编译前报错。

每条排序路线均在一次 kernel launch 内处理本次调用的全部行。连续末轴（I=1）直接排序；非末轴 N>1 且 I>1 时，先用 DSL 转置 `[O,N,I]→[O,I,N]`，沿连续末轴排序，再分别恢复 values 和 indices，共四个 kernel。N=1 或轴后仅 singleton 维度时不做物理转置；表中的 Host 路由号不是 kernel 数量。

这里四个 kernel 的计数只对应本例 L=N 且确需转置的配置。生成 L≠N 的接口时按 [布局判定](../../../../generic/shape-preprocessing.md#物理转置的判定与验收) 分别判断输入与输出；L=1 时输出恢复只需 reshape，不能直接套用本例的 `needs_transpose` 标志给两份输出各启动一次转置。

归并路线尚未适配。非末轴通过共用布局包装复用表中主体。two-stage 的两种恢复属于同一模板，通过编译期 `rank_inverse` 选择。局部单行 Radix 使用独立模板。轴长一按普通插入路线处理，比较循环为空。选路门槛与恢复阈值见 [选路与恢复策略](../../tiling.md#两阶段恢复硬约束与实现策略)。

## 入口与公共原语

| 文件 | 职责 |
|---|---|
| [entry.py](entry.py) | 公开入口、输入校验、Tiling 选型、workspace、编译及 Native 导出 |
| [sort_host.py](sort_host.py) | `@host` 按路由启动一个计算 kernel；多核 Radix 的逐字节阶段在 Kernel 内完成 |
| [sort_common.py](sort_common.py) | 容量常量、dtype 宽度、单调排序键和字节提取 |
| [sort_transpose.py](sort_transpose.py) | 非末轴输入转入及 values/indices 转回的共用 DSL 分块转置 |
| [radix.py](radix.py) | 共用的 SIMD 字节 permutation 与局部 LSD 排序原语 |

读取单个算法时，从对应模板开始；路由条件见 `entry.py`，实际启动见 `sort_host.py`，多核阶段依赖见 `RadixMoreCore.__call__`。

## 使用

```python
from entry import sort
values, indices = sort(x, dim=-1, descending=False, stable=True)
```

支持 rank≥1 的连续 NPU 张量、合法正负轴、升降序和 11 dtype：float16/bfloat16/float32、int8/16/32/64、uint8/16/32/64。输出 shape 与输入相同，indices 为 int64 原轴内坐标；`stable=False` 不要求等值元素顺序，本例仍复用稳定排序实现并返回稳定结果，不另设非稳定路线。空输入直接返回空输出。浮点正负零保留原位模式，NaN 按方向放在端点并保持输入次序。

公开入口对排序主体始终传 `inner=1`，非末轴经过 `PreparedSort` 的共用转置包装。Kernel 保留 `inner!=1` 的低层调用能力，独立验证见 `tests/test_device.py::DeviceTests.test_strided_schedules`；该直接调用不改变公开入口的统一布局规范。

复用编译与缓冲时使用 `PreparedSort.run()` / `close()`；下一次运行覆盖前次输出，关闭前须等待设备完成。公开 `sort()` 等待完成后关闭程序。

容量取设备 UB 与核数，two-stage 批次同时核对计数和打包键上限。浮点批处理候选域为 N≤256；全部 buffer 预算和同步见 [implementation-guide](../../implementation-guide.md)。MergeSort 尚未提供模板；当前多核路径使用全 launch 屏障组织阶段，不使用 lookback 协议，也不按字节拆分 kernel。

浮点插入门槛沿用参考基线，性能拐点尚未确定；低并发已有 Radix 更快的样例，不应将该门槛当作已验证的性能推荐。选路策略、证据边界与待调优状态见 [选路策略](../../tiling.md#dtype-路由表)。

`@export("sort_example")` 声明六个配置：float32 插入的两个方向、float32 局部 SIMD、float32 two-stage 逆映射、int32 two-stage 紧凑键、float32 多核 Radix。`@export("sort_transpose")` 另外声明 11 dtype 的通用转置配置；非末轴 Native 使用时须同时发布并注册这两个导出组。具体 width/batch/direction 见 `entry.py`；其他排序配置通过运行时编译使用。

## 多核 Radix 如何阅读

先读 `RadixMoreCore.__call__`：每轮处理排序键的一个字节，顺序固定为 `histogram → prefix → scatter`。偶数轮从 ping 读、向 pong 写，奇数轮反向；首轮从原输入生成保序键和原轴索引，末轮按原轴索引读取原值写入输出。

- `histogram` 对每个分块生成 256 桶计数，并将「块内原位置 × tile_size + 桶内排名」打包到 uint32 的 ranks；散射时用除余解码。tile_size 最大 8192，打包值小于 2^26。
- `prefix` 先求每桶跨所有分块的总量，再做 warp 内扫描及 warp 间累加。每个分块的偏移包含所有较小桶及此前分块的同桶元素数。
- `scatter` 将全局偏移与桶内排名相加，得到唯一目标位置；同时传递原轴索引，保证稳定性。最终值从 x 读取，保留原始位模式。

`WARP_LANES`/`WARP_SCAN_STEPS` 表达 32 lane 的扫描结构，`HISTOGRAM_THREADS` 表达每桶一线程，`RADIX_SCATTER_THREADS` 表达散射线程数；`DEFAULT_RADIX_TILE` 是多核默认块长，`INSERTION_MAX_BATCH_ROWS`/`INSERTION_THREADS` 分别限制插入批次与线程数；`LOCAL_RADIX_ALIGNMENT` 是局部排序的元素对齐粒度。这些常量的角色不同。三段之间的屏障始终在任务分支外，空闲核也必须参加。

## 同步约束

多核路径通过 `global_sync_all()`、SIMT L2 访问和生产 VF 的 `threadfence()` 建立阶段依赖；启动量限制在当前流有效 AIV 核数内。所有核均须到达屏障，不能把屏障放入有任务分支。

## 工作区实现状态

`PreparedSort` 的 route 3 保留单次 launch 和全 M 行 scratch。算法 GM 临时区为 `M×N×(2S+20)+16×M×ceil(N/T)×256` 字节；`workspace_bytes` 统计这些活跃 Tensor，不包含最终输出、布局区、缓存池或运行时内部占用；空输入为 0。非多核路线只传最小占位区。

当前全 launch 屏障实现尚未支持有界 P_act 批次复用。生成时应明确实际分配规模；内存优化须同时保持排序主体、同步和已验证的性能，不能以 Host 逐批 launch 代替，或把简单套入 kernel 的批次循环直接当作优化。具体算法和预算见 [活跃批次实现](../../implementation-guide.md#当前活跃批次的工作区复用)。

## 验证范围与复现

设备测试需要可用 NPU、已加载的 CANN 环境，以及包含 CANNBotDSL、PyTorch 和 torch_npu 的 Python 环境。**NPU 测试不在仓库普通 CI 中执行**；仓库的 `test_language_route.py` 检查资料入口与语言路由，不代表算子编译或上板验收。

在本目录执行，自包含设备用例不依赖库外输入文件：

```bash
python tests/test_device.py -v
```

| 验证资产 | 范围与边界 |
|---|---|
| `test_device.py` | 直接编译各路由、两种恢复、升降序、原轴索引、输入未修改、更新输入重跑、非末轴与低层 `inner=3` |
| 公开入口边界 | 空输入、length=1（尾轴与非末轴）、`stable=False` 仍使用稳定排序 |
| Host 拒绝与容量边界 | 未知路由（包括直接编译）、注入 UB 不足时明确拒绝、默认块容量恰好满足时放行 |
| 64 位 Radix | 跨 2^32、符号高半部、零/非零低半部与完整排序 |
| 多核工作区与提交粒度 | 全行 scratch、单次排序 launch 和更新输入重跑；有界 P_act 复用未实现，不宣称空间不随 M 增长 |
| 选路边界 | 整数插入门槛的下方/命中、two-stage 的首个匹配档位；浮点保留策略及紧凑键最大值 65535 的整批/尾批，均检查 Host 路线与精度 |

Argsort、TopK、KthValue 仍属于生成适配指导，本例未提供其独立接口或设备验收；后续须单独覆盖 first_rank/L 边界和输入/输出布局条件。全量性能矩阵不在上述测试范围；改变分核、批次或提交粒度时，性能须累计全部排序与布局搬运，并与可运行基线对比。软件版本、环境路径、个案结果与性能数据留在库外验证报告。

单核 `RadixOneCore` 每次仅排序一行，低层构建参数 `groups` 必须为 1；其他值在编译前拒绝。每核仍可顺序处理多行，合批排序使用两阶段模板。
