# 算子实际下发地址获取参考

## 1. 概念定义与适用场景

**算子实际内存地址 = 算子实际下发时输入（input）、输出（output）、工作区（workspace）三类 device 内存地址**（kernel launch 参数中携带的地址），静态图与动态图构成机制不同：

- **静态图（V1 task sink）**：task 下发地址 = 运行时区基址（模型加载时申请）+ 编译期 offset（内存分配器固化），一次加载全程不变
- **动态 shape（RT2.0）**：内存分配下沉为内存算子（Lowering 插入 `AllocMemHbm`/`AllocBatchHbm`/`FreeMemHbm` 等，注册见 GE 仓 [memory_kernel.cc](https://gitcode.com/cann/ge/blob/master/runtime/v2/kernel/memory/memory_kernel.cc)），地址随每次执行动态产生

适用：内存踩踏实址核查、地址对账、数据异常定位、零拷贝地址替换确认。

## 2. 路径一：imas addr（静态图首选）

```bash
cd imas && ./imas addr <debug日志目录>/plog/plog*
```

输出两部分（同源写 `mem_report/imas_mem_addr.csv`）：

**① 内存区基址（加载时实际申请）**

```
Process_11394,0,baseaddr,memaddr,0x12c041600000,mem_size,268470272,type,F,valid memaddr range[0x12c041600000~0x12c051608800]
```

| type | 含义 | 日志来源（函数名/关键字） |
|------|------|------------------------|
| F | feature map 静态图内存区 | `InitFeatureMapAndP2PMem ... MallocMemory type[F]`（DavinciModel） |
| V | 变量区（虚拟预留，按需占用） | `ReserveVirtualMemory virtual_active_addr_base` |
| W | workspace 独立分区 | — |

**② 算子地址映射行（input/output/workspace 三类齐备）**

```
name_<节点名>_<input_N/output_N/workspace_N>,type,F,size,S,memaddr,0x<实际地址>
```

**对账方法**：实际地址 − F 区基址 = 编译期 offset（imas_mem_graph.csv 同名块 offset），例：

| 算子 | 实际地址 | = F base + offset | 编译期 offset |
|---|---|---|---|
| allreduce_op input_0 | 0x12c041600000 | +0 | 0 |
| allreduce_op output_0 | 0x12c045600200 | +0x4000200 | 67109376 ✓ |
| allreduce_op workspace_0 | 0x12c049600400 | +0x8000400 | 134218752 ✓ |
| atomic_memset0_01 workspace_0 | 0x12c041600000 | +0 | 0（清零目标 = allreduce input 块，时序错开） |

零拷贝块（r:c:z:s 中 z=1）的地址分两层：imas addr / `logical_addr` 给出的是**加载期算出的逻辑地址**（编译期只有 offset，加载时换算为绝对地址；**零拷贝复用模式下该地址不实际分配内存**——F 区只申请非零拷贝部分，日志 `need <实际申请>/<布局总量> for feature-map without zero-copyable memory` 可判定，零拷贝 logical_addr 超出实际 F 区范围属正常）；**运行时实际地址的唯一权威来源是 `ConstructZeroCopyIoActiveBaseAddrs` 日志的 `user_addr`**（每次执行把当次用户 tensor 地址写入 active_mem_base，kernel 直接在该地址读写，logical_addr 从未被真正访问）：
- `[Input] index:N, user_addr:0x..., pls:device, is copy host:0` → 输入零拷贝直读用户 buffer
- `[Output] index:N, user_addr:0x...` → 输出零拷贝直写用户 buffer
- `is copy host:1` → 用户给 host tensor，走临时 aclrtMalloc + H2D 拷贝（**非零拷贝**，地址随执行变化）
- 无用户输入（离线评估）→ 不替换，逻辑地址即实际地址（grep `\[ZCPY\]` 无输出可判定此场景）
- **多轮执行时 input/output 槽值是否恒定由调用方 buffer 使用模式决定**（输入复用同一块则恒定；输出交替分配则轮换），非 GE 内部机制；两者经 `id` 与逻辑地址关联：`实际地址 = active_mem_base[id] = 当轮 user_addr`
- **未开零拷贝复用的分支**（如离线评估）：F 区按含零拷贝的布局总量**全量申请**（`MallocMemory type[F] mem_size[布局总量]`），零拷贝块有实际内存，逻辑地址即实际地址

**展示标注规则（必须遵守）**：地址表中零拷贝槽需同时展示**实际下发地址**、**logical_addr 占位**（含 id 与 refreshable）和 **user_addr 来源**（先 logical_addr 后 user_addr），格式示例：

```
    input_0   0x12c041241200  (8K)
              ← logical_addr 0x12c04124f000（未分配，零拷贝占位，id:3，refreshable=1）
              ← user_addr [Input] index:2 = 0x12c041241200（用户输入 buffer，即实际下发地址）
    input_1   0x12c041243600  (4B)
              ← logical_addr 0x12c041243600（已分配 W 区常量，refreshable=0 直传，id:9）
```

**refreshable 含义**（第 1 级 `[Args][Init]` 字段，**按 mem_type 静态判定的"支持刷新"能力标志，非"本轮需要刷新"**）：`1` = 所在区类型为 FeatureMap/ModelIo，**参与地址刷新管理**（id 取匹配的 allocation，实值经 `active_mem_base[id] + offset` 每轮重映射，值可能不变）；`0` = 常量等类型**不走映射**（id 强制 absolute，logical_addr 直传，值恒定）。所有槽位（含已分配 F 区槽）均需展示 refreshable。

**"需要刷新"标记 = UpdatePolicy（模型级，每轮执行时 CalcUpdatePolicy 动态计算，与 refreshable 正交）**，日志关键字 `Begin to update model args, policy <名>, fm_hit_count 0x.., model_io_hit_count:0x..`：

| policy 值 | 触发条件 | 行为 |
|----------|---------|------|
| `all-one-time`（kInitOneTime） | 首轮 | 全量初始化 |
| `fm-and-model-io`（kUpdateFmAndModelIo） | fm 区 base 变化 | 刷 fm + io 槽 |
| `model-io`（kUpdateModelIo） | 仅 model io base 变化 | 只刷 io 槽 |
| `no-need-update`（kNoNeedUpdate） | base 均未变 | 本轮跳过刷新 |

多轮执行时 policy 序列可判断"哪些区在变"：如 1_pro.log 为 `all-one-time` + 10×`model-io`（fm 恒定、仅用户 io buffer 每轮换）。

非零拷贝槽（GE 已分配内存）无需 user_addr 行，仅列实际地址。

**故障判定**：零拷贝复用模式下，**未分配的 logical_addr 若出现在实际下发中（dfx 槽实值 / D2H 回读值）即为故障特征**——算子将访问未分配内存导致执行报错，说明零拷贝替换未生效（地址刷新链路问题：刷新 kernel 未执行/时序错乱/替换遗漏），需排查；正常状态零拷贝槽实值 = 当轮 user_addr。

输出侧零拷贝经 `FilterZeroCopyAddrs`+`CpuTaskModelZeroCopy` 直写用户 buffer（日志 `Set CpuKernel model zero_copy task` 标记生效）。

## 3. 路径二：GE_DAVINCI_MODEL_PROFILING=2（V1 dfx 全量 args 表 + D2H 回读）

imas addr 只有地址映射；要看 **device 侧实收的 args 表内容**（逐 8 字节 input/output/workspace 地址 + D2H 回读比对），用 dfx 开关。

**第一步（必须先做）：从日志检查开关是否已开启**：

```bash
# 查开关状态行（InitModelProf 打印，INFO 级）
grep "Init davinci_model profiler flag" <日志>
# 输出: Init davinci_model profiler flag:0, get device model args table flag:1.
```

| 判定 | 含义 | 动作 |
|------|------|------|
| `get device model args table flag:1` | **已开启**（环境变量=2） | 直接进入下方分析 |
| `get device model args table flag:0` 或无此行 | **未开启** | **提示用户开启后重新运行取新日志**：`export GE_DAVINCI_MODEL_PROFILING=2` + `export ASCEND_GLOBAL_LOG_LEVEL=1`（两者都要；旧日志无 dfx 表，无法事后补） |

**第二步：开启方法（未开启时）**：

```bash
export GE_DAVINCI_MODEL_PROFILING=2   # 取值 2 才是 args 表打印；1 仅 profiling 统计
export ASCEND_GLOBAL_LOG_LEVEL=1      # 需 INFO 级
```

生效链路（ModelArgsManager）：`DavinciModel::InitModelProf` 读环境变量 → dfx_info_ 继承 → 执行时 LaunchKernel 后打印。关键输出（按日志关键字 grep）：

| 日志关键字 | 内容 |
|-----------|------|
| `Init davinci_model profiler flag` | 开关生效确认（第一步已查） |
| `[Args][Init] op_name:%s, pls dev addr:0x%llx` | 每算子 args device 地址（有 args 刷新时） |
| `Alloc model args ... addr=0x%llx for model %u(%s)` | 模型级 args 区地址 |
| `Print kernelLaunch Op args: model_offset_args_device_addr/.../workspace_addr/tiling_addr` | kernel launch 完整地址结构 |
| `Print model args host table ... model device_args_addr is 0x...` | **逐 8 字节 device args 表（实收 io 地址）** |
| `Print different args: device addr is ..., host addr is ...` | D2H 回读与 host 比对差异 |

**D2H 回读机制（device 实收地址的权威来源）**：dfx 开启后 `PrintKernelLaunchArgsDfxInfo` 在每次地址刷新（LaunchKernel）后自动执行：①`aclrtSynchronizeStream` 等待刷新 kernel 完成 → ②`aclrtMemcpy(ACL_MEMCPY_DEVICE_TO_HOST)` **整表回读 device args 表**（model_args_device_addr，长度 = 槽位数 × 8）→ ③逐槽与 host 表比对。判读规则：
- **`Print different args` 0 条** → 回读与 host 逐槽一致，`Print model args host table` 打印的槽值即 **device 实收地址**（D2H 已验证，如 1_pro.log 实测 264 条 0 差异）
- **出现 `Print different args`** → 该行 `device addr` 列为 D2H 回读的 **device 实际持有地址**（权威值），`host addr` 为 host 侧写入值——两者不一致需排查（刷新 kernel 未生效/时序问题）

触发条件：`get_model_args_device_table_flag && logLevel_ <= DLOG_INFO`，且模型有 args 刷新流程；纯固定地址模型走 `There are no args that need to be managed` 则无表打印。

**脚本一键提取**（上述全流程固化，含开关判定/槽位换算/分配状态/refreshable/user_addr 关联/D2H 判读）：

```bash
python3 graph/ge-memory-analysis/scripts/static_addr_extract.py <日志文件> [算子名过滤]   # 在 cannbot-skills 仓根目录执行
```

**args 表槽位 ↔ 算子 input/output 对应关系**（两级 `[Args][Init]` 日志建立）：

- 第 1 级（args_io_addrs_updater）：`op_name:X, logical_addr[i]:<加载期算出的逻辑地址>, id:N, offset:..., refreshable:...`——`logical_addr[i]` 按序 = 该算子 input_0..N / output_0..M；`refreshable=0` 为常量槽（值恒定）
- 第 2 级（GenModelArgsRefreshInfosForTask）：`op_name:X, pls dev addr:<段基址>, task args refresh info:[id, offset, io_index:K, args_offset:8K]`——io_index 为算子第 K 个 io，args_offset 为段内字节偏移
- **槽位公式**：`host table index = (算子段基址 − model_args_table_addr) / 8 + io_index`
- 注意：args 槽实值 = `active_mem_base[id] + offset` 的运行时映射（dfx 表含 `Print active mem base`），与 imas addr 逻辑地址在 smp≠0 时可能不同，`id` 为关联键
- **槽值恒定性判读**（多轮执行时）：input/常量/中间 tensor 槽的 active base 恒定（静态图内部基地址不变）；**OUTPUT 零拷贝槽每轮 = 当次用户传入的输出 buffer 地址**（`ConstructZeroCopyIoActiveBaseAddrs:[Output] index:N, user_addr:0x...` 日志直接给出，GE 把 user_addr 写入 active base，输出直写用户 buffer）——多轮取值不同是用户侧 buffer 地址在变，非 GE 内部轮换；smp 不影响该行为

## 4. 路径三：动态 shape RT2 执行路径

RT2.0 无模型级 args 表，地址由内存算子在执行时产生。**三类地址获取方法**：

**① output 地址（日志直接给出）**——内存算子为算子输出分配：

```
[KernelTrace][AllocMemHbm_<算子名>_<唯一id>][MEM]Alloc memory at stream N, block ..., address 0x<device地址>, size ..., tensor size ..., index N
```

`AllocMemHbm_<算子名>` 即**请求分配的算子**（其 output 落在该地址）；`AllocBatchHbm_*` 为批量分配；对应 `FreeBatchHbm`/`FreeMemHbm` 可追踪地址生命周期。

**①′ 下发实值（最直接，优先用）**——launch 时直接列出 input/output 的 device 地址实值：

```
[KernelTrace][LaunchKernelWithHandle_<算子名>_<id>]Input/Output addresses: <input地址...> <output地址...> , Input/Output sizes: ...
```

**input_N/output_N 归属划分**：同算子的 `Launch args-args addresses info` 行中 `ArgsInputsAddr(address/length)`、`ArgsOutputsAddr(address/length)` 给出输入/输出指针数（length/8），据此切分地址列表——前 N 个为 input_0..N-1，其余为 output_0..M-1。常量类 input 显示为 V 区地址（0x12c1...）；Reduce 类算子的 input_0 可能是 host/axis 区地址（如 0x3ffff...），数据 input 在 input_1。

**② input 地址（= 前驱的 output 地址）**——RT2 中 tensor 经数据流传递，算子输入地址即前驱算子 AllocMemHbm 分配的地址（或图输入/常量的加载地址：`Load file constant ... to addr` / `Copy constant ... to device addr`，DEBUG 级）。按计算图拓扑找到前驱算子后查其 AllocMemHbm 记录。

**③ workspace 地址（同样走 Alloc 算子）**——有独立 workspace 的算子，Lowering 同样插入 Alloc 请求，日志格式同 ①；elementwise 类小算子常无独立 workspace（如 dy.log 全图无 workspace 分配记录），属正常。

**launch 下发结构（辅助，host 侧）**：

```
[KernelTrace][LaunchKernelWithHandle_<算子名>_<id>]Launch args-args addresses info:
  args-args address base 0x<host>, args-args-size S
  ArgsCompiledAddr(address/length), ArgsInputsAddr(address/length), ArgsOutputsAddr(address/length)
```

即 launch args 的 host 段布局（InputsAddr/OutputsAddr 槽位长度 = 指针数 × 8，槽位内为 device 地址值，日志不展开具体值；需实值时用路径三 ①② 或 DEBUG 级 gdb）。RUNTIME 侧另有 `StreamLaunchKernelWithHandle: kernel info : ... addr1=0x...` 下发记录。

**获取方法汇总**：

```bash
# 方法一（结构化，推荐）：imas report → imas_mem_pool.csv
#   name 列 = 算子名，address 列 = 实际下发地址（output 侧），含 lt/size/realsize
cd imas && ./imas report <日志>
# 方法二（原生日志）：算子级分配/释放 + 物理页
grep 'KernelTrace.*Alloc.*address\|MallocPhysicalMemory\|Launch args' <日志>
# 方法三（脚本一键提取，按执行序 step 逐条输出，每条含时间戳/算子/档位/input_N/output_N 地址，
#         可直接对应哪次执行的地址；同档多轮地址是否漂移一眼可见）：
python3 graph/ge-memory-analysis/scripts/addr_extract.py <日志>   # 在 cannbot-skills 仓根目录执行
# 用户指定输出地址（hybrid 执行器，DEBUG 级）
grep 'user did specify output memory' <日志>
```

多迭代执行时同算子多次分配（地址可能复用），按 imas_mem_pool.csv 的 lt（生命周期）区分窗口。

## 5. 路径适用性

| 场景 | 路径一 imas addr | 路径二 dfx 开关 | 路径三 RT2 日志/imas pool |
|------|-----------------|----------------|--------------------------|
| 静态图（V1 task sink，如 log.tar.gz allreduce） | ✓ 首选（input/output/workspace 全量映射） | ✓（需有 args 刷新） | ✗ |
| RT2.0 动态执行（如 dy.log） | ✗（无静态区映射） | ✗ 不经 ModelArgsManager | ✓ 首选（output 直接查、input 查前驱、workspace 查 Alloc 记录） |

## 6. 注意

- `GE_DAVINCI_MODEL_PROFILING` 为内部调试环境变量（无对外 env_vars 文档），版本间可能变化
- imas addr 依赖 debug plog（info 级即可）；地址打印来自 GE 加载路径日志，缺失时检查日志阶段
- RT2 的 `Launch args` host 段布局打印为 executor tracer（KernelTrace）系列，需确认版本是否携带
- 多流场景地址生命周期与流并行相关，对账时结合 streamid（imas_mem_graph.csv 算子行）或 Alloc 日志的 stream 字段
