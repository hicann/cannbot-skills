# AICPU 负载均衡 Metadata：把不均匀任务变成可消费的调度记录

适用于设备上的长度、偏移等值决定任务数、分轮或分核的算子。Host 准备有界存储，AICPU 读取这些值并生成任务记录，AICore 消费记录完成计算。静态且均匀的任务可直接划分，无需为了结构一致增加 AICPU。

下列片段是独立的算法伪码；记录字段和成本模型由当前算子定义，不是一套固定 ABI。

## AICPU 声明接口

```text
from cannbotdsl.aicpu import aicpu_kernel, GmIn, GmOut, I32

class ScheduleArgs:
    lengths: GmIn(I32)
    records: GmOut(I32)
    record_capacity: I32

@aicpu_kernel
def build_schedule(args: ScheduleArgs):
    校验容量、生成记录，并按当前契约返回状态
```

使用不带括号的 `@aicpu_kernel`，参数注解必须可取得实际参数结构类型；上述字段也需要保留实际类型描述对象，不用未解析的字符串注解代替。Host 验证使用 `run_host(...)`，源码检查使用 `generate_sources(launch_mode="interface")`。设备构建接口见下文；其结果是 AICPU launcher，不是 `ProviderCallable`，按 launcher 自身接口管理启动与生命周期。

## 构建与启动：字段、地址、stream 分别明确

```text
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel
from cannbotdsl.aicpu import current_raw_stream

compiled = compile_aicpu_kernel(
    build_schedule, workdir=BUILD_DIRECTORY,
    launch_mode="interface", npu_arch=TARGET_ARCH,
)
compiled.launch(
    current_raw_stream(device_id),
    lengths=lengths.data_ptr(),
    records=records.data_ptr(),
    record_capacity=records.numel(),
)
```

这一路径需要目标 CANN 工具链。传入全部参数结构字段；GM 字段传设备地址，标量按声明的宽度校验。`.launch()` 的返回值表示包装层提交结果，不承载设备函数的业务状态；业务失败写入约定的输出状态字段，并由消费方处理。

生成源码和 Host 解释检查不能替代目标设备编译验证。

## 1. 分轮：用容量不等式选择本轮工作量

假设每个活动序列还有 `remaining[i]` 个 chunk，一个 chunk 占 `units_per_chunk` 个资源单位，本轮容量为 `slots`。

```text
usage(g) = sum(min(remaining[i], g) * units_per_chunk)

while 仍有 remaining[i] > 0:
    如果 usage(1) > slots：采用契约允许的分组策略，或报告容量不足
    否则在 [1, max(remaining)] 找到满足 usage(g) <= slots 的最大 g
    对每个活动序列 i：
        记录 take[i] = min(remaining[i], g)
        remaining[i] -= take[i]
```

在这一模型中 `usage` 单调，可二分查找。实现时先确认每轮至少能推进一个 chunk；不满足时不能继续寻找 `g=0`，否则可能死循环。不同资源成本需替换 usage 公式，不能套用相同常量。

## 2. 分核：按估算成本划连续任务范围

任务数相同不表示耗时相同。例如任务成本 `[8, 1, 1, 1, 1]`，两核按前 3 个和后 2 个分配，负载为 `10/2`；划为 `[0,1)` 与 `[1,5)`，负载为 `8/4`。

下面的贪心方法每次靠近剩余平均成本，并给后续核至少留一个任务。假定成本非负、核数为正。

```text
active = min(task_count, core_count)
start = 0; remaining_cost = sum(costs)
所有核的范围初始化为空
for core in [0, active):
    remaining_cores = active - core
    max_end = task_count - (remaining_cores - 1)
    end = start + 1; segment_cost = costs[start]
    if remaining_cores == 1:
        end = task_count; segment_cost = remaining_cost
    else:
        while end < max_end:
            before = abs(segment_cost * remaining_cores - remaining_cost)
            after = abs((segment_cost + costs[end]) * remaining_cores - remaining_cost)
            if after > before: break
            segment_cost += costs[end]; end += 1
    ranges[core] = [start, end)
    remaining_cost -= segment_cost; start = end
```

这是可解释的启发式，不能宣称全局最优。成本模型应包括实际起步开销、迭代量和尾块差异，并用测量校准。若比较不同拆分候选，可先比较最大核成本，再比较总成本和任务数；保留完整获选记录，避免范围与任务表来自不同候选。

**反例：** 仅按序列数量均分；估算成本忽略尾块；拆分后只更新任务数，未更新任务对应的输入区间与归并规则。

## 3. 落记录：先计数，再检查容量，再写入

以扁平记录为例：头部 `H` 个元素，`R+1` 个轮偏移，每轮 `F` 个字段，每个活动条目 `V` 个字段，共 `A` 个活动条目。

```text
第一遍：计算 R、A，以及生成期间所需 scratch 上限
required_elements = H + (R + 1) + F * R + V * A
检查乘加溢出、输出容量和 scratch 容量
容量不足：写入调用方可识别的失败状态；不得消费不完整记录
第二遍：用相同规则填写轮偏移、任务描述和分核范围
记录有效长度与协议版本；全部字段有效后发布成功状态
```

状态区必须预先分配且所有失败路径都可写，不能在容量连头部都放不下时再向越界位置写错误。`H/F/V` 是当前记录协议定义的字段数。字节容量还要乘元素字节数并满足对齐要求。第一遍和第二遍必须使用相同输入、配置与舍入规则；输入值在两遍之间不得被并发修改。

## 消费边界与检查

- 记录中保存分核使用的 `core_count`；AICore 消费方通过 `get_block_num()` 取得实际启动数量，并在读取按核记录前确认两者一致。不一致时走约定的失败路径，不能静默返回未写输出。该数量指消费方的 AICore block 数。空任务范围不代表取消对应 block；核数变化须重算记录，见 [编译与启动](compilation-launch.md)。
- 偏移单调且末尾等于有效条目数；所有核的范围在有效任务数内，完整覆盖且不重复。
- 生产与消费放在能保证先后关系的同一 stream，或建立明确的跨 stream 依赖。最后写成功状态本身不是内存屏障。
- 失败状态必须被实际消费路径处理，禁止继续算并返回未初始化结果。需要同步向 Host 报错时，明确状态读取和同步方式。
- 缓存生成记录时，长度值变化可能改变全部调度；仅以指针或 shape 为键不够。Host 负责分配的容量问题见 [Workspace](workspace.md)。

**检查：** 空任务、任务数小于核数、极端长短不均、容量恰好足够和差一个元素、零成本任务、相同 shape 但长度值改变。可记录任务数、每核估算成本与状态；诊断不再生成第二套调度记录。
