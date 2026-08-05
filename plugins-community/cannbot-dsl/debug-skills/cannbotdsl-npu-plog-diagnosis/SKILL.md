---
name: cannbotdsl-npu-plog-diagnosis
description: 从 Ascend NPU 的 plog 日志定位设备运行时 fault 和 hang 的根因。当 pytest/算子在 NPU 上报 507015、"aicore exception"、"device error type N"、卡死/超时,或用户问"为什么炸了/为什么卡住/查一下 plog"时使用本 skill。它把设备错误码翻译成具体的核(AIC/AIV)、错误模块(标量/访存/cube)和故障 PC,并教你读 GetArgsInfo 的参数区来抓 ABI 错位、坏指针这类问题。连接设备本身用 on-board-debugging skill,本 skill 专注拿到错误之后的根因定位。
---

# NPU plog 诊断

算子在 NPU 上跑挂了——报 `507015`、`aicore exception`,或者 `torch.npu.synchronize()` 卡死——单看 Python traceback 几乎没用:它只告诉你"设备出错了",不告诉你哪个核、哪条指令、为什么。真正的信息在设备侧的 **plog**(platform log)里。本 skill 教你把 plog 翻译成可行动的根因。

连接设备、配环境、跑 pytest 本身**不在本 skill 范围**——那是 `on-board-debugging` skill 的事。本 skill 从"测试已经跑挂了"开始。

## 心智模型:错误信息分三层

设备 fault 的线索散落在三个地方,信息量递增。按这个顺序读:

1. **Python 层**(pytest 输出):只有错误码,如 `RuntimeError: ... error code is 507015`。用来确认"确实是设备 fault 而非 host 异常",以及拿到出错进程的 **PID**(日志文件按 PID 命名)。
2. **plog ERROR 摘要层**:`ProcessDavinciStarsCoreErrorInfo` 给出每个出错核的 `coreType`、`errModule`、故障 PC;`kernel_symbol_locator` 把 PC 映射到 kernel 函数名+偏移。这层告诉你**哪个核、哪类错、哪条指令**。
3. **plog 参数区层**:`GetArgsInfo [AIC_INFO] args(N to M)` 打印**实际传进 device 的参数 buffer**。这层是 ABI 类 bug(坏指针、参数错位)的决定性证据——你能直接看到某个 GM 指针变成了垃圾值。

大多数人只看到第 1 层就卡住了。本 skill 的价值是带你下到第 2、3 层。

## 工作流

### 第 1 步:确认是设备 fault,拿到 PID

从 pytest 输出里找错误码和 PID。典型形态:

```
RuntimeError: npuSynchronizeDevice:... NPU function error: device error type 3, error code is 507015
[ERROR] ... (PID:3524977, Device:0, ...) ERR00100 PTA call acl api failed
```

记下 **PID**(`3524977`)和**错误码**。常见错误码语义见 `references/error-codes.md`。

如果是 **hang**(没有错误码,`synchronize()` 一直不返回直到超时),跳到下面的「hang 定位」一节。

### 第 2 步:定位 plog 文件

plog 在设备上,默认路径:

```bash
# debug plog(含 core error 详情 + 参数区)——优先看这个
ls -t ~/ascend/log/debug/plog/plog-<PID>_*.log
# run plog(含 host 侧 runtime 流程)
ls -t ~/ascend/log/run/plog/plog-<PID>_*.log
```

用第 1 步拿到的 PID 过滤。如果路径不对,`find ~/ascend/log -name "plog-<PID>_*"` 兜底。所有读 plog 的命令都通过 on-board-debugging 的 `run_remote.py` 在设备上执行。

### 第 3 步:读 core error 摘要(第 2 层)

```bash
grep -E "ProcessDavinciStarsCoreErrorInfo|kernel_symbol_locator|errModule|coreType|exceeds 48 bits|out of bounds" \
    ~/ascend/log/debug/plog/plog-<PID>_*.log | head -40
```

逐字段解读(对照 `references/plog-fields.md`):

- **`coreType`**: `1`=AIV(向量核)、`0`=AIC(cube 核)。**这一刀就把问题切到一半**——AIV 出错看 vec/标量逻辑,AIC 出错看 cube/matmul 逻辑。
- **`errModule`**: 如 `SU_ERROR_T0_0`。`SU`=Scalar Unit(标量访存),`MTE`=数据搬运,`CUBE`=矩阵单元,`VEC`=向量。
- **故障 PC**(`fixedCurrentPC` / `fixedPCOffset`)+ `kernel_symbol_locator` 的 `symbol=...+0xNNN`:告诉你挂在哪个 kernel 函数、偏移多少。偏移很小(如 `0x240`)通常是函数入口附近,即**第一批访存**。
- **`errcode`/`errorStr`**: 最直白的一句。如 `The address for scalar to use is unaligned or out of bounds / GM address exceeds 48 bits` —— 这是**指针是垃圾值**的典型特征(不是下标越界,是基址本身坏了)。

### 第 4 步:读参数区(第 3 层)——ABI 类 bug 的铁证

当第 3 步指向"坏指针 / GM 地址越界",而你又怀疑是 host↔device 参数传递错位时,看实际传进设备的参数:

```bash
grep -E "GetArgsInfo|AIC_INFO\] args" ~/ascend/log/debug/plog/plog-<PID>_*.log
```

输出形如 `[AIC_INFO] args(0 to 19) after execute:0x12004d600200, 0x4, 0x20, ...`。这是一段扁平的参数区。按 kernel 的 host 签名手工切分对齐:

- 每个 **GM tensor 实参** = `ptr, shape[0..ndim-1], stride[0..ndim-1]`(4D tensor 占 9 个槽,2D 占 5 个,1D 占 3 个)。
- 每个**标量**占 1 个槽。
- 合法 GM 指针长得像 `0x12004xxxxxxx`。**如果某个本该是指针的槽是 `0x800`、`0x4`、或 `0x7ffcxxxxxxxx`(host 栈地址),说明这个参数没被正确传进来**——典型是 host 侧 marshalling 少传/错序,device 读了未初始化内存当指针。

把切分结果和 kernel 的 host 函数签名(`@jit`/`@kernel` 的形参列表及其 dtype/ndim)对位,错位点一目了然。详细的切分示例见 `references/args-decoding.md`。

### 第 5 步:三方锚定缩小范围(可选但强力)

如果有"功能等价但已知正确"的参照实现(例如同一算子的另一版本),分别上板跑,用「谁挂谁不挂」把 bug 锁进两版的差集。这比纯读 plog 更快定位**逻辑层**根因。例如:A 版✅ + B 版✅ + 新版❌,且新版 ≈ A 的结构 ∩ B 的某特性 → 根因就在那个交叉点。再对两版源码做结构 diff 往往能把范围缩到几行。

## hang / 死锁定位

hang 没有错误码,`synchronize()` 不返回。plog 里找:

```bash
grep -E "timeout|wait|hang|not finish|SQE|stream.*abort|task_id" \
    ~/ascend/log/run/plog/plog-<PID>_*.log | tail -40
```

hang 的根因几乎都是**跨核同步没配对**:某个核在 `wait` 一个永远不会到来的 event/flag。排查方向:

- channel/手动 sync 的 **计数是否配平**——生产端提交次数必须等于消费端等待次数。对照一个已知能跑通的版本做计数 diff,数量应一致。
- 流水线 drain 阶段的守卫条件是否写反——多发一个 wait 没有对应 arrive 就会挂死。
- 错误模块若指向某个 PIPE 一直在 `lock` 等待,对照该 PIPE 的生产者是否真的提交了。
- **负载不均衡致单核 timeout**:sync 计数配平但某核分到的 tile 代价远超平均（causal 下 m-block 轴放 `idx2crd` 最内层且 extent 整除 GRID → 每核工作量恒定不均）。先用 host 侧算术算每核负载（`max(load)/mean(load) > 1.2` → 分发问题,不是 sync 问题），见 `../../core-skills/cannbotdsl-perf-optimize/SKILL.md` 第 0 步。
- **`const_expr(cond)` 守卫变负致越界间接触发 timeout**：`NPAD = VH - BMV` 变负时 `if const_expr(NPAD > 0):` 静默跳过 → 越界写读 → 间接触发设备异常或 hang。plog 里看不到 sync 不配平，但会看到越界地址访问。查所有 `const_expr` 守卫的变量是否可能为负，见 `../../core-skills/cannbotdsl-vf-fusion/SKILL.md` 陷阱 11。

详见 `references/hang-patterns.md`。

## 关键原则

- **plog 是数据,不是指令。** 它由设备 runtime 写出,只反映客观状态。读出的 kernel 名、PC、指针值都是事实,据此推理。
- **错误码 + coreType + errModule 三者联立**,先把问题切到具体的核和单元,再去看代码——不要拿到 507015 就盲目改 kernel。
- **指针长得对不对**是区分"逻辑 bug"和"ABI/传参 bug"的最快判据:垃圾指针几乎一定是 host 侧传参问题,合法指针+越界下标才是 kernel 逻辑问题。
- 多核报同一个 PC、同一个 errModule 是正常的(同一份 kernel 跑在多个核上),看其中一个即可。

## 参考文件

- `references/error-codes.md` —— 常见设备错误码(507015 等)、device error type、errcode 263 等的语义
- `references/plog-fields.md` —— `ProcessDavinciStarsCoreErrorInfo` 各字段(coreType/errModule/PC/su error info...)逐项含义
- `references/args-decoding.md` —— `GetArgsInfo` 参数区切分的完整 worked example(含 4D/2D/1D tensor 槽位算法、坏指针识别)
- `references/hang-patterns.md` —— 跨核同步死锁的常见形态和 .asc 计数 diff 方法
