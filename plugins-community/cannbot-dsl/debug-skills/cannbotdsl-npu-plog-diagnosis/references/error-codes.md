# 常见 NPU 设备错误码语义

设备 fault 时 Python 层抛 `RuntimeError: ... device error type N, error code is XXXXXX`。下面是高频码的含义。错误码是 host runtime 给的粗分类,真正的细节在 plog 的 `errcode`(设备侧)里。

## 顶层 error code(host runtime, `rtDeviceSynchronizeWithTimeout` 返回)

| code | 名称 | 含义 | 第一反应 |
|---|---|---|---|
| `507015` | aicore exception | AI Core 执行时硬件异常(访存越界/非法指令/未对齐)。**最常见**。 | 读 debug plog 的 `ProcessDavinciStarsCoreErrorInfo`,看 coreType + errModule |
| `507018` | aivec exception | 向量核异常,通常 vec 指令操作数/地址非法 | 同上,coreType 多为 1(AIV) |
| `107015` | 等待超时 | task 超时未完成,常见于 hang/死锁 | 走「hang 定位」,不是 fault |
| `507899` | device 通用错误 | runtime 内部错误,信息少 | 看 run plog 的完整调用链 |

`device error type N`(PTA 层的分类)只是粗标签:`type 3` 一般对应 aicore exception。以 plog 里的具体 errcode 为准。

## 设备侧 errcode(plog `ProcessDavinciStarsCoreErrorInfo` 里的 `error code = N`)

| errcode | 含义 |
|---|---|
| `263` | 标量/访存地址非法。`errorStr` 通常是 `The address for scalar to use is unaligned or out of bounds / The GM address exceeds 48 bits, or the on-chip buffer address exceeds the size of the buffer`。**强烈指向坏指针或 ABI 错位**(基址本身是垃圾),而非单纯下标越界。 |
| 其它 | 对照 `errorStr` 字段直读,它通常是一句人类可读的描述。 |

## 怎么用这张表

1. Python 层错误码 → 确认是 fault 还是 timeout(hang)。
2. fault → 去 debug plog 拿设备侧 errcode + errorStr。
3. errorStr 提到 "exceeds 48 bits / out of bounds for scalar" → 高度怀疑 **host 侧传参把某个 GM 指针传成了垃圾值**,直接跳到参数区(SKILL 第 4 步)验证。

## 关键区分:坏指针 vs 越界下标

- **坏指针**(基址错):`errorStr` 说 "exceeds 48 bits"。参数区里能看到某个 ptr 槽是 `0x800` / host 栈地址。根因在 **host marshalling / 传参**,不在 kernel 计算。
- **越界下标**(基址对、offset 错):指针合法但加了过大偏移。根因在 **kernel 的下标计算**(分核公式、循环上界、tile 坐标)。这类要回去读 kernel 逻辑,核对下标推导。

先判断是哪一类,能省掉大量走错方向的时间。
