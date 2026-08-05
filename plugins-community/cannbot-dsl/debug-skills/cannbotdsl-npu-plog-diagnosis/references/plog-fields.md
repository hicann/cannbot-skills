# plog 关键字段逐项解读

设备 fault 时,debug plog(`~/ascend/log/debug/plog/plog-<PID>_*.log`)里有两条关键 ERROR 行。本文逐字段拆解,配真实样例。

## 行 1:`ProcessDavinciStarsCoreErrorInfo`

每个出错的核都打一行。样例(已折行):

```
ProcessDavinciStarsCoreErrorInfo:The error from device(chipId:0, dieId:0),
serial number is 46, there is an aicore error exception, core id is 0,
error code = 263, dump info: pc start: 0x120041000000, current: 0x12004100023c,
sc error info: 0xffffffffffff, su error info: 0xb7fbdefb...,0xdbff4fda...,
mte error info: 0x989deb7f..., vec error info: 0, cube error info: 0,
l1 error info: 0xdf7b..., aic error mask: 0x395856, para base: 0x12004ba00000, ...
```

| 字段 | 含义 / 怎么用 |
|---|---|
| `core id` | 出错的物理核号。多核报错正常(同份 kernel 多核跑),看一个即可。 |
| `error code` | 设备侧 errcode,见 `error-codes.md`。`263`=标量访存非法。 |
| `pc start` / `current` | kernel 起始 PC 和故障 PC。`current - start` 就是函数内偏移(下面 `kernel_symbol_locator` 会直接给符号+偏移)。 |
| `su error info` | **非 0 说明 Scalar Unit 报错**——标量 load/store 出问题。这是坏指针/标量访存越界的信号。 |
| `mte error info` | 非 0 说明 MTE(数据搬运 DMA)报错——GM↔L1/L0 拷贝地址或参数非法。 |
| `vec error info` | 非 0 说明向量单元报错。 |
| `cube error info` | 非 0 说明 cube(matmul)单元报错。 |
| `l1 error info` | L1 buffer 相关。 |
| `para base` | 参数区基址。`GetArgsInfo` 打印的就是从这里开始的内存。 |

**读法**:看哪个 `* error info` 非 0,就知道是哪个执行单元挂的。`su` 非 0 + errcode 263 = 标量访存读了非法地址,十有八九是坏指针。

## 行 2:`kernel_symbol_locator`

把故障 PC 映射到符号:

```
[Dump][Exception] Error PC information. coreId=62, coreType=1,
originalCurrentPC=0x120041001264, fixedCurrentPC=0x120041001240,
rawPCOffset=0x240, fixedPCOffset=0x1240, errModule=SU_ERROR_T0_0.

[Dump][Exception] Error symbol information. coreId=62, coreType=1,
symbol=_Z18kernel_fa_kernel_0..._mix_aiv+0x240.
```

| 字段 | 含义 / 怎么用 |
|---|---|
| `coreType` | **`0`=AIC(cube 核),`1`=AIV(向量核)**。决定去看 cube 逻辑还是 vec 逻辑。 |
| `errModule` | 错误模块。`SU_ERROR_*`=标量单元,`MTE_ERROR_*`=搬运,`CUBE_ERROR_*`=矩阵,`VEC_ERROR_*`=向量。 |
| `symbol` | 故障 kernel 函数名。`_mix_aiv` 后缀=AIV 侧函数,`_mix_aic`=AIC 侧。函数名里能看到参数类型签名(mangled)。 |
| `+0xNNN` / `fixedPCOffset` | 函数内偏移。**偏移小(几百)= 函数入口附近 = 第一批指令/访存**。kernel 刚进来就挂,通常是最早读的那个 GM 标量(如读 seqused_kv/block_table 指针)。 |

## 联立推理示例

`coreType=1`(AIV)+ `errModule=SU_ERROR_T0_0`(标量单元)+ errcode `263`(地址非法)+ `errorStr` "GM address exceeds 48 bits" + symbol `..._mix_aiv+0x240`(AIV 函数入口附近):

→ **AIV 核在函数刚开始时,用一个垃圾基址做标量 GM 读**。AIV 侧的 GM 标量读通常就是读那几个 int64 元信息张量(block_table / seqused_kv 之类)。指针是垃圾 → host 没把这个张量正确传进来 → 去第 4 步看参数区验证。

这一串推理不碰 kernel 计算逻辑,直接指向 host 传参,省掉所有在 kernel 里瞎找的时间。
