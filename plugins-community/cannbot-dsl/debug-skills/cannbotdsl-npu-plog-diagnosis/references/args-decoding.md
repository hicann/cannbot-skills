# 参数区(GetArgsInfo)切分 worked example

这是 ABI 类 bug(坏指针、传参错位)的决定性证据来源。plog 里 `GetArgsInfo` / `[AIC_INFO] args` 打印的是**实际进了设备的参数 buffer**——把它和 kernel 的 host 签名对位,错位点就暴露了。

## 槽位算法

参数区是一段扁平的 8 字节槽序列。按 host 函数签名顺序,逐参数切分:

| 参数类型 | 占用槽数 | 布局 |
|---|---|---|
| GM tensor(ndim 维) | `1 + 2*ndim` | `ptr, shape[0..ndim-1], stride[0..ndim-1]` |
| 4D tensor | 9 | ptr + 4 shape + 4 stride |
| 2D tensor | 5 | ptr + 2 shape + 2 stride |
| 1D tensor | 3 | ptr + 1 shape + 1 stride |
| 标量(f32/i64) | 1 | 值本身(按值传) |

## 指针长相判据

- **合法 GM 指针**:`0x12004xxxxxxx` 量级(设备 GM 虚拟地址空间,`0x100000000000` 往上)。
- **垃圾值警报**:
  - `0x800`、`0x4`、`0x20` 这种小整数出现在**本该是 ptr 的槽** → 这个参数没传进来,读到的是相邻槽/未初始化值。
  - `0x7ffcxxxxxxxx` → **host 栈地址**。绝不该出现在设备参数区的 ptr 位置。出现即铁证:host marshalling 漏传了前面某个参数,导致后续全部错位、读了 host 自己的栈。

## 完整样例(真实 507015 案例)

plog:
```
[AIC_INFO] args(0 to 19) after execute:0x12004d600200, 0x4, 0x20, 0x100, 0x80,
  0x100000, 0x8000, 0x80, 0x1, 0x12004ce00000, 0x4, 0x20, 0x100, 0x80,
  0x100000, 0x8000, 0x80, 0x1, 0x12004e200000, 0x20.
[AIC_INFO] args(20 to 39):0x20, 0x80, 0x80, 0x80000, 0x4000, 0x80, 0x1,
  0x120050400000, 0x20, 0x20, 0x80, 0x80, 0x80000, 0x4000, 0x80, 0x1,
  0x7ffc3db504f3, 0x120052600400, 0x80, 0x20.
[AIC_INFO] args(40 to 49):0x20, 0x1, 0x800, 0x7ffc64cb5f30, 0x7ffc64cb5cc0, ...
```

host 签名(从 kernel 的 `@jit`/`@kernel` 形参列表读到):
`(out:4D, q:4D, k:4D, v:4D, scale:f32, mask:2D, block_table:2D-i64, seqused_kv:1D-i64, win_left:i64, win_right:i64)`

按算法切分:

| 槽范围 | 参数 | 校验 |
|---|---|---|
| 0–8 | out(4D) | ptr `0x12004d600200` ✓,shape(4,32,256,128)=`0x4,0x20,0x100,0x80` ✓ |
| 9–17 | q(4D) | ptr `0x12004ce00000` ✓ |
| 18–26 | k(4D) | ptr `0x12004e200000` ✓,shape(32,32,128,128) ✓ |
| 27–35 | v(4D) | ptr `0x120050400000` ✓ |
| 36 | scale(f32) | `0x7ffc3db504f3` —— 标量按值,host 栈值正常,**OK** |
| 37–41 | mask(2D) | ptr `0x120052600400` ✓,shape(128,32)=`0x80,0x20`,stride(32,1)=`0x20,0x1` ✓ |
| **42** | **block_table.ptr** | **`0x800` —— 垃圾!** 应是 `0x12004...` |
| 43+ | seqused_kv 等 | `0x7ffc64cb5f30`(host 栈)—— 全部错位 |

**结论**:参数区从 block_table(槽 42)开始全坏。前面 mask 正常,说明 host marshalling 传到 mask 为止都对,之后 block_table / seqused_kv **没被传进来**。根因锁定在 host 侧:这两个参数是用关键字传的,而 marshalling 只遍历位置参数,把它们漏了 → device 读栈垃圾当指针 → AIV 标量访存越界 → 507015。

## 注意

- `scale` 那种标量槽出现 `0x7ffc...` 是**正常**的(标量按值传,值恰好是 host 栈上算出来的)。只有**本该是 ptr 的槽**出现栈地址才是 bug。
- 切分前务必拿到准确的 host 签名和每个 tensor 的 ndim——kernel 的 `@jit`/`@kernel` 形参列表及其声明的 dtype/ndim 就能给到。ndim 错了整个切分就错位了。
- 参数区会在 fault 时自动落 plog(runtime 行为),不需要额外开关。
