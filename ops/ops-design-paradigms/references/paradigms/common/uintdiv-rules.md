# UintDiv 快除规则

> 固定除数除法/取余：`GetUintDivMagicAndShift` + `UintDiv` 组合，把运行期除法替换为乘法+移位。坐标解码（flat index → 多维坐标）、按固定块数切分等场景的标准手段。

---

## 1 总览

### 1.1 API 清单

| API | 执行域 | 作用 |
|-----|--------|------|
| `AscendC::GetUintDivMagicAndShift<T>(magic, shift, divisor)` | **仅 scalar 端**（`__aicore__`，host 禁止） | 预计算除数 `divisor` 的 magic/shift 快除参数 |
| `AscendC::Simt::UintDiv<T>(n, magic, shift)` | **仅 VF 端**（`__simt_vf__`） | 返回 `n / divisor`（等价除法商） |
| 取余 | VF 端组合 | `n % divisor = n - UintDiv(n, magic, shift) * divisor` |

```cpp
// scalar 端（Process 内）：预计算
IDX_T magic = 0;
IDX_T shift = 0;
AscendC::GetUintDivMagicAndShift<IDX_T>(magic, shift, static_cast<IDX_T>(divisor));

// VF 端（kernel 内）：O(1) 快除
IDX_T q   = AscendC::Simt::UintDiv<IDX_T>(n, magic, shift);   // n / divisor
IDX_T rem = n - q * divisor;                                   // n % divisor
```

### 1.2 能力边界（负面清单）

- **`GetUintDivMagicAndShift` 出参为 `T&`，禁止直接传 `__ubuf__` 左值**——必须先写入 Local Memory 标量，再写入 UB/寄存器中转；编译器会拒绝 `__ubuf__` 引用绑定
- `GetUintDivMagicAndShift` 是 device 端 API，**禁止在 host（tiling）侧计算**——magic/shift 只能在 kernel 的 scalar 端（`__aicore__` 函数）预计算
- `UintDiv` 为 **VF 专用**，在 scalar 端调用无效
- `divisor = 0` 未定义——空张量场景（如 outSize=0）必须在预计算前早退，否则 strides 可能含 0 除数

---

## 2 预计算与传参模式（UB 中转）

快除参数（magic/shift）从 scalar 端传入 VF 有两种形态，按数量选择：

### 2.1 数组型（多个除数，如逐维坐标解码）

TilingData 数组字段（如 strides）与对应快除参数经 UB 中转传 VF：

```cpp
// scalar 端：UB 分配 + 逐个预计算 + 写入 UB
LocalTensor<IDX_T> magicUb = ubAlloc.Alloc<IDX_T>(MAX_DIM);
__ubuf__ IDX_T* magicPtr = (__ubuf__ IDX_T*)magicUb.GetPhyAddr();
for (int32_t d = 0; d < rank - 1; ++d) {
    IDX_T magic = 0;                     // Local Memory 标量中转（禁止直接写 __ubuf__）
    IDX_T shift = 0;
    AscendC::GetUintDivMagicAndShift<IDX_T>(magic, shift,
                                            static_cast<IDX_T>(strides[d]));
    magicPtr[d] = magic;
    shiftPtr[d] = shift;
}
DataSyncBarrier<MemDsbT::UB>();          // scalar 写 UB 后必须同步，保证 VF 可见

// VF 端：按维解码
IDX_T c = AscendC::Simt::UintDiv<IDX_T>(rem, outMagic[d], outShift[d]);
```

### 2.2 标量型（单个除数）

单个除数的 magic/shift 直接以 VF 标量参数传递，无需 UB：

```cpp
IDX_T inStrideMagic = 0;
IDX_T inStrideShift = 0;
AscendC::GetUintDivMagicAndShift<IDX_T>(inStrideMagic, inStrideShift,
                                        static_cast<IDX_T>(inStrideDim));
asc_vf_call<Kernel>(dim3(THREAD_NUM), ..., inStrideMagic, inStrideShift, ...);
```

---

## 3 值域限制（32/64 位路径分派）

- **32 位路径（`IDX_T = uint32_t`）**：除数与被除数均须 `≤ INT32_MAX`，超界即整体切换 64 位
- **64 位路径（`IDX_T = uint64_t`）**：uint64 乘法高（`__umul64hi` 类）为多指令模拟、占双寄存器，**高迭代次数全展开会触发寄存器溢出（spill）**——含快除的循环若需编译期展开，单实例展开次数建议 ≤ 3~4 次；更多次时保留运行期截断分支作为编译器调度边界
- 判定示例（gather 类）：`use64Bit = (outSize > INT32_MAX) || (inputNum > INT32_MAX)`；任何 `strides[d] ≤ outSize`、`dimSize ≤ inputNum` 均被这两个总量覆盖，无需逐项判定

---

## 4 典型用法

| 场景 | 公式 | 说明 |
|------|------|------|
| flat → 多维坐标 | `c_d = UintDiv(rem, magic_d, shift_d); rem -= c_d * stride_d` | 行主序前向逐维解码 |
| 多维 → flat（组合） | `flat = Σ c_d * stride_d` | 逆操作无需快除（纯乘加） |
| 固定块切分 | `block = UintDiv(i, magicB, shiftB); off = i - block * blockSize` | 每线程按块索引定位 |
| 整切片收集 | `o = UintDiv(i, magicInner, shiftInner)`（内层大小）+ 再除索引数 | 两段快除见 IndexGather 范式 type-a |

---

## 5 消费方

- IndexGather 范式（坐标解码/合轴后解码）：`references/paradigms/indexgather/generic/decode-bounds.md`
