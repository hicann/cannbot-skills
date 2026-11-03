# IndexGather Kernel 入口与双重载分派

> 入口注册 / idxMode 分派 / 编译期特化的收益与边界（spill 教训）。设计文档 §4.1 的真值源。

---

## 1 入口结构（kernel 入口文件）

```cpp
template <uint32_t idxMode>
__global__ __aicore__ void xxx_gather(GM_ADDR input, GM_ADDR index, GM_ADDR output,
                                      GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(XxxTilingData);
    GET_TILING_DATA_WITH_STRUCT(XxxTilingData, tilingData, tiling);
    if constexpr (idxMode == 0) {
        NsXxxGather::Process<DTYPE_INPUT, DTYPE_INDEX, uint32_t>(
            input, index, output, workspace, &tilingData);
    } else {
        NsXxxGather::Process<DTYPE_INPUT, DTYPE_INDEX, uint64_t>(
            input, index, output, workspace, &tilingData);
    }
}
```

- `DTYPE_INPUT / DTYPE_INDEX` 宏由 def.cpp `DynamicCompileStaticFlag(true)` 激活，dtype 组合（示例 6×2=12 组，由 spec 决定）在 idxMode 的 2 个二进制实例内编译期展开
- 核函数签名参数顺序与 REG_OP 原型一致（input, index, output + 框架 workspace/tiling）

## 2 Process 内分派（双重载模型）

```cpp
if constexpr (sizeof(IDX_T) == 4) {
    switch (td->inputRank) {                       // 编译期 RANK 特化（IDX32）
        case 1: asc_vf_call<Kernel<T, INDEX_T, IDX_T, 1>>>(...); break;
        // ... case 8
    }
} else {
    asc_vf_call<Kernel<T, INDEX_T, IDX_T>>(         // 运行期 rank（IDX64）
        ..., td->inputRank, ...);
}
```

A 型止步 idxMode 双重载（结构与 rank 无关，无 rank/K switch；batch_dims 组合另加 `batchDim > 0` 运行期分支——双级分派同 B 型，b=0 路径零退化，见 [type-a.md](type-a.md) §3），B 型为双级分派（`batchDim > 0` × `switch (td->k)`，K 范围由需求确定，四 kernel 模板见 [type-b.md](type-b.md) §3），C 型用 rank——**分派维度与 idxMode 正交**。

## 3 编译期特化的收益与边界（spill 教训）

### 3.1 为什么特化

坐标解码是逐维除法链（每元素 rank-1 次 UintDiv）。编译期 RANK 使循环 `for (d < RANK-1)` 固定次数全展开，消除运行期截断分支与空转——IDX32 实测提速 27%~36%。

### 3.2 为什么 IDX64 禁止全展开（负知识，违反必退化）

- uint64 乘法高位无原生指令（`__umul64hi` 多指令模拟），单次解码成本 ≈ uint32 的 2.7 倍
- RANK=5~8 全展开时，编译器把 5~7 条 uint64 乘高链**跨迭代交错调度** + 常量数组整体上提，寄存器占用超 `__launch_bounds__(1024)` 预算 → **spill 到局部内存**
- 实测：rank∈{5,7,8} + int64 index 全展开退化 +35%/+62%/+87%
- **运行期 `if (d < rank-1)` 截断分支是解药**：它构成编译器调度边界，阻止跨迭代交错与常量上提，寄存器压力有界

### 3.3 边界结论

| IDX_T | 循环写法 | 依据 |
|-------|---------|------|
| uint32 | `for (d < RANK-1)` 编译期特化（switch 分派 8 实例） | native `__umulhi` 单指令、单寄存器，全展开纯受益 |
| uint64 | 展开 MAX_RANK-1 次 + 运行期截断（3 参重载） | 乘高多指令模拟；截断分支作调度屏障 |

**特例**：int64 index + `inputNum/outSize/indexNum/taskNum` 均 ≤ INT32_MAX 时走 IDX32 特化（use64Bit 不看 indexDtype，越界语义由原值比较保障，见 [decode-bounds.md](../../generic/decode-bounds.md) §1/§2）——因此 IDX64 路径仅服务超大张量，存量极少。

## 4 空 tensor 早退（顺序约束）

```cpp
if (td->outSize == 0) {
    return;   // 必须先于快除预计算：outSize==0 时 outStrides 可能含 0 除数
}
```
