# 静默错误清单

**编译干净、不 crash、结果错**的已知陷阱。遇到数值对不上时先查这里，可能直接命中。

每条给出：现象 → 为什么会这么写 → 正确写法 → 怎么验证。

---

## 1. `tile_view` 的 coord 是 tile 索引，不是元素偏移

**现象**：数据取到了完全不相干的位置，rel≈0.93。不报错。

**为什么会写错**：签名 `tile_view(input, tiler, coord)` 看起来像「从 coord 处取一块
tiler 大小」，很自然写成元素偏移 `(0, c * CHUNK)`。

源码注释写得很明确（`tensor.py`）：

> `coord` is a tile index, not an element coordinate

```python
# ✗ 跳到第 c*CHUNK 个 tile
q_tile = tile_view(q_slice, (M, CHUNK), (0, c * CHUNK))
# ✓
q_tile = tile_view(q_slice, (M, CHUNK), (0, c))
```

**验证**：单 chunk 探针，把 chunk 内容设成可识别的值（如全 c），看取回来的是第几块。

---

## 2. PV 的 L1→L0B 需要 `transpose=True`，QK 不需要

**现象**：PV 结果错，rel≈1.36。不报错。

**为什么会写错**：两者都是「K/V tile 送进 L0B」，代码看起来对称。而且蓝本
（D=N=128 方阵）里看不出区别——方阵下转不转置形状都一样。

L0B 要的是 B 操作数的 `(K, N)` 且按 B^T 打包：

| matmul | GM 中的源 | 语义 | transpose |
|---|---|---|---|
| QK | K 是 `[S_kv, d]` | 要算 `K^T`，GM 布局已是 `(N, K)` | **不加** |
| PV | V 是 `[S_kv, d]` = `(K, N_out)` | fractal load 期望 B^T 打包 | **`transpose=True`** |

**验证**：`probe-recipes.md` §6 的单变量 A/B 探针，两版只差 transpose。

---

## 3. 共用 L0 channel：形状只在特定 tile 配置下偶然重合 ⚠️

**本清单里最贵的一条。** 详细复盘见 SKILL.md §3。

**现象**：换一个 shape 参数就静默算错，其他 shape 全对；四个隔离探针全绿。

两个 matmul 的 L0 操作数形状本来就不同，只在某个 tile 配置下碰巧一致：

```
QK:  L0A=(M, d_chunk)   L0B=(d_chunk, tile_n)
PV:  L0A=(M, tile_n)    L0B=(tile_n, d_chunk)
                              ↑ 只在 d_chunk == tile_n 时重合
```

**修法**：显式判断，不要默认可以复用：

```python
if d_chunk == tile_n:
    self.l0a_pv, self.l0b_pv = self.l0a, self.l0b
else:
    self.l0a_pv = Channel(MemLoc.L0A, shape=(tile_cube_m, tile_n), ...)
    self.l0b_pv = Channel(MemLoc.L0B, shape=(tile_n, d_chunk), ...)
```

**通用规律**：算子里有两个以上 matmul 时，把每个的 `(M, K, N)` 三元组写出来逐一对比，
不要凭「看起来都是方阵」的印象共用 buffer。

---

## 4. split-M 的空分区 AIV 会污染共享状态

**现象**：query 长度 S ≤ `tile_vec_m` 时**所有行**都错（不只是某一半），误差看起来均匀。
边界锐利：`tile_vec_m=16` 时 S=16 全错、S=17 全对。

**机制**：cube 的 M tile 由 2 个 AIV 分摊。S ≤ tile_vec_m 时其中一个分不到行——但它
**不是什么都不做**：照样在 padding 上算 rowmax/rowsum，并写入**共享的** running
max/sum，于是把另一个 AIV 的行也带偏。这解释了为什么误差是「均匀」而非「一半对一半错」。

**修法**：缩小 M tile 直到两个 AIV 都有行（下限是 NZ fractal 的 16 行）；再短就在 host
侧把 query 补到阈值以上，算完切掉。

**注意**：不只整个张量的 S 要满足，**每个 m-tile 都要**。S=65、tile=64 时第二个
m-tile 只有 1 行，同样触发（错行恰好是 `[64]`）。把 S 补齐到 tile 的整数倍可一次性消除。

---

## 5. causal 场景补 query 行必须**前置**

**现象**：补齐 query 行后，非 causal 全对、causal 系统性偏差。

**机制**：causal mask 右下角对齐（`j <= i + (S_kv - S)`）。**追加** padding 时真实行
位置不变但 S 变大，可见上界从 `i + (S_kv - S)` 缩到 `i + (S_kv - S - n)`——**每行都少看
n 个 key**。**前置**则真实行 i 移到 i+n，`(i+n) + (S_kv - (S+n))` 恒等于原值。

输出侧相应取**尾部** `out[:, pad:]` 而非 `out[:, :orig_S]`。

**教训**：验证 padding 类修复**必须带 causal 变体**。非 causal 全绿会给出虚假安全感——
这个 bug 就是这样躲过所有定向验证、直到全量 sweep 才暴露的。

---

## 6. 归约轴的零填充列污染 softmax 分母

**现象**：`S_kv % tile_n != 0` 时精度不达标，误差随**被 padding 的列数**增长
（补 1 列能过、补 64 列错得最凶），而非随尾块大小。

**机制**：K tile 的缺失列被 `nd2nz` 零填充 → 这些列 score=0 → softmax 里
`exp(0 - rowmax)` 是个**非零权重**，被计进 rowsum 分母。V 对应行是 0 所以分子不受影响，
但分母被撑大，输出整体偏小。

causal **只减轻不消除**：mask 跟的是因果对角线，不是 S_kv 边界，越过 S_kv 的 padding
列在对角线以下时仍然可见。

**修法**：让 rowmax/rowsum 不跨过 `actual_n`（在无运行期分支的前提下把越界 lane 压成
`-inf`）。属 raw-VF 精密改动，动前先写只测该逻辑的探针。

---

## 7. 用同 dtype golden 当精度基准会误判

见 SKILL.md §6。低精度参考实现自身误差可能比被测 kernel 还大
（实测 fp16 golden 2.10e-3 vs kernel 1.64e-3）。必须用 fp64 golden + 官方 checker。

---

## 8. 高层 `cast(int8_dst, fp16_src)` 反交织

**现象**：fp16→int8 的高层 `cast` 输出按 2 反交织（`res[k] == ref[2k]`），编译不报错、
结果静默错。

**规避**：退到 raw vf 用 `vstore_pack(mode=B32_TO_B8)` 产连续 int8。

---

## 9. `@kernel` launch 按位置绑定，不按名字

**现象**：host wrapper 里传参顺序与 `@kernel` signature 错位 → 张量被静默错路由
（如 `q_scale` 落进 `key` 槽）→ NaN。编译和 launch 都成功、无 device fault。

**修法**：逐槽核对 launch 调用顺序与 signature。看似关键字的对应关系**不生效**。

---

## 10. 显式 `addr=` 别名 channel 跨迭代竞态

**现象**：两个 Channel 用 `addr=` 显式别名，单迭代测试全对；一旦相邻操作数 `depth≥2` 让相邻迭代重叠，立刻静默串数据（编译无告警、运行无 fault）。

**机制**：别名 channel 的地址重叠对同步层完全不可见，无 pass 串行化；`depth≥2` 让相邻迭代的写操作同时发射且写同一批字节。别名操作数自己 `depth=1` 不构成保护。

**修法**：先算不别名装不装得下 → 调 tile → 仍不行才别名，且必须论证跨迭代不重叠。详见 `../../core-skills/cannbotdsl-op-design/SKILL.md` §2.0。

---

## 11. `const_expr(cond)` 守卫变负失效

**现象**：形如 `if const_expr(NPAD > 0):` 的保护，当 `NPAD = VH - BMV` 因上游参数变化而变负时**静默跳过**，而它本该保护的越界写读照常发生 —— 守卫非但没报警，还掩盖了唯一线索。

**机制**：`const_expr` 在 trace 期求值 cond；cond 变负时 `NPAD > 0` 为 `False`，整个 `if` 分支被跳过。但被保护的代码（如 padding 填充）本应执行，跳过后越界写读无声发生。

**修法**：给守卫加下界断言（`assert VH >= BMV`），或让参数在 host 侧推导而非硬编码。比守卫本身更可靠。

**验证**：枚举上游参数变化域，确认 NPAD 变负时是否有断言拦截。详见 `../../core-skills/cannbotdsl-vf-fusion/SKILL.md` 陷阱 11。

---

## 12. raw-VF 操作数序猜错

**现象**：raw-VF 算子的操作数序与 lane 布局，签名里看不出来，猜错就是静默算错。编译通过、无 device fault、结果数值范围正常但全错。

**机制**：这些 op 的 Python 签名只有形参名（`acc`/`lhs`/`rhs`），源码直接建 MLIR op、无语义注释；两种读法都类型合法、都能编译、都返回有限值。

| op | 实测语义 | 猜错的后果 |
|---|---|---|
| `vmadd(acc, lhs, rhs)` | `acc*lhs + rhs` | 自然读法 `acc + lhs*rhs`，两种都能编译，写错即静默错数 |
| `vexp_sub(a, b)` | `exp(a - b)` | 反了得 `exp(+Δ)`，有限、看着合理、全错 |
| 窄化 `vcast`（fp32→fp16） | 64 个结果落在偶数 lane，与 `reg_layout` 无关 | 以为 `ZERO`+`ONE`+`vmerge` 能拼成 128 lane → 纯几何置换。正解：两次 `ZERO` cast 再 `vdeintlv` |
| `vreduce_max/sum` | 不广播，只有 lane0 有效 | 少一个 `vdup_lane0` 就跨行污染 |
| `vload_brc` | 广播 4 字节偏移处的单个元素 | 按"每 datablock 一个"理解会静默混行 |

**修法**：写 20 行探针钉死语义，比在 500 行 kernel 里查便宜一个数量级。测操作数序用**三个互不相同、不满足任何对称性**的值（如 2.0 / 3.0 / 5.0）。详见 `../../core-skills/cannbotdsl-vf-fusion/SKILL.md` 陷阱 10。

---

## 非静默但容易误判的两条

**`const_expr(range(n))` 不是循环包装器** —— 返回 bool，报
`TypeError: 'bool' object is not iterable`。静态展开写 `for c in tuple(range(n))`。
`const_expr` 只用于 `if const_expr(cond)` 这类编译期分支。

**`@jit` 参数不能是 Python `bool`** —— 报
`TypeError: unsupported runtime value for @jit parameter`。编译期开关放模块级常量或
`@kernel` 类的 `__init__` 属性。
