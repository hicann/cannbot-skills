# IndexGather DSL：Kernel 实现写法

> A 型三因子分解 / C 型合轴逐维解码 / 越界零填 / canonical 域锚定。语言公共纪律（grid-stride / metadata 直读 / 域锚定）见 [language-rules.md](language-rules.md)。

---

## 1 通用骨架（两型共享）

```python
@kernel
class MyGatherDynKernel:
    def __init__(self, <结构键参数>):        # 零 shape 数值
        ...
    def __call__(self, gm_in, gm_idx, gm_out, out_size, in_num):
        # metadata 只在 VF 外读（8 维预留静态索引）；out_size/in_num 由 @host 派生
        with vf(mode="simt", thread=THREAD_NUM):
            t = simt.thread_idx()[0]
            tdim = simt.thread_dim()[0]
            stride = simt.grid_dim()[0] * tdim
            tid = simt.block_idx()[0] * tdim + t
            if const_expr(self._use64):
                tid = simt.cast(tid, dtype=dtypes.int64)
                stride = simt.cast(stride, dtype=dtypes.int64)
            <循环外：out_s / in_s / gds 由 metadata 静态索引取至局部>
            for i in cannbotdsl.range(tid, out_size, stride):   # grid-stride 铁律
                j = gm_idx[i]
                if const_expr(self._need_cast):
                    j = simt.cast(j, dtype=self._j_dtype)       # 检查后转换
                if (j < 0) or (j >= gds):       # 原值域检查（use64 域对齐）
                    gm_out[i] = self._dt(0)     # 越界零填（dt(0) 常量）
                else:
                    <解码 + 搬移>
                    gm_out[i] = gm_in[in_off]
```

要点：
- **域锚定**：解码链以 `rem = i + dtypes.int64(0)` 起（canonical 算术链，language-rules.md §4）
- **in_off / j 乘法**在 i64 域统一（值域安全：in_num ≤ 2^31 于 32 位路径、use64 路径本就 i64）
- **纯字节搬移**：`gm_out[i] = gm_in[in_off]` 对任意 dtype 成立（NaN/Inf/±0 位级透传）

## 2 A 型：三因子分解（index_select / take / embedding）

```
q = i // inner_size          # 两次运行期整除（统一整除，见 §4）
o = q // gather_size
j = q - o * gather_size      # index 元素位置
inner_off = i - q * inner_size
src = (o * gather_dim_size + k) * inner_size + inner_off
```

- **除法统一整除**：被除数 `i` 源自 range 循环变量（恒裸载体）→ 统一 `//`
- 0 维标量退化：outer=inner=gds=1，分解自然成立
- 展平 take 变体：inner_size=1 → q=i、单除法退化

## 3 C 型：合轴后逐维解码（gather）

```
rem = i + dtypes.int64(0)
in_off = dtypes.int64(0)
for d in cannbotdsl.range_constexpr(rk - 1):    # rk = merge_rank（合轴后）
    c = rem // out_s[d]                        # 解码除数 = index 空间步长
    rem = rem - c * out_s[d]
    if const_expr(d != dim):
        in_off = in_off + c * in_s[d]          # 偏移乘子 = input 空间步长
if const_expr(rk - 1 != dim):
    in_off = in_off + rem                       # 末维（stride 恒 1）
in_off = in_off + j * in_sdim                   # dim 维坐标被 j 替代
```

- **两套步长分离**（generic/decode-model.md §1）：`out_s`（index 空间解码）与 `in_s`（input 空间偏移）两套步长（metadata 直读）分离消费——广播正确性的根基，禁止合并
- **range_constexpr 解码循环**：d 为 trace 期 Python int，`const_expr(d != dim)` 编译期剪枝、步长槽位静态索引——合轴收益（rk-1 次除法）直接体现
- **合轴不变量**由 wrapper 保证（generic/decode-model.md §2.2）

## 4 除法实现：统一整除（与 A 型对照）

**固定除数统一使用 `//`**（正确性优先；不设快除分支）。

## 5 退化场景核对表

| 场景 | 行为 | 核对点 |
|---|---|---|
| out_size == 0 | wrapper 早退（返回空张量） | 早退先于步长预计算 |
| rank == 1（C 型） | 解码循环零次；in_off = j × in_sdim | dim=0 特化 |
| 0 维标量（A 型） | outer=inner=gds=1 | 输出保持 0 维 |
| 全广播（C 型） | merge_rank = rank（不可合轴） | 段头豁免 + 两套步长 |
| int64 index + 32 位规模 | need_cast：j 收窄 i32 | 值域由 use64 判定保证 |
| int32 index + use64 | need_cast：j 加宽 i64 | 防 k 截断 |

## 6 golden 设计

- 界内：bit-exact 对照 `torch.gather / torch.index_select`（CPU）
- 越界零填：对照值 = ref 的 clamp 版本 + 越界位置置零（torch 原生报错不可直比）
- dtype 别名（fp64/complex/fp8/bool）：torch 层 view 后按位比较
