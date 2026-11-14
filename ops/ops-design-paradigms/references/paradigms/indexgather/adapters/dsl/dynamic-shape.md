# IndexGather DSL：动态 shape 程序设计

> 结构键设计 / 8 维结构化契约 / 零元张量（`dt(0)` 常量）/ 零重编译验证。动态侧真值源（语言公共部分见 [language-rules.md](language-rules.md) §2）。

---

## 1 结构键（编译一次，shape 零参与）

| 分型 | 结构键 | 说明 |
|---|---|---|
| **A 型**（index_select / take / embedding） | `(data_dtype, idx_dtype, use64, need_cast)` | 三因子坐标分解与 rank 无关——键最少 |
| **C 型**（gather） | `(data_dtype, idx_dtype, merge_rank, dim, use64, need_cast)` | 合轴结构（`_merge_axis` 保留于 wrapper，动态化不放弃合轴）进键——同合轴结构的所有 shape 共享程序 |

- **use64**：`outSize/inNum > INT32_MAX`（余量 2×grid 上限）；idx_dtype 不参与判定（int64 index 只是触发 need_cast/域加宽）
- **need_cast / j_dtype**：32 位路径遇 int64 index 收窄 / 64 位路径遇 int32 index 加宽
- 键空间规模由分型结构与 dtype 组合决定，不随 shape 数值增长

## 2 规格与运行期通道

```python
MAXD = 8
pv_in  = [Dim(f"V{d}", min=1) for d in range(MAXD)]   # 8 维预留（1-padding）
pv_idx = [Dim(f"I{d}", min=1) for d in range(MAXD)]
pv_out = [Dim(f"O{d}", min=1) for d in range(MAXD)]
program = cannbotdsl.compile(
    self.run,
    TensorSpec(tuple(pv_in), data_dt),   # 输入（真实维在前缀、尾部补 1）
    TensorSpec(tuple(pv_idx), idx_dt),   # index
    TensorSpec(tuple(pv_out), data_dt),  # 输出
)
```

- 张量由 wrapper 以 8 维预留视图传入（`t.view(*t.shape, *[1] * (8 - t.dim()))`，零拷贝；受限 dtype 先 torch 层 view 别名）
- **零 params 零标量**：out_size / gds / 解码常量由 kernel VF 外 metadata 直读（§3），或由 `@host` 闭式派生；launch 在 `@host` 计算后 `[launch]` 发射

## 3 结构化 8 维契约与 metadata 直读（现行）

**声明**：var/index/out 全部按 `MAXD = 8` 维声明（1-padding；真实维在前缀 `[0, rank)`、尾部补 1）——单程序覆盖全 rank，rank/形状变化**零重编译**。

| 分型 | metadata 直读（VF 外静态索引） | `@host` 派生标量 |
|---|---|---|
| **C 型**（gather） | in_stride_dim / gds / out_strides[0..rk-2] / in_strides[0..rk-2]（`gm.stride[d]`、`gm.shape[dim]`） | out_size / in_num（+ launch） |
| **A 型**（index_select） | outer / gds / inner（按 8 维槽位静态索引） | out_size / launch |

- 维度循环一律 `range_constexpr(MAXD)` 展开 + 运行卫 `d < rank`（metadata 须静态索引）
- dim/merge_rank 进结构键（编译期 int 才能静态索引 metadata）；`view(-1)` 不支持——扁平视图由 8 维乘积显式构造

## 4 零元张量（越界零填的值源）

越界零填 = **`dt(0)` 编译期常量**（R1 整改：`zero_t` 已删除——gather/index_select 的 work dtype 集合无无符号/bool，字面量位转换缺陷不触达，且零 H2D）：

```python
if (j < 0) or (j >= gds):
    gm_out[i] = self._dt(0)       # 编译期常量（work dtype 原生 0）
```

## 5 零重编译验证要求（设计文档必含）

1. 同结构键异 shape：二次调用不触发重编译，程序缓存保持 1 份
2. 异结构键（不同 rank/dim/dtype）：各自编译，互不干扰
3. 正确性：对 dtype/rank/dim/越界/use64/空输出等组合做 golden 对照（bit-exact 对照 torch.gather / torch.index_select；越界位置期望零）

## 6 wrapper 侧清单

1. 校验（dtype/rank/contiguous/dim 归一化/广播约束仅 C 型）→ 空输出早退
2. **合轴（C 型）**：`_merge_axis` 保留——merge_rank/dim 进结构键；**gather_dim_size 于合轴前捕获**（dim 将被重赋值为合并轴）
3. use64 判定 + need_cast 推导
4. 8 维预留视图构造（零拷贝）+ `@host` 派生（out_size/gds/launch）
5. blocks 核数两步法（下限保护 PER_CORE_MIN——算法公式见 [../../generic/thread-layout.md](../../generic/thread-layout.md) §2）
6. compiled-program 发射（零元张量已删除：`dt(0)` 常量）
7. 受限 dtype（fp64/complex/fp8/bool）torch 层 view 进出
