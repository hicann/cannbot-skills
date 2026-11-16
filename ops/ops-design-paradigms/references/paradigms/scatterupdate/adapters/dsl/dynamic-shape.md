# ScatterUpdate DSL：动态 shape 程序设计

> 结构键 / det·nondet 双程序族 / workspace 与 8 维 metadata 契约。语言公共部分见 [language-rules.md](language-rules.md) §2。

---

## 1 结构键（编译一次）

| 维度 | 键内容 | 说明 |
|---|---|---|
| 写语义 | assign（本范式固定） | reduce 归约写划界至 AtomicUpdate 范式（独立程序族设计见姊妹范式） |
| dtype 组合 | (val_dtype, idx_dtype) | 值域 × 索引域 |
| KeyT 位宽 | i16 / i32 / i64 | generic/conflict-domain.md §3 自适应；哨兵占用键值域最大值（skip 形态） |
| index 形态衍生 | 覆盖全部维（key 域 = var 元素域）/ 覆盖单维（key = index 值域）/ nd（key = 前 K 维 lin 域） | 决定 Phase1 线性化结构与 key 值域 |
| 计数域 | u32 / i64 | N > 2^30 时 i64（程序族 ×2） |
| det 附加 | SUBSET 源 / dedup（标量源值同，仅重复度驱动）/ axis 位置 | det 程序族细分 |
| 确定性 | det / nondet | **双程序族**（同 wrapper 双路径派发，§3） |

- N/shape 数值**零参与**键——单 ws 张量 + 8 维结构化 metadata + @host 派生标量运行期流动（params 已从交付形态移除）
- det 程序族量级由分型维度（dtype/key 位宽/形态衍生/计数/det 细分）决定，不随 shape 数值增长；懒编译终身复用

## 2 workspace 与参数通道

- **workspace 双形态**：MTE 段（b8/hist/cumsum/excl/gh）共享单 ws（32B 对齐分段，`seg()` 视图传入；b8 段加宽 `tc×TILE`——DSL ub2gm 回写固定满 tile 长度无尾裁剪的 workaround）；SIMT 寻址段（lin/sorted/perm/DB）独立张量传入
- **结构化 8 维 TensorSpec**（1-padding）：var/index/src 等全部张量按 `MAXD = 8` 维声明；Phase1 逐维 strides/dims 由 kernel VF 外从 metadata 静态直读（`range_constexpr(8)` 展开 + 运行卫 `d<rank`）——**零 params 零 H2D**
- **标量入参**：`@host` 从 shape 闭式派生 n/totalIndexNum、s_sz/var_sz、active_cores、launch（排布策略见 [../../generic/thread-layout.md](../../generic/thread-layout.md)）后经 kernel 参数传入（寄存器零访存）
- **标量 value / 源承载**：assign 语义的标量源以 1 元素张量承载；无 params 中转

## 3 det / nondet 双程序族与派发

```python
# wrapper（门控 = 确定性开关 + 重复度；assign 无 dtype 白名单，与 ASC 语义同源）
deterministic = torch.are_deterministic_algorithms_enabled()   # 开关传导
use_nondet = (not deterministic) and (dup_ratio <= THRESHOLD)  # 重复度（鸽笼下界）
if use_nondet:
    nondet_program(var8, index, src, ...)                      # 直写覆盖
else:
    det_program(var8, index, src, ..., ws_segments...)         # 三段式段尾覆盖
```

- 两程序族**独立懒编译**，各自结构键缓存
- det 为兜底：任何 nondet 不满足条件（开关开 / 高重复）都由 det 接住
- **越界语义跨路径一致**（skip 经哨兵沉底 / 直跳——同输入同行为；无 err flag）
- 标量源（值同）单程序族 + `dedup` 结构键位（仅重复度驱动，确定性开关不触发）

## 4 结构键设计核对表

- [ ] 键内零 shape 数值（N 经 @host 派生标量流动）
- [ ] 8 维预留结构化契约（1-padding）：rank 不进键；dim/axis 进键
- [ ] KeyT 位宽按冲突域值域自适应（含哨兵余量）
- [ ] 计数域 u32/i64 分族（N > 2^30）
- [ ] det 细分维度（SUBSET/dedup/axis）显式列全
- [ ] 双路径派发判定与 ASC 场景判定的确定性分治对表
- [ ] 零重编译断言（同键异 shape）入验证计划
