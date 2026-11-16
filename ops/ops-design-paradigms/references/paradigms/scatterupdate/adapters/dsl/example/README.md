# ScatterUpdate DSL example

ScatterUpdate 范式 DSL 侧的**代码组织形态参照**（基例：tensor_scatter_nd_update_dsl，assign × nd 双路径）。

## 定位声明

本 example 展示 DSL 覆盖写算子的**代码形态**——nondet 直写 + det 三段式段尾覆盖双程序族、wrapper 视图 pack、自包含排序闭环。**设计知识（场景判定、越界形态、覆盖顺序）一律以范式 `generic/` + `adapters/dsl/` 文档为真值源**，两者有出入时以文档为准。

## 基例：tensor_scatter_nd_update_dsl（assign + nd 坐标元组）

| 文件 | 形态要点 | 知识真值源 |
|------|---------|-----------|
| `tensor_scatter_nd_update_dsl/tensor_scatter_nd_update.py` | wrapper：双门控（`torch.are_deterministic_algorithms_enabled()` → det；否则 nondet 直写——无 dtype 白名单、无 dupRatio）→ **视图 pack**（reshape+view 零拷贝；W 递降 `8//elem_bytes`，slice 整除 ∧ 首地址对齐双判据；inplace var 不可拷贝对齐→降档 W=1；f8/bool→uint8 位级别名）→ **结构化 8 维预留（1-padding）直传**（零 params 零标量）→ det tiling（仅 ws 定形 + 程序变体；key_dt 三档）→ workspace：MTE 段（b8 加宽 `tc×TILE` / hist / cumsum / excl / gh）共享单 ws 32B 分段 **NPU 设备端创建** + SIMT 寻址段（lin/sorted/perm/db_v/db_i）独立张量 → 发射（帧契约 nondet 3 槽 / det 13 槽） | dsl/dynamic-shape.md §3、dsl/design-spec.md（pack/async 条目） |
| `tensor_scatter_nd_update_dsl/tensor_scatter_nd_update_kernel.py` | det 三段式：Phase1 nd 直读（无除法反解；**R3-E：VF 外 8 维预留 metadata 静态读取，维度循环 8 块展开 + 运行时卫 `d<k_dim`**）→ Phase2 嵌入排序 → **Phase3 段尾覆盖（O(1) 段尾自判：全局末位或与下一索引不同）**；nondet：单 VF grid-stride 直写（任意赢，无原子无 ws） | dsl/deterministic.md §1-3、ascendc/type-assign.md §4.1 |
| `tensor_scatter_nd_update_dsl/tsnu_sort/` | **自包含排序闭环**（随算子目录整体移动，无跨算子依赖）：`sort_dsl.py` 编排（Pass1 全局直方图 / Pass2 值置换 / Pass3 跨块合并）+ `sort_kernel.py`（over-launch safe + 三态 lookback（bit30\|hist=就绪 / bit31\|value=前缀 / 0=未发布）+ B1 发布→barrier→B2 lookback/散射屏障分离 + excl 循环外发布）+ `radix_tiling.py` / `radix_passes.py` | dsl/deterministic.md §2 |

## 共通骨架

```
wrapper: 校验 → 双门控（det/nondet）→ 视图 pack（W 递降）
         → 结构化 8 维预留直传 → det tiling（ws 定形 + 变体）→ 发射（enqueue-and-return）
@host:   shape 派生 n_idx/k_dim/s_num → 闭式 active_cores/launch
         （tile/is_single_core 为变体常量；零标量帧槽）
kernel:  [det] metadata 直读(VF 外) → Phase1(nd 直读 + 哨兵 skip) → 屏障
         → 嵌入排序（lookback dcci）→ 屏障 → Phase3(段尾 O(1) 自判直写)
         [nondet] metadata 直读 → 单 VF grid-stride 直写
launcher: 结构键懒编译 ×2 族（det/nondet）；自包含 sys.path 注入
```

## 0.8.3 适配说明

- 本基例按 cannbot-dsl 0.8.3 编写：片上流水用 `RingBuffer`（`next_write()/next_read()`）、`Buffer` 新签名、dtype 视图用 `view(dtype=)`/`from_tensor`、原子累加用 `copy_options`
- det Phase1/Phase3 的 grid-stride 三边界全部提升到 i64（0.8.3 已支持 SIMT VF 内 i64 归纳变量，issue #79 已修）
- `@host` 由 shape 派生的 n_idx/k_dim/s_num 须显式 `dtypes.int64(...)` 构造（脱离 shape-dim 溯源，否则 range 边界仍会被按 i32 索引域窄化并在 VF 内产生截断）
- hist 工作区 dtype 用 uint16、A 相 u32 通道用 uint32（GM2UB 同 dtype 校验）
- 2D workspace 取行先 `view` 成一维 `*_flat` 再 `tile_slice`
