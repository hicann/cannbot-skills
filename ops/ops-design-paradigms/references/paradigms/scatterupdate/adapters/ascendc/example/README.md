# scatter_update example

ScatterUpdate 范式的**代码组织形态参照**（基例：tensor_scatter_nd_update，assign × nd 双模板）。

> ⚠️ **交叉组合无独立 example**：assign + dim（scatter(reduce=none) 等）与标量源模式的关键差异内联于 [type-assign.md](../type-assign.md) / [linearize.md](../linearize.md)，本目录仅作文件组织/注册/入口骨架参照。reduce + dim（custom_scatter_add）基例属归约写范式，不在本目录。

## 定位声明

本 example 仅展示 scatter 覆盖写算子的**代码形态**——双模板多模板注册（TilingRegistry priority）、SortLib 集成、三段式段尾覆盖编排、TilingData/TilingKey 场景位结构。**设计知识（为什么这样写、场景判定、越界形态、覆盖顺序）一律以范式 `references/` 文档为真值源**，两者有出入时以 references/ 为准。

## 基例：examples-tensor-scatter-nd-update-code（assign + nd）

assign 覆盖写 + 坐标元组形态：**双模板（默认 nondet 直写任意赢 + det 三段式段尾覆盖 last-write-wins）**、哨兵 skip 越界、纯搬运 binary 比对。

| 文件 | 形态要点 | 知识真值源 |
|------|---------|-----------|
| `op_host/tensor_scatter_nd_update_def.cpp` | REG_OP + dtype 映射（20 组，由 spec 决定） | overview.md §4 |
| `op_host/tensor_scatter_nd_update_infershape.cpp` | shape 推导 + 双端校验（K ∈ [1, rank]、updates shape 拼接） | tiling-preprocess.md §1 |
| `op_host/arch35/tensor_scatter_nd_update_tiling.cpp` | Tiling4Xxx → TilingRegistry::DoTilingImpl 入口 + CompileInfo 平台信息 | kernel-entrance.md §2 |
| `op_host/arch35/tensor_scatter_nd_update_tiling_base.cpp` | Base 公共校验/shape 解析（纯只读，降级安全） | kernel-entrance.md §2 |
| `op_host/arch35/tensor_scatter_nd_update_det_tiling.*` | 确定模板 + priority=10 注册（IsCapable ≡ true 兜底） | kernel-entrance.md §2 |
| `op_host/arch35/tensor_scatter_nd_update_nondet_tiling.*` | IsCapable（GetDeterministic()==0，无 dtype 白名单）+ priority=0 | type-assign.md §2 / kernel-entrance.md §2 |
| `op_kernel/arch35/tensor_scatter_nd_update_tiling_data.h` | assign 型双 TilingData（det + Nondet 子集） | overview.md §3.2 |
| `op_kernel/arch35/tensor_scatter_nd_update_tiling_key.h` | countMode + keyDtype + deterministic 三场景位（7 SEL；nondet 占位值） | overview.md §4 |
| `op_kernel/arch35/tensor_scatter_nd_update_common.h` | 共享工具：THREAD_NUM / PackType / ComputePackParam（两路径单一事实源） | kernel-entrance.md §3 |
| `op_kernel/arch35/tensor_scatter_nd_update_nondet_simt.h` | 单 VF 直写覆盖（任意赢，无原子无 ws 无 SyncAll） | type-assign.md §4.1 |
| `op_kernel/arch35/tensor_scatter_nd_update_det_simt.h` | Phase1 坐标元组直读（无除法）+ Phase3 段尾覆盖 | linearize.md §3 / deterministic.md §3 |
| `op_kernel/tensor_scatter_nd_update_apt.cpp` | 顶层 if constexpr (deterministic) 二分 + 同步三连 | kernel-entrance.md §1/§4/§5 |

## sort_lib 依赖

基例携带一份 `sort_lib`（`op_host/arch35/sort_lib_tiling.h` + `op_kernel/arch35/sort_lib*`，radix sort SIMD 库），为基例的构建自包含依赖；SortLib 的功能契约与 tiling 接口见 [tiling-preprocess.md](../tiling-preprocess.md) §3。
