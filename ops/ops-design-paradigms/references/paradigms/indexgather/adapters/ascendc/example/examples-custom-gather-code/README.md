# examples-custom-gather-code

IndexGather 范式的**代码组织形态参照**（C 型实现：custom_gather，对标 torch.gather，全量 dtype × 形态矩阵回归覆盖）。

> ⚠️ **A/B 型开发者注意**：本 example 仅覆盖 C 型（逐元素对齐 + 广播 + 合轴）。A 型（整维切片两段快除）与 B 型（N 元组 K 特化）的 tiling/kernel 结构与 C 型差异显著（A 型零数组 TilingData、无 rank/K switch），代码形态参照意义有限——请以 `references/index-gather-type-{a,b}.md` 文字指引 + 内联代码为主，本 example 仅作文件组织/注册/入口骨架参照。

## 定位声明

本 example 仅展示 IndexGather 算子的**代码形态**——文件组织、REG_OP 注册、infershape 双端校验、tiling 全流程（含 MergeAxis 合轴）、TilingData/TilingKey 结构、VF kernel 双重载、`__global__` 入口。**设计知识（为什么这样写、三型差异、变体推导）一律以范式 `references/` 文档为真值源**，两者有出入时以 references/ 为准。

## 文件对照

| 文件 | 形态要点 | 知识真值源 |
|------|---------|-----------|
| `CMakeLists.txt` | 构建注册形态 | — |
| `op_host/custom_gather_def.cpp` | REG_OP + 12 组 dtype 映射 + ATTR(dim) | overview.md §4 |
| `op_host/custom_gather_infershape.cpp` | shape 推导 + 双端校验 | tiling-preprocess.md §1 |
| `op_host/arch35/custom_gather_tiling.cpp` | MergeAxis + 元信息 + use64Bit + 核数两步法 | tiling-preprocess.md §2-§4 |
| `op_kernel/arch35/custom_gather_tiling_data.h` | TilingData 结构（40 int32） | overview.md §2 |
| `op_kernel/arch35/custom_gather_tiling_key.h` | idxMode 场景位 TPL 宏 | overview.md §4 |
| `op_kernel/arch35/custom_gather_simt.h` | 双重载 VF kernel + 越界原值比较 + 坐标解码 | decode-bounds.md / kernel-entrance.md |
| `op_kernel/custom_gather.cpp` | 入口 + idxMode `if constexpr` 分派 | kernel-entrance.md §1 |

A 型（index_select 两段快除）/ B 型（gather_nd K 特化 + batch_dims）的 kernel 内部差异不设 example，关键代码内联于 `references/index-gather-type-{a,b}.md`。
