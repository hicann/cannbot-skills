# flash-linear-attention-npu 参考规范

默认开源参考仓库：

```text
https://github.com/flashserve/flash-linear-attention-npu
```

本文路径和搜索建议基于以下参考版本记录：

```text
reference_url=https://github.com/flashserve/flash-linear-attention-npu
reference_commit=e7a05dc24ede580fd74f3ffc97d0ea71738e91fb
last_verified=2026-08-10
```

如果远程仓目录结构变化，先 checkout 到上述 commit 复核路径；若用户指定其他 commit/tag，交付文档必须记录实际使用版本。

命中 GDN/KDA/retention/RWKV 线性 Attention 算子时，优先按该远程仓库对齐算法语义和 NPU 工程规范。不要依赖本机绝对路径；如需本地读取源码，应由执行环境尝试 clone 到当前工作区。clone 是 best effort，不是设计/生成门禁。

本地参考例外：只有用户 prompt 明确给出本地实现参考路径（例如“本地参考实现 / 实现参考 / source-of-truth / 按此 pipeline 对齐”）时，才允许改用该路径作为 primary reference。用户给出的 baseline / 评测 / 性能对比路径不属于本地实现参考。否则不得自动扫描或使用开发机已有本地仓库、历史算子目录、当前工作区外同名实现。

推荐 clone 位置：

```text
tmp/open_source_refs/flash-linear-attention-npu
```

交付文档必须记录远程 URL、commit/tag、clone 路径；使用用户本地实现参考时，必须记录用户给出的路径和该路径的 git commit 或文件状态。

如果 clone 失败，交付文档记录 `clone_status=UNAVAILABLE`、失败原因，并使用本文的建议读取位置、可继承规范/不能直接继承内容、KDA dAv/GDN 同族规则和远程 URL 作为开源参考摘要继续执行。不得因为 clone 失败改用未由用户显式指定的开发机本地实现。

## 建议读取位置

| 目标 | 仓内相对路径或搜索方式 | 用途 |
|---|---|---|
| KDA NPU 实现 | `fla/ops/ascendc/kda` 或搜索 `kda` | host tiling、AIC/AIV 协作、workspace、runner、测试布局；路径基于本文 `reference_commit` |
| Python API/算子入口 | 搜索 `chunk_kda`、`chunk_bwd`、`gated_delta` | 输入输出契约、layout、stage 名称 |
| backward dAv | 搜索 `dAv`、`bwd_dav`、`chunk_kda_bwd` | `dA/dV` stage 语义、GVA/varlen/scale 处理 |
| varlen | 搜索 `cu_seqlens`、`chunk_indices` | 变长序列 chunk 映射和 offset 规则 |

## 可继承的规范

- Python API 的输入输出 tensor、dtype、layout、shape 命名。
- stage operator 与 full-flow operator 的边界划分。
- `cu_seqlens`、`chunk_indices`、`BT/chunk_size`、GVA `HV/HK` 的语义。
- host 侧 tiling、workspace size、case 数据目录、runner 参数、报告字段。
- baseline unsupported/MISSING 的状态记录方式。

## 不能直接继承的内容

- 不能把手写 AscendC 循环直接作为 Catlass 主路径；Catlass 算子仍需用 `Kernel{}(params)`、`BlockMmad`、`BlockEpilogue`、`Tile` 等模板拼装。
- 不能把 Triton block/grid 一一机械翻译为 Catlass TileShape；必须重新核算 L1/L0/UB 容量和 AIC/AIV 同步。
- 不能把本地调试路径、临时输出目录、历史 runner 二进制路径写入 skill 或交付文档。

## KDA dAv 生成规范

- `dAv` 是 backward stage operator，通常只覆盖 `dA` 与 `dV`，不是完整 KDA backward。
- 常见计算顺序：`dA_raw = dO @ V^T`，AIV 做 causal/gate/scale/mask，随后 `dV = A^T @ dO`。
- 若 `dV` 消费 AIV 后处理结果，保持 `RunDa -> signal -> wait -> RunDv`；未证明 flag/credit/resource 安全前不要重排。
- varlen/partial 场景区分物理 `BT` 计算块和 `validRows` 写回/掩码。Catlass TLA 主路径先保持 full-BT 物理块，尾块收敛必须单独 profile 和全量验收。
- GVA/GQA 场景按 `hk = hv / (HV/HK)` 管理 K-side cache scope，设计中记录 `HV % HK == 0` 约束或异常处理。
