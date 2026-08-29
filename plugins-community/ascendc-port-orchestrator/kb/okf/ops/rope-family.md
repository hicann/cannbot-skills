---
schema_version: okf.v1
kind: operator
type: operator
source_family: ops
title: "Rope 类算子族（5 个泛化算子）：布局/语义/aclnn 接口速查"
description: "RopeWithSinCosCache / inplace_partial_rotary_mul / rotary_position_embedding / apply_rotary_pos_emb / kv_rms_norm_rope_cache 的旋转约定、数据布局、aclnn 接口与参数限制速查，供 porter 生成/转译时快速对齐语义。"
tags: [rope, rotary, position-embedding, kv-cache, rms-norm]
created_at: 2026-08-27T00:00:00Z
updated_at: 2026-08-27T00:00:00Z
---

# Rope 类算子族速查

基于 5 个已 case 泛化 Rope 算子在 Ascend 910B3 上的生成/转译实践。完整性能经验见
生成器 skill 的 references/best-practices/rope-developer-mode.md。

## 语义速查表

| 算子 | 输入组织 | 旋转约定 | 特殊语义 |
|---|---|---|---|
| `RopeWithSinCosCache` | `(N,H,head_size)` 3D | half 半切 | 3D 整头 whole-head 拷贝；hs>rd 尾部直通 |
| `inplace_partial_rotary_mul` | `(B,N,S,D)` 4D | half / interleave 按配置 | partial slice 之外的数据不得修改（in-place） |
| `rotary_position_embedding` | `(N,nh,D)` + bs1d cos | neox / 非 neox | cos/sin 按 head 维广播；小 case 需 MTE2 repeat 广播 |
| `apply_rotary_pos_emb` | `(nt,1,2,rh)` 4D 交错 | half（first/second 或 even/odd） | dim2 第 0/1 片分别放两半；kernel 内零重排 |
| `kv_rms_norm_rope_cache` | `(B,S,dkv)` | cat（kv 特有） | rms-first：`kv[..., :rms]`=RMSNorm、`kv[..., rms:]`=rope；结果写回 cache |

## 关键参数限制

- **aclnnKvRmsNormRopeCache**：cos/sin 第一维必须等于 B（`[B,N,S,Dk]` 或
  `[B,1,1,Dk]`），`[1,...]` 报 tiling 错误 `561002`（cos or sin shape is
  invalid）；V1 结果写回 k_cache/ckv_cache（k_rope/c_kv 独立输出可能为 None）
- **aclnnApplyRotaryPosEmbV2**：可 C wrapper 直调；禁止 torch_npu Python 绑定
  （会分解出 Cast/BroadcastTo/ZerosLike 子 op，msprof 抓到子 op 使性能对比失真，
  实测 kv_rms_norm_rope_cache 0.46x → C wrapper 修正 1.17x）
- **布局分发原则**：3D 源输入 `(N,H,hs)` 走整头 whole-head 拷贝；天然 4D 交错
  `(nt,1,2,rh)` 走零重排；禁止按 hs==rd 把 3D 强转 4D（加速比 2.52x → 1.71x）
- **精度**：fp16 一律 fp32 中间计算后 cast 回（max_diff≈0.004）；验证须覆盖
  head_size=256 边界（rotary 切片按 head 维 `view(N,H,hs)[:,:,:rd]`）
