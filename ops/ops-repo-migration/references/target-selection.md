# category 与目标仓选择

本表覆盖 spec 的 26 类；category 是计算范式，不是开源仓归属。先尊重用户目标，再结合 spec 的功能、公式和接口选择具体分类。
同名或关键词只用于找到同类算子，不能把出现 `attention`、`sparse` 或 `fused` 当作完整语义证据。

## 决策顺序

1. 原 prompt 明确拒绝迁移 → `SKIP / user_opt_out`；拒绝同时指定目标仓，或明确指定多个互斥目标 → 任务失败，不能默认消解。
2. 显式指定目标：四个支持仓中选该仓；不受 category 默认仓限制。指定 ops-rand 或其他不支持仓 → `SKIP / unsupported_target`，保留原仓名。支持仓仍须根据本表和布局确定有效 `op_class`；无法确定时 `SKIP / ambiguous_target`，保留已指定仓及歧义证据，不另选仓。
3. 没有显式目标：先匹配下述**领域规则**，再用 category 默认项；无对应实现领域或规则不足 → `SKIP / unsupported_category` 或 `ambiguous_target`。
4. RUN 必须同时给出四仓之一及其合法分类；把匹配规则、spec 原文和代表算子路径写入 evidence。不得把 26 类全部无条件映射到四仓。

## 领域规则（优先于 category 默认项）

- 已知神经网络激活 → nn/activation；张量列表操作 → nn/foreach；循环网络状态递推 → nn/rnn；哈希 embedding 状态 → nn/hash。
- attention / KV-cache / MoE 路由 / FFN / 位置编码 / Mamba / mHC / 通算融合，按实际功能分别落 transformer 的 attention / moe / ffn / posembedding / mamba / mhc / mc2。grouped matmul 落 gmm；普通 matmul 仍在 nn/matmul。
- 上一条不是关键词优先级。例如多尺度 deformable attention 在 nn/vfusion；ROI pooling 在 cv/objdetect；查清功能与同类来源后裁决。
- 视觉图像重采样 → cv/image；ROI、检测与几何框操作 → cv/objdetect，但真实同类在其他分类时以源码证据为准（如 nn/vfusion/normalize_bbox、cv/image/nms_with_mask）。
- 普通 gather/scatter/index 原语 → nn/index；KV-cache 专用 gather/scatter → transformer/attention。不会因为计算包含索引就统一落入 math。
- 融合按所属功能族判定：`fused_mul_add` 一类数值融合可在 math/math；norm/matmul/loss 融合分别在 nn 对应分类；一般神经网络向量融合才用 nn/vfusion。`FusedComposite` 没有无条件默认仓。

## 26 类映射及代表证据

路径均为真实 checkout 下的同类算子，用来证明领域存在，不意味着同名新算子可覆盖它。版本见各仓布局页。

| category | 默认或条件目标 | 代表证据 / 限制 |
|---|---|---|
| Elementwise | ops-math/math | `math/abs`；relu/gelu 等激活遵循 nn/activation 领域规则 |
| Broadcast | ops-math/math | `math/add` |
| Reduction | ops-math/math | `math/reduce_sum`；复合归一化走下一行 |
| ReductionComposite | ops-nn/norm | `norm/layer_norm`、`norm/rms_norm` |
| Contraction | ops-nn/matmul | `matmul/mat_mul_v3`；分组矩阵乘为 transformer/gmm/grouped_matmul |
| LayoutTransform | ops-math/conversion | `conversion/transpose`；领域专用变换遵循领域规则 |
| IndexGather | ops-nn/index | `index/gather_v2`、`index/gather_nd`；KV-cache 例外见 transformer/attention/gather_pa_kv_cache |
| ScatterUpdate | ops-nn/index | `index/tensor_scatter_update`、`index/scatter_update` |
| MaskPredicate | ops-math/math 或 conversion，按同类目录 | `conversion/masked_fill`；数据依赖输出的 masked_select 按 VariableOutput |
| SortSelect | ops-math/math | `math/top_k_v2`；MoE 专用路由筛选走 transformer/moe |
| FusedComposite | 按领域，无默认 | math/math/fused_mul_add；nn/vfusion/modulate；transformer/attention/flash_attention_score、ffn/ffn |
| RandomSampling | ops-math/random | `random/stateless_random_uniform_v2`、`random/drop_out_v3_grad`；不选 ops-rand |
| SlidingWindow | ops-nn/conv 或 pooling | nn/conv 的卷积、nn/pooling 的池化；ROI pooling 走 cv/objdetect |
| Collective | 有证据的通算融合 → ops-transformer/mc2 | `mc2/matmul_all_reduce`、`mc2/all_gather_matmul`；纯分布式原语 all_reduce/all_gather 不凭此自动迁移，记 unsupported_category |
| ControlFlow | 无通用默认 | nn/control 实际有 assert、sleep、浮点状态查询，不能据此认定 cond/while_loop 图控制流可迁；后者记 unsupported_category |
| Recurrence | 数值扫描 → ops-math/math；循环网络 → ops-nn/rnn | math/math/cumsum；nn/rnn/dynamic_rnn、gru；Mamba 按领域规则 |
| Interpolation | ops-cv/image 或 objdetect | `image/grid_sampler2_d`、`image/resize_bilinear_v2`；ROI 为 `objdetect/roi_align` |
| Quantization | 独立量化 → ops-nn/quant | `quant/quantize`、`quant/dequantize`；融合量化按 matmul、norm、attention 等所属功能族 |
| Stateful | ops-nn/optim、norm 或 hash | `optim/apply_adam_w`、`optim/sgd`、含 running stats 的 batch norm、`hash/embedding_hash_table_lookup_or_insert`；其他状态语义有歧义则跳过 |
| ArgReduce | ops-math/math | `math/arg_max_v2` |
| Sparse | 按领域，无默认 | math/conversion/coalesce_sparse、sparse_reshape；nn/matmul/sparse_tensor_dense_mat_mul、optim/sparse_apply_adagrad；transformer/attention/block_sparse_attention |
| Padding | ops-math/conversion | `conversion/mirror_pad`；以同类 pad 的实际目录为准 |
| VariableOutput | ops-nn/index | `index/non_zero`、`index/unique_consecutive` |
| Spectral | ops-math/math | `math/rfft1_d` |
| Histogram | ops-math/math | `math/histogram_v2` |
| AtomicUpdate | ops-nn/index | `index/scatter_add`、`index/scatter_nd_add`；特定领域融合仍按领域规则 |

## migrate-decision.json

```json
{
  "decision": "RUN",
  "target_repo": "ops-cv",
  "op_class": "image",
  "reason_code": "category_match",
  "reason": "输入像素由浮点坐标插值得到，归入图像重采样",
  "source_package": "/absolute/work-dir/op",
  "evidence": {
    "prompt_file": "/absolute/prompt.txt",
    "matched_statement": null,
    "category": "Interpolation",
    "routing_rule": "Interpolation: image resampling",
    "spec_evidence": "引用该算子 spec 中的实际公式或功能描述",
    "representative": "ops-cv/image/grid_sampler2_d"
  }
}
```

RUN reason_code 为 `explicit_target` / `domain_match` / `category_match`；SKIP 为上述四种原因，`op_class=null`，未确定仓时 `target_repo=null`。
`unsupported_target` 必须保留用户指定的不支持仓名；显式支持仓但分类有歧义也保留该仓。prompt 文件缺失/为空只表示没有显式要求，不能编造用户语句。
运行 `python3 "$SKILL_DIR/scripts/migration_contract.py" decision "$WORK_DIR/.workflow/migrate-decision.json"` 校验字段；语义和证据仍由执行者与 verifier 核实。
