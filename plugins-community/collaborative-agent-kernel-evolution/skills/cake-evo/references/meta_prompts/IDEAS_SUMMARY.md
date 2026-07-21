# IdeaPool优化思想汇总

本文档汇总了从43个生产级算子中提取的优化策略。

## Attention

共41个优化策略，来自以下算子:

- **multi_scale_deformable_attention_grad**: 9个策略
- **multi_scale_deformable_attn_function**: 10个策略
- **scaled_masked_softmax_grad_v2**: 12个策略
- **scaled_masked_softmax_v2**: 10个策略

## Elementwise

共69个优化策略，来自以下算子:

- **clipped_swiglu**: 10个策略
- **foreach_abs**: 8个策略
- **foreach_add_list**: 12个策略
- **foreach_add_scalar**: 12个策略
- **foreach_add_scalar_list**: 7个策略
- **foreach_addcdiv_list**: 9个策略
- **modulate**: 11个策略

## Indexing

共54个优化策略，来自以下算子:

- **embedding_dense_grad_v2**: 8个策略
- **gather_elements_v2**: 11个策略
- **linear_index**: 9个策略
- **masked_scatter_with_position**: 9个策略
- **scatter_elements_v2**: 9个策略
- **sparse_to_dense**: 8个策略

## Normalization

共109个优化策略，来自以下算子:

- **add_rms_norm_cast**: 11个策略
- **add_rms_norm_dynamic_quant**: 10个策略
- **batch_norm_v3**: 9个策略
- **deep_norm**: 17个策略
- **gemma_rms_norm**: 9个策略
- **inplace_add_rms_norm**: 10个策略
- **layer_norm_v3**: 13个策略
- **layer_norm_v4**: 9个策略
- **rms_norm_grad**: 9个策略
- **rms_norm_quant**: 12个策略

## Optimizer

共18个优化策略，来自以下算子:

- **apply_adagrad_d**: 8个策略
- **apply_adam_w_v2**: 10个策略

## Quantization

共75个优化策略，来自以下算子:

- **ascend_quant_v2**: 8个策略
- **dequant_bias**: 9个策略
- **dynamic_block_quant**: 13个策略
- **dynamic_mx_quant**: 8个策略
- **dynamic_quant_update_scatter_v2**: 9个策略
- **fake_quant_affine_cachemask**: 10个策略
- **grouped_dynamic_mx_quant**: 9个策略
- **trans_quant_param_v2**: 9个策略

## Reduction

共48个优化策略，来自以下算子:

- **adaptive_avg_pool3d**: 10个策略
- **adaptive_max_pool3d_grad**: 10个策略
- **max_pool_grad_with_argmax_common**: 9个策略
- **max_pool_with_argmax_v3**: 10个策略
- **norm_common**: 9个策略

