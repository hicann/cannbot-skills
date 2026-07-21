# INDEX.md — 自动生成，勿手动编辑
# 由 scripts/build_index.sh 从 hypotheses/ frontmatter 生成
# 更新方法：新增/修改 hypothesis 后执行 bash scripts/build_index.sh

## 全量假设列表

| ID | Title | symptom | when | root_cause | evidence | escalate_to |
|---|---|---|---|---|---|---|
| [H01](hypotheses/H01_cache_line_write_conflict.md) | 多核 GM 写冲突（小输出 tensor < 32B） | `zero_output` | `multicore_only` | `cache_line_conflict` | `code` | null |
| [H02](hypotheses/H02_workspace_shape_cache.md) | Workspace 跨 Shape 缓存越界 | `crash` | `cross_shape_reuse` | `workspace_undersize` | `code` | null |
| [H03](hypotheses/H03_cross_tile_accumulation.md) | D-axis 跨 Tile 累加未用 scalar 变量 | `precision_bias` | `large_D_only` | `cross_tile_accum` | `code` | null |
| [H04](hypotheses/H04_phase_workspace_missing.md) | 多 Phase 融合算子跨 Phase 中间 tensor 未写 workspace | `precision_bias` | `large_D_only` | `phase_ws_missing` | `code` | null |
| [H05](hypotheses/H05_einsum_transpose_semantics.md) | einsum 转置语义错误（矩阵索引方向写反） | `precision_bias` | `always` | `index_semantics` | `code` | null |
| [H06](hypotheses/H06_ub_buffer_overalloc.md) | UB buffer 按 D 而非 dimTile 分配导致溢出 | `crash` | `large_D_only` | `ub_overalloc` | `code` | null |
| [H07](hypotheses/H07_tiling_hardcode.md) | Tiling 参数硬编码导致大 shape 错误 | `precision_bias` | `large_shape_only` | `tiling_hardcode` | `code` | null |
| [H08](hypotheses/H08_idle_core_no_exit.md) | 空闲核未安全退出（tokens < coreNum） | `crash` | `large_shape_only` | `idle_core_no_exit` | `code` | null |
| [H09](hypotheses/H09_remainder_token_split.md) | 核间 token 余数分配逻辑错误 | `precision_bias` | `large_batch_only` | `remainder_split_error` | `code` | null |
| [H10](hypotheses/H10_datacopy_direction_barrier.md) | DataCopy 方向写反 / PipeBarrier 缺失 | `zero_output` | `always` | `datacopy_direction` | `code` | null |
| [H11](hypotheses/H11_sync_missing.md) | SetFlag/WaitFlag 不配对或 SetFlag 顺序错误 | `multicore_mismatch` | `intermittent` | `sync_missing` | `tool_sanitizer` | mssanitizer |
| [H12](hypotheses/H12_event_id_reuse.md) | 事件 ID 复用导致硬件状态机混乱 | `hang` | `multicore_only` | `event_id_reuse` | `tool_sanitizer` | mssanitizer |
| [H13](hypotheses/H13_alignment_violation.md) | 内存对齐违规（向量化操作数据不对齐） | `precision_bias` | `large_shape_only` | `alignment_violation` | `code` | mssanitizer |
| [H14](hypotheses/H14_aicore_dump.md) | AI Core Dump 分析（msaicerr 工具链） | `crash` | `always` | `aicore_dump_analysis` | `tool_msaicerr` | msaicerr |
| [H15](hypotheses/H15_tbuf_compiler_merge.md) | TBuf 多写被 CANN 编译器合并（只保留最后一次写入） | `compiler_merged_output` | `always` | `tbuf_compiler_merge` | `code` | null |
| [H16](hypotheses/H16_inplace_alias.md) | CANN 框架 Inplace 别名（input0 与 output0 共享物理内存） | `precision_bias` | `always` | `inplace_alias` | `code` | null |
| [H17](hypotheses/H17_tque_depth1_deadlock.md) | TQue DEPTH=1 多 lifetime 死锁（内外循环共用同一 queue） | `hang` | `loop_body_only` | `tque_depth1_deadlock` | `code` | null |
| [H18](hypotheses/H18_vector_api_min_count.md) | 向量 API 最小操作数违规（Cast/DataCopy/Duplicate 元素数低于硬件下限） | `hang` | `small_shape_only` | `vector_api_min_count` | `code` | null |
| [H19](hypotheses/H19_reduce_ws_type_mismatch.md) | ReduceMax/ReduceSum workspace buffer 类型与 src 不匹配 | `precision_bias` | `always` | `reduce_ws_type_mismatch` | `code` | null |
| [H20](hypotheses/H20_aicore_aiv_guard_missing.md) | ASCEND_IS_AIV 守卫缺失 → AIC/AIV 数据竞争 | `multicore_mismatch` | `always` | `aicore_aiv_guard_missing` | `code` | null |
| [H26](hypotheses/H26_double_rounding_cast_path.md) | 多阶段 Cast 路径双重舍入（Double-Rounding via FP16 Intermediate） | `precision_bias` | `fp16_intermediate_cast_only` | `double_rounding_cast_path` | `code` | null |

---

## 按症状索引（symptom）

### `compiler_merged_output`
- **H15** [when=`always`] TBuf 多写被 CANN 编译器合并（只保留最后一次写入） → ev=`code`

### `crash`
- **H02** [when=`cross_shape_reuse`] Workspace 跨 Shape 缓存越界 → ev=`code`
- **H06** [when=`large_D_only`] UB buffer 按 D 而非 dimTile 分配导致溢出 → ev=`code`
- **H08** [when=`large_shape_only`] 空闲核未安全退出（tokens < coreNum） → ev=`code`
- **H14** [when=`always`] AI Core Dump 分析（msaicerr 工具链） → ev=`tool_msaicerr`

### `hang`
- **H12** [when=`multicore_only`] 事件 ID 复用导致硬件状态机混乱 → ev=`tool_sanitizer`
- **H17** [when=`loop_body_only`] TQue DEPTH=1 多 lifetime 死锁（内外循环共用同一 queue） → ev=`code`
- **H18** [when=`small_shape_only`] 向量 API 最小操作数违规（Cast/DataCopy/Duplicate 元素数低于硬件下限） → ev=`code`

### `multicore_mismatch`
- **H11** [when=`intermittent`] SetFlag/WaitFlag 不配对或 SetFlag 顺序错误 → ev=`tool_sanitizer`
- **H20** [when=`always`] ASCEND_IS_AIV 守卫缺失 → AIC/AIV 数据竞争 → ev=`code`

### `precision_bias`
- **H03** [when=`large_D_only`] D-axis 跨 Tile 累加未用 scalar 变量 → ev=`code`
- **H04** [when=`large_D_only`] 多 Phase 融合算子跨 Phase 中间 tensor 未写 workspace → ev=`code`
- **H05** [when=`always`] einsum 转置语义错误（矩阵索引方向写反） → ev=`code`
- **H07** [when=`large_shape_only`] Tiling 参数硬编码导致大 shape 错误 → ev=`code`
- **H09** [when=`large_batch_only`] 核间 token 余数分配逻辑错误 → ev=`code`
- **H13** [when=`large_shape_only`] 内存对齐违规（向量化操作数据不对齐） → ev=`code`
- **H16** [when=`always`] CANN 框架 Inplace 别名（input0 与 output0 共享物理内存） → ev=`code`
- **H19** [when=`always`] ReduceMax/ReduceSum workspace buffer 类型与 src 不匹配 → ev=`code`
- **H26** [when=`fp16_intermediate_cast_only`] 多阶段 Cast 路径双重舍入（Double-Rounding via FP16 Intermediate） → ev=`code`

### `zero_output`
- **H01** [when=`multicore_only`] 多核 GM 写冲突（小输出 tensor < 32B） → ev=`code`
- **H10** [when=`always`] DataCopy 方向写反 / PipeBarrier 缺失 → ev=`code`

---

## 按根因索引（root_cause）

### `aicore_aiv_guard_missing`
- **H20** ASCEND_IS_AIV 守卫缺失 → AIC/AIV 数据竞争

### `aicore_dump_analysis`
- **H14** AI Core Dump 分析（msaicerr 工具链）

### `alignment_violation`
- **H13** 内存对齐违规（向量化操作数据不对齐）

### `cache_line_conflict`
- **H01** 多核 GM 写冲突（小输出 tensor < 32B）

### `cross_tile_accum`
- **H03** D-axis 跨 Tile 累加未用 scalar 变量

### `datacopy_direction`
- **H10** DataCopy 方向写反 / PipeBarrier 缺失

### `double_rounding_cast_path`
- **H26** 多阶段 Cast 路径双重舍入（Double-Rounding via FP16 Intermediate）

### `event_id_reuse`
- **H12** 事件 ID 复用导致硬件状态机混乱

### `idle_core_no_exit`
- **H08** 空闲核未安全退出（tokens < coreNum）

### `index_semantics`
- **H05** einsum 转置语义错误（矩阵索引方向写反）

### `inplace_alias`
- **H16** CANN 框架 Inplace 别名（input0 与 output0 共享物理内存）

### `phase_ws_missing`
- **H04** 多 Phase 融合算子跨 Phase 中间 tensor 未写 workspace

### `reduce_ws_type_mismatch`
- **H19** ReduceMax/ReduceSum workspace buffer 类型与 src 不匹配

### `remainder_split_error`
- **H09** 核间 token 余数分配逻辑错误

### `sync_missing`
- **H11** SetFlag/WaitFlag 不配对或 SetFlag 顺序错误

### `tbuf_compiler_merge`
- **H15** TBuf 多写被 CANN 编译器合并（只保留最后一次写入）

### `tiling_hardcode`
- **H07** Tiling 参数硬编码导致大 shape 错误

### `tque_depth1_deadlock`
- **H17** TQue DEPTH=1 多 lifetime 死锁（内外循环共用同一 queue）

### `ub_overalloc`
- **H06** UB buffer 按 D 而非 dimTile 分配导致溢出

### `vector_api_min_count`
- **H18** 向量 API 最小操作数违规（Cast/DataCopy/Duplicate 元素数低于硬件下限）

### `workspace_undersize`
- **H02** Workspace 跨 Shape 缓存越界

