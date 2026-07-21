# AscendC Op Debug — Taxonomy（受控词汇表）

> 本文件是整个 skill 体系的「宪法」。
> **调试阶段**：只能从下表合法值中选，validate 脚本会拒绝未登记的值。
> **复盘阶段**：LLM 发现真正新模式时，先在本文件新增条目，再写 hypothesis。
> 新增流程：① 确认现有值无法覆盖 → ② 在对应维度末尾追加行 → ③ 重新执行 `build_index.sh`。

---

## 维度 1：symptom（输出现象，必填）

| value | 含义 | 常见同义描述 |
|---|---|---|
| `zero_output` | 输出全零 | all zeros / 全0 / 结果为零 |
| `nan_inf` | 输出 NaN 或 Inf | 数值爆炸 / inf / 无穷大 |
| `precision_bias` | 精度持续偏差 | max_diff 大 / 结果偏 / match_rate 低 |
| `crash` | 运行时崩溃 / 报错退出 | DDR out-of-range / coredump / exception |
| `hang` | 挂起或超时 | 507034 / kernel timeout / 卡死 |
| `multicore_mismatch` | 多核结果不一致 | 某些核错误 / 偶发错误 / 非确定性 |
| `wrong_shape` | 输出 shape 或维度错误 | 维度不对 / size mismatch |
| `compiler_merged_output` | 输出仅包含最后一次写入的值 | 所有输出等于最后一个常数 / TBuf 合并 |

---

## 维度 2：when（触发条件，必填）

| value | 含义 |
|---|---|
| `always` | 任何情况都稳定复现 |
| `multicore_only` | 单核正确，多核失败 |
| `large_shape_only` | 小 shape 正确，大 shape 失败 |
| `cross_shape_reuse` | 同进程先跑小 shape 再跑大 shape 才失败 |
| `large_D_only` | D 超过单 tile 容量上限时失败 |
| `multi_tile_only` | 单 tile 正确，多 tile 失败 |
| `large_batch_only` | 小 batch 正确，大 batch 失败 |
| `intermittent` | 偶发，无稳定复现条件 |
| `second_run_only` | 第一次调用正确，后续调用失败 |
| `loop_body_only` | 循环体内第 2 次以上迭代才失败（初始化路径正常）|
| `small_shape_only` | 小 shape（d/N < 硬件最小操作数）才失败，大 shape 正常 |
| `fp16_intermediate_cast_only` | 经过 FP16 中间步骤的多阶段 Cast（FP32→FP16→INT8）才触发 |

---

## 维度 3：root_cause（根因分类，必填）

| value | 含义 |
|---|---|
| `cache_line_conflict` | 多核 GM 写粒度 < 32B，相邻 token 落同一 cache line |
| `workspace_undersize` | workspace 按当前 shape 计算而非 maxShape，CANN 缓存导致越界 |
| `cross_tile_accum` | 跨 tile 的归约/累加未使用 scalar 变量，用了 buffer 累加 |
| `tiling_hardcode` | tiling 参数（dimTile/loopCount 等）硬编码而非动态推导 |
| `index_semantics` | 矩阵索引方向错误 / einsum 转置语义误解 |
| `ub_overalloc` | UB buffer 按完整 D 而非 dimTile 分配，大 D 时 UB 溢出 |
| `sync_missing` | SetFlag/WaitFlag/EnQue/DeQue/PipeBarrier 缺失或不配对 |
| `alignment_violation` | 写入粒度未满足 32B 对齐要求（half: 16元素, float: 8元素）|
| `idle_core_no_exit` | token 数 < 核数时，空闲核未提前 return |
| `remainder_split_error` | totalTokens 不整除 coreNum 时余数分配逻辑错误 |
| `phase_ws_missing` | 多 Phase 融合算子中，跨 Phase 使用的中间 tensor 未暂存到 workspace GM |
| `event_id_reuse` | 事件 ID 被复用导致硬件状态机混乱 |
| `datacopy_direction` | DataCopy(dst, src) 方向写反 |
| `aicore_dump_analysis` | 根因未知，需通过 AI Core dump 工具链定位（兜底入口）|
| `tbuf_compiler_merge` | CANN 编译器合并 TBuf 上的多次 Duplicate+DataCopy，只保留最后一次 |
| `inplace_alias` | CANN 框架将 input0 和 output0 映射到同一物理内存（InferShape 触发）|
| `tque_depth1_deadlock` | TQue DEPTH=1 时外层 tensor 未 Free 导致内层 AllocTensor 永久阻塞 |
| `vector_api_min_count` | Cast/DataCopy/Duplicate 元素数低于硬件最小值（< 64/32B/bufSize）|
| `reduce_ws_type_mismatch` | ReduceMax/ReduceSum 的 workspace buffer 类型与 src 不一致 |
| `aicore_aiv_guard_missing` | AIV-only kernel 缺少 ASCEND_IS_AIV 守卫，AIC 子核也执行产生竞争 |
| `double_rounding_cast_path` | 多阶段 Cast 路径每步都用 CAST_ROUND，导致 FP16 中间值被二次舍入 |

---

## 维度 4：evidence（证据层，必填）

| value | 含义 | 对应 protocol |
|---|---|---|
| `code` | 读源码可定位，无需工具（Layer 1）| protocols/read_code.md |
| `log` | 需要 plog / 错误码辅助定位（Layer 2）| protocols/parse_log.md |
| `tool_sanitizer` | 需要 msSanitizer 检测（Layer 3）| protocols/run_tools.md |
| `tool_msaicerr` | 需要 msaicerr AI Core dump 分析（Layer 3）| protocols/run_tools.md |
| `env_probe` | 需运行环境探针 kernel 排除硬件/驱动问题（非代码层可修复）| protocols/run_tools.md |

---

## 复盘阶段新增 tag 规范

新增条目时必须提供：
1. `value`：全小写，下划线分隔，英文，≤ 30 字符
2. `含义`：一句中文说明，≤ 20 字
3. `常见同义描述`（symptom/when 维度必填，root_cause/evidence 可选）

**禁止**：缩写不一致（`ws` vs `workspace`）、中英混用 value、复数形式（用 `conflict` 不用 `conflicts`）。
