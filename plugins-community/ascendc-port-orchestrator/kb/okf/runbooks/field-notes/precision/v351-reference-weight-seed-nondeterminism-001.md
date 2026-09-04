---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "reference/candidate weight seed via hash(torch.dtype) is address-based and process-unstable — golden/fixture and eval-process weights diverge"
description: "npubench tasks that seed weights with torch.manual_seed(hash(key)) where key contains torch.dtype get DIFFERENT weights in every process (hash(torch.dtype) is address-based, not value-based; PYTHONHASHSEED does not control it). Frozen golden/fixture and the candidate's eval-process weights then disagree, producing fake precision FAILs whose failing-case identity hops across O5 rounds on a byte-identical candidate tree."
phenomenon: precision_issue
signal:
  - "failing-case IDENTITY hops across O5 rounds while candidate source SHA stays byte-identical"
  - "near-miss failures with matched_ratio > 0.99 and MERE within ~1.3x rtol on a candidate that once scored 50/50"
  - "reference task contains torch.manual_seed(hash(key)) with a key tuple that includes x.dtype or x.device"
  - "pre-generated weight npz / golden / phase_golden.json disagrees with the runtime process weights"
confidence: single_run
original_id: v351-reference-weight-seed-nondeterminism-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, precision, weight-seed, nondeterminism, reference, npubench, hash]
created_at: 2026-08-26T12:10:00Z
updated_at: 2026-08-26T12:10:00Z
---
## 现象 / 触发

参考任务（npubench task .py）或 candidate model 里出现：

```python
torch.manual_seed(hash(key) & 0xFFFFFFFF)   # key 含 x.dtype / x.device
```

症状（55 pp-7 / 57 kw-28,34 实踩）：

1. 字节级相同的候选树，跨轮 O5 失败 case 身份反复漂移（55：50→49→48→49→48→48，失败 case 7→15→20→14/41→23/33）；
2. 预生成 golden/fixture（input-gen 进程）与评测进程的 candidate 权重不一致 → 假精度 FAIL；
3. 跨进程 `--golden phase_golden.json` 相位对比无效（57 kw-34：连 PASS 的 case45 都被假报 q_ch 发散）。

## 根因 / 教训

2026-08-26 实测：`hash(torch.dtype)` 三个进程三个值（地址基哈希，C++ 对象地址）；`hash(torch.device('cpu'))` 稳定（255）。
**`PYTHONHASHSEED=0` 管不住它**（只影响 str/bytes 哈希）。凡是 seed 依赖含 torch 对象的元组哈希，跨进程必漂移。

## 正确修法（推荐 ③）

1. **冻结 capture 判分**：golden/fixture 一次生成后落盘复用，评测只比对不重生成；
2. **参考权重落盘复用**：权重 npz 与评测进程共享；
3. **key 剔除 dtype/device（本卡采用）**：seed 只取 key 的稳定 int-only 部分：

```python
# 55：torch.manual_seed(hash((attn_dim, num_heads, kernel_size, padding, stride)) & 0xFFFFFFFF)
# 57：torch.manual_seed(hash((channel, c2)) & 0xFFFFFFFF)
```

task 与 candidate **双侧同改**，改完必须重冻结 reference bundle（stage_npubench_inputs + 旋转 .opgen_state.json 的 reference 块）。

## 判定纪律

- 失败 case 身份在字节级相同候选上跨轮漂移 ≠ 回归，是评测噪声；
- 跨轮 PASS 数 / MERE 直接对比无意义（权重不同）；
- 边界 near-miss 的"修复"必须防翻转：matched_ratio ≥ 0.93 且修法有机制证据，否则是噪声拟合（55 kw-18/21/22 教训）。
