# 预加载深度 N 的选择

`preload_num` = 同时在飞的 macro 数。本文回答"该选 2 还是 3 还是 4"，以及它怎么和 L1 / UB 预算耦合。

## 各深度的物理含义

| N | 名字 | 同时在飞 | 说明 |
| --- | --- | --- | --- |
| 1 | 串行 | 1 | 同一 macro 内 cube QK → vec softmax → cube PV → vec update_o 严格依赖 |
| 2 | macro DB | 2 | 当前 macro 在 vec 上做 softmax 时，cube 已经开始下一个 macro 的 QK |
| 3 | macro TB | 3 | vec softmax(now) ↔ cube PV(now-1) ↔ cube QK_prefetch(now+1) 三段同时 |
| 4+ | 深流水 | 4+ | 罕见 — L1 cost 急剧上升，cube/vec 通常已经塞满 |

**FA 用 N=3 的实际理由**：softmax 需要保留 previous max/sum 做 rescale，本来就要 triple-buffer。把 p_l1、sp_l1 也跟着 N=3，槽位轮转与 softmax 状态机周期对齐——索引算法和同步通道都简单。N=2 时 vec 没空间塞 softmax_max 的"前一拍"。

## L1 cost 曲线

加 preload 主要吃 L1（p_l1 与 mxfp8 的 sp_l1）。一个典型形态（M=128, tile_n_qk=128, K_pv_total=256, fp8）：

| N | p_l1 | sp_l1 (mxfp8) | 增量（相对 N=2） |
| --- | --- | --- | --- |
| 2 | 64 KB | 4 KB | baseline |
| 3 | 96 KB | 6 KB | +34 KB |
| 4 | 128 KB | 8 KB | +68 KB |

L1 总上限 512 KB；其他必要占用（Q/K/V/sQ/sK/sV + L0 staging）约 250 KB。N=3 时 L1 ~340 KB（余 172 KB），N=4 时 ~408 KB（余 104 KB）。再深就紧。

## UB cost

softmax 状态三 buffer（max / sum / exp）随 N 线性增长，但每个 slot 只 (M, 1) fp32 = 0.25 KB。N=3 时增 ~2.25 KB，可忽略。UB 不是 preload 的瓶颈——UB 的压力来自 qk_ub、p_ub、tmp_* 这些"working set"。

## 性能收益曲线（FA 估算）

不上 NPU 实测不知精确数，但量级判断：

| N | cube 利用率 | vec 利用率 | 说明 |
| --- | --- | --- | --- |
| 1 | 30-40% | 30-40% | 大量串行等待 |
| 2 | 60-70% | 55-65% | cube tile DB 已经把 QK/PV 内部填满，宏观仍有缝 |
| 3 | 80-90% | 75-85% | 三 buffer 把 softmax/QK/PV 三段几乎完全错开 |
| 4 | 82-90% | 78-85% | 边际收益小，主要因为 vec softmax 已经是关键路径 |

3 → 4 的提升小于 2 → 3 的提升。**除非 profile 明确 cube 还有空隙，否则不要往更深走**。

## 决策树

```
profile 显示 cube 利用率 < 70%？
├─ 否 → 当前 preload 够了，不要加深
└─ 是 → 看 vec 是不是关键路径
    ├─ 否 → 不是 preload 问题，先优化 cube tile 形态
    └─ 是 → preload 有用
        L1 余量 > 100 KB？
        ├─ 是 → preload_num = 3
        └─ 否 → 先缩 K_pv_total（折半通常立即让出 32 KB）
                再走 preload_num = 3
```

## 其他配置维度

- **n_macros 太小**：preload_num=3 而 n_macros=4 时，steady-state 只跑 2 个 macro，warmup+drain 各 2 个，"在飞"的窗口很短，收益打折。host 端 `assert n_macros >= preload_num + 1`，否则 fallback 到 DB 形态。
- **subblock 拆分**：subblock 把每个 batch_head 的 M 维再切——每个 subblock 内的 n_macros 不变。subblock 不影响 preload_num 的选择，但会让 grid 大小、`subblock_idx` 在 Vector 内的 offset 计算复杂化。
- **softmax_max/sum/exp 必须 = preload_num**：不要"p_l1 = TB 但 softmax_*_ub_tb = DB"。索引轮转一拍配不上一拍，会读到错位的 previous。

## 何时回退到更浅

| 信号 | 选 |
| --- | --- |
| L1 现在就紧（剩 < 100 KB） | DB（preload_num=2） |
| n_macros 总是 ≤ 3 | DB |
| sparse 强（causal 把后半行 tile 砍掉） | DB —— 实际有效 macro 太少 |
| 短 S2（≤256） | DB —— n_macros 太少 |

variant D 的 Channel depth=3 蓝本当前只支持 `sparse_mode=0`、`S2 >= 512`。要往更小 shape 推时先在 host 端做 fallback。旧 `gqa_mxfp8_nbuffer_preload` 名称只能用于识别历史方案，不得复制其已删除的多槽 API。

## 调深的代价不只是 L1

- **代码量**：warmup + drain 各 N-1 次展开，steady-state 的生产/消费点也增多；slot 由 Channel 管理，不手写 `%N` 索引。
- **首次正确性**：stage 延迟错一个 +1 / -1 就可能产生中间 NaN。N=3 调通后再加深，每加一级都要重新过 unit_scale 与死锁检查。
- **可表达性**：Channel 只提供 FIFO 生产/消费语义。如果 N=4 设计要求随机槽读写而不能重写为 FIFO，当前前端不支持，不得伪造 `slot_now/slot_prev` 类 API。

## 常用快查

```
preload_num = 3 时（FA 标配）：
  写当前 slot   slot_now   = macro_idx % 3
  读 1 个之前   slot_prev  = (macro_idx + 2) % 3
  读 2 个之前   slot_pprev = (macro_idx + 1) % 3
```

`(macro_idx + N - k) % N` 等于 `(macro_idx - k) % N`（Python `%` 是非负的）——两种写法都对，但**显式 +N 形式更不容易看错**。
