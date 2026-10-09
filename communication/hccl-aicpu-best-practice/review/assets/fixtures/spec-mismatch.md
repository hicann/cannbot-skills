<!-- 评审器自测夹具：一份**故意与 ins_temp_all_reduce_mesh_1D_one_shot.cc 对不上**的 dataflow spec。
     只保留 review_template.py --spec 会读的那几行，用来验证 S01/S02/S03 三条规则确实会报。 -->

## 4. Buffer 布局

| 项 | 值 |
|---|---|
| **scratch 倍数** | `2` |
| 理由 | 与源码的 `templateRankSize_` 对不上 → 应触发 S01 |

## 5. 切片

| 符号 | 值 | 说明 |
|---|---|---|
| `RPT` | `4` | 声明了多次 repeat，但源码没有 repeat 循环 → 应触发 S02 |
| `IRS` / `ORS` | `S*N` / `S*N` | |
| `ISS` / `OSS` | `S` / `S` | |

## 6. 数据流

```
T0: LocalCopy   IN[0,S] -> OUT[0,S]
sync main -> sub
T[i]: SendRecvBatchWrite  IN[0,S] -> SCR@p[R*S, S]
sync main -> sub
sync sub -> main
```
