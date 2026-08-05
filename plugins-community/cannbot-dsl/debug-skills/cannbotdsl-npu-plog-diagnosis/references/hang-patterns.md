# hang / 跨核同步死锁的常见形态

hang 和 fault 不同:没有错误码,`torch.npu.synchronize()` 一直不返回,直到 host 侧超时(常见顶层码 `107015`)。设备没"出错",它在**等一个永远不来的同步事件**。

## plog 里找什么

```bash
grep -E "timeout|wait|not finish|SQE|stream.*abort|task_id|Synchronize" \
    ~/ascend/log/run/plog/plog-<PID>_*.log | tail -40
```

run plog 会显示 host 侧发了 task、stream 在等结果但迟迟不返回。它通常**不直接告诉你哪个 wait 挂了**——hang 的根因要回到生成的 .asc / kernel 逻辑里查同步配对。

## 根因模式:同步原语没配平

NPU 上 cube 核(AIC)和 vector 核(AIV)跨核协作,靠 event/flag 同步。死锁几乎都是**生产端和消费端的同步次数对不上**:某个核 `wait` 一个没有人 `arrive`/`commit` 的 event。

### 排查方法:与已知能跑通的版本做计数 diff

把两版 kernel 源码里的同步原语抽出来计数:

```bash
# channel 版的同步原语
grep -oE "\.(acquire|commit|wait|release)|(vec|cube)_sync_block_(arrive|wait)" kernel.py \
  | sort | uniq -c
```

和一个**已知能跑通的等价版本**(例如同算子的另一实现)做同样统计。两边数量应当一致——某一类原语数量对不上的地方,就是同步没配平的嫌疑点。

> 经验:同一算子的 dense 版能跑、新版挂死时,对比两版的 acquire/commit/wait/release 序列(序列应完全相同)。差出来的那条就是漏发/多发的同步。

### 具体形态

- **drain 守卫写反**:流水线尾部 drain 各 stage 的欠账时,守卫条件(`if remain_idx < K`)写反 → 给一个没产出的 phantom tile 发 `wait`,但没有对应的 `arrive`/`commit` → 挂死。检查 drain 循环每个 stage 的边界条件。
- **生产端少 commit**:消费端 `wait` N 次,生产端只 `commit` N-1 次 → 消费端最后一次 wait 永远阻塞。
- **PIPE lock 等不到**:`errModule`/日志指向某 PIPE 一直 `lock` 等待 → 该 PIPE 的生产者实际没提交。对照生产者路径是否在某个分支被跳过。
- **延迟数组槽数不足**:多级流水用 IntArray 缓存跨拍标量,若数组深度 < 最深 lag+1,槽位被提前覆写,可能间接导致同步对象索引错乱。深度须 = 最深 lag + 1。

## 与 fault 的区分

| 现象 | 类型 | 入口 |
|---|---|---|
| 有 `error code` + `aicore exception` | fault | debug plog,看 core error |
| `synchronize()` 不返回 / 超时 `107015` | hang | run plog + .asc 同步计数 diff |
| 返回了但结果全 0 / NaN | 数据正确性 | 不在本 skill,走数值排查 |

hang 比 fault 更依赖**对照一个能跑通的版本**——纯看挂死现场的 plog 信息量有限,diff 才是利器。
