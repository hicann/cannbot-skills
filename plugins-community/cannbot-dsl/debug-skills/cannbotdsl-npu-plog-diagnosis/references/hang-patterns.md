# hang / 跨核同步死锁的常见形态

hang 和 fault 不同:没有错误码,`torch.npu.synchronize()` 一直不返回,直到 host 侧超时。设备没"出错",它在**等一个永远不来的同步事件**。

## plog 里找什么

```bash
grep -E "timeout|wait|not finish|SQE|stream.*abort|task_id|Synchronize" \
    ~/ascend/log/run/plog/plog-<PID>_*.log | tail -40
```

run plog 会显示 host 侧发了 task、stream 在等结果但迟迟不返回。它通常**不直接告诉你哪个 wait 挂了**——hang 的根因要回到生成的 .asc / kernel 逻辑里查同步配对。

## 根因模式:同步原语没配平

NPU 上 cube 核(AIC)和 vector 核(AIV)跨核协作,靠 event/flag 同步。死锁几乎都是**生产端和消费端的同步次数对不上**:某个核在等一个没有人 `arrive` 的 event。

### 排查方法:与已知能跑通的版本做计数 diff

把两版 kernel 源码里的同步原语抽出来计数:

```bash
# channel 版的同步原语
grep -oE "(vec|cube)_sync_block_(arrive|wait)|sync_(intra|notify)_[a-z]+" kernel.py \
  | sort | uniq -c
```

和一个**已知能跑通的等价版本**(例如同算子的另一实现)做同样统计。两边数量应当一致——某一类原语数量对不上的地方,就是同步没配平的嫌疑点。

> 经验:同一算子的 dense 版能跑、新版挂死时,对比两版的同步原语序列(序列应完全相同)。差出来的那条就是漏发/多发的同步。

### 具体形态

- **drain 守卫写反**:流水线尾部 drain 各 stage 的欠账时,守卫条件(`if remain_idx < K`)写反 → 给一个没产出的 phantom tile 发 `wait`,但没有对应的 `arrive` → 挂死。检查 drain 循环每个 stage 的边界条件。
- **生产端少发一次**:消费端等待 N 次,生产端只发 N-1 次 → 消费端最后一次等待永远阻塞。
- **PIPE lock 等不到**:`errModule`/日志指向某 PIPE 一直 `lock` 等待 → 该 PIPE 的生产者实际没提交。对照生产者路径是否在某个分支被跳过。
- **延迟数组槽数不足**:多级流水用 IntArray 缓存跨拍标量,若数组深度 < 最深 lag+1,槽位被提前覆写,可能间接导致同步对象索引错乱。深度须 = 最深 lag + 1。
- **负载不均衡致单核 timeout**:sync 原语计数配平、但某核分到的 tile 代价远超平均（causal 下 m-block 轴放 `idx2crd` 最内层且 extent 整除 GRID → 每核工作量恒定不均）。与上述 sync 不配平不同,这不是死锁而是**最慢的核拖垮全核**。先用 host 侧算术算每核负载（`max(load)/mean(load) > 1.2` → 分发问题）,见 `../../../core-skills/cannbotdsl-perf-optimize/SKILL.md` 第 0 步。
- **`const_expr` 守卫变负致越界间接 hang**：`NPAD = VH - BMV` 变负时 `if const_expr(NPAD > 0):` 静默跳过,被保护的越界写读照常发生 → 可能写坏 sync 状态或踩到其他 channel 的缓冲区 → 间接触发 hang。plog 里看不到 sync 不配平,但会看到越界地址访问。查所有 `const_expr` 守卫的变量是否可能为负,见 `../../../core-skills/cannbotdsl-vf-fusion/SKILL.md` 陷阱 11。

## 与 fault 的区分

| 现象 | 类型 | 入口 |
|---|---|---|
| 有 `error code` + `aicore exception` | fault | debug plog,看 core error |
| `synchronize()` 不返回 / 超时 | hang | run plog + .asc 同步计数 diff |
| 返回了但结果全 0 / NaN | 数据正确性 | 不在本 skill,走数值排查 |

hang 比 fault 更依赖**对照一个能跑通的版本**——纯看挂死现场的 plog 信息量有限,diff 才是利器。
