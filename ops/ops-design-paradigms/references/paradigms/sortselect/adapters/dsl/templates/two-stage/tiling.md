# DSL 两阶段排序 Tiling

## 1 输入与恢复策略

M 为独立行数，N 为轴长，C 为可用向量核数，U 为扣除 SIMT 32768 字节后的 UB，S 为键存储宽度：int64/uint64 为 8，其余为 4。`P=ceil(M/C)`。

先应用 [two-stage 经验档位](../../tiling.md#dtype-路由表)。当前浮点 N≤32、整数 N≤64 使用逆映射；更长轴使用紧凑键。浮点候选可扩展至 N≤256。

## 2 容量与键范围

每批 B 行，逻辑元素数 E=BN，物理局部宽度 W=align256(E)。预算为 `W(2S+19)+1024≤U`。局部桶计数 uint16，因此最大 W 是 65280：

```text
Emax = min(65280, floor((U−1024)/(2S+19)/256)×256)
Bmax = min(P, floor(Emax/N))
若为紧凑键：Bmax = min(Bmax, isqrt(65536//N))
```

紧凑键最大为 `B²N−1`。isqrt 使用整数平方根，避免浮点舍入放行非法键；尾批按有效行数 v 和 e=vN 生成键。

## 3 批次均衡搜索

```text
best = 无; baseRounds = 未设置
for B 从 Bmax 递减到 2:
  Q = ceil(M/B)
  active = min(C,Q)
  rounds = ceil(Q/active)
  idle = rounds×active−Q
  首个候选建立 baseRounds
  若 rounds != baseRounds：停止
  若 idle 更小：更新 best；同分保留较大 B
若无候选：返回其他排序路线
```

本模板只枚举 B≥2；Bmax<2 时返回其他排序路线。最终编译 width=W、groups=B、rank_inverse=所选恢复分支，运行 blocks=min(C,Q)。

## 4 演算

取 int32、N=256、M=1088、C=17、U=131072，S=4。局部容量上界 Emax=4608，P=64；紧凑键把 Bmax 限为 16。W=4096 时预算 `4096×27+1024=111616`，Q=68，active=17，rounds=4、idle=0。B=15 的 Q=73、rounds=5，停止搜索，选 B=16。

若浮点 N=64，则走紧凑键，B≤32；B=32 时最大键为 65535。检查这两个恢复边界，以及局部宽度补齐后超 UB 的情形。

来源：[two_stage_batch](../../example/sort/entry.py)。局部缓冲分解见 [Buffer](dag-buffers.md)，恢复算法见 [Kernel](kernel-template.md)。

本例的任务与输出追踪见 [Kernel §7](kernel-template.md#7-贯穿最后一批的原位置与名次)，工作区与存活阶段见 [Buffer §4](dag-buffers.md#4-阶段角色与预算余量)。
