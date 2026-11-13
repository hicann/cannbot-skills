# DSL 两阶段排序 Tiling

## 1 输入与恢复策略

M 为独立行数，N 为轴长，C 为可用向量核数，U 为扣除 SIMT 32768 字节后的 UB，S 为键存储宽度：int64/uint64 为 8，其余为 4。`P=ceil(M/C)`。

先应用 [two-stage 经验档位](../../tiling.md#dtype-路由表)。已通过档位检查的候选中，N≤32 使用逆映射；更长轴使用稳定行号分组。当前浮点档位最大 N=256；整数上限按各 dtype 表判断，不能只依据恢复阈值放行。

## 2 容量与键范围

每批 B 行，逻辑元素数 E=BN，物理局部宽度 W=align256(E)。预算为 `W(2S+14)+1024≤U`。局部桶计数 uint16，因此最大 W 是 65280：

```text
Emax = min(65280, floor((U−1024)/(2S+14)/256)×256)
Bmax = min(P, floor(Emax/N))
若为行号分组：Bmax = min(Bmax, 256)
```

行号键范围为 `[0,B−1]`；B≤256 保证只需一个字节。尾批仍按有效行数 v 和 e=vN 遍历。稳定性使 Stage1 排名无需再次编码。

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

本模板只枚举 B≥2；Bmax<2 时返回其他排序路线。选中 B 后，W=align256(BN)。编译入口传 width=W、groups=B 和 route=4/5；`sort_host.py` 据 route 向 `SmallAxisTwoStage` 固定传入 rank_inverse=True/False，不是在 VF 内运行时选择。运行 blocks=min(C,Q)。

## 4 演算

取 int32、N=256、M=1088、C=17、U=131072，S=4。局部容量上界 Emax=5888，P=64，Bmax=23。首个候选 B=23 的 Q=48、rounds=3、idle=3；B=22 的 Q=50、idle=1，因此选 B=22。B=21 需要 4 轮，停止搜索。W=5632，预算 `5632×22+1024=124928` 字节。

浮点 N=64 走行号分组，B 的上限由 UB、行数和 256 行共同决定；不再受 B²N≤65536 限制。验证 255/256/257 行的低层字节切换、尾批稳定性与局部宽度补齐后的 UB 容量。

来源：[two_stage_batch](../../example/sort/entry.py)。局部缓冲分解见 [Buffer](dag-buffers.md)，恢复算法见 [Kernel](kernel-template.md)。

本例的任务与输出追踪见 [Kernel §7](kernel-template.md#7-贯穿最后一批的原位置与名次)，工作区与存活阶段见 [Buffer §4](dag-buffers.md#4-阶段角色与预算余量)。
