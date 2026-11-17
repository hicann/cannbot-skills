# 可选已修复 Ring 样本

仅在需要此蓝本时展开，先校验optional的all-gather-ring-example卡；卡不可用不阻止独立设计。

HCCL `ff0ab744b7481086aeeb8a0bdf52e95dc915a202`（2026-10-20 核对），相对路径：

- `src/ops/all_gather/algorithm/template/ccu/ccu_temp_all_gather_ring_1D_mem2mem.{h,cc}`。
- `src/ops/all_gather/algorithm/template/ccu/kernel/ccu_kernel_all_gather_ring1d_mem2mem.{h,cc}`。

样本为 CCU_SCHED、单 kernel、prev/next 相邻建链；N≥3 两条 channel，N=2 去重一条，N=1 无对端。启动 9 个槽/缓存 11 个槽只属于此实现，不覆盖 Mesh 的 15/17 槽布局。

Ring step k=0..N-2：sendSlot=(r-k+N)%N。step0 无等待，后续先等 prev 的前一步到达再转发，本地 Write 完成后再发 Notify；最后确认末收。按每个 **接收 channel 的表** 检查同步：

| 信号 | 槽/位 | 收表 |
|---|---|---|
| ADDR：output/token | slot0，bits1/2；XN1/2 | from-next |
| DATA(k) | slot=k/16，bit=k%16 | from-prev |
| POST | slot3，bit15 | 两个方向 |

N=64 时 DATA0..62+POST 共64位，恰好占满 from-prev 的4×16位；ADDR 位于另一张表。N=2 合并收表后只有 DATA0/ADDR1,2/POST，不冲突。`assets/sync-contract.example.json` 覆盖这两个极点。不能把不同收表错误合并，也不能在 N=2 仍按两表证明。上界变化需重算峰值，不能只测 N=8。

该样本是接口与证明参考，不能作为新的算法功能/性能证据；不要复活修复前的 POST=5 或 DATA=1+k/16 布局。
