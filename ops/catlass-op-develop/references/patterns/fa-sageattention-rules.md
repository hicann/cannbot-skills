# FA-SageAttention 硬规则（不变量 + 运行纪律）

> 违反任意一条 = 挂死或错数。写代码前通读，写完逐条对照。

## A. 8 条不变量（I1-I8，违反即挂死/错数）

| # | 不变量 | 违反后果 |
|---|--------|---------|
| I1 | **L0A/L0B 槽距对齐**: FAIQK 读 Shape=(M,N,K)、FAIPV 读 (M,K,N); 两 block 的 L0A/L0B 槽都从 0 布局且共享事件 id {0..3}, 槽距必须完全相等(都=32KB: QK 垫 K_static=256) | P≥2 竞态挂死/错数 |
| I2 | **×8 冗余张量索引**: SQ/SK/mu 的 GM 元素下标必须 ×8 (mu 是 ·d·8) | j>0 块全读块 0 的 scale |
| I3 | **MM1/MM2 释放只 set 本步槽**: 双 set 产生 surplus → 下下个 PV 提前通过、覆盖还在被读的 oUb | O 的 rows 32-63 损坏 |
| I4 | **游标 jCur**: SageCursorNext 返回前已 j++, 调用方一律用 c.jCur | 步序 [1,2,0] 错乱 |
| I5 | **状态链分槽**: step s 读 lastMax/sumL 的 (s-1)%N 槽, 写 s%N 槽 | 多步精度劣化 |
| I6 | **CrossCore mode4**: AIC set id 和 16+id (双 AIV 各一份); AIV set 单 id (硬件按 sub-core 路由); AIC wait id 和 16+id | 挂死或单 AIV 数据缺失 |
| I7 | **AIC 上禁 `PipeBarrier<PIPE_ALL>`** (AIC 管集合无 V/MTE3) | 必挂死 |
| I8 | **pL1 槽复用 handshake**: AIC PV 后 set PV_CONSUMED(id 12-15), AIV SM 覆盖 pL1[m3] 前 wait; P-Cast 前要 `WaitFlag<MTE3_V>(m3Prev)` | P>槽数时挂死 |

## B. 运行纪律（血泪教训，Never/Always）

### Always
- 一律用安全运行器 `SAGE_DEV=N bash scripts/srun.sh <dir>`（内置 rc + o.bin 新鲜度检查，防 stale 欺骗）
- 同卡挂 2 次立即换卡；npu-smi Health 可能仍显示 OK 但核已污染
- 用 `--bench` 测真实 kernel 用时（单次 srun 的 7~27s 是每进程 `aclrtSetDevice`/`Reset` 开销，非 kernel 卡顿）
- 新增 tiling 字段必须同步加进 kernel 内 `td` 逐字段拷贝列表（漏→设备读栈垃圾→P=0→输出全零）
- 改 .hpp/被 `#include` 的文件后，必须 touch 引用它的 .asc TU 源文件再 make（CMake 不跟踪 include 依赖，旧 exe 假验证），并确认 exe mtime > 源 mtime

### Never
- 绝不直接跑 `./build/sage_fa` 后解读输出 —— kernel 挂死被 timeout 杀掉后 o.bin 不更新，读到的是旧文件的假结果（一次事故污染全部 8 张卡）
- timeout 后绝不能解读任何输出文件
- AIC 上绝不用 `PipeBarrier<PIPE_ALL>`（见 I7）
- 新版 catlass 下：kernel 侧勿再手动 `CrossCoreWaitFlag MM1/MM2`（catlass `BlockMmadTla` 内部已 wait，双重 wait 永久阻塞，见 troubleshooting §F P31）
- PV_CONSUMED 勿用循环前 `PIPE_FIX` 预置（冷 set 不传播，首 SM 永卡）；改 step 守卫（见 troubleshooting §F P32）
