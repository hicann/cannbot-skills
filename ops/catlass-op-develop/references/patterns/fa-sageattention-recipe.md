> **定位**：Ascend950（dav-3510）量化注意力 FA-SageAttention（K 平滑 + INT8 QK + online softmax + FP16 PV）
> 的完整开发配方：数学语义、双 kernel 架构、CrossCore mode4 协议、精度取证与排障，全部在本文件与四个
> 同目录参考（[architecture](fa-sageattention-architecture.md) / [rules](fa-sageattention-rules.md) /
> [troubleshooting](fa-sageattention-troubleshooting.md) / [precision](fa-sageattention-precision.md)）内自包含。
> 与 FA 族其它算子配方同层（覆盖路由见 [fa-kernel-handcraft.md](fa-kernel-handcraft.md) §12.14-C）。


# fa-sageattention（Ascend950/dav-3510 量化注意力）开发流程与逐步骤配方

> 本 skill 是在 Ascend950PR 上从零开发 FA-SageAttention 算子的完整方法论，
> 由一次完整开发实践（约 20+ 个 bug 的全量取证与修复、22 case 功能全 PASS、
> 精度 cos 0.9996~0.9999、序列长至 200K）沉淀而来。v1（INT8 QK + FP16 PV）已达标。
> **全部开发经验固化在 四个 fa-sageattention-* 参考文件**（架构 / 8 不变量 / 坑全集 / 精度取证），自包含。

## 第 1 步 —— 算子定义与量化路径

```
输入: Q ∈ R^{B×H×S_q×D}, K ∈ R^{B×H_kv×S_kv×D}, V ∈ R^{B×H_kv×S_kv×D}  (fp16/bf16, HND)
输出: O = softmax(QKᵀ·sm_scale + causal_mask)·V ∈ R^{B×H×S_q×D}  (fp16)

SageAttention 量化路径 (v1, quantScheme=0):
  1) K 平滑:    每 128 行 KV block, μ = mean(K[blk], axis=row) ∈ R^D
                K_s = K − 1·μᵀ;  smax = max(amax|K_s|, amax|μ|);  sK = 127/smax
                K̂ = round(K_s·sK) → INT8
  2) Q 量化:    sQ = 127/amax(|Q[blk]|);  Q̂ = round(Q·sQ) → INT8
  3) QK^T:      S_int = (Q̂·K̂ᵀ)_{INT8×INT8→INT32}
                S = S_int·(sm/(sQ·sK_j)) + comp_j + mask
                其中 comp_j[r] = dot(Q̂[r,:], μ_j)·(sm/sQ)   ← K 平滑补偿, sK 代数消去
  4) softmax:   online (行 max 减法 + exp + 缩放), P ∈ [0,1] → FP16
  5) PV:        O_acc^{FP32} += P·V  (FP16 Cube)
```

**关键代数**（comp 补偿项，勿丢）：`QK_sᵀ = QKᵀ − (Q·μ)·1ᵀ`，故 `S_true = S_int·dScale + dot(Q[r,:],μ)·(sm/sQ)`。
μ 以 fp32 原值存 GM（×8 冗余），comp 在 AIV 侧 fp32 向量 dot（整数 dot ≤2M 在 fp32 精确），与旧"n=144 GEMM 增广列"逐位等价。完整公式/内存布局见 `fa-sageattention-architecture.md`。

## 第 2 步 —— 开发工作流（七步）

```
Step 1  环境与先验: CANN 9.x + dav-3510; 克隆 catlass(gitcode.com/cann/catlass);
        读 catlass/docs/（库定位与组件约束）; CrossCore flag 协议/组件选型/内存布局
        全部以本 skill references/ 为准（自包含，不查任何外部样例/算子源码）
Step 2  工程搭建: .asc 单 TU (量化kernel + FA kernel + host main);
        CMakeLists 把头文件加进 add_executable sources (否则头改动不触发重编!)
        ★ 新版 catlass: 必须设 CATLASS_ARCH=3510 (见 troubleshooting §F P29)
Step 3  Kernel A (量化, AIV-only): architecture.md §2
        → SAGE_SKIP_FA=1 + numpy 比对 qhat/khat/μ/scale 全量数值验证
Step 4  Kernel B (主算子, __mix__(1,2)): architecture.md §3
        → 分相位门控冒烟 (QK only → +SM → +PV → 全开), 每步 srun.sh 验证
Step 5  精度取证: precision-verification.md 的 numpy 仿真对拍,
        逐层定位 (S_int → comp → 状态链 → O), 直到 cos ≥ 0.99
Step 6  功能全覆盖: causal/GQA/MQA/D=64/B>1/bf16/尾块/长序列(200K)
Step 7  性能与封装: split-KV → KV tile 加宽 → (ACLNN 封装 Owner 已跳过)
```

## 第 3 步 —— 写代码前先背 8 条不变量 + 运行纪律

违反任意一条不变量 = 挂死或错数。完整 I1-I8 + Never/Always 见 `fa-sageattention-rules.md`。要点：
- **Always**：用 `srun.sh`（rc+新鲜度检查）；同卡挂 2 次换卡；`--bench` 测时；新增 tiling 字段同步加拷贝列表。
- **Never**：直跑二进制后解读输出；timeout 后解读任何文件；AIC 用 `PipeBarrier<PIPE_ALL>`；**新版 catlass 下 kernel 手动 wait MM1/MM2**（catlass 内部已 wait，双重 wait 死锁）；**PV_CONSUMED 用循环前 PIPE_FIX 预置**（冷 set 不传播）。

## 第 4 步 —— 规划 cube/vec PIPE sync（CrossCore mode4）

AIC 与 AIV 子核间的 4 相位流水（QK→SM→PV→RS）用 CrossCore mode4 flag 同步。完整 flag 表（id 分配、set/wait 双份语义、PV_CONSUMED handshake）见 `fa-sageattention-architecture.md` §3.4。要点：
- AIC set `id` 和 `16+id`（双 AIV 各一份）；AIV set 单 id（硬件按 sub-core 路由）；AIC wait `id` 和 `16+id`。
- pL1 槽复用：AIC PV 后 set PV_CONSUMED(id 12-15)，AIV SM 覆盖 pL1[m3] 前 wait（step 守卫：首次填充跳过、复用才 wait，勿用循环前预置）。

## 第 5 步 —— 排障路由

| 症状 | 去处（`fa-sageattention-troubleshooting.md`）|
|------|------|
| 单块对、多块错/挂 | §L0 槽位类 (P2/P3) + 状态链 (I4/I5) |
| 挂死无输出 | precision-verification.md §挂死取证 (SAGE_WATCH 分相位标记) |
| 精度随步数劣化 (0.9999→0.99→0.88) | I2(×8索引) → I5(状态链) → I3(MM2 surplus) 顺序排查 |
| 特定行段损坏 (rows 32-63) | AIV 子核半槽竞态 / MulDstAdd 误用 / merge sub 映射 (P33) |
| case0 splitK sub1 全 inf | merge 紧凑化只 copy sub0 (P33) → copy 512 全 slot |
| D=64 崩溃 507015 | RS 相位行距硬编码 128, 须 D-aware |
| bf16 cos≈0.28 | host 未转 bf16→fp16 (P34) |
| 编译错 ScaleGranularity 未声明 | CMakeLists 没设 CATLASS_ARCH=3510 (P29) |
| kernel 挂死、看门狗 ~27s | AIC 重复 wait MM1/MM2 (P31) 或 PV_CONSUMED 冷预置 (P32) |
| 单次 srun 7~27s 疑卡顿 | 实为每进程 setDevice/Reset 开销 (P35)，用 --bench |

## 第 6 步 —— 验证与验收

精度取证方法论（numpy 精确仿真作真值 + kernel 中间量 GM dump + 假设拟合定位）见 `fa-sageattention-precision.md`。验收标准（v1 已达标，2026-08 重建上板复现）：
- [x] **kernel 级精度: vs FP32 golden cosine ≥ 0.99** — 22/22 case PASS，cos 0.999592~0.999909
- [x] **覆盖: causal/GQA(8:1,4:1,2:1)/MQA/D=64/B>1/bf16/尾块/长序列 32K~200K 不爆内存**
- [x] **性能基线**（`--bench`，28 AIC+56 AIV）：128²≈31μs / 1k²≈73μs / 4k causal≈505μs / 200k≈719ms
- [ ] quantScheme=1 (FP8 PV) — 撞 catlass FP8 MX API 墙（P36），须先用探针 kernel 实测 catlass FP8 MX 组件真实数据流再复刻，非刚需、暂缓
- [ ] 规格补全: lse / varlen(`sageattn_varlen`) / NHD 布局 / D=96（真功能缺口，优先于 FP8）

## 第 7 步 —— 新版 catlass 适配（重建必读）

原开发用旧版 catlass；重建环境装的是 gitcode 最新 catlass，`BlockMmadTla` 行为变了。重建→22/22 PASS 修了 6 处（详见 `fa-sageattention-troubleshooting.md` §F P29-P35）：
1. **构建胶水**：CMakeLists 设 `CATLASS_ARCH=3510`；`aclrtMalloc` 的 `void*` 须经 `uint8_t*` + `(void**)&` cast 喂 `GM_ADDR`。
2. **★双重 wait 死锁**：新版 `BlockMmadTla` 内部已 `CrossCoreWaitFlag(MM1_RES/MM2_RES_INTRA_EVENT[taskId])`（FAIQK:223 / FAIPV:197，ID 与本算子全同 {9,10}/{7,8}）→ kernel 侧勿再手动 wait MM1/MM2。
3. **PV_CONSUMED bootstrap**：勿用循环前 `PIPE_FIX` 预置；改 step 守卫。
4. **merge 紧凑化**：copy 512 全 slot（sub0+sub1 的 m|l），只 copy 128 漏 sub1 → inf。
5. **bf16 输入**：host 须先 bf16→fp16(RNE) 再喂 kernel。
6. **计时纪律**：单次 srun 的 7~27s 是每进程 `aclrtSetDevice`+`Reset` 开销，非 kernel 卡顿；用 `--bench`。

## References

| 文档 | 何时读 |
|------|--------|
| `fa-sageattention-architecture.md` | 完整架构：数学公式、双 kernel、内存布局（UB/L1/L0 精确字节）、flag 表、8 条不变量、新版 catlass 适配注记 §8 |
| `fa-sageattention-rules.md` | 写代码前：8 条不变量 I1-I8 + 运行纪律 Never/Always |
| `fa-sageattention-troubleshooting.md` | 排障/重建时：平台坑全集 A~F 类（含新版 catlass 适配坑 P29-P36）：症状→根因→解法 |
| `fa-sageattention-precision.md` | 精度取证：numpy 精确仿真、GM dump 布局、假设拟合定位、挂死取证 |
