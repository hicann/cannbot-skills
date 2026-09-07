# FA 族手搓 kernel 坑清单（AIC/AIV 双 launch / 双子核 / online softmax / 多 batch）

> **适用**：FA 族形态真无整经验证 kernel（如 FFA 双 KV 等变体）必须手搓时，与 [fa-kernel-handcraft.md](fa-kernel-handcraft.md)（手搓手册：§1–9 复刻教程、§10 变体映射法、§12.5 实证库）**配套读取**——手册教怎么写对，本清单教别踩哪。**标准 FA / MLA 有整经验证内核路径，优先采用**（见 [flash-attention.md](flash-attention.md) §0.3 / [fa-mla-paged.md](fa-mla-paged.md) §0）；本清单全部条目来自 MLA 手搓工程 / FA 封装工程 手搓工程实测（910B3），与配方册 §12.5 的 自写 FFA 工程 实证互补。
> **来源**：MLA 手搓工程 / FA 封装工程 手搓工程全链路实测（910B3 / dav-2201），全部二分实证；最终手搓版 38/38+32/32 精度 PASS（对照组：标杆 kernel 性能是手搓的 3~9×，手搓只作正确性/兜底与组件无覆盖形态）。

---

## 1. 七个架构级坑（新 W 级，每个都挂死或错数过）

| # | 坑 | 现象 | 对策 |
|---|---|---|---|
| H1 | **FAQK 双实例不可共存** | 同一 kernel 进程 new 第二个 FAQK 对象，仅构造（未调用）即破坏 HardEvent 配平 → 出口挂死 | QK 全部走单实例；多任务在实例内分趟 |
| H2 | **AIV→AIC `CrossCoreSetFlag` 双子核都必须发** | 单子核发 flag（MODE 0x2 需 2 Set:1 Wait 配平）→ AIC WaitFlag 永阻塞 | 2Set:1Wait：AIV 双子核都 Set；反向 AIC→AIV 单 Set 可满足双 Wait |
| H3 | **设备 printf / int→float cast 不可用** | 设备侧打印或隐式转换直接编译/运行异常 | 调试用 GM dump 槽 + host 读回；数值比较全在 host |
| H4 | **双子核共享同核 UB** | 两个 subblock 各拿 `GetSubBlockIdx()` 以为独立 UB，实际同一块 → 互相覆盖 | 写缓冲按 `subIdx × SEG` 显式分区（如 `subIdx*32768`） |
| H5 | **ubMax/ubSum 需 512B 对齐** | 64B 对齐时 CrossCore 读写越界覆盖相邻行 | 归约 buffer 一律 512B 对齐起址 |
| H6 | **循环内 V↔MTE 复用小缓冲竞态** | Cast/DataCopy 交替复用同一 UB 段 → 数据 +1 行移位；`PIPE_ALL` 都挡不住 | 两段式：全部计算完 → 屏障 → 全部搬运；计算/搬运缓冲物理分开 |
| H7 | **CMake 不追踪 `.asc` 内嵌 include 的 kernel.cpp 改动** | 改了 kernel 源不重编，stale binary 假象（"改了没效果"） | 改头文件后 touch 主 .asc，或 `rm -rf build` 全量 |

## 2. 多块 online softmax 三修复（单块 PASS、多块（kv>128）数值错时按序查）

1. **跨块 rescale（m 递推）**：Launch A 每块把运行 max `m_j` 存 GM 数组（`mArr[task][blk][16][rLo]`）；Launch B 用 `m_final = 末块 m + Σ m_j`，按 `w = exp(m_j − m_final)` 缩放各块 `O_j` 再累加。只做块内 softmax 不做跨块 rescale → l 对、O 错 ~5% 且分布不规则。
2. **P 行 padded 步长**：P 行步长必须 pad 到 `nBlocks×BLOCK`（尾块写满 BLOCK、masked 位置 0 正确写入）；按 `valid` 截断会丢列，FA 宽容差（atol=0.02）掩盖、严格 atol（1e-3）下暴露。
3. **跨任务 oTmp 单槽覆盖竞态**：末块不 Wait 累加完成就复用 oTmp，多任务同核时下一任务块 0 覆盖上一任务末块未读数据（fp16 靠时序侥幸、bf16+MTP 必现）→ **每块都 WaitFlag(accDone)**（flag 配平仍 2Set:1Wait）。

## 3. 单活跃子核陷阱（与 H2 同源，单独列出因为文档曾写反）

`if (subIdx != 0) return` 单活跃子核 + flag MODE 仍 0x2 → AIC WaitFlag 只收 1 Set 永阻塞。修复：**双子核都活跃**（UB 按 subIdx 分区、各处理一半行），保留 0x2 双 Set。设计文档若写"单活跃子核 + flag 1:1"而代码 MODE 是 0x2，**以 MODE 实测配平为准**。

## 4. 多 batch bug（B=1 矩阵永远测不出，见评测 doc §4）

- **S-slot 核间地址撞车**：双缓冲需 2 槽，`sBase = coreIdx×TILE×BLOCK` 单槽 → core1-slot0 与 core0-slot1 同地址，非确定性时对时错。修复：槽位 ×2。
- **host BuildTiling 缺 `cu += qSeq[b]`**：B>1 全 batch 用 cu=0，读错 Q 写错 O（b0 对、b1 错）。从宿主工程复制 tiling 代码时最容易丢这一行。

## 5. 轻量纯向量 kernel 是单子核（勿假设双子核）

merge/归一化类轻量纯向量 kernel 编译为**单上下文**，`GetSubBlockIdx()` 恒 0——与主 kernel（双子核）行为不同。按双子核写 merge 会 rows 8-15 全零。写法：单上下文一次处理全部 16 行。

## 6. kv-split 并行优化的三条教训（v4 亲测）

1. **host seg planning 写死 SEGS=1/BPS=1 → 虚假性能**：只处理 block0，实测 45.6us（53.7×）全是假的，tiling dump 才发现 sed 没生效。**split 优化先 dump tiling 验证分段真写入了再谈性能**。
2. **sed 批量改 kernel 源不可回滚**：v3 源码被 sed 覆盖无法 diff、v4 bug 无从对比。kernel 迭代用编辑器逐处改 + 备份（`.bak`），或从基线骨架重写。
3. **精度不过的优化不可用**：kv-split 53.7× 但多块 FAIL，最终回退双 launch V^T 版（32/32 PASS）。性能数字必须以精度 PASS 为前提归档。

## 7. 手搓架构模板（双 launch，最终全 PASS 形态）

```
Launch A（FAQK 流式）: 逐块 Q·K^T → 双子核在线 softmax（m/l 递推）
                      → 写 P_unnorm（padded 步长）+ mArr + l
Launch B（BlockGemm PV + AIV 累加归一）: 读 P/mArr/l → w=exp(m_j−m_final)
                      → PV 累加 → /l → O
AIV→AIC 就绪 flag 2Set:1Wait；accDone flag 每块 Wait
```

调试套路：环境变量分相位 dump（如 `MLAHC_PHASE=1/3`）把 P/l/O 中间量对 numpy 参考二分定位——比整链比对快一个量级。

---

## 8. 工程级元坑（第三轮 kfc 单 TU 实证补充，2026-08-27）——坑的不是 kernel，是你的观测

| # | 坑 | 现象 | 对策 |
|---|---|---|---|
| M1 | **stale binary 假验证** | `.asc` TU 的 include 依赖不被 CMake 跟踪；`rm -rf CMakeFiles/<target>.dir` 静默破坏 make 规则图（旧 exe 残留）→ 改了代码数字逐位不变 / 假 PASS | 重建脚本：`touch TU源 + cmake .. 再生成 + make` + **断言 exe mtime > 源 mtime**；exe 打 REV 版本串运行确认 |
| M2 | **stale dump 假验证** | kernel 挂死（timeout）后 verify 读上一次 dump → 逐位相同的假 PASS/FAIL，误导归因数小时 | **verify 前必删 dump 文件**；运行必查 sync 完成标记 |
| M3 | **memset 排队污染计时** | 争用卡上 aclrtMemsetAsync（全量 ws 几十 MB）= 25ms/launch 且与大小无关（纯排队）；把它算进 kernel 时间 → 全部归因错 | 事件对分离 `ev0→memset→ev1→kernel→ev2` 分别报 MEMSET_US/KERNEL_US；kfc 区只需清 64KB |
| M4 | **device lambda** | ASC 编译器不吃 device lambda（报 undeclared member） | 写私有成员函数 |
| M5 | **验证脚本自己的语法错** | 测试代码里 `replace('raise SystemExit(main())','')` 造成空 if 块 → 把文件语法错误误当被测对象失败 | 测试 harness 的字符串替换要落 'pass' 并单独语法校验 |
| M6 | **对标参数未生效** | 逐 case 精度与性能分别起两次 exe，第二次忘了带同一个 env 旋钮 → 测的是默认配置还当成旋钮效果 | 旋钮写入 tiling 并在运行日志回显；一次进程内完成"生成+计时" |
| M7 | **链接文件≠目录文件（tiling 来源坑）** | 工程根目录同名文件从未被 CMake 链接（实际链的是别处版本）→ 据目录文件打包/移植后行为完全不同（kvSplit 恒错档、小 kv 更慢） | 移植/打包前核对 build 实际链接的文件：CMake 源列表指向 + quoted-include 解析顺序 |
