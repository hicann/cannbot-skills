# FA 族性能评测与 gate 判定方法论（计时口径陷阱 / 官方标杆 / 时长分布泛化套件）

> **适用**：FA 族（FlashAttention / MLA / FFA / paged 变体）生成后的性能评测与 gate 判定。本文全部来自 FA 封装工程（FA 内核封装）+ MLA 手搓工程（MLA 内核封装）两工程实测踩坑（910B3），每条都付过假结论的代价。
> **核心原则**：**计时口径错 → 一切加速比都是假的**。先读 §1 陷阱，再采数。

---

## 1. 计时四陷阱（按危害排序，全部实测复现过）

### 陷阱 1：torch 进程内 ctypes kernel 逐次 event 计时**漏捕**（最危险，产出过 391× 假加速）

`record → ctypes launch → record → sync` ×10 取 min，对外来（非 torch_npu 注册）kernel **偶发漏捕**：min 恒 ~1.6-1.7us、与 case 规模无关 → 加速比虚高数百倍。median 表现正常，极具迷惑性。

**正解**：
- kernel 侧时长用 **exe 纯回放 aclrtEvent 组计时**：3 组 × 10 次/组，组均取 min（与 msprof Task Duration 同带，已互证）
- torch event 只用于纯 torch op 的 baseline（median + min 双报，min 单报必查是否触底恒定值）

### 陷阱 2：ATB 融合算子标杆每次 launch 重建 executor（host 开销吞掉一切）

`npu_multi_head_latent_attention` 等 ATB 算子无 executor 复用：单次 event 计时 4061us vs device 实际 76us。对比这类标杆**必须用 profiler op_summary 的 device 时长**，event/墙钟口径一律无效。

> **★NNAL ATB env 前置（2026-09 实测）**：跑官方 ATB 标杆前必须 `source /usr/local/Ascend/nnal/atb/set_env.sh`
> （与 cann set_env 同 source），否则 `torch_npu.atb.npu_multi_head_latent_attention` 抛
> "Please check that the nnal package is installed"——bench 直接退化为 0/0。墙钟口径下还会把
> executor 重建算进标杆时长（短 kv 放大 5~40×），gate 虚高。

### 陷阱 3：torch_npu profiler 惰性解析

op_summary 只在 `export_chrome_trace`（或等价导出）触发后才生成——只 start/stop 不导出会以为"没有 op_summary"。

### 陷阱 4：profiling 异常值先查环境再下结论

曾出现 msprof 采出 FAInferFp16 15us（真实 72-78us）：根因是**磁盘满导致 trace 截断**，不是工具盲区。结论"工具有固有盲区"之前，先查磁盘/设备状态（`df -h`、npu-smi）。

**ATB 标杆采集附加**：逐 case **独立进程**跑（单进程累积采 20 case 会被 OOM 杀，采一半丢全部）。

---

## 2. gate 判定口径

```
gate = geomean(标杆 device 时长 / 本侧 device 时长)   # 20 case 泛化矩阵
```

- **对比对象必须在报告显式命名**：「vs torch 分解链」与「vs 融合算子标杆」是两个完全不同的数字（MLA 实测：分解链 33.5× 墙钟 vs 官方融合 1.16× device）。写"比标杆快 N 倍"前先确认口径。
- 公平下界 = **device 侧 sum**（op_summary 逐 kernel 求和）；墙钟含分解链 host 调度间隙（实测 d18：441 kernel/iter，19.7ms 墙钟中 8.3ms 是间隙）。
- 几何平均（geomean）为 gate，算术均值只在附录给（长尾 case 会拉高算术均值造成误读）。
- 同源对比（本侧即标杆 kernel 同源封装）gate ≈ 1.0 属正常（FA 内核封装差 1.025）；此时 gate 意义 = 回归检测，不是优化空间。

实测参考（910B3）：
| 算子 | kernel | gate vs 标杆 |
|---|---|---|
| fa-page-attention | FA 内核封装 | **1.025**（同源，封装差） |
| fa-mla | MLA 内核封装 | **3.047**（序列长扫描至 kv=204800 同设备复测版，长段 ratio 最高 7.16×；batch 放大口径 1.161） |
| （手搓版对照组） | 双 launch AIC/AIV | 0.107~0.364（正确性优先，预期内） |

---

## 3. 序列长分布泛化套件（当前口径；早期时长分布口径已废弃）

**当前口径（2026-08-24 起）**：case **按 kv 序列长分桶**，不按时长分桶。序列长尽可能长（≥20 万 token），以不爆内存为限：
- fa（单序列 kv 无上限，受内存限制）：20 case 升序 kv 扫描，桶 `≤16k ×5 / 16k~64k ×5 / 64k~200k ×5 / >200k ×5`，最大 kv 458752（k+v ≈229MB/case，实测安全）
- mla（接口 gen_data 声明范围仅 [128,16384]，**实测非硬约束，kernel/ATB 均可正确跑至 20万+**）：桶 `≤2k / 2k~16k / 16k~100k / 100k~204800(20万) ×5`；超 16384 属标杆未验证区，精度必须逐 case 自测；B×kv 总 token 放大（如 B16×16384）可另列演示 case，勿混进 kv 扫描
- B 固定小值（fa B=1、mla B=2）使 kv 成为唯一变量；B>1 约束仍满足（mla 侧）
- 内存预算：host 每 case bin ≈ `kv × D × 2B × 2(k+v)`；device 同量级 + workspace。生成前 `df -h` + `npu-smi` 双查

**历史口径（已废弃，仅存档）**：按时长分桶 `<100us:10% / 100-500:20% / 500-1000:40% / >1000:30%` → 20 case 精确 2/4/8/6，需用吞吐模型定标构造（fa `t≈0.004×kv`、mla `U=Σ T·H·kv_b` 吞吐表），模型 2 轮迭代收敛。废弃原因：定标迭代成本高、且时长是导出量（随 kernel 优化漂移，case 集要反复重建）；序列长是输入量，一次构造长期有效。

**case 重生成坑**：新旧 case 同名时脚本会"先 gen 新、后清理误删新留旧"——重生成前先手动清旧名目录；换口径用新目录（如 `cases_len/`）并保留旧目录存档。

---

## 4. ★泛化套件的隐藏价值：强制 B>1 是多 batch bug 探测器

单 batch（B=1）语义矩阵会**掩盖**多 batch bug——泛化/性能套件强制构造 B>1 case 后，FA 封装工程 一次暴露两个此前 10/10 PASS 从未触发的 bug：

| bug | 现象 | 根因 |
|---|---|---|
| **S-slot 核间地址撞车**（非确定性：时对时错） | `sBase = coreIdx×TILE×BLOCK`，但双缓冲需 2 槽 → core1-slot0 与 core0-slot1 同地址 | 双缓冲槽位 ×2（`sBase×2`） |
| **host BuildTiling 缺 `cu += qSeq[b]`**（确定性：b0 对 b1 错） | token 偏移不按 batch 递增，B>1 全部 batch 用 cu=0 | 补累积行（MLA 手搓工程 有、FA 封装工程 漏——复制宿主代码时最容易丢的一行） |

**规则**：任何 FA 族 kernel 交付前，评测矩阵必须含 B>1 且各 batch qSeq/kvSeq 不等的 case。

---

## 5. 归档清单

评测完成后归档（两工程模板：`FA 封装工程/perf_test_list.md`）：
- 逐 case 表：case 名（`<op>-NNN` 三位编号）/ shape 布局与参数 / 本侧 device us / 标杆 device us / 加速比
- 末行 gate（几何平均）+ 均值（算术，标注口径）
- 时长分布命中表（2/4/8/6 ✓）、精度 N/N PASS、三方互证 err 范围
- 数据 tsv 留 `/tmp` 副本（case 名与目录名、meta.json `case` 字段三处同步，重命名用脚本一次改齐）

---

## 6. 争用设备计时口径（2026-08-27 第三轮实证补充）

协同租户占卡 87-95% 时的公平口径，否则归因全错：

1. **本侧**：exe 内事件对把 memset 与 kernel **分开计时**（`ev0→memsetAsync→ev1→launch→ev2`，各报 MEMSET_US / KERNEL_US，取 3 组×10 次的组内 min）。争用排队发生在第一个操作上，不分离时 25ms 级的 memset 排队会把 kernel 归因完全污染。
2. **标杆侧**（torch_npu op-plugin 路径）：同卡同窗口**背靠背回放均摊**——warmup 3-5 次 → sync → `perf_counter` 包住 N 次连发（N=5-10）→ sync，除以 N，取 3 组 min。torch `cuda.Event` 在 NPU 是 dummy 类直接抛异常，不可用；`time.perf_counter` 单发含 host 调度间隙（几十至几百 μs）必须均摊。
3. **逐 case 同窗口交替**：本侧与标杆紧邻背靠背跑（同争用水平），不要隔着几分钟各跑各的——争用波动 > 30%。
4. **旋钮一致性**（M6）：性能与精度若分别起进程，必须带同一套 env 旋钮；旋钮值打进运行日志回显。
5. 序列长分桶套件的标杆侧同样可能受 H≤16（sink 参数限制）/同进程随机 161001 约束：逐 case 子进程隔离（见配方册 §12.11-F）。
