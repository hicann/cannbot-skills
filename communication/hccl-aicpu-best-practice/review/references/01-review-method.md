# 01 · 评审方法

## 1. 四层顺序，不要跳

```
① 机械层 W01–W07     编不过 / 接不上的先清掉        ┐ review_template.py
        ↓  有 BLOCKER 就先让作者改，语义评了也白评   │ 默认一起跑
② 语义层 C/M/N/R/H/X 四族不变式 + 通用不变式         ┘
        ↓
③ 人工读 references/02..06 的清单    脚本判不了的：偏移、尾块、错峰顺序、cost model
        ↓
④ --spec 对照（作者给了 spec 时）    代码与 dataflow spec 还对不对得上
```

跳过 ① 直接看语义，常见的结果是花半小时评一个**根本没登记进
`src/scatter_aicpu_kernel.cmake`、运行时找不到符号**的文件。
`--semantic-only` 只在"接线已经由别的工具查过"时才用。

## 2. 一条意见要有三样东西

| 缺了它就没用 | 例子 |
|---|---|
| **证据** | `ins_temp_xxx.cc:76`，贴上那 3 行 |
| **后果** | "`hcclBuff` 按 1 倍切、实际写 N 倍 → 越界踩内存"，而不是"不符合规范" |
| **可执行的修法** | "返回 `templateRankSize_`"，或"确有理由请写进 spec 第 4 章并说明" |

只有"后果"能让作者判断优先级。**写不出后果的意见，多半是规则背诵，删掉。**

## 3. 严重度的判据

| 级别 | 判据 | 典型 |
|---|---|---|
| BLOCKER | 逻辑上必然错 / 必然不生效，且不依赖运行环境 | X01 选路登记不完整；W01 没登记进 CMake；S02 spec 声明多 repeat 但代码没循环 |
| MAJOR | 与该算法族的定义冲突，或与本仓硬约定冲突，但存在"作者有特殊理由"的可能 | M01 one-shot 倍数不是 N；C01 倍数乘 `repeatNum`（确定性性能退化，非算错）；C07 单边同步 |
| MINOR | 建议项；存量里普遍如此 | C05 `count == 0` 早退；C11 `channels.count()`；C03 没写 repeat 循环 |
| INFO | 需要人看一眼、脚本给不出结论；以及**无文献背书又无实现验证**的本仓推测 | R07 RS/AG 是否共用分片表；R05 / R06 / H06 见 `references/05` §5 |

## 4. 基线：先分清"他写的"和"仓里本来就有的"

局部评审优先用 `--base`。base 要指向改动发生前的版本：工作树改动用 `HEAD`，最近一次提交用
`HEAD^`，分支 / PR 用目标分支（如 `origin/master`）。默认只展开本次新增 finding；
存量只做汇总，需要逐条排查时才加 `--show-baseline`。

```bash
python3 "$SKILL_DIR/scripts/review_template.py" --repo "$HCCL_REPO" --all --quiet | tail -20
```

最近一次实测（commit `6ebb413b`，2026-09-04）：**52 个** AICPU template，
**0 BLOCKER、0 MAJOR、194 MINOR、15 INFO**。

**这是历史观察，不是当前事实**——用户给的仓在哪个 commit 上，数字就会变。
引用前先自己跑一次 `--all`；也别拿未提交的工作树当基线。

数字会随真仓增删 template 漂，引用前先跑一次 `--all`，别照抄本节；
**也别拿未提交的在建 template 当基线**——它带来的 BLOCKER/MAJOR 是那份代码的待办，不是基线变了。

按规则汇总里排前几名的（`W04` 30 处、`C11` 29 处、`C03` 25 处、`C05` 22 处、`W05` 20 处、`C06` 18 处）
都是**全仓性欠账**：

- `C05 / C06`（`count == 0`、单 rank 早退）：多数 template 靠 executor 保证不会拿 0 进来；
- `C11`（`channels.at()` 前不校验）：连蓝本 `ins_temp_all_reduce_mesh_1D_one_shot.cc:164` 都没写；
- `W04 / W05`（include guard 命名、缺默认构造）：仓内命名不统一，且不走 FastLaunch 的 template 本就不需要默认构造；
- `C03`（没写 repeat 循环）：只服务单层、不打算被分层 executor 复用的实现可以省（见 `references/00` §7）。

**这些在评审新代码时提一句就够，不要当成阻塞项。**
如果哪天基线里冒出了 BLOCKER / MAJOR，说明**要么真仓演进了、要么规则口径过严**，
两种都得改本 skill，而不是去改 HCCL 仓。

## 5. 规则失效与校准

`review_template.py` 的规则贴着真仓 API，**最容易随仓演进而失效的是这几处**：

| 依赖 | 出现在 | 真仓来源 |
|---|---|---|
| 选路四个登记点 | X01 | `src/common/alg_parse.h/.cc`（实时解析，不写死） |
| `CalcRes` 的三个资源字段名 | C08 | `AlgResourceRequest` |
| 原语名单（`SendRecvBatchWrite` 等） | C12 / C14 | `src/ops/op_common/template/wrapper/` |
| 软归约三步 | C09 / C10 | `HcommBatchModeEnd / Start / HcommThreadJoin` |
| 蓝本文件名 | `selftest.sh` 的故障注入 | `ins_temp_all_reduce_mesh_1D_one_shot.cc` |

校准新规则的流程，**顺序不能反**：

1. 先在真仓 `--all` 上跑，看它在存量上报几条；
2. 报得满仓都是 → 要么降级成 MINOR，要么规则本身错了（先去读源码，别急着加白名单）；
3. 在 `assets/fixtures/` 加一对正例 / 反例，或在 `selftest.sh` 里加一处故障注入；
4. `bash scripts/selftest.sh "$HCCL_REPO"` 必须全过；记录实际通过、失败和跳过项，不固定测试数量。

## 6. 常见误报与它们的正解

| 现象 | 原因 | 已有对策 |
|---|---|---|
| 变体子类（`class A : public InsTempXxx`）被报"缺 XXX" | 大半实现在父类 | 脚本沿继承链把父类 `.h/.cc` 一起扫（`family_text()`）；仍误报就人工判掉 |
| `return scratchMultiple;` 被当成"倍数不明" | 返回局部变量 | `resolve_expr()` 回溯赋值；多分支时会显示成 `0 \| templateRankSize_` |
| AllGather / ReduceScatter 的 NHR 倍数是 N 却被报 | in/out 不等大，N 倍是对的 | 只有 **AllReduce 语义**才按 MAJOR 报，其余降 MINOR |
| `src/ops/all_reduce/` 下的 AllGather 构件被按 AllReduce 判 | **目录 ≠ 语义**：`ins_temp_all_gather_nhr_dpu_inter.cc` 是 2 级 AllReduce 的 inter 段 | 按**类名**认语义（`infer_op`），且 `*AicpuReduce*` 变体天然要 N 份暂存 |
| 拓扑认成 `unknown` | 类名里没有拓扑词 | 对单个目标用 `--topo` 显式指定；它不能与 `--all` 或多目标组合 |
| `PreSync` 比 `PostSync` 多一次 | 存在故意不对称的用法（如 post-copy 线程握手） | 两边都非 0 时按 MINOR 报，人工确认每一处配对 |

## 7. 评审 ring / HD 的额外一步

先核对目标版本是否有可用实现；**不把旧版本没有实现的观察当作当前事实**：

1. 先查 X01 的四个登记点——枚举、`ALGO_TYPES`、两张名字映射表逐项核对；
   再人工确认按族分流逻辑是否需要扩展，缺任一必需登记算法就配不上或选不上；
2. Ring 先按算子语义与形态选择 `references/05` 的适用规则；HD 对照 `06`；
3. 拿 `assets/fixtures/ins_temp_all_reduce_{ring,hd}_good.cc` 当"应该长什么样"的参照——
   它们**不参与编译**，只是把不变式写成了可读的代码形态。


## 8. 证据复用与 finding 统计

首次评审保存关键入口、状态赋值/消费、wrapper 完成语义和资源消费的证据索引，绑定源码身份。
作者提供的索引用于定位，Reviewer 仍独立核验关键结论。复审按 finding → 修复 → 影响范围 →
新证据核对；依赖未变时复用定位结果，跨层行为变化时扩展检查。

给每条独立 finding 分配稳定 ID，复审沿用。分别记录有效性（确认/误报/待确认）、
归属（本次/存量）、处置（整改/接受约束/未解决）。同一根因的重复告警关联到同一 ID；
真实存量问题不是误报，纯脚本提示不是已确认缺陷。报告总数应能由明细重算。

评测分别记录已知缺陷检出率、合法实现误报率、人工取证耗时与复核耗时。
单次零新增 finding 不证明脚本无用，也不能从有效意见比例推导漏报率。
案例与评测方法见 [09-regression-cases.md](09-regression-cases.md)。
