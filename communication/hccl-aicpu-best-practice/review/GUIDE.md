# HCCL AICPU Template 评审指南

## 0. 这是什么

这是 `hccl-aicpu-best-practice` 的专项检视参考：对着 `$HCCL` 里的 AICPU algorithm template（`src/ops/{算子}/algorithm/template/aicpu/ins_temp_*.h/.cc`，
基类 `InsAlgTemplateBase`）出一份分级评审意见。领域背景在
`references/00-domain-primer.md` 里自带一份，检查器 `scripts/review_template.py` 自带机械层与语义层。

四层，顺序不能反（①②③ 必做，④ 有 spec 时才做）：

| 层次 | 谁做 | 管什么 |
|---|---|---|
| ① 机械层（W01–W07） | `review_template.py` | 命名、include guard、版权头、必须实现的方法、`props` / `AlgoType`、构造签名、**两处 CMake 接线** |
| ② 语义层（C / M / N / R / H / X） | `review_template.py` | scratch 倍数、repeat 与 stride、同步配对、notify 自洽、channel 取用、四族各自的算法不变式 |
| ③ 人工层 | 按 `references/02..06` 的清单复核 | 脚本判不了的：切片偏移、错峰顺序、尾块归谁、cost model 合不合适 |
| ④ 契约层（S00–S03，可选） | `review_template.py --spec` | 作者给了 dataflow spec 时，先校验 spec 完整性，再核代码与 spec 是否一致 |

一个没接进 CMake 的文件，语义评得再细也没有意义——所以 ① 排在最前，且默认就跑
（已经用别的工具查过接线，可以 `--semantic-only` 跳过）。

本指南仅支持检视，不授予源码写入或流程裁决权限；具体执行角色由上层 Agent 定义。

## 0.1 覆盖范围：四个算法族

| 族 | 仓内现状 | 判据来源 |
|---|---|---|
| **mesh**（Mesh1D，含 one-shot / two-shot / mesh-chunk / omnipipe / Z 轴绕行） | 存量最多 | 从真仓源码归纳，`references/03` |
| **NHR**（含 NHR_AICPU_REDUCE、多 jetty、PCIe 读模式） | 存量次多 | 从真仓源码归纳，`references/04` |
| **ring** | 不要求仓内已有蓝本；以目标文件和绑定 executor 的已核验契约为入口 | 按算子语义分派轮转分片 / 链式广播；映射到本仓 API，`references/05` |
| **HD**（recursive halving-doubling） | 不要求仓内已有蓝本；只定位本次实现与必要调用方 | 算法定义（Rabenseifner / MPICH 的非 2 幂降级）→ 映射到本仓 API，`references/06` |

ring / HD 的部分判据尚未获得 CANN 实现验证；先核对目标版本接口与 executor 契约，再做两项额外检查：
**(a)** 逐项确认选路登记是否齐全：`AlgoType` 枚举、`ALGO_TYPES`、
`GetAlgoTypeToNameMap()`、`GetAlgoNameToTypeMap()`，并人工检查按族分流逻辑是否需要扩展；
缺任一必需登记都会导致算法配不上或选不上（规则 X01，BLOCKER）；
**(b)** 逐条对着 `references/05` / `06` 的不变式读代码，而不是"看着像 ring 就过"。

脚本会在 ring / HD 输出中直接提示：R/H 规则含尚未获 CANN 官方实现验证的映射；
不要把它们与 X01、W/C 这类仓内可核验事实混成同等确定的结论。

## 1. 技术输入

### 1.0 仓路径必须已确认

仓路径来自用户、调用工作流已确认的路径，或环境变量 `HCCL_REPO` / `HCCL_ROOT`。
复用已确认的路径；缺失时返回输入缺口，不猜常见路径、不扫盘。用户交互由调用 Agent 负责。
脚本接受 `--repo`，未传时只读取 `HCCL_REPO`；HCCLBot 工作流显式传 `--repo "$HCCL_ROOT"`。

```bash
export HCCL_REPO=<用户给的 HCCL 仓根目录>
REVIEW_DIR=<本 GUIDE.md 所在目录>         # 脚本在这里，不在 HCCL 仓里
```

只支持当前 `src/ops/<op>/algorithm/template/aicpu/` 布局。旧版平铺目录、CCU/AIV
不在检查器覆盖范围，遇到不支持的目标应报告未覆盖，不能将未检查当作通过。

> 脚本在 skill 目录里。一旦 `cd` 进 `$HCCL_REPO` 再写 `python3 scripts/xxx.py` 就会 `can't open file`，
> **始终用 `"$REVIEW_DIR/scripts/review_template.py"` 的绝对路径**。

### 1.1 确定评审模式（优先复用任务上下文）

| 模式 | 情况 | 怎么评 |
|---|---|---|
| **A** | 新写 / 新生成的 template | 全套 ①②③；有 spec 就加 ④。BLOCKER 与 MAJOR 都要作者答复 |
| **B** | 存量 template 的**局部改动** | 用 `--base <ref>`（见 §2）：与 base 快照的 finding 集合比对，只把新出现的算作者的 |
| **C** | "这个 template 写得对吗"（纯只读） | ①②③ 全跑，但结论按「现状说明 + 风险点」写，不要求整改 |

报告模板与 `references/07` 里说的「模式 B」就是上表这一行：**没有 spec 的局部改动**。

## 2. 命令

```bash
python3 "$REVIEW_DIR/scripts/review_template.py" --repo "$HCCL_REPO" <相对路径>.cc
```

**模式 B（存量 template 的局部改动）用 `--base`**，它替你区分「作者这次引入的」和「存量欠账」。
base 必须是改动发生前的版本，按场景选择：

```bash
# 未提交 / 已暂存的工作树改动
python3 "$REVIEW_DIR/scripts/review_template.py" --repo "$HCCL_REPO" --base HEAD

# 最近一次已提交改动
python3 "$REVIEW_DIR/scripts/review_template.py" --repo "$HCCL_REPO" --base HEAD^

# 分支 / PR 相对目标分支的全部改动
python3 "$REVIEW_DIR/scripts/review_template.py" --repo "$HCCL_REPO" --base origin/<目标分支>
```

归属方式是**在 base 快照上再评一遍、比对 finding 身份**，不看行号——纯删除、只改 `.h`、
只改 CMake 这些改动本来就没有"新增行"可锚，按行号归属必然漏掉。

自动选目标时不只看 `.cc/.h`：局部 CMakeLists、全局 `scatter_aicpu_kernel.cmake`、
executor、`src/common`（`AlgoType` 真源）变了，都会把受影响的 template 拉进来重评。
已提交、已暂存、未暂存、未跟踪四种改动都覆盖。

默认只展开 **★ 本次改动引入**，存量欠账只在末尾汇总，避免全局文件改动带出几百行旧问题；
确实需要逐条查看存量时加 `--show-baseline`。**退出码只看本次引入**，可直接当门禁。
取不到 base 版本时一律按"全部本次引入"计（宁可多报）。

作者给了 dataflow spec 时再加一次（先跑 S00，再跑 S01–S03）：

```bash
python3 "$REVIEW_DIR/scripts/review_template.py" --repo "$HCCL_REPO" <相对路径>.cc --spec <spec>.md
```

| 开关 | 用处 |
|---|---|
| `--topo mesh\|nhr\|ring\|hd` | 单个目标认成 unknown 时强制指定；不能与 `--all` 或多个目标组合，它不是筛选器 |
| `--all` | 扫全仓，配 `--quiet` 只看有 BLOCKER/MAJOR 的 |
| `--semantic-only` | 跳过机械层 W01–W07（已用别的工具查过接线时） |
| `--base <ref>` | 与 base 快照比对 finding 集合，只把新出现的算本次引入（模式 B）；退出码只看这部分 |
| `--show-baseline` | 配合 `--base` 展开存量 finding；默认只显示本次引入并汇总存量 |
| `--strict` | 让 MAJOR 也返回非 0，CI 门禁用 |
| `--rules` | 打印全部规则目录（带 references 出处） |

**脚本不修改 HCCL 工作树**，也不要求它能编译。唯一的写动作是 `--base`：
它在临时目录里解一份 base 快照（`git archive`，不建 worktree、不碰 `.git`），跑完即删。

## 3. 严重度：怎么定，怎么用

| 级别 | 含义 | 处理 |
|---|---|---|
| **BLOCKER** | 必然错或必然不生效，证据确凿 | 合入前必须改 |
| **MAJOR** | 很可能错，或与该算法族的定义冲突 | 作者必须答复；确有理由就写进 spec |
| **MINOR** | 建议改；存量里普遍如此，新代码应该做得更好 | 列出来，不阻塞 |
| **INFO** | 脚本判不了、需要人看一眼的点 | 评审人自己看，别原样甩给作者 |

**存量基线**：真仓有几百条 MINOR 的全仓性欠账（`count == 0` 早退、`channels.count()` 校验、
repeat 循环、include guard 命名之类），**别一股脑贴给作者，只提他这次改到的**——
用 `--base`（§2）让脚本替你分。除去这些欠账，**新代码上出现任何 BLOCKER / MAJOR 都是信号**。

> 具体数字会随真仓增删 template 漂，本文不留快照；要引用就现跑 `--all`，
> 判据与最近一次实测见 `references/01` §4。

## 4. 先判族，再按需加载

四族的完整不变式在 `references/03..06`，**别凭印象评，判完族就把对应那篇读进来**。
只加载通用规则、目标算法族及涉及的 Spec/资源小节，不安排通读全部参考。
已有来源/适用范围记录可复用来定位；独立核对本次变更的关键结论，缺口才扩大源码搜索。
判族的依据（脚本用的是同一套，`--topo` 可强制指定）：

| 族 | 一眼分辨 | 打开 |
|---|---|---|
| **mesh** | 类名/文件名带 `Mesh`；一步到位、没有 step 列表 | `references/03` |
| **NHR** | 带 `NHR`，或出现 `GetNHRStepNum` / `stepInfoList` | `references/04` |
| **ring** | 带 `Ring`；邻接通信，先区分轮转分片与链式转发 | `references/05` |
| **HD** | 带 `Hd` / `HalvingDoubling`；有 2 的幂降级与 `idx ^ (1 << k)` | `references/06` |

通用不变式（与拓扑无关）逐条在 `references/02`，读代码时优先盯这几处最容易错的：

- `CalcScratchMultiple` 按**单次 repeat** 报——放大是 executor 的活，自己再乘一遍编译和 ST 都不报错
- 地址要乘 `rpt`、预算不要乘；scratch 跨度自己算，只有 `inBuffType == HCCL_BUFFER` 时才用入参 stride
- scratch→out 本地拷贝先查 executor 各段 `outBuffType` 与两侧 offset；只有证明数据已在目标槽位才可跳过，见 C16 / R09
- `PreSyncInterThreads` / `PostSyncInterThreads` 成对；线程与 notify 的三个字段一起填
- `channels` 的 key 是**通信域 userRank**（`subCommRanks_[0][idx]`），不是算法内序号；`.at()` 前先 `.count()`
- 新增支持性限制或模式分支：人工追踪状态赋值→选路消费→执行消费，穷举目标版本模式枚举；见 `references/02` §11。
- 归约类要交代 64-bit 与 `HCCL_REDUCE_PROD` 谁兜底；跨 rank 边只写一个方向（读/写模式二选一）

## 5. 评审报告怎么写

模板在 `assets/review-report.md`。要点：

1. **先说静态评审结论**：脚本检查通过但人工检查待完成；或人工检查完成后的静态评审通过；或改完 BLOCKER 再评；或需要作者答复 N 个问题；
2. 每条意见给**证据**（`文件:行号` + 代码片段）和**后果**（"scratch 越界"比"不符合规范"有用得多）；
3. 分级排序，BLOCKER 在最前；MINOR 折叠成一段，不逐条展开；
4. 脚本报的原样引用规则号（如 `C01`），人工发现的另起一节，注明是人看出来的；
5. 没有实际执行验证时，不写“可以合入”；列出人工检查、编译与功能验证各自的完成状态。
   本指南不执行功能验收；AICPU 功能验收须满足引擎入口的新算法定向验证与存量回归双轨契约。静态评审和编译通过不替代运行验证。

### 检视输入与证据

本指南只覆盖 template 专项；executor、selector 及跨层一致性另按 AICPU 引擎契约核对。仅做独立只读检视时不要求环境安装、开发或构建前置。

- 输入：仓路径、目标文件、模式 A/B/C、固定的 base commit（模式 B）、各 template 对应的 spec（可选），以及对应代码版本的验证证据。
- 增量复审沿用首次评审的 base commit，不随修复提交移动 `HEAD` 基线；首次将用户指定 ref 解析为 commit 并记入报告。
- spec 一份对应一个 template，分别执行 `--spec`；没有 spec 如实标注契约层未执行。
- 脚本只输出检查结果到终端；证据可按 `assets/review-report.md` 汇总，实际报告路径和状态由调用 Agent/Plugin 指定。
- 已确认未解决的 BLOCKER、未澄清的 MAJOR 和必需检查未完成须分别列明；未完成不能冒充代码缺陷或通过。
- 脚本退出码 0 不等于整体评审通过；`--strict` 会让 MAJOR 也返回非 0，不能替代证据复核。功能验证状态单列，静态通过不等于任务验收通过。
- 本指南不修改被审源码，也不决定修复或复测角色。

## 6. 红线

- **不改代码**。评审产出是意见，不是 patch；要动手改是另一件事，先和作者确认
- **不猜仓路径**（同 §1.0）
- **不把存量欠账算到作者头上**：先跑一次基线，区分"他引入的"和"仓里本来就有的"
- **ring / HD 不许"看着像就过"**：按算法形态逐条核对适用不变式、目标接口与 `AlgoType` 枚举缺口；不为寻找同类蓝本强制全仓扫描
- **先确认规则适用性**：同一算法形态内的例外用源码证据与 spec 解释；结构性不适用应修正规则分派，
  补正反例，不要求每次评审重复豁免同一误报。
- 改了 `review_template.py` 或 references 里的规则口径，**必须跑 `bash scripts/selftest.sh "$HCCL_REPO"`**

## 7. 按需加载的参考

| 文件 | 何时读 |
|---|---|
| `references/00-domain-primer.md` | **不熟 HCCL template 就先读这篇**：架构、接口契约、数据结构、原语、资源模型 |
| `references/01-review-method.md` | 评审流程、证据标准、误报怎么处理、新规则怎么校准 |
| `references/02-common-checks.md` | W01–W07（机械层）与 C01–C16（通用语义）逐条：怎么看、为什么错、怎么改 |
| `references/03-mesh.md` | 评 mesh 族（M01–M05） |
| `references/04-nhr.md` | 评 NHR 族（N01–N05） |
| `references/05-ring.md` | 评 ring 族（R01–R09）按算子语义与形态分派判据，先读这篇 |
| `references/06-hd.md` | 评 HD 族（H01–H06）★无经 CANN 验证的蓝本，先读这篇 |
| `references/07-spec-conformance.md` | 代码 ↔ dataflow spec 逐章对照（S00–S03） |
| [references/09-regression-cases.md](references/09-regression-cases.md) | 维护规则或评测人工语义检出能力时，读取复盘正反例 |
| [references/08-maintenance.md](references/08-maintenance.md) | 维护来源、规则耦合点与回归方法 |

**`references/00` 是自带的领域速成**：分层架构、接口契约、`TemplateDataParams` / `BuffInfo` 字段、
原语表、thread/notify 模型、scratch 与 loop、repeat/stride、两处 CMake、验证命令。
对 HCCL template 不熟就先读它，不需要去翻别处的文档。
