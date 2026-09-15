# `kb/` 知识组织约束规范

**这是硬规则，不是建议。** 任何 agent 或人在改动 `kb/` 之前必须先读本文件。
每条规则都标了**依据**（哪个文件哪一行强制它）与**违反后果**。

> 想先理解各目录是干什么的，读 [`README.md`](README.md)；本文件只讲**约束**。
> 规则的权威来源是同仓的 `plugins-community/cannbot-knowledge/`（门禁 + 检索引擎），
> **不是** NPU-Kernel-Wiki 的 SPEC-*.md（那描述的是另一个引擎版本，已明确排除）。

---

## R1 · `kb/` 是只读的 b 层，运行时产生的知识一律写 c 层

**规则**：任何 agent **禁止**在运行时写入 `kb/` 下的任何文件。

**依据**：`kb/` 随插件分发。插件升级/重装会覆盖整个目录 —— 写进去的内容**必然丢失**，
且丢失时没有任何报错。

**正确去处**：c 层用户本地 KB，根目录 `$ASCENDC_PORT_USER_KB`（默认 `~/.ascendc-port/user_kb`，
由 `init.sh:594` 创建）。

**例外**：维护者在 git 仓库里做的**离线编辑**不受此限（那是在改分发内容本身）。
区分标准是「这次改动会不会进 commit」——不进 commit 的写入一律是错的。

**已知踩坑**：`agents/aog-hardware-probe.md` 曾要求把新 probe 结果写进
`kb/hardware/probe_findings/`，2026-09-05 已更正为写 c 层。

---

## R2 · 只有三个目录进检索语料，且只收 `.md`

**规则**：只有 `okf/reference/`、`okf/ops/`、`okf/runbooks/` 三棵树下的 `.md` 会被索引。
每一级的 `index.md` **被排除**。放在这三棵树之外的任何东西都不进语料。

**依据**：`cannbot-knowledge/skills/knowledge-query/scripts/retrieval/config.py:31`
的 `CONTENT_ROOTS`；`retrieval/cards.py:_concept_paths_under` 的过滤条件。

**违反后果**：文件"在知识库里"但**永远检索不到**，而且没有任何提示。
（历史案例：迁移期 `_migration` 下的 4 份核销底账、`cann_classification` 下的 `.txt` 清单——
前者已删，后者已移到引用它的卡旁边。两者当时都在语料外，检索不到。）

---

## R3 · 目录深度：doc-id 的斜杠数 ≤ 3

**规则**：卡片相对 `reference/` 的路径最多 3 个斜杠，即最深 `bundle/section/category/card.md`。

**依据**：`knowledge_lint.py:130` —— `if nid.count("/") > 3: blocker`。
`nid` 对 reference 卡是**相对 `reference/` 的路径**（`cards.py` 剥掉了 `reference/` 前缀）。

**违反后果**：每个超深文件一个 blocker。

**更深的分类怎么办**：用 `tags`，或把第 4 段并进文件名前缀
（如 `struct/layout/MakeNDLayout.md` → `tensor_layout/make_nd_layout.md`）。

---

## R4 · bundle 名会改变适用规则 —— 取名前先查这四张表

**规则**：`bundle` = `reference/` 下的**第一级目录名**（`retrieval/facets.py:15-17`）。
它同时是 `knowledge_lint.py` 四张规则表的 key：

| 表（`knowledge_lint.py:81-84`） | 成员 | 落进去的后果 |
|---|---|---|
| `GIT_BUNDLES` | `asc-devkit`, `ops` | 每张卡 `resource` 必须匹配 `https://gitcode.com/.+/(blob\|tree)/[0-9a-f]{40}/`（**40 位 sha**），且 bundle 根 `index.md` 必须有 `upstream_repo`/`upstream_ref`/`upstream_commit` pin |
| `DOCSITE_BUNDLES` | `ascend-c-op-dev-guide`, `ascend-c-profiling` | `resource` 须为 hiascend.com 文档 URL |
| `EMPTY_OK_BUNDLES` | `glossary`, `runbooks` | `resource` **允许为空** |
| `SNAKE_BUNDLES` | `asc-devkit`, `glossary`, `ascend-c-op-dev-guide`, `ascend-c-profiling` | 文件名含大写 → warn |

**不在任何表里的 bundle 名**：`resource` 只需匹配 `^https?://`，无命名约束。

**检索侧还有一处硬编码**：`retrieval/relevance.py:131-133` 对
`path.startswith("asc-devkit/api/")` 在 `api_usage` 意图下 +0.08。**与 lint 用同一个字符串**，
所以拿加分就必须担 40 位 sha 的约束，二者不可分离。

**当前取名与理由**：
- `asc-devkit-vendored`（511）—— 故意**不**叫 `asc-devkit`。上游确切 commit 不可考
  （`Ascend/agent-skills` PR #103 的 sha 已随分支删除；`cann/asc-devkit` HEAD 文档树已改版；
  四个版本分支内容均不匹配），叫 `asc-devkit` 就必须**编造**一个 40 位 sha。
- `porter`（35）—— 本插件自产，`resource` 填本仓自身 permalink。

---

## R5 · frontmatter：9 个必填字段，`kind` 只能 9 选 1

**必填**（`knowledge_lint.py:139-142`）：
`schema_version` `kind` `type` `source_family` `title` `description` `tags` `created_at` `updated_at`。
另加 `resource`（bundle ∉ `EMPTY_OK_BUNDLES` 时为**硬 blocker**）。

**`kind` 受控词表（9 个，`knowledge_lint.py:87`）**：
`api` / `guide` / `example` / `operator` / `glossary` / `operator_optimization` /
`implementation_trap` / `debugging_journey` / `cross_skill_gap`

> ⚠️ **陷阱**：`retrieval/config.py:58` 里的 `KNOWN_KINDS` 有 **14** 个（多了 `index` 与
> legacy 的 `header`/`field_note`/`runbook`/`other`）。**以 lint 的 9 个为准** ——
> 只有 lint 产生 blocker。特别是**不要用 `field_note`**，检索认它但 lint 会 blocker。

**`type` → `kind` 的规范映射**（`retrieval/config.py:65` 的 `TYPE_KIND`）：
`api_reference`→api、`code_example`→example、`devkit_guide`/`programming_guide`/`profiling_guide`/`migration_guide`→guide、
`term`/`paradigm`→glossary、`operator_spec`→operator、`optimization_runbook`→operator_optimization、
`implementation_trap`→implementation_trap、`debugging_journey`→debugging_journey、`cross_skill_gap`→cross_skill_gap。

**`source_family` 受控词表（`knowledge_lint.py:92`）**：
`asc_devkit` / `ascend_c_op_dev_guide` / `ascend_c_profiling` / `ops` / `curated` / `community`。

**`created_at`/`updated_at` 格式**：UTC ISO-8601 带 `Z`，如 `2026-05-16T00:00:00Z`
（`RE_UTC_Z`，`knowledge_lint.py:95`）。

---

## R6 · `description` 是检索结果的门面，写法有硬要求

**规则**：一句话，说清「这张卡讲什么」，让检索到它的 agent 立刻判断相关性。

- **必须依据正文**。正文里没有的信息**不得编造** —— 尤其禁止臆测性能数字、芯片支持范围、版本约束。
  正文有「产品支持情况」表的，可以照抄该表的事实。措辞用「**标注**支持 X」而非「支持 X」。
- **信息密度优先**：写具体 API 语义、约束、适用场景。
  ❌「介绍 asc_add 接口的使用方法。」
  ✅「按元素矢量加法 dst=src0+src1，UB 指针形态，支持 half/float/int16/int32，含 count 形与 repeat 形。」
- **合法 YAML 双引号标量**：句内不得有未转义的 `"`（改用「」），不得有换行，
  不得有 `\(` 这类 YAML 不认的转义（会让 `yaml.safe_load` 抛 ScannerError）。
- **不得含 `TODO` / `待填` / `PENDING`** —— `knowledge_lint.py` 的 `D 正文` 会报 warn。

**`title` 同理**：不得含 HTML 残留（`<a name=...>`、`<span>`）。`title` 在检索里权重 **3.0**
（`FIELD_WEIGHTS` 最高），塞垃圾等于稀释最强信号。

---

## R7 · 正文的两条硬禁令

1. **禁止嵌入图片** —— `![...](...)`、`<img>` 均为 blocker（`knowledge_lint.py:98,211`）。
   改为指向上游图片的文字链接。
2. **至多一对 `# 相关` 托管块**（`<!-- okf:related:start … end -->`）。
   该块由 `okf_graph.py inject` 管理，**不要手写**。

---

## R8 · 非卡片资产：与引用它的卡同目录，不单开资产目录

**规则**：`.txt` / `.json` 等非 Markdown 文件（按 R2 永不进语料）与**引用它们的卡放同一目录**。

**理由**：证据紧邻声称。改卡时不容易漏掉配套数据；读卡的人一眼能看到证据就在旁边。

**现存两处**：
- `okf/runbooks/operator-optimization/cube_required_ops.txt` —— 与
  `cann-op-family-arch-classification.md` 同目录（该卡正文自述 "Full op list ... in this dir"）
- `okf/reference/porter/patterns/fa_a3_multicore_mc_{verify,perf}.json` —— pb-56 卡
  「20/20 精度、disk-verifiable」那句声称的原始数据

**反例（已撤销）**：曾建过 `kb/assets/` 集中收纳，结果是证据与声称分离、
卡里的散文引用集体失效且门禁抓不到（不是 markdown 链接）。

---

## R9 · 同名 ≠ 重复。合并前必须逐组核验

**规则**：跨目录同名文件**不得**仅凭文件名判定为重复而合并。

**判据**：去掉 frontmatter 与溯源注释后逐字节比对。差异只有以下三种形态才可合并：
空行、相对链接的 `../` 层数、链接被剥成纯文本。

**反例**：`asc_add.md` 在 `vector_compute/` 与 `reg_vector/` 各有一份，差 >20 行 ——
它们是 **memory-base 与 reg-base 两种编程模型下的两个不同 API**，合并即知识丢失。
同类情况全库 61 组。

---

## R10 · 改动后必须跑的门禁

```bash
P=plugins-community/ascendc-port-orchestrator
$P/engine/src/scripts/okf/okf_kb.sh build     # 重建索引（改了卡就必须重建，否则 verify 报指纹陈旧）
$P/engine/src/scripts/okf/okf_kb.sh lint      # 确定性检查 + 聚合 verify
python3 $P/scripts/md_link_validator.py       # MD-01 相对链接，必须 0 断链
```

**当前基线（2026-09-05）**：`lint` blocker = **1**，即 `okf_graph verify` 的
「cards never judged」。该项已论证排除出范围（图谱层在本插件无消费方、不影响检索、
且属存量状态），详见 `README.md` §6 第 5 条。**除它之外新增任何 blocker 都算回归。**

> ⚠️ `okf_kb.sh lint` **当前不在任何 CI 里跑**（`tests/run-tests.sh` 与 `.gitcode/` 零引用）。
> 所以它拦不住你 —— 得自己跑。

---

## R11 · 已知的操作陷阱（都实际踩过）

| 陷阱 | 表现 | 正确做法 |
|---|---|---|
| **macOS `sort`/`uniq` 对中文按 collation 比较** | 不同的中文字符串被判为重复，计数偏高（实测同一份数据：默认 locale 20 组 vs `LC_ALL=C` 13 组） | 含 CJK 的统计一律 `LC_ALL=C sort \| LC_ALL=C uniq`，并用 python `collections.Counter` 复核 |
| **改了路径不改相对链接** | 目录重组后所有 `../` 层数失效，MD-01 报一片断链 | 按**旧路径**解析链接目标 → 查映射表 → 按**新路径**重算相对路径 |
| **改了卡不重建索引** | `knowledge_query verify` 报 `content fingerprint stale` | 改完卡先 `okf_kb.sh build` 再 `lint` |
| **指向目录的链接** | 扁平化后目录没了，链接失效；而 MD-01 认目录也算解析成功，改动前查不出来 | 链接一律指向具体 `.md` 文件；指向多变体 API 族时没有正典成员，保留文字去掉链接 |
| **补 frontmatter 把正文行挤出固定行窗** | 消费方按固定行数读文件头（`briefs/kb_scope.py` 只在前 20 行找 `applies_to: soc=`）。给一份老文件补上 frontmatter 后，那行落到第 21 行 → 读不到 → **fail-open**，SoC 作用域过滤静默失效。**泄漏方向取决于被门控那张卡的 scope**：本仓被门控的是 a3 专用卡，故方向是 a3 模板发给 a5 worker（2026-09-05 实测）。**实测影响面**：1326 张卡里 161 张在补 frontmatter 前后判定不同，全部是 fail-open 方向；但生产上真正被门控的那张卡并未被打破（tag 在第 14 行，旧窗口尚余 6 行）。行为级测试可能会红，但它报的是症状不是这个原因 | 补 frontmatter 后，grep 一遍谁按行窗读这些文件（`lines[:N]` / `head -N`），行窗一律**从 frontmatter 结束处起算** |
| **给「不该命中」写测试，却没给「该命中」写** | 只验 fail-open 分支通过，正例失效时测试照绿 | 每个作用域过滤至少一对用例：命中 + 不命中；再做一次变异测试（把实现改回旧行为，确认测试真的红） |
| **`git mv` 同时改内容** | rename 相似度下降，`git log --follow` 追不回历史 | 大规模重组拆两个 commit：先纯 `git mv`，再改内容 |

---

## R12 · 「lint 全绿」不等于「知识合格」

`knowledge_lint.py:14-15` 自陈：**distilled-not-copied、description accuracy、
index progressive-disclosure 三条语义原则 out of scope**，只在 `--sample N` 的
LLM 深审里查。

所以：
- `reference/asc-devkit-vendored/` 的 511 张目前是**上游原文的搬运**，**未蒸馏**。
  lint 查不出来，但它确实不符合 `ops-knowledge-reference-ingest/SKILL.md` §1 的「蒸馏非照搬」。
- 报告合规状态时**必须区分**这两个维度，不要用「lint 全绿」暗示内容质量已达标。
