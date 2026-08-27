# CANNBay v2 设计（atomgit 数据集仓 + proxy jsonl 格式）

## 背景

v1：gitcode `CANNBay` 仓，上传 `.db` 快照（8 表导出），全量 clone，`master` + 20 条轮转。
v2：atomgit `cannbay2` 数据集仓（**公开**），上传 proxy 捕获 jsonl 文件夹，partial clone，`main` 分支。
v1 的上传/解析代码与 API 原样保留（界面上不再出现），用于读取以前存的文件。

## 需求

| # | 需求 | 验收标准 |
|---|------|---------|
| R1 | 上传目标 = `https://atomgit.com/guanxinghua/cannbay2.git`（公开库） | push 成功、布局正确 |
| R2 | 上传载体 = claude-jsonl 文件夹格式：proxy 捕获与 native claude **原文件直传**；opencode 会话由 insight DB **导出 jsonl** 再传 | `sessions/<sid>/<sid>.jsonl` (+ `subagents/`) |
| R3 | 千级文件夹下选一条加载要快 | 列表 <2s（热）/ <5s（含 fetch）；导入单条 <5s |
| R4 | 上传一条 = O(新会话)，与库存量无关 | 上传全程 <5s（push 实测 1.5s） |
| R5 | 上传前数据治理，防止 key 上传（公开库，安全红线）：**唯一强制关口，所有文件一律视为不可信输入** | 治理后仍检出疑似密钥 → 熔断拒绝上传 |
| R6 | 旧 v1 上传/解析保留、UI 不可见 | v1 API 可用、UI 无入口 |
| R7 | 分支 `main`；pull 匿名 HTTPS（公开库）；push 凭据 **base64 密文常量内嵌**（与 v1 同方式），默认零配置，换机/CI 无需任何设置 | |
| R8 | `*.jsonl` 不走 LFS；mirror 初始化时自动剔除模板 `.gitattributes` 中的 LFS 行 | 纯 git 客户端读到真实 JSON |

非目标：不迁移 gitcode 旧数据；不做轮转/归档；不引入 git-lfs；不接入
"CANNbay 数据之湖"（涉密内容去向，settings 页已有配置入口 `cannbay-config.ts`，
当前仅保存凭据留后用）。

## 仓库布局

```
cannbay2/  (main)
  .gitattributes        ← 无 *.jsonl LFS 行
  sessions/
    <sid>/
      <sid>.jsonl        ← 主会话（proxy 落盘原文件·治理后副本）
      subagents/
        sub-xxxx.jsonl
        sub-xxxx.meta.json
```

一个会话一个文件夹。导入时在缓存物化层重排成 adapter 认的 proxy 布局
（`<sid>.jsonl` 与 `<sid>/subagents/` 并列），`importSession(..., 'claude-jsonl')`
零适配复用：subagent 切分、Full Context（system/tools）自动，比 v1 的 .db
快照**多保留 Full Context**。

## opencode 会话导出（无原始 jsonl 的来源）

opencode-db 导入的会话没有捕获文件，上传前由 insight DB 导出 claude-jsonl：

```
Turn 行 → claude 行（user/assistant；assistant 行带 message.id=uuid、model、
usage 用 anthropic 字段名、duration_ms/finishReason）
按 subagentSessionId 分组 → subagents/<sub>.jsonl + meta.json
（toolUseId 从 InteractionBridge 桥接数据取，best-effort）
```

保真边界：导出件含 turns / toolCalls / token 级数据，**不含** system prompt /
Full Context（opencode 原生 db 里没有）——重新导入后这些面板为空。proxy 捕获
会话不受影响（原文件直传，全保真）。导出件同样过治理关口（无条件）。

## 架构

单一持久镜像（partial clone），上传/下载共用：

```
tmp/cannbay2-cache（env CANNBAY2_CACHE_DIR）
  = git clone --filter=blob:none --no-checkout <匿名HTTPS URL>
    ├─ .git/objects         ← blob 按需到达后永久缓存（越用越热）
    ├─ materialized/<sid>/  ← 导入时 cat-file 物化（proxy 布局）
    └─ staging/<sid>/       ← 上传时治理后的待传副本

push URL = base64 密文常量（与 v1 `upload-session/route.ts` 同方式），解码后为
https://<user>:<token>@atomgit.com/guanxinghua/cannbay2.git，凭据随代码走，
换机/CI 零配置。
```

- **list**：`fetch origin main`（增量元数据）→ `git ls-tree -r --name-only
  origin/main sessions/` 得 sid 清单 → 一次 `git log` walk 构建
  文件→(提交人, 内容描述, 时间) 映射（单子进程）。全程不碰 blob。
- **import 一条**：`git ls-tree -r origin/main sessions/<sid>` → 逐文件
  `git cat-file -p <oid>` 写入 `materialized/<sid>/`（重排）→
  `importSession(..., 'claude-jsonl')`。
- **upload 一条**：见下节数据治理流程。`git sparse-checkout set --no-cone
  '/sessions/<sid>/**'` 只物化自己的文件夹 → cp 治理副本 → `git add
  sessions/<sid>`（scoped，禁 `git add .`）→ `git commit -F <msgfile>`（提交人:/
  内容描述: 约定不变）→ fetch+merge 整合并发 → `git push <凭据URL> main`。

写操作（sparse/commit/push）进程内 mutex 串行；读操作（ls-tree/cat-file）无锁。

## 上传前数据治理（R5，fail-closed）

治理是上传的**唯一强制关口**，所有待传文件一律视为不可信输入，不因来源
不同而豁免：

- **proxy 捕获文件**：落盘时 `redactor.ts` 已在 `dispatchEmit` 清洗过 ——
  那只是前端预处理，治理阶段照常全量重扫（幂等，已脱敏文本二次清洗无变化），
  覆盖 redactor 上线前的存量捕获与规则漏网形态。
- **native claude-jsonl 文件**（`~/.claude/projects/**`，从未经过我们的清洗）：
  agent 本身理论上不主动落 key，但**用户往对话里贴 key、agent 执行
  `env`/读配置把密钥回显进上下文**都会让它进会话记录 —— 治理对这类文件
  是第一道也是唯一一道防线。

**公开库，密钥出仓即泄露，故熔断优先于可用性。**

```
源捕获文件（~/.cannbot-insight/proxy/…）
  │ 1. 逐行 JSON.parse → 深度遍历 redactInPlace（结构性键名）
  │    + 每个字符串字段过 redactString（厂家前缀/32hex/Bearer/环境变量回显/URL query）
  │    —— 只写 staging 副本，永不改写原文件；幂等（对已脱敏文本二次清洗无变化）
  ▼
staging/<sid>/ 治理后副本
  │ 2. 独立复检（与清洗分离的第二遍扫描）：疑似密钥残留？
  │    - 否 → 继续 push
  │    - 是 → 熔断：拒绝上传，返回检出项（文件/行号/形态，值已掩码）
  ▼
sessions/<sid>/ → commit → push
```

- 清洗核心移植自 `proxy/src/redactor.ts`（该模块无 proxy 内部依赖）到
  `src/lib/redactor.ts`；**golden parity 测试**锁定两份实现：同一 fixture
  文件两套代码必须产出字节相同的脱敏结果，漂移即测试失败。
- 治理范围：主 jsonl + subagents 全部 jsonl + meta.json（全扫，无一例外）。

## 改动清单

新增（v2 全部收进 `src/lib/cannbay2/` 子目录，不再往 `src/lib/` 根撒扁平文件）：
- `src/lib/cannbay2/mirror.ts` — git 交互层：partial clone mirror
  （ensure/fetch/写 mutex）、`ls-tree` 列表、`cat-file` 物化重排、
  `git log` 元数据 walk、sparse scoped 提交、SSH push、`.gitattributes` 自修
- `src/lib/cannbay2/governance.ts` — 自 `proxy/src/redactor.ts` 移植的
  redactor（parity 锁）+ 上传前扫描/复检熔断
- `src/lib/cannbay2/export.ts` — Turn/ToolCall/Bridge → claude-jsonl
  （opencode 来源导出）
- `src/lib/cannbay2/index.ts` — list / import / upload 编排，route 唯一入口
- `src/app/api/ingest/cannbay2/route.ts` — `action: list | import | upload`

修改：
- `LocalFileImport.tsx`（CANNBay tab → v2）、`SessionList.tsx`（上传按钮 → v2，
  三类来源均可传：proxy 捕获 / native claude 用 `sourcePath` 原文件直传，
  opencode 走 DB 导出 jsonl）、
  CLI `client.ts`/`commands/upload.ts`、TUI `App.tsx`
- `src/lib/version.ts` → 1.82、README ×2

不动（R6）：
- v1 `upload-session`/`import-from-cannbay` route、`export-service`、
  `cannbay-archive`、cannbot-insight adapter、核心管线

## 测试方案（IT）

- **闭环 IT**（不碰真实 atomgit）：本地 `git init --bare` 伪远端 → v2 上传
  fixture 捕获（含 subagent + 注入 key 各形态）→ 断言远端文件已脱敏、
  列表元数据正确 → import 断言 turns/subagent execution/bridge 落库 +
  Full Context 可读
- **native jsonl 上传 IT**：无 `subagents/` 的 native claude 会话文件（内含
  用户贴 key / env 回显场景）→ 治理后上传 → 远端文件无明文密钥 →
  导入回读正常
- **opencode 导出上传 IT**：opencode-db fixture 会话 → DB 导出 jsonl →
  上传 → 导入回读 turns/toolCalls/subagent 数据一致
- **熔断 IT**：构造清洗后仍残留的形态 → 上传被拒、远端无新 commit
- **parity IT**：`proxy/src/redactor.ts` 与 `src/lib/redactor.ts` 对同一
  fixture 输出字节一致
- v1 route 回归（旧 .db list/import 不受影响）

## 性能预算与实测

| 操作 | 预算 | 真实环境实测① | 千级库存实测② |
|------|------|--------------|--------------|
| list 冷启动（全新 partial clone） | — | 796ms | 167ms |
| list 热缓存 | <2s | ~500ms（含 fetch RTT） | ~160ms（本地）+ 网络 RTT ≈ <1s |
| import 一条（物化 + 管线） | <5s | 478ms + 755ms | 69ms + 108ms |
| upload 一条 | <5s | 4.35s（3.36M 数据 + 首次 LFS 行剔除 + push） | 150ms（库存 1000 条时新增 1 条，O(新会话)） |
| 磁盘（partial 镜像 vs 全量 clone） | — | — | 4.4M vs 130M（30×） |

① atomgit cannbay2 真实仓，上传/回读一条真实 proxy 会话（1.48M 主文件 + 10 subagent 文件）。
② 本地 bare 仓合成 1000 会话 / 7000 文件 / 1001 commit（每条 ~200KB、一条一 commit、提交信息按约定格式）。合成内容压缩率偏高，只影响"全量 vs partial"对比幅度；列表/物化/上传测的是元数据与对象数，结论不受影响。

压测记录：千级压测发现 listSessions 存在 O(N²)（每 sid 全量 filter 一次
fileCommit 映射），1000 会话时热列表 1794ms；改为 log walk 单遍按 sid 分组后
159ms（11×）。当前各操作均在预算 1/10 以内，余量可支撑万级。

梯度压测（1000→2000，每档 +100，本地 bare 仓同仓增量增长；每档"上传一条"
走完整真实管线：治理→sparse→commit→push）：

| 库存 | 热列表 | 物化冷会话 | 导入一条 | 上传一条 | 镜像磁盘 |
|------|--------|-----------|---------|---------|---------|
| 1000 | 152ms | 55ms | 117ms | 145ms | 5M |
| 1100 | 173ms | 58ms | 113ms | 165ms | 5M |
| 1200 | 197ms | 67ms | 103ms | 167ms | 5M |
| 1300 | 227ms | 66ms | 97ms | 174ms | 6M |
| 1500 | 266ms | 66ms | 102ms | 190ms | 6M |
| 1600 | 294ms | 75ms | 96ms | 189ms | 7M |
| 1700 | 328ms | 66ms | 98ms | 196ms | 7M |
| 1800 | 362ms | 75ms | 104ms | 203ms | 8M |
| 1900 | 405ms | 72ms | 117ms | 225ms | 8M |
| 2000 | 456ms | 240ms* | 98ms | 226ms | 8M |

- **查询/新增单条 = 常数**：物化 ~55-75ms（240ms 为 2000 档单次首触抖动）、
  导入 ~100ms、上传 145→226ms（+1000 库存仅 +80ms，来自 index/tree 规模
  的缓增），与库存量基本无关
- **热列表线性缓增**：~0.3ms/会话（log walk 输出随 commit 数增长），
  2000 档 456ms + 网络 fetch RTT ≈ <1s；外推 10000 档 ~2.3s + RTT
- 镜像磁盘随元数据缓增（~3KB/会话），blob 仍只存看过/传过的

真实尺寸梯度压测（每条 1-10MB 循环、均值 5.5MB，主 70% + 3 subagent 各 10%；
同法增量 1000→2000，总耗时 13.9min；远端压缩后 2000 档 172M）：

| 库存 | 热列表 | 物化冷会话(1MB档) | 导入一条(1MB档) | 上传一条(6MB档) |
|------|--------|------------------|----------------|----------------|
| 1000 | 171ms | 85ms | 762ms | 486ms |
| 1200 | 229ms | 110ms | 633ms | 533ms |
| 1500 | 305ms | 111ms | 651ms | 530ms |
| 1800 | 387ms | 116ms | 627ms | 579ms |
| 2000 | 477ms | 374ms* | 949ms | 620ms |

（1100/1300/1600/1700/1900 档与相邻档一致，表内省略；*为 2000 档单次抖动）

- **blob 大小不影响列表**：1-10MB 会话的热列表（171→477ms）与 100KB 会话
  的（152→456ms）几乎重合 —— 列表只读元数据，两种尺寸实测互证
- 单条操作随"该条自身大小"线性、与库存无关：1MB 物化 85-125ms / 导入
  ~650ms；6MB 上传（本地 push）~550ms；真实 3.36M 会话参照真实环境轮
  （物化 478ms / 上传 4.35s 含网络）
- 压测方法修正：本地路径 clone 默认硬链接、忽略 `--filter`，本地压测/IT
  的镜像实为全量（物化退化成本地读）——`cloneMirror` 已加 `--no-local`
  修复；真实 HTTPS 镜像不受影响。本地压测中"镜像磁盘"列因此不采信

真实远端梯度压测（atomgit `test_cannbay` 专用测试仓，1-10MB/条同法合成，
1000→2000 每档 +100，总耗时 15.1min；含完整网络传输）：

| 库存 | 热列表 | 物化冷会话(1MB档) | 导入一条(1MB档) | 上传一条(6MB档) |
|------|--------|------------------|----------------|----------------|
| 1000 | 641ms | 4847ms* | 1370ms | 3353ms |
| 1200 | 644ms | 1051ms | 1278ms | 2809ms |
| 1500 | 781ms | 1165ms | 1233ms | 3285ms |
| 1800 | 882ms | 1334ms | 1268ms | 3532ms |
| 2000 | 1073ms | 1503ms | 1303ms | 3815ms |

（*首触含 TLS 冷启动；1100/1300/1600/1700/1900 档与相邻一致。补种吞吐：
首批 1000 条（5.5GB 原始→压缩 ~116M）305s，之后每 +100 条 40-110s。）

- 真实网络下全部在预算内：查询/新增单条与库存无关（上传微增来自 index
  规模），热列表 = fetch RTT(~500ms) + ~0.25ms/会话元数据项，外推万级 ~3s
- 物化成本为**协议协商固定开销主导**（每 blob 一次 HTTPS 协商；实测批量
  fetch 无收益：1MB 会话 1.6-1.7s vs 逐取 1.1-1.5s，已试已回退）。大尺寸
  会话物化 ~2s（8-10MB 档，传输开始占主导）。若未来要再压：方向是减少单
  会话文件数（合并为单文件）或 bundle 化，而非 git 参数调优

## 风险与开放问题

| 风险 | 应对 |
|------|------|
| 公开库内容敏感性不止 key（源码片段、内部路径、人名随会话公开） | 已决策：涉密内容未来走"CANNbay 数据之湖"（settings 页已有配置入口，仅保存、暂不接入）；公开仓上传由用户按非涉密自行把关 |
| atomgit 单仓体积/单文件数配额 | 撞到再启轮转或分仓（v1 planArchive 可复用） |
| 超大单会话（>100MB）撞平台单文件限制 | 上传前检查文件大小，超限报错跳过 |
| 同 sid 重复上传 | 按"覆盖更新"语义（新 commit 替换文件夹）；内容无变化时优雅提示 |
| 捕获进行中上传 = 快照不完整 | 允许（git 可再传覆盖），列表不做特殊标注 |
| git ≥2.19 / 服务端 filter 能力 | atomgit 已实测支持；镜像损坏删库重建（秒级） |
| push 凭据 base64 密文内嵌源码（与 v1 同级风险） | 可接受（用户决策，与 v1 保持一致）；泄露时改密 + 换常量 |
