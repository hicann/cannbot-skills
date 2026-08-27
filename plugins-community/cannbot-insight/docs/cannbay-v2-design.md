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

## 性能预算

| 操作 | 预算 | 依据 |
|------|------|------|
| list（千会话，热缓存） | <2s | ls-tree + 内存映射，零 blob |
| list（含增量 fetch） | <5s | 只拉新 commit 元数据 |
| import 一条 | <5s | ~10 blob 按需下载（实测单 blob 0.6s） |
| upload 一条 | <5s | push 实测 1.5s，O(新会话) |

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
