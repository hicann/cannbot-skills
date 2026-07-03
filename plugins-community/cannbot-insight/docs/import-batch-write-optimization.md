# P0 优化方案：导入写入从逐条 create 改为批量 createMany + $transaction

> 目标：将单次 session 导入的 ~100+ 次独立 SQLite INSERT 合并为 1 次事务内的 8 个批量 INSERT，
> 预期性能提升 **10-50x**。

---

## 1. 当前问题

`src/lib/ingest/data-service.ts` 中，新建 session 的写入路径（L377-L524）全部使用逐条 `await client.xxx.create()`：

```
session.create              × 1
turn.create                 × N (每个 turn 一条，N 通常 30-100)
toolCall.create             × M (每个 toolCall 一条)
skillEvent.create           × K (每个 skillEvent 一条)
interactionBridge.create    × B (每个 bridge 一条)
execution.create            × E (每个 execution 一条)
executionSkill.create       × S (嵌套在 execution 循环内)
sessionSkill.create         × L (每个 skill 一条)
```

典型 session（50 turns + 30 toolCalls + 10 skillEvents + 5 bridges + 3 executions + 2 executionSkills + 4 sessionSkills）= **~104 次独立 INSERT**。

SQLite 每次自动提交事务约 1-3ms（含 WAL journal + fsync），加上 Prisma ORM 的 JS→Rust→C 序列化开销，实际单次 create 约 5-10ms。104 次 ≈ 520ms-1040ms。

---

## 2. 优化方案

### 2.1 核心改动：createMany + $transaction

将所有逐条 `create` 替换为 `createMany`，并将所有写入包裹在 `$transaction` 中。

**Prisma 对 SQLite 的 `createMany` 支持情况**：Prisma 6.x + SQLite 完全支持 `createMany`，底层生成单条 `INSERT INTO ... VALUES (...), (...), (...)` 语句。

**`$transaction` 的作用**：将多个操作合并为 1 次 SQLite 事务提交，避免每次 INSERT 都触发 journal write + fsync。

#### 改动后的写入流程（新建 session 路径）

```ts
// ── 1. 准备所有数据（纯内存操作，不涉及 DB）──
const sessionData = { ... };          // session 行
const turnsData = turns.map(t => {    // 批量 turn 行
  const { cost, ...rest } = t;
  return { ...rest, sessionId: createdSessionId, createdAt_ts: ..., completedAt: ... };
});
const toolCallsData = toolCalls.map(tc => ({ ...tc, startedAt: ..., completedAt: ... }));
const skillEventsData = skillEvents.map(se => ({ ...se, startedAt: ..., completedAt: ... }));
const bridgesData = bridges.map(b => ({ sessionId: createdSessionId, ... }));
const executionsData = executions.map(e => ({ ...e, sessionId: createdSessionId }));
const executionSkillsData = [];       // 扁平化，不再嵌套在 execution 循环中
for (const [execId, skills] of executionSkillsMap) {
  for (const es of skills) {
    executionSkillsData.push({ executionId: execId, ...es });
  }
}
const sessionSkillsData = uniqueSkillNames.map(skillName => ({ sessionId: createdSessionId, ... }));

// ── 2. 一次性事务写入 ──
const createdSessionId = await client.$transaction(async (tx) => {
  const session = await tx.session.create({ data: sessionData });
  const sid = session.id;

  // 给所有子行填入正确的 sessionId
  for (const t of turnsData) t.sessionId = sid;
  for (const b of bridgesData) b.sessionId = sid;
  for (const e of executionsData) e.sessionId = sid;
  for (const s of sessionSkillsData) s.sessionId = sid;

  await tx.turn.createMany({ data: turnsData });
  await tx.toolCall.createMany({ data: toolCallsData });
  await tx.skillEvent.createMany({ data: skillEventsData });
  await tx.interactionBridge.createMany({ data: bridgesData });
  await tx.execution.createMany({ data: executionsData });
  await tx.executionSkill.createMany({ data: executionSkillsData });
  await tx.sessionSkill.createMany({ data: sessionSkillsData });

  return sid;
});
```

**关键细节**：
- `session.create` 必须在事务内先执行，拿到 `createdSessionId` 后才能给子表填外键
- `createMany` 不返回创建的行（SQLite limitation），但我们的 ID 都是预生成的（`generateId()`），不需要依赖 DB 生成的 ID
- `TurnRow`、`ToolCallRow` 等类型中的 `id` 字段已由 `generateId()` 预生成，`createMany` 会直接使用

---

### 2.2 需处理的类型适配

#### 2.2.1 TurnRow 的 `cost` 字段

Prisma schema 中 `Turn` 没有 `cost` 字段（`data-service.ts:406` 已有 `const { cost, ...rest } = turn` 剚除）。
批量写入时同样需要统一剔除：

```ts
const turnsData = turns.map(t => {
  const { cost: _cost, ...rest } = t;
  return { ...rest, sessionId: sid, createdAt_ts: ..., completedAt: ... };
});
```

#### 2.2.2 日期字段转换

当前逐条 create 时，日期字段是在每条写入时单独转 `new Date(...)`。
批量写入需要预先转换：

```ts
function toDate(v: string | null): Date | null {
  return v ? new Date(v) : null;
}
```

#### 2.2.3 ExecutionSkill 扁平化

当前 `executionSkill` 是嵌套在 `execution` 循环内逐条写入的。改为批量写入需要先将 `executionSkillsMap` 扁平化为数组：

```ts
const executionSkillsData: ExecutionSkillCreateInput[] = [];
for (const [execId, skills] of executionSkillsMap) {
  for (const es of skills) {
    executionSkillsData.push({
      executionId: execId,
      skillName: es.skillName,
      skillVersion: es.skillVersion,
      isPrimary: es.isPrimary,
      user: es.user,
    });
  }
}
```

---

### 2.3 增量导入路径（dedup/merge）同样需要优化

`data-service.ts` L222-L366 的增量导入路径（session 已存在时追加新 turn/toolCall/skillEvent）也是逐条写入：

```ts
for (const turn of newTurns) {
  await client.turn.create({ data: ... });
}
for (const tc of newToolCalls) {
  await client.toolCall.create({ data: ... });
}
for (const se of newSkillEvents) {
  await client.skillEvent.create({ data: ... });
}
```

同样改为 `createMany` + `$transaction`：

```ts
const sid = dedupResult.existingSessionId!;
const newTurnsData = newTurns.map(t => {
  const { cost: _cost, ...rest } = t;
  return { ...rest, sessionId: sid, createdAt_ts: toDate(rest.createdAt_ts), completedAt: toDate(rest.completedAt) };
});
const newToolCallsData = newToolCalls.map(tc => ({ ...tc, startedAt: toDate(tc.startedAt), completedAt: toDate(tc.completedAt) }));
const newSkillEventsData = newSkillEvents.map(se => ({ ...se, startedAt: toDate(se.startedAt), completedAt: toDate(se.completedAt) }));

await client.$transaction(async (tx) => {
  await tx.turn.createMany({ data: newTurnsData });
  await tx.toolCall.createMany({ data: newToolCallsData });
  await tx.skillEvent.createMany({ data: newSkillEventsData });
});
```

---

## 3. 涉及的文件

| 文件 | 改动 |
|------|------|
| `src/lib/ingest/data-service.ts` | **主要改动**：新建路径 L377-L524 和增量路径 L222-L366 的所有 `for + create` 改为 `createMany` + `$transaction` |

其他文件（`opencode-db.ts`、`turn-split.ts`、`bridge-builder.ts`、`execution-split.ts`、`normalize.ts`、`merge.ts`）均为纯内存操作，**无需改动**。

---

## 4. 不改动的部分

- **`listSessions()` / `readSession()` 的 N+1 查询问题** → 属于 P1 优化，不在本方案范围内
- **DB 连接复用** → P2 优化，不在本方案范围内
- **`splitIntoTurns` 的 O(n²)** → P3 优化，不在本方案范围内

---

## 5. 预期性能收益

| 场景 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| 单 session（50 turns 等，~104 次 INSERT） | ~520-1040ms（5-10ms × 104） | ~5-15ms（1 次事务 + 批量 INSERT） | **35-100x** |
| 大 session（200 turns，~300 次 INSERT） | ~1500-3000ms | ~10-20ms | **75-300x** |
| 增量导入追加 5 个新 turn | ~25-50ms | ~2-5ms | **5-25x** |

> 注：实际收益取决于 SQLite WAL 配置和磁盘 I/O。SQLite 默认在 WAL 模式下批量 INSERT 极快，
> 单次事务内 INSERT 100 行仅需数毫秒。

---

## 6. 验证方案

1. **单元测试**：现有 `tests/cli/unit/commands/import.test.ts` 应能直接运行通过（功能不变，只是写入方式改变）
2. **性能基准测试**：手动对比导入前后耗时
   - 在本地 `opencode.db` 上导入一个 session，记录耗时
   - 应用优化后再次导入同一 session，对比耗时
3. **数据完整性校验**：导入后查询 DB，验证所有 turn/toolCall/skillEvent/bridge/execution/sessionSkill 数量与逐条写入结果一致
4. **增量导入校验**：对已存在的 session 再次导入，验证 merge 逻辑正确追加新数据

---

## 7. 风险与注意事项

### 7.1 `createMany` 不返回创建的行

Prisma 的 `createMany` 返回 `{ count: number }`，不返回创建的行数据。我们的场景中所有 ID 都是预生成的（`generateId()`），不依赖 DB 返回的 ID，所以无影响。

### 7.2 `createMany` 的 `skipDuplicates` 选项

如果增量导入中可能存在重复数据，可以使用 `createMany({ data, skipDuplicates: true })`。但当前 merge 逻辑已确保只传入新数据，因此不需要 `skipDuplicates`。

### 7.3 SQLite 的 SQL 变量限制

SQLite 单条 INSERT 的 VALUES 列表无硬性上限（Prisma 会生成 `INSERT INTO ... VALUES (...), (...), ...`），但极端大 session（>1000 turns）可能导致单条 SQL 过长。Prisma 内部对此有限制（约 999 个变量/参数），超限时需分批。

**应对方案**：添加分批写入辅助函数：

```ts
async function batchCreateMany(
  tx: PrismaTransactionClient,
  model: 'turn' | 'toolCall' | 'skillEvent' | ...,
  data: any[],
  batchSize: number = 500
): Promise<void> {
  for (let i = 0; i < data.length; i += batchSize) {
    const batch = data.slice(i, i + batchSize);
    await tx[model].createMany({ data: batch });
  }
}
```

正常 session 不会触发分批，但以防万一需要此保护。

### 7.4 `$transaction` 的交互式写法

Prisma 的 `$transaction(async (tx) => { ... })` 交互式事务在 SQLite 下完全支持。注意事务内的所有操作必须使用 `tx` 而非 `client`。

---

## 8. 实现步骤

1. 在 `data-service.ts` 中新增 `toDate()` 辅助函数和 `batchCreateMany` 辅助函数
2. 改写新建 session 的写入路径（L377-L524）：逐条 `create` → `createMany` + `$transaction`
3. 改写增量导入的写入路径（L222-L366）：逐条 `create` → `createMany` + `$transaction`
4. 运行现有测试确认功能正确
5. 手动性能对比测试
