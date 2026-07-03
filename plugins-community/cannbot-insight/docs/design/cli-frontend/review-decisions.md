# CLI 前端架构评审决策文档

> **评审日期**: 2026-06-14  
> **基于**: design.md v1.0 + test-plan.md v1.0  
> **当前版本**: v0.31 (src/lib/version.ts)

---

## 问题逐条分析

---

### 问题1: Commander.js 与 Ink 的 stdin 冲突 [P0]

**设计文档现状**:
design.md L810-812 `runTui()` 直接调用 `ink.render()`，没有任何 stdin 接管边界处理。Commander 解析 argv 期间 Node.js 进程 stdin 处于 cooked mode，而 Ink 需要 raw mode 来捕获键盘事件。

**严重程度**: P0 — 不处理会导致 TUI 模式下键盘事件完全失效或抛出 EAGAIN 错误。

**PoC 验证结果**: Ink 版本确认为 **v7.0.6**（不是 v5），React 19.2.7，需要 ESM（package.json `"type": "module"`），运行工具用 tsx。Ink v7 render() 返回实例方法有变化：没有 lastFrame/frames/output，新增 waitUntilRenderFlush。stdin 切换方案（resume + setRawMode）保持不变。

**决策方案**: Commander 完成 argv 解析后，stdin 处于 paused 状态。需要在 `runTui()` 中显式处理 stdin 切换。Ink v7 的 render() 实例 API 如下：

```typescript
// Ink v7 render() 返回实例方法（PoC 验证）
// - rerender: function          — 重新渲染组件
// - unmount: function           — 卸载组件
// - waitUntilExit: function     — 等待退出
// - waitUntilRenderFlush: function — 新增：等待渲染刷新完成
// - cleanup: function           — 清理资源
// - clear: function             — 清除终端输出
// 注意：Ink v7 原生 render() 没有 lastFrame()、frames、output
// 测试时需用 ink-testing-library 的 render() 才有 lastFrame()
```

```typescript
// src/cli/tui/App.tsx
import { render } from 'ink';
import React from 'react';

export function runTui(config: CliConfig): Promise<void> {
  // Commander 解析完 argv 后 stdin 处于 paused 状态
  // 需要先 resume，再让 Ink 接管 raw mode
  if (process.stdin.isTTY) {
    process.stdin.resume();
    process.stdin.setRawMode(true);
  }

  const { waitUntilExit } = render(<App config={config} />, {
    exitOnCtrlC: false, // 我们自己处理 Ctrl+C
    patchConsole: false, // 不 patch console.log，避免和命令模式冲突
  });

  return waitUntilExit().then(() => {
    // Ink 退出后恢复 stdin
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(false);
      process.stdin.pause();
    }
  });
}
```

关键点：
1. `process.stdin.resume()` — Commander 解析 argv 后 stdin 处于 paused，必须先 resume
2. `process.stdin.setRawMode(true)` — 让 Ink 能捕获键盘事件
3. `exitOnCtrlC: false` — App 中 useInput 自己处理 q/Ctrl+C，避免 Ink 和 App 双重处理
4. 退出时恢复 stdin — 防止进程退出后终端状态异常
5. `patchConsole: false` — 命令模式和 TUI 模式可能交替使用 console，不 patch 更安全
6. Ink v7 需要 ESM — package.json 必须 `"type": "module"`

**是否修改文档**: 是，修改 design.md L810-812 的 `runTui()` 实现，并标注 Ink v7 + ESM 要求。

---

### 问题2: Ink 第三方组件库成熟度 [P0]

**设计文档现状**:
design.md 推荐了 ink-table, ink-select, ink-text-input, ink-spinner, ink-box, ink-big-text 等第三方组件。这些库大多针对 Ink v3/v4 开发，2-3 年未更新，与 Ink v7 (React 19+) 的兼容性存疑。PoC 验证 Ink 实际版本为 v7.0.6。

**严重程度**: P0 — 如果第三方库不兼容，TUI 核心交互（选择、输入、表格）将无法工作。

**PoC 验证结果**: Ink 版本确认为 **v7.0.6**（不是 v5），第三方库兼容性问题更严重。PoC 验证了 Ink v7 基础渲染正常（表格布局、颜色输出、中文字符、borderStyle）。

**评估结论**:

| 组件库 | Ink v7 兼容性 | 替代方案 |
|--------|-------------|----------|
| ink-spinner | 不兼容 v7 | 自写 3 行代码即可 |
| ink-text-input | 不兼容 v7 | 自写（30 行） |
| ink-select-input | 不兼容 v7 | 自写 SelectInput（< 50 行） |
| ink-table | 不兼容 v7 | 自写 DataTable（design.md 已有设计） |
| ink-box | 不兼容 v7 | Ink 内置 `<Box>` 已够用 |
| ink-big-text | 不兼容 v7 | 用 chalk.bold 替代，不需要大字 |

**决策方案**: **全部自实现**，不依赖任何第三方 Ink 组件库。理由：

1. design.md 已经设计了 DataTable、TabBar、ConfirmDialog、AsciiBar、TreeView 等组件，它们不依赖第三方库
2. 缺少的 Spinner 和 TextInput 极简，自写成本低：
   - Spinner: 一个 `<Text>` + useState 轮换字符序列，10 行代码
   - TextInput: 一个 `<Box>` + useInput 捕获字符 + useState 保存输入值，30 行代码
3. 自写组件可以完全控制行为和样式，适配 Ink v7 无风险
4. 避免依赖链问题（Ink v7 的第三方库生态确实不成熟）

```typescript
// src/cli/tui/components/Spinner.tsx — 自写示例
const SPINNER_FRAMES = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'];

function Spinner({ label }: { label?: string }) {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setFrame(f => (f + 1) % SPINNER_FRAMES.length), 80);
    return () => clearInterval(timer);
  }, []);
  return <Text>{SPINNER_FRAMES[frame]} {label ?? 'Loading...'}</Text>;
}

// src/cli/tui/components/TextInput.tsx — 自写示例
function TextInput({ value, onChange, placeholder, focus = true }: TextInputProps) {
  useInput((input, key) => {
    if (!focus) return;
    if (key.backspace || key.delete) onChange(value.slice(0, -1));
    else if (key.return) { /* 提交 */ }
    else if (!key.ctrl && !key.meta) onChange(value + input);
  }, { isActive: focus });
  return (
    <Box>
      <Text color="gray">{placeholder ?? ''}</Text>
      <Text>{value}</Text>
      <Text color="gray">█</Text>
    </Box>
  );
}
```

**是否修改文档**: 是，删除 design.md 中对第三方组件库的引用，新增 Spinner.tsx 和 TextInput.tsx 的自写设计。

---

### 问题3: ink-testing-library 兼容性 [P0] — **决策修正**

**设计文档现状**:
test-plan.md 大量依赖 `ink-testing-library` (v4) 做 TUI 组件渲染测试（4.1-4.9 全部使用 `renderTui` helper）。原评审认为该库不兼容 Ink v7，决策删除依赖。

**严重程度**: P0 — TUI 组件测试（20% 测试比例）的核心工具。

**PoC 验证结果**: `ink-testing-library` **完全兼容 Ink v7**！PoC 3 验证：
- ink-testing-library 可以正常 import
- ink-testing-library 的 render() 返回的实例有 lastFrame() 方法
- lastFrame() 能正确返回渲染后的文本内容

**关键发现**: Ink v7 原生 render() 没有 lastFrame()（PoC 2 验证），但 ink-testing-library 的 render() 提供了 lastFrame()。两者 API 不同：

```
Ink v7 原生 render() 返回:          ink-testing-library render() 返回:
  - rerender                         - rerender
  - unmount                          - unmount
  - waitUntilExit                    - waitUntilExit
  - waitUntilRenderFlush (新增)       - lastFrame ← 关键差异
  - cleanup                          - frames
  - clear                            - stdin (可模拟按键)
  ❌ 没有 lastFrame/frames/output      ✅ 有 lastFrame + stdin
```

**修正决策**: **保留 ink-testing-library 依赖！** 采用双 render 策略：

- **测试代码**: 使用 `ink-testing-library` 的 render() — 有 lastFrame() + stdin 模拟
- **生产代码**: 使用 Ink 原生 render() — 有 waitUntilExit/waitUntilRenderFlush/clear

```typescript
// tests/helpers/render-tui.tsx — 使用 ink-testing-library（PoC 验证兼容）
import React from 'react';
import { render } from 'ink-testing-library';

export function renderTui(element: React.ReactElement) {
  const instance = render(element);
  return {
    ...instance,
    // ink-testing-library 的 lastFrame() 返回 ANSI 含色码的输出
    lastFrame: () => instance.lastFrame(),
    // 获取纯文本（去除 ANSI 色码）
    getPlainText: () => instance.lastFrame()?.replace(/\x1b\[[0-9;]*m/g, '') ?? '',
    // stdin 可模拟按键（ink-testing-library 提供）
    pressKey: (key: string) => instance.stdin.write(key),
    pressEnter: () => instance.stdin.write('\r'),
    pressEscape: () => instance.stdin.write('\x1b'),
    pressUp: () => instance.stdin.write('\x1b[A'),
    pressDown: () => instance.stdin.write('\x1b[B'),
    pressCtrlC: () => instance.stdin.write('\x03'),
  };
}
```

```typescript
// src/cli/tui/App.tsx — 生产代码使用 Ink 原生 render()
import { render } from 'ink';

export function runTui(config: CliConfig): Promise<void> {
  // ... stdin 处理 ...
  const { waitUntilExit } = render(<App config={config} />, { exitOnCtrlC: false });
  return waitUntilExit();
}
```

**层2: handler 逻辑独立测试**（仍保留） — 对复杂 useInput 组件提取 handler 逻辑，不依赖 stdin 模拟：

```typescript
// tests/cli/components/ConfirmDialog.test.tsx — handler 逻辑测试
import { describe, it, expect, vi } from 'vitest';

describe('ConfirmDialog logic', () => {
  it('calls onConfirm when input is y', () => {
    const onConfirm = vi.fn();
    const handler = createInputHandler({ onConfirm, onCancel: vi.fn() });
    handler('y', {});
    expect(onConfirm).toHaveBeenCalled();
  });
});
```

**是否修改文档**: 是，修改 test-plan.md 的 4.1 render-tui.tsx helper 改用 ink-testing-library，保留所有组件测试（4.2-4.9）的现有方案。修改 design.md 中 Ink 版本标注为 v7。

---

### 问题4: 类型定义重复 [P1]

**设计文档现状**:
design.md 在 `src/cli/types.ts` 定义了 20+ 个接口（SessionListItem、TurnItem、BridgeItem 等），而 `src/lib/shared/types.ts` 已有 SessionListItem（字段不同：shared 有 id/createdAt/firstQuery/turnCount/modelName，CLI 有 sessionId/startTime/endTime/totalTokens 等）。

**严重程度**: P1 — 字段名和结构不一致会导致维护困难，但 CLI 类型来自 API response 格式，和 shared 内部类型本就不同。

**决策方案**: CLI 类型源自 API response JSON 结构，和 shared 的内部数据库映射类型（snake_case → camelCase 转换前）确实不同。**不应强行合并**，而是通过明确命名区分：

```typescript
// src/cli/types.ts — CLI 视图类型（来自 API response）
// 前缀 Api 命名，与 shared 内部类型明确区分

// 不复用 shared 的 SessionListItem（字段完全不同）
// shared: { id, createdAt, firstQuery, turnCount, modelName }
// CLI API: { sessionId, taskId, query, startTime, endTime, totalTokens, ... }

// 保留 CLI 专用类型，但增加注释说明来源
/** 来自 /api/observe/data response — 与 shared/SessionListItem 不同 */
export interface ApiSessionListItem { ... }

/** 来自 /api/observe/session/turns response */
export interface ApiTurnItem { ... }
```

对于确实重叠的部分（如 TokenUsage 结构），从 shared 导入：

```typescript
// src/cli/types.ts
import type { TokenUsage } from '@/lib/shared/types';

// CLI 专用类型中复用共享的子结构
export interface ApiTurnItem {
  // ... CLI 专用字段
  tokenUsage: TokenUsage;  // 复用 shared 的 TokenUsage
}
```

**是否修改文档**: 是，修改 design.md 的类型定义章节，给 CLI 类型加 `Api` 前缀，标注哪些复用 shared。

---

### 问题5: InsightClient 每次渲染都重新实例化 [P1]

**设计文档现状**:
design.md L816 App.tsx 中 `const client = new InsightClient(config.server, { timeout: config.timeout })` 直接在组件函数体内创建实例，每次 re-render 都会创建新 client。

**严重程度**: P1 — 导致 useApi deps `[client]` 每次都是新引用，缓存 key 变化，缓存失效。

**决策方案**: 使用 `useMemo` 或 `useRef` 保持 client 单例：

```typescript
// src/cli/tui/App.tsx — 修复方案
function App({ config }: TuiAppProps) {
  // useMemo: config 不变时 client 不变
  const client = useMemo(
    () => new InsightClient(config.server, { timeout: config.timeout }),
    [config.server, config.timeout]  // 只依赖实际变化的字段
  );
  // ... 其他逻辑不变
}
```

选择 `useMemo` 而非 `useRef` 的理由：config 变化时应该重建 client（比如用户修改了 server URL），useMemo 自动处理这种场景。

**是否修改文档**: 是，修改 design.md L816 的 App.tsx。

---

### 问题6: useApi 缓存 key 性能问题 [P1]

**设计文档现状**:
design.md L1277 `const cacheKey = JSON.stringify(deps)` 每次 render 都执行。deps 是数组，每次 render 可能是新引用（即使内容相同），导致 JSON.stringify 白跑。

**严重程度**: P1 — 每次 render 都做 JSON.stringify 是不必要的开销，且如果 deps 是新引用但内容相同，会产生相同 key（这恰好是正确行为），但 stringify 本身浪费 CPU。

**决策方案**: 使用稳定化 deps + 只在 deps 变化时计算 key：

```typescript
// src/cli/hooks/useApi.ts — 优化方案
export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[],
  ttl: number = DEFAULT_TTL,
): { data: T | null; loading: boolean; error: Error | null; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // 只在 deps 实际变化时才重新计算 cacheKey
  const cacheKey = useMemo(() => JSON.stringify(deps), [deps]);

  // ... fetchData 用 useCallback + [fetcher, cacheKey] 依赖
  // （与原设计相同，但 cacheKey 计算被 memo 住了）
}
```

更进一步，如果 deps 中包含函数（如 `() => client.listSessions({ pageSize: 100 })`），每次 render 函数引用都不同，会导致 cacheKey 不稳定。解决方案是将 fetcher 也稳定化：

```typescript
// 调用方应该稳定化 fetcher
const fetchSessions = useCallback(
  () => client.listSessions({ pageSize: 100 }),
  [client]  // client 已通过问题5 的 useMemo 稳定化
);
const { data } = useApi(fetchSessions, [client]);
```

**是否修改文档**: 是，修改 design.md useApi 实现和调用示例。

---

### 问题7: 虚拟滚动 vs 分页 [P1]

**设计文档现状**:
design.md 的 DataTable 实现了虚拟滚动（只渲染 visibleRows 行），但 Ink 是全量重绘框架——每次状态变化都重绘整个输出，虚拟滚动只减少了 React 组件数量，但不减少终端渲染面积。1000+ 条数据时终端仍会卡顿。

**严重程度**: P1 — 虚拟滚动在 Ink 环境下收益有限（只减少 React reconcile 成本，不减少终端 I/O 成本）。

**决策方案**: 改为**分页加载 + 服务端分页**：

```typescript
// src/cli/tui/screens/SessionList.tsx — 分页方案
function SessionList({ client, onSelect }: SessionListProps) {
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  const { data, loading } = useApi(
    useCallback(() => client.listSessions({ page, pageSize: PAGE_SIZE }), [client, page]),
    [client, page]
  );

  useKeyboard({
    onNavigateUp: table.selectUp,
    onNavigateDown: table.selectDown,
    custom: {
      'n': () => setPage(p => p + 1),  // Next page
      'p': () => setPage(p => Math.max(1, p - 1)),  // Previous page
    },
  });

  // DataTable 不再需要虚拟滚动，每页最多 20 行
  return (
    <Box flexDirection="column">
      <DataTable columns={SESSION_COLUMNS} data={data?.items ?? []} selectedIndex={table.state.selectedIndex} />
      <Text color="gray">Page {page}/{Math.ceil((data?.total ?? 0) / PAGE_SIZE)} │ n: Next │ p: Prev │ Total: {data?.total ?? 0}</Text>
    </Box>
  );
}
```

对于 Turn 列表等场景（数据量可能很大），同样采用分页：

```typescript
// API 调用时传递 page 参数
client.getTurns(taskId, { page: 1, pageSize: 50 })
```

如果后端 API 不支持 page 参数，则在前端做分页切割（一次性加载全部数据，前端 slice 分页展示），但限制单次加载上限（如最多 500 条）。

**是否修改文档**: 是，修改 design.md 的 DataTable 设计（移除虚拟滚动），修改 SessionList 和 TurnsTab 的数据加载方式改为分页。

---

### 问题8: E2E TUI 测试在 CI 中不可靠 [P1]

**设计文档现状**:
test-plan.md 的 E2E 测试（5.2-5.4）模拟完整用户流程，但实际上是 API 级别的 E2E（InsightClient + MockServer），并不涉及真实 PTY/TUI 渲染。

**严重程度**: P1 — 原设计的 E2E 实际上是 API 集成测试（不涉及 PTY），所以 CI 可靠性问题不存在于当前设计中。但文档声称是 E2E 测试，名称误导。

**决策方案**: 当前 test-plan.md 的 E2E 测试实际上是**API 流程集成测试**，不需要 PTY。调整分类和命名：

1. 将现有 E2E 测试（5.2-5.4）归类为**API 流程集成测试**，移到 integration 目录
2. TUI 交互测试降级为**组件级交互测试**（mock useInput），不做真实 PTY E2E
3. 如果未来需要真实 TUI E2E，使用 `node-pty` + `child_process.spawn` 在本地手动执行，不进 CI

```typescript
// tests/cli/integration/api-flow.test.ts — 原 E2E 重命名
describe('API flow: session list → detail → turns', () => {
  // ... 原内容不变，只是重命名和移目录
});

// 不再在 CI 中运行真实 TUI E2E
// 本地手动测试脚本放在 scripts/test-tui-manual.sh
```

**是否修改文档**: 是，修改 test-plan.md 将 E2E 测试改为 API 流程集成测试，删除 TUI E2E 章节（或标注为手动测试）。

---

### 问题9: GitHub Actions CI 不适用 [P1]

**设计文档现状**:
test-plan.md L2830-2875 配置了 `.github/workflows/cli-test.yml`，使用 GitHub Actions。项目实际托管在 GitCode 上。

**严重程度**: P1 — CI 配置完全无法使用。

**决策方案**: GitCode 使用 GitLab-compatible CI（`.gitlab-ci.yml`），改为适配 GitCode 的 CI 配置：

```yaml
# .gitlab-ci.yml (GitCode CI)
stages:
  - test

cli-test:
  stage: test
  image: node:22
  only:
    changes:
      - src/cli/**/*
      - tests/cli/**/*
  script:
    - npm ci
    - npx prisma generate
    - npm run test:cli:unit
    - npm run test:cli:components
    - npm run test:cli:integration
    - npm run test:coverage
  artifacts:
    paths:
      - coverage/
```

同时保留 `.github/workflows/cli-test.yml` 作为备用（如果有人 fork 到 GitHub），但主 CI 使用 GitCode 配置。

**是否修改文档**: 是，修改 test-plan.md 的 CI 配置改为 GitCode/GitLab-compatible 格式。

---

### 问题10: 版本号硬编码 [P2]

**设计文档现状**:
design.md L442 Commander `.version('0.18')` 和 StatusBar `<Text> v0.18</Text>` 硬编码了版本号。当前实际版本是 v0.31。

**严重程度**: P2 — 每次版本更新都要手动改两处，容易遗漏。

**决策方案**: 从 `@/lib/version.ts` 导入版本号：

```typescript
// src/cli/index.ts
import { VERSION } from '@/lib/version';

program.version(VERSION);

// src/cli/tui/components/StatusBar.tsx
import { VERSION_DISPLAY } from '@/lib/version';

<Text> {VERSION_DISPLAY}</Text>
```

**是否修改文档**: 是，修改 design.md 的两处硬编码。

---

### 问题11: cbin 短别名未配置 [P2]

**设计文档现状**:
design.md L441 `.alias('cbin')` 在 Commander 中配置了别名，但 package.json 的 bin 字段没有配置。

**严重程度**: P2 — npm install -g 后只能用 `cannbot-insight` 命令，无法用 `cbin`。

**决策方案**: 在 package.json 中配置双入口 bin：

```json
{
  "bin": {
    "cannbot-insight": "./src/cli/index.ts",
    "cbin": "./src/cli/index.ts"
  }
}
```

实际执行时需要编译为 JS 或使用 tsx 运行。建议在 bin 入口脚本中使用 `#!/usr/bin/env node` + 预编译路径，或用 `tsx` 直接运行：

```json
{
  "bin": {
    "cannbot-insight": "./dist/cli/index.js",
    "cbin": "./dist/cli/index.js"
  }
}
```

**是否修改文档**: 是，在 design.md 中新增 package.json bin 配置说明。

---

### 问题12: 缺少认证机制 [P2]

**设计文档现状**:
design.md 的 ClientConfig 只有 baseUrl/timeout/retries/retryDelay，没有 authToken。远程访问时需要认证。

**严重程度**: P2 — 当前项目是本地工具（localhost:21025），远程访问场景非核心需求。但预留接口成本极低。

**决策方案**: 在 ClientConfig 中预留 authToken，但不实现复杂认证流程：

```typescript
// src/cli/client.ts
export interface ClientConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  retryDelay: number;
  authToken?: string;  // 预留：Bearer token
}

// request 方法中：如果有 authToken，自动添加 header
private async request<T>(...): Promise<T> {
  const init: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...(this.config.authToken ? { Authorization: `Bearer ${this.config.authToken}` } : {}),
    },
    signal: AbortSignal.timeout(this.config.timeout),
  };
  // ...
}

// 配置来源：环境变量 CANNBOT_TOKEN 或配置文件
export interface CliConfig {
  server: string;
  timeout: number;
  theme: 'dark' | 'light' | 'auto';
  keybindings: Record<string, string>;
  authToken?: string;  // 预留
}
```

**是否修改文档**: 是，修改 design.md 的 ClientConfig 和 CliConfig。

---

### 问题13: 缺少性能基准 [P2]

**设计文档现状**:
design.md 和 test-plan.md 都没有提到性能基准测试。Ink TUI 在大数据量下的渲染性能、API Client 的请求延迟等没有量化目标。

**严重程度**: P2 — 性能基准对 CLI 工具很重要，但可以在开发中逐步建立。

**决策方案**: 新增简易性能基准测试，不引入额外框架：

```typescript
// tests/cli/bench/render-perf.test.ts
describe('TUI render performance baseline', () => {
  it('DataTable renders 20 rows within 100ms', () => {
    const data = createMockSessionList(20);
    const start = performance.now();
    const output = renderSnapshot(<DataTable columns={columns} data={data} selectedIndex={0} />);
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(100);
  });

  it('API Client listSessions completes within 2s with mock server', async () => {
    const start = performance.now();
    await client.listSessions({ pageSize: 20 });
    const elapsed = performance.now() - start;
    expect(elapsed).toBeLessThan(2000);
  });
});
```

性能目标：

| 指标 | 目标 | 测试方式 |
|------|------|----------|
| DataTable 20 行渲染 | < 100ms | renderSnapshot 计时 |
| DataTable 100 行渲染 | < 300ms | renderSnapshot 计时 |
| API 单次请求（mock） | < 100ms | mock server 计时 |
| useApi 首次加载→渲染 | < 200ms | render + waitFor 计时 |
| 命令模式 sessions 输出 | < 500ms | parseAsync + console 计时 |

**是否修改文档**: 是，在 test-plan.md 中新增性能基准章节。

---

### 问题14: 中文宽度对齐问题 [P1]

**PoC 验证结果**: PoC 4 验证了表格渲染时中文字符宽度对齐有问题。中文字符在终端中占 2 列宽（CJK 双宽字符），但 Ink 的 `<Text width={N}>` 按 N 个字符位计算，导致含中文的列错位。

**严重程度**: P1 — CLI 前端需要显示中文内容（query、contentSummary 等），列错位影响可读性。

**决策方案**: 采用三层策略：

**层1: 关键表格列使用英文标签** — 表头和固定列用英文，避免宽度问题

```typescript
// 列定义中 label 用英文，宽度可控
const SESSION_COLUMNS = [
  { key: '#', label: '#', width: 3 },
  { key: 'startTime', label: 'Date', width: 14 },  // 英文标签
  { key: 'query', label: 'Query', width: 30 },      // 内容可能含中文，需要截断
];
```

**层2: 使用 string-width 库** — 对含中文的动态内容计算实际显示宽度

```typescript
// src/cli/utils/format.ts — 新增
import stringWidth from 'string-width';

export function padEndVisual(str: string, width: number): string {
  const visualWidth = stringWidth(str);
  const padding = width - visualWidth;
  if (padding <= 0) return str.substring(0, width); // 简截断，中文时仍可能不精确
  return str + ' '.repeat(padding);
}

export function truncateVisual(str: string, maxVisualWidth: number): string {
  if (stringWidth(str) <= maxVisualWidth) return str;
  // 逐字符截断直到视觉宽度不超过 maxVisualWidth
  let result = '';
  let currentWidth = 0;
  for (const char of str) {
    const charWidth = stringWidth(char);
    if (currentWidth + charWidth > maxVisualWidth - 1) {
      return result + '…';
    }
    result += char;
    currentWidth += charWidth;
  }
  return result;
}
```

**层3: 使用 cli-truncate 处理截断** — 替代手写截断逻辑

```typescript
import cliTruncate from 'cli-truncate';

export function truncateVisual(str: string, maxWidth: number): string {
  return cliTruncate(str, maxWidth, { trimMidspace: false });
}
```

**新增依赖**: `string-width` + `cli-truncate`（都是 Sindre Sorhus 维护的轻量 ESM 库）

**是否修改文档**: 是，修改 design.md 的格式化工具（新增 padEndVisual/truncateVisual）、表格渲染工具（使用视觉宽度），新增中文宽度处理章节。

---

## 最终决策表

| # | 问题 | 决策 | 修改文件 | 优先级 |
|---|------|------|----------|--------|
| 1 | Commander 与 Ink stdin 冲突 | runTui() 中显式 resume stdin + setRawMode，退出时恢复；Ink v7 render 配置 exitOnCtrlC=false；需要 ESM + tsx | design.md §3.1 App.tsx | P0 |
| 2 | Ink 第三方组件库成熟度 | 全部自实现（Spinner 10行、TextInput 30行），不依赖 ink-table/ink-select/ink-spinner 等第三方库；Ink v7 无兼容库 | design.md 新增 §3.10 Spinner、§3.11 TextInput，删除第三方库引用 | P0 |
| 3 | ink-testing-library 兼容性 | **修正：保留 ink-testing-library！** PoC 验证完全兼容 Ink v7，有 lastFrame()+stdin；测试用 ink-testing-library，生产用 Ink 原生 render() | test-plan.md §4.1 render-tui.tsx（改用 ink-testing-library）、§8.1 依赖包 | P0 |
| 4 | 类型定义重复 | CLI 类型加 Api 前缀与 shared 区分；TokenUsage 等子结构从 shared 导入复用；注释标注来源 | design.md §1.1.3 Response 类型定义 | P1 |
| 5 | InsightClient 每次渲染重新实例化 | 改为 useMemo 创建 client，依赖 [config.server, config.timeout] | design.md §3.1 App.tsx L816 | P1 |
| 6 | useApi 缓存 key 性能问题 | cacheKey 用 useMemo 包裹；fetcher 用 useCallback 稳定化；调用示例更新 | design.md §4.3 useApi.ts | P1 |
| 7 | 虚拟滚动 vs 分页 | 改为服务端分页（pageSize=20）+ 前端翻页（n/p 键）；DataTable 移除虚拟滚动逻辑 | design.md §3.4 DataTable、§5.1 SessionList、§5.4 TurnsTab、§8.1 性能 | P1 |
| 8 | E2E TUI 测试在 CI 中不可靠 | 将 E2E 测试重分类为 API 流程集成测试；TUI 交互测试改为组件级（ink-testing-library render+stdin）；真实 TUI E2E 仅本地手动 | test-plan.md §5 E2E 章节 → integration 目录 | P1 |
| 9 | GitHub Actions CI 不适用 | 改为 GitCode/GitLab-compatible CI（.gitlab-ci.yml）；保留 GitHub Actions 作为备用 | test-plan.md §8.5 CI/CD 集成 | P1 |
| 10 | 版本号硬编码 | 从 @/lib/version.ts 导入 VERSION / VERSION_DISPLAY | design.md §1.3 index.ts、§3.2 StatusBar.tsx | P2 |
| 11 | cbin 短别名未配置 | package.json bin 字段配置 cannbot-insight + cbin 双入口 | design.md 新增 §1.9 package.json bin 配置 | P2 |
| 12 | 缺少认证机制 | ClientConfig 和 CliConfig 预留 authToken 字段；request 方法中自动添加 Authorization header；支持 CANNBOT_TOKEN 环境变量 | design.md §1.1 ClientConfig、§1.5 CliConfig | P2 |
| 13 | 缺少性能基准 | 新增简易性能基准测试（ink-testing-library lastFrame 计时、mock server 计时）；定义 5 个性能目标 | test-plan.md 新增 §9 性能基准 | P2 |
| 14 | 中文宽度对齐问题 | 使用 string-width 库计算显示宽度；关键表格列用英文标签；cli-truncate 处理截断 | design.md 新增 §8.5 中文宽度处理、§1.7 表格渲染工具、§1.6 格式化工具 | P1 |

---

## 实施优先级建议

1. **P0 立即执行**（开发前必须完成）: 问题1、2、3 — stdin 切换（含 ESM+tsx）、组件自实现、测试方案（**保留 ink-testing-library**）
2. **P1 开发中调整**（第一个迭代内完成）: 问题4-9、14 — 类型命名、client/useApi 修复、分页、测试重分类、CI、**中文宽度对齐**
3. **P2 小改进**（后续迭代完成）: 问题10-13 — 版本号、bin 配置、authToken、性能基准
