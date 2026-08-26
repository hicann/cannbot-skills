# claude-jsonl 适配 cannbot-proxy 格式：重构方案

## 要解决什么

`claude-jsonl.ts` 本来是解析 claude-code jsonl 的核心适配器。后来为了显示 cannbot-proxy 捕获的 System/Tools/Memory/Skills（Full Context 面板），往里加了 `readFullContext`——它读 proxy 在 assistant 行上挂的两个扩展字段 `system` / `tools`。

问题：proxy 特有的逻辑住进了核心解析文件。proxy 以后再加扩展字段（metadata / cache / retry…），又得改 `claude-jsonl.ts`。核心对 proxy 的演进没关上。

目标：把 proxy 特有的 `readFullContext` 剥出去，核心只管 claude 格式的通用解析，对修改闭合、对扩展开放。

## 现状

`claude-jsonl.ts` 908 行，真正跟 proxy 耦合的就两样：

- `readFullContext`（39 行）—— 读 `line.system` / `line.tools` 扩展字段 + 扫消息取 memory/skills
- `FullContext` / `FullContextTool` 两个类型

其余（行解析、turn 构建、子代理目录约定、`stripSystemReminders`）都是 claude 格式本身的东西——claude 原生 jsonl 也走这套，不是 proxy 特有，留在核心。

顺带一提：类型层其实已经闭合了。`ClaudeJsonlLine` 接口没声明 `system`/`tools`，`readFullContext` 是靠 `(line as { system?: unknown })` 强转访问的。所以核心数据契约对 proxy 无感，只是 `readFullContext` 这个函数物理上住错了地方。

## 改完长什么样

```
src/lib/ingest/adapters/
├── claude-jsonl.ts                 核心。逻辑不动，只把几个 helper 改成 export
└── claude-jsonl-full-context.ts     proxy 扩展层。读扩展字段 + 组装 Full Context
```

依赖单向：扩展层 → 核心。核心不 import 扩展层。

## 具体怎么改

**1. 核心 `claude-jsonl.ts`：改可见性 + 删 proxy 符号**

把这几个现在 private 的纯函数和类型改成 `export`，给扩展层复用：

- `parseJsonlLines`（行解析基座）
- `extractSystemText`、`findMemorySection`、`findSkillsSection`（消息扫描）
- `ClaudeJsonlLine`、`ContentBlock`（类型）

它们本来就是 claude 格式概念，export 出去不算破坏封装。

然后删掉 `readFullContext` + `FullContext` / `FullContextTool`——搬到扩展层。

**2. 新建 `claude-jsonl-full-context.ts`**

把 `readFullContext` 和两个类型搬过去，逻辑逐行等价。顺手把消息扫描那段抽成 `scanMessageContext` 内部函数（复用、可单测）。proxy 以后加新扩展字段，就在这个文件里改，核心不动。

```ts
import { parseJsonlLines, extractSystemText, findMemorySection, findSkillsSection,
  type ClaudeJsonlLine, type ContentBlock } from './claude-jsonl';

export interface FullContextTool { name: string; description: string }
export interface FullContext { systemPrompt: string; tools: FullContextTool[]; memoryFiles: string; skills: string }

export function readFullContext(filePath: string): FullContext | null {
  const lines = parseJsonlLines(filePath);
  if (lines.length === 0) return null;
  // 扩展字段：第一条带 tools 的 assistant 行（proxy 捕获）
  let systemPrompt = '', tools: FullContextTool[] = [], foundExt = false;
  for (const line of lines) {
    if (foundExt || line.type !== 'assistant') continue;
    const sys = (line as { system?: unknown }).system;
    const tls = (line as { tools?: unknown }).tools;
    if (Array.isArray(tls) && tls.length > 0) {
      foundExt = true;
      systemPrompt = extractSystemText(sys as string | ContentBlock[] | undefined) ?? '';
      tools = (tls as Array<{ name?: string; description?: string }>)
        .map(t => ({ name: t.name ?? '', description: t.description ?? '' })).filter(t => t.name);
    }
  }
  const { memoryFiles, skills } = scanMessageContext(lines); // 抽出来的扫描 helper
  if (!foundExt && !memoryFiles && !skills) return null;
  return { systemPrompt, tools, memoryFiles, skills };
}
```

**3. 调用方 import 改向**

`turns/[turnId]/route.ts` 和测试文件：`readFullContext` 从扩展层 import，`listSubagentSessions` 还从核心 import。`readSessionFullContext` 本身逻辑不动。

## 影响范围

纯搬移 + 改可见性，逻辑等价：

- proxy 扩展 jsonl：解析结果不变
- 原始 claude jsonl：不受影响（`readFullContext` 没扩展字段时优雅降级，返回空 system/tools + 从消息扫出来的 memory/skills，跟现在一样）
- `stripSystemReminders`、子代理目录约定：留在核心，不动
- API 响应、UI 显示：零变化

## 风险

- 循环 import：依赖单向，`tsc`/`eslint` 能检出
- 漏改 import：`npm run build` 编译期就报
- 回滚：单 commit，`git revert` 即可，无 DB/数据变更

## 验收

1. `npm run test` 全绿（insight + proxy），proxy 目录存活（不被测试删）
2. 实跑 session 4bcecc74 turn 13（subagent），Full Context 仍是子代理自己的（13 tools / 4195 system）

## 不做

- `stripSystemReminders` / 子代理目录约定不搬——claude 原生也用，是核心行为
- 不搞策略注册表 / 接口抽象——目前就一个扩展点，注册表是过度设计，等真有第二种非-claude 格式要 Full Context 再说
