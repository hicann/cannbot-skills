// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi } from "vitest";
import type { PrismaClient } from "@prisma/client";
import {
  stripSkillContent,
  recoverSkillBody,
  recoverStateMdPlan,
  recoverPlanFileDeclaration,
  recoverMainAgentWorkflowBody,
  resolveWorkflowSkillName,
  recoverWorkflowDeclaration,
  buildAuditArgs,
  buildStructuredRecords,
  buildTranscriptArgs,
  auditKindsForEvents,
  isDispatchOnlyAgent,
  dispatchOnlySkillNames,
  getInflightAudit,
} from "@/lib/sift-audit";

/** 造一个最小 mock prisma：只实现 recoverSkillBody 用到的两个方法。 */
function makePrisma(opts: {
  skillEvent?: { turnId: string } | null
  toolCalls?: Array<{ toolName: string; argsJson: string | null; resultJson: string | null }>
}) {
  return {
    skillEvent: {
      findFirst: async () => opts.skillEvent ?? null,
    },
    toolCall: {
      findMany: async () => opts.toolCalls ?? [],
    },
  } as unknown as Parameters<typeof recoverSkillBody>[2];
}

describe("stripSkillContent", () => {
  it("剥掉 <skill_content name=…> 外壳、返回内部正文（trim）", () => {
    const resultJson =
      '<skill_content name="ascendc-crash-debug">\n# Skill: ascendc-crash-debug\n调试算子卡死\n</skill_content>';
    expect(stripSkillContent(resultJson)).toBe("# Skill: ascendc-crash-debug\n调试算子卡死");
  });

  it("name 属性带别的值也能剥", () => {
    expect(stripSkillContent('<skill_content name="foo-bar">BODY</skill_content>')).toBe("BODY");
  });

  it("多行正文完整保留", () => {
    const inner = "line1\n\nline2\n- a\n- b";
    expect(stripSkillContent(`<skill_content name="x">${inner}</skill_content>`)).toBe(inner);
  });

  it("没有 skill_content 外壳 → null", () => {
    expect(stripSkillContent("just some text without wrapper")).toBeNull();
  });

  it("空串 / null-ish → null", () => {
    expect(stripSkillContent("")).toBeNull();
    expect(stripSkillContent(null as unknown as string)).toBeNull();
  });
});

describe("recoverSkillBody", () => {
  const SID = "ses_cuid_1";

  it("invoke 事件 + 同 turn 的 skill ToolCall(resultJson 含正文) → 剥壳返回", async () => {
    const prisma = makePrisma({
      skillEvent: { turnId: "t1" },
      toolCalls: [
        {
          toolName: "skill",
          argsJson: '{"name":"ascendc-crash-debug"}',
          resultJson: '<skill_content name="ascendc-crash-debug">\nBODY\n</skill_content>',
        },
      ],
    });
    expect(await recoverSkillBody(SID, "ascendc-crash-debug", prisma)).toBe("BODY");
  });

  it("无 invoke 事件（只有 dispatch）→ null", async () => {
    const prisma = makePrisma({ skillEvent: null });
    expect(await recoverSkillBody(SID, "x", prisma)).toBeNull();
  });

  it("invoke 事件在、但 ToolCall 的 resultJson 空 → null", async () => {
    const prisma = makePrisma({
      skillEvent: { turnId: "t1" },
      toolCalls: [{ toolName: "skill", argsJson: '{"name":"x"}', resultJson: null }],
    });
    expect(await recoverSkillBody(SID, "x", prisma)).toBeNull();
  });

  it("toolName 是 task/dispatch（非 invoke）→ 被滤掉、无候选 → null", async () => {
    const prisma = makePrisma({
      skillEvent: { turnId: "t1" },
      toolCalls: [{ toolName: "task", argsJson: '{"name":"x"}', resultJson: "<skill_content>x</skill_content>" }],
    });
    expect(await recoverSkillBody(SID, "x", prisma)).toBeNull();
  });

  it("按 argsJson.name 精确匹配（同 turn 多 skill 时挑对的那个）", async () => {
    const prisma = makePrisma({
      skillEvent: { turnId: "t1" },
      toolCalls: [
        { toolName: "skill", argsJson: '{"name":"other"}', resultJson: '<skill_content name="other">WRONG</skill_content>' },
        { toolName: "skill", argsJson: '{"name":"want"}', resultJson: '<skill_content name="want">RIGHT</skill_content>' },
      ],
    });
    expect(await recoverSkillBody(SID, "want", prisma)).toBe("RIGHT");
  });
});

describe("buildAuditArgs", () => {
  it("拼出 sift audit 的参数数组：位置参 skill 目录 + --db/--session/-o + --kind skill", () => {
    // --kind skill 必带（--db 路是 per-skill 对账）：不带会拿本 skill 的 SKILL.md 去对账整条
    // session（含别的 skill 跑的 turn）。sift 靠 SKILL.md 的 name + --kind skill 切本 skill 的段。
    expect(
      buildAuditArgs({
        skillPath: "/tmp/x/myskill",
        dbPath: "/a/b.db",
        sessionId: "ses_1",
        outputDir: "/tmp/out",
      }),
    ).toEqual([
      "audit",
      "/tmp/x/myskill",
      "--db",
      "/a/b.db",
      "--session",
      "ses_1",
      "-o",
      "/tmp/out",
      "--kind",
      "skill",
      "--turn-refs",
    ]);
  });
});

describe("buildTranscriptArgs", () => {
  it("kind=skill：拼出 --transcript 参数 + --kind skill + --turn-refs", () => {
    expect(
      buildTranscriptArgs({
        skillPath: "/tmp/x/myskill",
        transcriptPath: "/tmp/s.json",
        outputDir: "/tmp/out",
        kind: "skill",
      }),
    ).toEqual([
      "audit",
      "/tmp/x/myskill",
      "--transcript",
      "/tmp/s.json",
      "-o",
      "/tmp/out",
      "--kind",
      "skill",
      "--turn-refs",
    ]);
  });

  it("kind=agent：--kind agent + --turn-refs（审被 dispatch 的 agent，按 agent 归属切）", () => {
    expect(
      buildTranscriptArgs({
        skillPath: "/tmp/x/developer",
        transcriptPath: "/tmp/s.json",
        outputDir: "/tmp/out",
        kind: "agent",
      }),
    ).toEqual([
      "audit",
      "/tmp/x/developer",
      "--transcript",
      "/tmp/s.json",
      "-o",
      "/tmp/out",
      "--kind",
      "agent",
      "--turn-refs",
    ]);
  });
});

describe("auditKindsForEvents", () => {
  // Skills tab 每行按 events 的 eventType 决定该跑哪种对账:invoke/use → skill 面
  // (Skill 调用,正文从 session 恢复);dispatch → agent 面(被 dispatch,.md 从 AGENTS_ROOT 读)。
  // dispatch-only 行(如 developer)原来挂 skill 对账按钮必 404(无 invoke 正文可恢复)→ 路由到 agent 对账。
  it("invoke-only → [skill](纯 skill,对 skill 声明)", () => {
    expect(auditKindsForEvents([{ eventType: "invoke" }, { eventType: "invoke" }])).toEqual(["skill"]);
  });

  it("'use' 也算 skill 面(Skill 调用的别名 eventType)", () => {
    expect(auditKindsForEvents([{ eventType: "use" }])).toEqual(["skill"]);
  });

  it("dispatch-only → [agent](纯 agent,如 developer;原 skill 对账会 404)", () => {
    expect(auditKindsForEvents([{ eventType: "dispatch" }, { eventType: "dispatch" }])).toEqual(["agent"]);
  });

  it("dual-nature(invoke+dispatch 同名,如 st-verifier)→ [skill, agent] 两面", () => {
    expect(auditKindsForEvents([{ eventType: "invoke" }, { eventType: "dispatch" }])).toEqual(["skill", "agent"]);
  });

  it("空 events → [](无锚点,不显示对账按钮)", () => {
    expect(auditKindsForEvents([])).toEqual([]);
  });

  it("未知 eventType → [](既非 invoke/use 也非 dispatch,不当工具用)", () => {
    expect(auditKindsForEvents([{ eventType: "load" }, { eventType: "whatever" }])).toEqual([]);
  });
});

describe("recoverMainAgentWorkflowBody", () => {
  // 主 agent 通常只 dispatch、不 invoke skill，其 workflow 声明是 session 首条 user turn
  // （isSubagent=false、role=user、turnIndex 最小）的注入系统提示。内容过短（<500）→ null。
  const body = "# 基于 ACLNN 的算子开发工作流\n\n你是纯编排者（Orchestrator），" + "x".repeat(500);
  const shortQuery = "帮我写个 hello world";
  const makeTurnPrisma = (content: string | null) =>
    ({ turn: { findFirst: vi.fn().mockResolvedValue(content === null ? null : { content }) } }) as unknown as PrismaClient;

  it("首条 user turn 内容 ≥500 → 返回正文", async () => {
    const prisma = makeTurnPrisma(body);
    const r = await recoverMainAgentWorkflowBody("ses_1", prisma);
    expect(r).toBe(body);
    expect(prisma.turn.findFirst).toHaveBeenCalledWith({
      where: { sessionId: "ses_1", role: "user", isSubagent: false },
      orderBy: { turnIndex: "asc" },
    });
  });

  it("首条 user turn 过短（<500）→ null（视为普通短查询，非 workflow 声明）", async () => {
    const prisma = makeTurnPrisma(shortQuery);
    expect(await recoverMainAgentWorkflowBody("ses_1", prisma)).toBeNull();
  });

  it("无 user turn → null", async () => {
    const prisma = makeTurnPrisma(null);
    expect(await recoverMainAgentWorkflowBody("ses_1", prisma)).toBeNull();
  });

  it("content 为空 → null", async () => {
    const prisma = { turn: { findFirst: vi.fn().mockResolvedValue({ content: null }) } } as unknown as PrismaClient;
    expect(await recoverMainAgentWorkflowBody("ses_1", prisma)).toBeNull();
  });
});

describe("recoverStateMdPlan", () => {
  // 主 agent 编排 具体计划：从 session 文件工具调用（Read/Write of STATE.md）恢复最长内容，
  // 剥行号 + [x]→[ ] 归一 plan 态。无 STATE.md / 过短 → null。
  const stateBody = "# 工作流 STATE\n\n## Tasks\n\n- [x] 1. 创建开发工作区\n- [ ] 2. 初始化开发日志\n" + "x".repeat(120);
  const makeStatePrisma = (tcs: Array<{ toolName: string; argsJson: string; resultJson: string | null }>) =>
    ({ toolCall: { findMany: vi.fn().mockResolvedValue(tcs) } }) as unknown as PrismaClient;

  it("Read of STATE.md → 剥行号 + [x]→[ ] 归一 plan 态", async () => {
    const withLineNums = stateBody.split("\n").map((l, i) => `${i + 1}: ${l}`).join("\n");
    const prisma = makeStatePrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: `<path>/x/STATE.md</path><content>${withLineNums}</content>` },
    ]);
    const r = await recoverStateMdPlan("ses_1", prisma);
    expect(r).not.toBeNull();
    expect(r).toContain("[ ] 1. 创建开发工作区"); // [x] 归一为 [ ]
    expect(r).not.toContain("[x]");
    expect(r).not.toMatch(/^\s*\d+:/m); // 行号已剥
  });

  it("Write of STATE.md（argsJson.content）→ 取 content", async () => {
    const prisma = makeStatePrisma([
      { toolName: "write", argsJson: JSON.stringify({ filePath: "/x/STATE.md", content: stateBody }), resultJson: null },
    ]);
    expect(await recoverStateMdPlan("ses_1", prisma)).toContain("创建开发工作区");
  });

  it("多个 STATE.md 读 → 取最长那份", async () => {
    const short = "# short\n\n## Tasks\n\n- [ ] a\n" + "x".repeat(120);
    const long = "# long\n\n## Tasks\n\n- [x] task A\n- [ ] task B\n" + "x".repeat(200);
    const prisma = makeStatePrisma([
      { toolName: "read", argsJson: '{"filePath":"/a/STATE.md"}', resultJson: `<content>${short}</content>` },
      { toolName: "read", argsJson: '{"filePath":"/b/STATE.md"}', resultJson: `<content>${long}</content>` },
    ]);
    expect(await recoverStateMdPlan("ses_1", prisma)).toContain("task A");
  });

  it("无 STATE.md 工具调用 → null", async () => {
    expect(await recoverStateMdPlan("ses_1", makeStatePrisma([]))).toBeNull();
  });

  it("内容过短（<100）→ null", async () => {
    const prisma = makeStatePrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: `<content>1: 短</content>` },
    ]);
    expect(await recoverStateMdPlan("ses_1", prisma)).toBeNull();
  });
});

describe("recoverPlanFileDeclaration（D 混合方案：文件名模式 + 内容特征）", () => {
  // 主 agent 编排 计划文件泛化提取：单次查所有 read/write 工具调用，内存双桶分流。
  // named 桶（文件名命中 PLAN_FILE_NAMES）优先；named 无 substantial → scan 桶（内容 plan-like）。
  // 返回 { content, source }：source = basename（named）或 "scan:"+basename（scan）。
  const planBody = "# 工作流\n\n## Tasks\n\n- [x] 1. 创建工作区\n- [ ] 2. 初始化日志\n- [ ] 3. 派发 developer\n" + "x".repeat(120);
  const makePlanPrisma = (tcs: Array<{ toolName: string; argsJson: string; resultJson: string | null }>) =>
    ({ toolCall: { findMany: vi.fn().mockResolvedValue(tcs) } }) as unknown as PrismaClient;

  // ── 阶段 1：文件名模式（named 桶）──

  it("STATE.md Read → named 桶，source=STATE.md，剥行号 + [x]→[ ]", async () => {
    const withNums = planBody.split("\n").map((l, i) => `${i + 1}: ${l}`).join("\n");
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: `<content>${withNums}</content>` },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r).not.toBeNull();
    expect(r!.source).toBe("STATE.md");
    expect(r!.content).toContain("[ ] 1. 创建工作区");
    expect(r!.content).not.toContain("[x]");
    expect(r!.content).not.toMatch(/^\s*\d+:/m);
  });

  it("TODO.md Write → named 桶，source=TODO.md（argsJson.content）", async () => {
    const prisma = makePlanPrisma([
      { toolName: "write", argsJson: JSON.stringify({ filePath: "/x/TODO.md", content: planBody }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("TODO.md");
    expect(r!.content).toContain("创建工作区");
  });

  it("TODO.md / TASKS.md / WORKFLOW.md 等其他计划文件名 → named 桶", async () => {
    for (const name of ["TODO.md", "TASKS.md", "WORKFLOW.md", "BACKLOG.md"]) {
      const prisma = makePlanPrisma([
        { toolName: "write", argsJson: JSON.stringify({ filePath: `/x/${name}`, content: planBody }), resultJson: null },
      ]);
      const r = await recoverPlanFileDeclaration("ses_1", prisma);
      expect(r!.source).toBe(name);
    }
  });

  it("文件名大小写不敏感（state.md / Plan.md）→ named 桶，source 保留原大小写", async () => {
    const prisma = makePlanPrisma([
      { toolName: "write", argsJson: JSON.stringify({ filePath: "/x/state.md", content: planBody }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("state.md");
  });

  it("file_path 字段（非 filePath）→ 仍能提取 basename", async () => {
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"file_path":"/x/STATE.md"}', resultJson: `<content>${planBody}</content>` },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("STATE.md");
  });

  it("多个计划文件读 → named 桶取最长", async () => {
    const short = "# short\n\n## Tasks\n\n- [ ] a\n" + "x".repeat(120);
    const long = "# long\n\n## Tasks\n\n- [ ] a\n- [ ] b\n" + "x".repeat(200);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/a/STATE.md"}', resultJson: `<content>${short}</content>` },
      { toolName: "read", argsJson: '{"filePath":"/b/TODO.md"}', resultJson: `<content>${long}</content>` },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.content).toContain("long");
  });

  // ── 阶段 2：内容特征（scan 桶，fallback）──

  it("非标命名 + checklist ≥3 → scan 桶，source=scan:文件名", async () => {
    const body = "# my plan\n\n- [ ] task A\n- [ ] task B\n- [ ] task C\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/work-plan.md"}', resultJson: `<content>${body}</content>` },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r).not.toBeNull();
    expect(r!.source).toBe("scan:work-plan.md");
    expect(r!.content).toContain("task A");
  });

  it("非标命名 + 计划标题 ## Tasks → scan 桶（标题强信号）", async () => {
    const body = "## Tasks\n\n这是任务清单的说明文字，无 checklist 无编号，但有计划标题。" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/notes.md"}', resultJson: `<content>${body}</content>` },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("scan:notes.md");
  });

  it("非标命名 + 编号 ≥5 → scan 桶（编号弱信号需多条）", async () => {
    const body = "1. task one\n2. task two\n3. task three\n4. task four\n5. task five\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/steps.md"}', resultJson: `<content>${body}</content>` },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("scan:steps.md");
  });

  it("非标命名 + checklist <3（仅 2 个）→ null（不误提）", async () => {
    const body = "- [ ] only one\n- [ ] two\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/code.ts"}', resultJson: `<content>${body}</content>` },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  it("非标命名 + 编号 <5（仅 2 条）→ null（源码列表不误提）", async () => {
    const body = "1. first\n2. second\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/util.ts"}', resultJson: `<content>${body}</content>` },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  it("非标命名 + 无 plan 特征（普通源码）→ null", async () => {
    const body = "export function foo() {\n  return 42;\n}\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/foo.ts"}', resultJson: `<content>${body}</content>` },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  it("源码含 1 个 `- [ ]` TODO 注释 → null（checklist <3）", async () => {
    const body = "export function foo() {\n  // - [ ] TODO: refactor\n  return 42;\n}\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/foo.ts"}', resultJson: `<content>${body}</content>` },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  // ── 双桶优先级 + 混合 ──

  it("named 桶优先于 scan 桶（即使 scan 更长）", async () => {
    const namedShort = "# short STATE\n\n## Tasks\n\n- [ ] a\n" + "x".repeat(120);
    const scanLong = "## Tasks\n\n- [ ] a\n- [ ] b\n- [ ] c\n" + "x".repeat(300);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: `<content>${namedShort}</content>` },
      { toolName: "read", argsJson: '{"filePath":"/x/work.md"}', resultJson: `<content>${scanLong}</content>` },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("STATE.md");
    expect(r!.content).toContain("short STATE");
  });

  it("named 桶过短 → 退 scan 桶（named 不 substantial）", async () => {
    const namedShort = "1: 短 STATE";
    const scanLong = "## Tasks\n\n- [ ] a\n- [ ] b\n- [ ] c\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: `<content>${namedShort}</content>` },
      { toolName: "read", argsJson: '{"filePath":"/x/work-plan.md"}', resultJson: `<content>${scanLong}</content>` },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("scan:work-plan.md");
  });

  // ── 边界：跳过 + 容错 ──

  it("无文件工具调用 → null", async () => {
    expect(await recoverPlanFileDeclaration("ses_1", makePlanPrisma([]))).toBeNull();
  });

  it("计划文件过短（<100 字）→ null（两桶都不 substantial）", async () => {
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: `<content>1: 短</content>` },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  it("todowrite 仅 1 todo（<3 阈值）→ 不够 substantial，null", async () => {
    const prisma = makePlanPrisma([
      { toolName: "todowrite", argsJson: '{"todos":[{"content":"task A","status":"in_progress"}]}', resultJson: null },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  it("Read resultJson 无 <content>（文件不存在）→ 跳过该 tc", async () => {
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: "<path>/x/STATE.md</path><error>not found</error>" },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  it("Write argsJson 非 JSON → 跳过该 tc", async () => {
    const prisma = makePlanPrisma([
      { toolName: "write", argsJson: "not-json", resultJson: null },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  it("Write argsJson 无 content 字段 → 跳过该 tc", async () => {
    const prisma = makePlanPrisma([
      { toolName: "write", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: null },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  it("argsJson 为 null → 跳过（不崩）", async () => {
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: null, resultJson: `<content>${planBody}</content>` },
    ]);
    // argsJson null → extractFilePath null → basename null → 不进 named 桶；
    // 但 content 有 plan 特征 → scan 桶，source=scan:unknown
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r).not.toBeNull();
    expect(r!.source).toBe("scan:unknown");
  });

  it("Write content 含 [x] → 归一为 [ ]（plan 态）", async () => {
    const body = "## Tasks\n\n- [x] done task\n- [ ] todo\n- [ ] another\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "write", argsJson: JSON.stringify({ filePath: "/x/STATE.md", content: body }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.content).toContain("[ ] done task");
    expect(r!.content).not.toContain("[x]");
  });

  // ── todowrite 提取（主 agent 自己的任务计划）──

  it("todowrite ≥3 todos → source=todowrite，格式为 plan 态 checklist", async () => {
    const todos = [
      { content: "1.1 开发准备：创建算子目录、LOG.md、issues目录、环境检查", status: "in_progress", priority: "high" },
      { content: "1.2 需求分析：生成REQUIREMENTS.md和aclnnAPI接口文档", status: "pending", priority: "high" },
      { content: "1.3 方案设计：生成DESIGN.md和PLAN.md（跳过CP1）", status: "pending", priority: "high" },
      { content: "1.4 测试设计：生成TEST.md和测试用例", status: "pending", priority: "high" },
    ];
    const prisma = makePlanPrisma([
      { toolName: "todowrite", argsJson: JSON.stringify({ todos }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r).not.toBeNull();
    expect(r!.source).toBe("todowrite");
    expect(r!.content).toContain("- [ ] 1.1 开发准备");
    expect(r!.content).toContain("- [ ] 1.4 测试设计");
    expect(r!.content).not.toContain("in_progress"); // status 不入正文
  });

  it("todowrite cancelled 项保留（是计划的一部分）", async () => {
    const todos = [
      { content: "1.3R 方案评审：跳过", status: "cancelled", priority: "low" },
      { content: "1.4 测试设计", status: "pending", priority: "high" },
      { content: "1.5 开发实现", status: "pending", priority: "high" },
    ];
    const prisma = makePlanPrisma([
      { toolName: "todowrite", argsJson: JSON.stringify({ todos }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.content).toContain("- [ ] 1.3R 方案评审：跳过");
  });

  it("多个 todowrite 调用 → 取 todos 最多的那份（最新最完整）", async () => {
    const few = [{ content: "task A", status: "pending" }, { content: "task B", status: "pending" }, { content: "task C", status: "pending" }];
    const many = [
      { content: "task A", status: "completed" },
      { content: "task B", status: "completed" },
      { content: "task C", status: "completed" },
      { content: "task D", status: "pending" },
      { content: "task E", status: "pending" },
    ];
    const prisma = makePlanPrisma([
      { toolName: "todowrite", argsJson: JSON.stringify({ todos: few }), resultJson: null },
      { toolName: "todowrite", argsJson: JSON.stringify({ todos: many }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.content).toContain("task E"); // 取 many（5 > 3）
  });

  it("todowrite argsJson 非 JSON → 跳过", async () => {
    const prisma = makePlanPrisma([
      { toolName: "todowrite", argsJson: "not-json", resultJson: null },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  it("todowrite argsJson 无 todos 数组 → 跳过", async () => {
    const prisma = makePlanPrisma([
      { toolName: "todowrite", argsJson: '{"other":"data"}', resultJson: null },
    ]);
    expect(await recoverPlanFileDeclaration("ses_1", prisma)).toBeNull();
  });

  // ── 优先级：named-Read > todowrite > named-Write > scan ──

  it("S1 场景：STATE.md Read + plan-like 优先于 todowrite（dispatch 动作）", async () => {
    // STATE.md 有 ## Tasks + checklist（plan-like）→ named-Read 桶，最高优先
    // todowrite 是 dispatch 动作（"Dispatch developer to..."），不如 STATE.md 详细
    const stateRead = "# 工作流 STATE\n\n## Tasks\n\n- [x] 1. 创建开发工作区\n  - 执行者：developer\n  - 验收标准：test -d\n- [ ] 2. 初始化日志\n- [ ] 3. 派发 developer\n" + "x".repeat(120);
    const todos = [
      { content: "Dispatch developer to create docs", status: "in_progress" },
      { content: "Dispatch developer to init log", status: "pending" },
      { content: "Dispatch verifier to check", status: "pending" },
    ];
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: `<content>${stateRead}</content>` },
      { toolName: "todowrite", argsJson: JSON.stringify({ todos }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("STATE.md"); // plan-like 的 STATE.md 优先
    expect(r!.content).toContain("创建开发工作区");
  });

  it("S2 场景：PLAN.md 已从 PLAN_FILE_NAMES 移除 → 不识别，退 todowrite", async () => {
    // PLAN.md 不在 PLAN_FILE_NAMES（产物，从来不是 workflow）→ 不进 named 桶
    const planRead = "# BenchNoHarnessLstm 迭代执行计划\n\n## 迭代一穿刺列表\n\n| 任务类型 | TilingKey | Dtype | 验证目标 |\n|---------|-----------|-------|---------|\n| 主线 | FLOAT,1,0,0,0 | fp32 | 完整正确性 |\n" + "x".repeat(120);
    const todos = [
      { content: "1.1 开发准备：创建算子目录", status: "in_progress", priority: "high" },
      { content: "1.2 需求分析：生成REQUIREMENTS.md", status: "pending", priority: "high" },
      { content: "1.3 方案设计：生成DESIGN.md和PLAN.md", status: "pending", priority: "high" },
      { content: "1.4 测试设计：生成TEST.md", status: "pending", priority: "high" },
      { content: "设计阶段完成后暂停", status: "pending", priority: "high" },
    ];
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/PLAN.md"}', resultJson: `<content>${planRead}</content>` },
      { toolName: "todowrite", argsJson: JSON.stringify({ todos }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("todowrite"); // PLAN.md 不识别 → 退 todowrite
    expect(r!.content).toContain("1.1 开发准备");
  });

  it("无 todowrite → 退 named-Read（STATE.md Read + plan-like 兜底）", async () => {
    const stateRead = "# 工作流 STATE\n\n## Tasks\n\n- [x] 1. 创建开发工作区\n- [ ] 2. 初始化日志\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "read", argsJson: '{"filePath":"/x/STATE.md"}', resultJson: `<content>${stateRead}</content>` },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("STATE.md");
  });

  it("S2 场景：PLAN.md Write（产物，已移除）+ todowrite → todowrite 胜出", async () => {
    const planWrite = "# 方案设计 PLAN\n\n## Tasks\n\n- [ ] 步骤 A\n- [ ] 步骤 B\n- [ ] 步骤 C\n" + "x".repeat(120);
    const todos = [
      { content: "1.1 开发准备：创建算子目录", status: "in_progress", priority: "high" },
      { content: "1.2 需求分析：生成REQUIREMENTS.md", status: "pending", priority: "high" },
      { content: "1.3 方案设计：生成DESIGN.md和PLAN.md", status: "pending", priority: "high" },
      { content: "1.4 测试设计：生成TEST.md", status: "pending", priority: "high" },
      { content: "设计阶段完成后暂停", status: "pending", priority: "high" },
    ];
    const prisma = makePlanPrisma([
      { toolName: "write", argsJson: JSON.stringify({ filePath: "/x/PLAN.md", content: planWrite }), resultJson: null },
      { toolName: "todowrite", argsJson: JSON.stringify({ todos }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    expect(r!.source).toBe("todowrite"); // todowrite > named-Write
    expect(r!.content).toContain("1.1 开发准备");
  });

  it("PLAN.md Write 但无 todowrite → 不识别（已从 PLAN_FILE_NAMES 移除）→ 退 scan 或 null", async () => {
    const planWrite = "# 方案设计\n\n## Tasks\n\n- [ ] 步骤 A\n- [ ] 步骤 B\n- [ ] 步骤 C\n" + "x".repeat(120);
    const prisma = makePlanPrisma([
      { toolName: "write", argsJson: JSON.stringify({ filePath: "/x/PLAN.md", content: planWrite }), resultJson: null },
    ]);
    const r = await recoverPlanFileDeclaration("ses_1", prisma);
    // PLAN.md 不在 PLAN_FILE_NAMES → 不进 named-Write；内容 plan-like → scan 桶
    expect(r).not.toBeNull();
    expect(r!.source).toBe("scan:PLAN.md");
    expect(r!.content).toContain("步骤 A");
  });
});

describe("resolveWorkflowSkillName + recoverWorkflowDeclaration", () => {
  // root 对账声明 = workflow skill 的编排规程（slice.py："root 声明常是 workflow 级 SKILL.md"）
  // 从主 agent Skill invoke 定位 skillName → recoverSkillResourceFile / recoverSkillBody 恢复
  const makeWfPrisma = (opts: {
    skillEvent?: { skillName: string; turnId?: string } | null
    resourceTcs?: Array<{ toolName: string; argsJson: string; resultJson: string | null }>
    skillBodyTcs?: Array<{ toolName: string; argsJson: string | null; resultJson: string | null }>
    turn0?: { content: string } | null
  }) => ({
    skillEvent: {
      findFirst: vi.fn()
        // resolveWorkflowSkillName 调用（isSubagent=false, eventType=invoke）
        .mockResolvedValueOnce(opts.skillEvent ?? null)
        // recoverSkillBody 调用（如果退到 fallback）
        .mockResolvedValueOnce(opts.skillEvent?.turnId ? { turnId: opts.skillEvent.turnId } : null),
    },
    toolCall: {
      findMany: vi.fn()
        // recoverSkillResourceFile 调用
        .mockResolvedValueOnce(opts.resourceTcs ?? [])
        // recoverSkillBody 调用（如果退到 fallback）
        .mockResolvedValueOnce(opts.skillBodyTcs ?? []),
    },
    turn: {
      findFirst: vi.fn().mockResolvedValue(opts.turn0 ?? null),
    },
  }) as unknown as PrismaClient;

  const resourceBody = "# Task 调用参数\n\n## 1.1 开发准备\n\n- [ ] 开发日志已创建\n- [ ] 问题目录已创建\n- [ ] 环境检查已执行\n" + "x".repeat(120);
  const skillBodyContent = "# Skill: ops-registry-invoke-workflow\n\n## 核心原则\n\n- 测试驱动\n- 阶段递进\n- 阶段门控\n" + "x".repeat(120);

  it("resolveWorkflowSkillName → 主 agent 首 invoke 的 skillName", async () => {
    const prisma = {
      skillEvent: { findFirst: vi.fn().mockResolvedValue({ skillName: "ops-registry-invoke-glacier" }) },
    } as unknown as PrismaClient;
    const name = await resolveWorkflowSkillName("ses_1", prisma);
    expect(name).toBe("ops-registry-invoke-glacier");
  });

  it("resolveWorkflowSkillName → 无 invoke → null", async () => {
    const prisma = {
      skillEvent: { findFirst: vi.fn().mockResolvedValue(null) },
    } as unknown as PrismaClient;
    expect(await resolveWorkflowSkillName("ses_1", prisma)).toBeNull();
  });

  it("recoverWorkflowDeclaration → skill resources 文件（task-prompts.md）优先", async () => {
    const prisma = makeWfPrisma({
      skillEvent: { skillName: "ops-registry-invoke-workflow" },
      resourceTcs: [
        { toolName: "read", argsJson: '{"filePath":"/x/ops-registry-invoke-workflow/resources/task-prompts.md"}', resultJson: `<content>${resourceBody}</content>` },
      ],
    });
    const r = await recoverWorkflowDeclaration("ses_1", prisma);
    expect(r).not.toBeNull();
    expect(r!.source).toBe("task-prompts.md");
    expect(r!.content).toContain("开发准备");
  });

  it("recoverWorkflowDeclaration → 无 resources → 退 SKILL.md body", async () => {
    const prisma = makeWfPrisma({
      skillEvent: { skillName: "ops-registry-invoke-workflow", turnId: "t1" },
      resourceTcs: [], // 无 resource 文件
      skillBodyTcs: [
        { toolName: "skill", argsJson: '{"name":"ops-registry-invoke-workflow"}', resultJson: `<skill_content name="ops-registry-invoke-workflow">${skillBodyContent}</skill_content>` },
      ],
    });
    const r = await recoverWorkflowDeclaration("ses_1", prisma);
    expect(r).not.toBeNull();
    expect(r!.source).toBe("ops-registry-invoke-workflow (SKILL.md)");
    expect(r!.content).toContain("核心原则");
  });

  it("recoverWorkflowDeclaration → 无 Skill invoke → null", async () => {
    const prisma = makeWfPrisma({ skillEvent: null });
    expect(await recoverWorkflowDeclaration("ses_1", prisma)).toBeNull();
  });

  it("recoverWorkflowDeclaration → resources 内容过短 → 退 SKILL.md body", async () => {
    const prisma = makeWfPrisma({
      skillEvent: { skillName: "ops-registry-invoke-workflow", turnId: "t1" },
      resourceTcs: [
        { toolName: "read", argsJson: '{"filePath":"/x/ops-registry-invoke-workflow/resources/task-prompts.md"}', resultJson: `<content>1: 短</content>` },
      ],
      skillBodyTcs: [
        { toolName: "skill", argsJson: '{"name":"ops-registry-invoke-workflow"}', resultJson: `<skill_content name="ops-registry-invoke-workflow">${skillBodyContent}</skill_content>` },
      ],
    });
    const r = await recoverWorkflowDeclaration("ses_1", prisma);
    expect(r!.source).toBe("ops-registry-invoke-workflow (SKILL.md)");
  });

  it("recoverWorkflowDeclaration → filePath 不含 skillName → 不算 resource", async () => {
    const prisma = makeWfPrisma({
      skillEvent: { skillName: "ops-registry-invoke-workflow", turnId: "t1" },
      resourceTcs: [
        // filePath 不含 skillName → 不是 skill 的 resource 文件
        { toolName: "read", argsJson: '{"filePath":"/x/other-file.md"}', resultJson: `<content>${resourceBody}</content>` },
      ],
      skillBodyTcs: [
        { toolName: "skill", argsJson: '{"name":"ops-registry-invoke-workflow"}', resultJson: `<skill_content name="ops-registry-invoke-workflow">${skillBodyContent}</skill_content>` },
      ],
    });
    const r = await recoverWorkflowDeclaration("ses_1", prisma);
    // resource 不匹配 → 退 SKILL.md body
    expect(r!.source).toBe("ops-registry-invoke-workflow (SKILL.md)");
  });

  it("S1 型：无 Skill invoke → 退 turn0 ≥500（注入编排规程）", async () => {
    // S1 的 turn0 是 1809 字的编排规程（"你是纯编排者，读取 STATE.md..."）
    const turn0Body = "# 基于 ACLNN 的算子开发工作流\n\n你是纯编排者（Orchestrator），读取 STATE.md 按顺序派发 SubAgent。" + "x".repeat(500);
    const prisma = {
      skillEvent: { findFirst: vi.fn().mockResolvedValue(null) }, // 无 Skill invoke
      turn: { findFirst: vi.fn().mockResolvedValue({ content: turn0Body }) },
    } as unknown as PrismaClient;
    const r = await recoverWorkflowDeclaration("ses_1", prisma);
    expect(r).not.toBeNull();
    expect(r!.source).toBe("turn0 (注入编排规程)");
    expect(r!.content).toContain("纯编排者");
  });

  it("S1 型：无 invoke + turn0 <500 → null（短用户任务不是编排规程）", async () => {
    const prisma = {
      skillEvent: { findFirst: vi.fn().mockResolvedValue(null) },
      turn: { findFirst: vi.fn().mockResolvedValue({ content: "帮我写个 hello world" }) },
    } as unknown as PrismaClient;
    expect(await recoverWorkflowDeclaration("ses_1", prisma)).toBeNull();
  });
});

describe("isDispatchOnlyAgent", () => {
  // Skills 表据此过滤：仅 dispatch（子代理分派，如 blackbox-designer）不展示为 skill。
  it("dispatch-only → true（子代理，应从 Skills 表隐藏）", () => {
    expect(isDispatchOnlyAgent([{ eventType: "dispatch" }, { eventType: "dispatch" }])).toBe(true);
  });

  it("dispatch + unload → true（unload 不算 skill 调用）", () => {
    expect(isDispatchOnlyAgent([{ eventType: "dispatch" }, { eventType: "unload" }])).toBe(true);
  });

  it("有 invoke → false（真 skill）", () => {
    expect(isDispatchOnlyAgent([{ eventType: "dispatch" }, { eventType: "invoke" }])).toBe(false);
  });

  it("有 load/use → false（真 skill）", () => {
    expect(isDispatchOnlyAgent([{ eventType: "load" }])).toBe(false);
    expect(isDispatchOnlyAgent([{ eventType: "use" }])).toBe(false);
  });

  it("空 events → true（无 skill 调用证据，按子代理处理）", () => {
    expect(isDispatchOnlyAgent([])).toBe(true);
  });
});

describe("dispatchOnlySkillNames", () => {
  // 全页统一过滤：返回仅 dispatch 的 skillName 集合，Overview/Skills 图表据此排除子代理。
  it("只列出仅有 dispatch 事件的 name", () => {
    const ev = [
      { skillName: "blackbox-designer", eventType: "dispatch" },
      { skillName: "real-skill", eventType: "invoke" },
      { skillName: "real-skill", eventType: "dispatch" },
      { skillName: "bare-dispatch", eventType: "dispatch" },
    ];
    const set = dispatchOnlySkillNames(ev);
    expect(set.has("blackbox-designer")).toBe(true);
    expect(set.has("bare-dispatch")).toBe(true);
    expect(set.has("real-skill")).toBe(false);
    expect(set.size).toBe(2);
  });

  it("load/use 也算 skill 事件（不归入排除集）", () => {
    const set = dispatchOnlySkillNames([
      { skillName: "a", eventType: "load" },
      { skillName: "b", eventType: "use" },
      { skillName: "c", eventType: "dispatch" },
    ]);
    expect(set.has("a")).toBe(false);
    expect(set.has("b")).toBe(false);
    expect(set.has("c")).toBe(true);
  });

  it("空 events → 空集", () => {
    expect(dispatchOnlySkillNames([]).size).toBe(0);
  });
});

describe("getInflightAudit", () => {
  // SiftAuditDialog 的 POST 在 useEffect 里,dev 下 remount(HMR / Fast Refresh 回退全页刷新)
  // 会重跑 effect → 不去重就再发一次不可中止的 POST(服务端 execFileSync 单跑可达 30 分钟)→
  // 堆叠抢 LLM 端点限流(实测 opdef-developer:两轮重叠,先起的被杀)。去重:同 key 在飞就复用。

  it("首次调用 → 执行 fetcher、缓存 promise", async () => {
    const cache = new Map<string, Promise<string>>();
    let calls = 0;
    const p = getInflightAudit("k", () => { calls++; return Promise.resolve("R"); }, cache);
    expect(calls).toBe(1);
    expect(cache.has("k")).toBe(true);
    await expect(p).resolves.toBe("R");
  });

  it("同 key 在飞 → 复用同一 promise,不再调 fetcher(去重核心)", async () => {
    const cache = new Map<string, Promise<string>>();
    let calls = 0;
    let resolve1!: (v: string) => void;
    const p1 = getInflightAudit("k", () => { calls++; return new Promise<string>(r => { resolve1 = r; }); }, cache);
    const p2 = getInflightAudit("k", () => { calls++; return Promise.resolve("OTHER"); }, cache);
    expect(calls).toBe(1);          // 第二次没调 fetcher
    expect(p2).toBe(p1);            // 复用同一个 promise
    resolve1("R");
    await expect(p1).resolves.toBe("R");
  });

  it("settle 后清缓存 → 下次 open 重新调 fetcher(不复用旧的)", async () => {
    const cache = new Map<string, Promise<string>>();
    let calls = 0;
    await getInflightAudit("k", () => { calls++; return Promise.resolve("R"); }, cache);
    expect(cache.has("k")).toBe(false);     // resolve 后清了
    await getInflightAudit("k", () => { calls++; return Promise.resolve("R2"); }, cache);
    expect(calls).toBe(2);                  // 第二次重新调了
  });

  it("不同 key → 独立(各自调 fetcher,不互斥)", async () => {
    const cache = new Map<string, Promise<number>>();
    let calls = 0;
    await getInflightAudit("a", () => { calls++; return Promise.resolve(1); }, cache);
    await getInflightAudit("b", () => { calls++; return Promise.resolve(2); }, cache);
    expect(calls).toBe(2);
  });

  it("reject 也 settle(清缓存 + 复用同一拒绝,失败不重发)", async () => {
    const cache = new Map<string, Promise<string>>();
    let calls = 0;
    const p1 = getInflightAudit("k", () => { calls++; return Promise.reject(new Error("boom")); }, cache);
    const p2 = getInflightAudit("k", () => { calls++; return Promise.resolve("X"); }, cache);
    expect(calls).toBe(1);
    await expect(p1).rejects.toThrow("boom");
    await expect(p2).rejects.toThrow("boom");   // 复用同一拒绝,没重发
    expect(cache.has("k")).toBe(false);
  });
});

describe("buildStructuredRecords", () => {
  // 最小 prisma mock:turn.findMany + toolCall.findMany(忽略 where,按数组序返回)。
  function makeStructuredPrisma(opts: {
    turns: Array<{
      id: string; turnIndex: number; role: string; content: string | null; agentName: string | null;
      isSubagent?: boolean; subagentSessionId?: string | null; parentExecutionId?: string | null;
    }>;
    toolCalls: Array<{ turnId: string; toolName: string; argsJson: string | null; resultJson: string | null }>;
  }) {
    return {
      turn: { findMany: async () => opts.turns },
      toolCall: { findMany: async () => opts.toolCalls },
    } as unknown as PrismaClient;
  }

  it("Turn+ToolCall → {format, records}:按 turnIndex 排序,tool_calls 映射 name/input;不带 result(audit 不读工具结果)", async () => {
    const prisma = makeStructuredPrisma({
      turns: [
        { id: "t1", turnIndex: 1, role: "user", content: "分析崩溃", agentName: null },
        { id: "t2", turnIndex: 2, role: "assistant", content: "调用 skill 排查", agentName: "kernel-developer" },
      ],
      toolCalls: [
        { turnId: "t2", toolName: "skill", argsJson: '{"name":"ascendc-crash-debug"}', resultJson: '<skill_content name="x">B</skill_content>' },
        { turnId: "t2", toolName: "read", argsJson: '{"file_path":"x.py"}', resultJson: "content" },
      ],
    });
    const out = await buildStructuredRecords("ses_1", prisma);
    expect(out.format).toBe("sift-records");
    expect(out.records).toHaveLength(2);
    expect(out.records[0]).toMatchObject({ role: "user", text: "分析崩溃", parent_tool_use_id: null });
    const asst = out.records[1] as Record<string, unknown>;
    expect(asst).toMatchObject({ role: "assistant", text: "调用 skill 排查", agent: "kernel-developer" });
    const tcs = asst.tool_calls as Array<Record<string, unknown>>;
    // result(resultJson)恒不带:audit 判定只读 user_turns + tool 输入(input)+ assistant 文本,
    // 工具结果是死重量。即便 resultJson 非空也不回传(见下条 payload 守卫测试)。
    expect(tcs[0]).toEqual({ name: "skill", input: { name: "ascendc-crash-debug" } });
    expect(tcs[1]).toEqual({ name: "read", input: { file_path: "x.py" } });
    expect("result" in tcs[0]).toBe(false);
    expect("result" in tcs[1]).toBe(false);
  });

  it("resultJson 巨大也不进 payload(audit 不读工具结果 → 不回传 result,免 10MB+ 撑爆结构化输入)", async () => {
    // 真实 session 的 resultJson 常达 10MB+(Read 整文件 / 大输出),而 audit 判定只读
    // user_turns + tool 输入 + assistant 文本——工具结果对 verdict 零贡献。回传它会让
    // --transcript 的 session.json 撑到 20MB+,序列化/落盘/解析全慢。守卫:再大也不进 payload。
    const prisma = makeStructuredPrisma({
      turns: [{ id: "t1", turnIndex: 0, role: "assistant", content: null, agentName: null }],
      toolCalls: [
        { turnId: "t1", toolName: "read", argsJson: '{"file_path":"big.py"}', resultJson: "X".repeat(2_000_000) },
      ],
    });
    const out = await buildStructuredRecords("ses_1", prisma);
    const tc = ((out.records[0] as Record<string, unknown>).tool_calls as Record<string, unknown>[])[0];
    expect(tc).toEqual({ name: "read", input: { file_path: "big.py" } });
    expect("result" in tc).toBe(false);
    expect(JSON.stringify(out).length).toBeLessThan(100_000);   // 2MB resultJson 没泄进 payload
  });

  it("无 turn → records 空(仍带 format 标记)", async () => {
    const prisma = makeStructuredPrisma({ turns: [], toolCalls: [] });
    const out = await buildStructuredRecords("ses_x", prisma);
    expect(out.format).toBe("sift-records");
    expect(out.records).toEqual([]);
  });

  it("argsJson 坏 / 空 → input 为 {}", async () => {
    const prisma = makeStructuredPrisma({
      turns: [{ id: "t1", turnIndex: 0, role: "assistant", content: null, agentName: null }],
      toolCalls: [
        { turnId: "t1", toolName: "bash", argsJson: "not-json", resultJson: null },
        { turnId: "t1", toolName: "write", argsJson: null, resultJson: null },
      ],
    });
    const out = await buildStructuredRecords("ses_1", prisma);
    const tcs = (out.records[0] as Record<string, unknown>).tool_calls as Array<Record<string, unknown>>;
    expect(tcs[0].input).toEqual({});
    expect(tcs[1].input).toEqual({});
    // result 恒不带(audit 不读工具结果)→ 无 result 键
    expect("result" in tcs[0]).toBe(false);
    // content 为 null 的 turn → 不带 text 键
    expect("text" in (out.records[0] as Record<string, unknown>)).toBe(false);
  });

  it("子 agent turn 的 parent_tool_use_id = subagentSessionId;主 turn 仍 null(scope-aware 切段)", async () => {
    // sift 切片器(slice.py _scope_of)按 parent_tool_use_id 归作用域:None=主,其他值=
    // 某子 agent。之前 buildStructuredRecords 恒 null → 子 agent 里跑的 skill 段被主 scope 的
    // last-invoked-owns 污染(主 scope 噪声混进段)。修:子 agent turn 指到自己的 subagentSessionId。
    // 零嵌套数据下即正确(无需合成 Agent tool_use;见 slice.py 的 scope 归段 + depth-1 closure)。
    const prisma = makeStructuredPrisma({
      turns: [
        { id: "t1", turnIndex: 0, role: "user", content: "main turn", agentName: null },
        { id: "t2", turnIndex: 1, role: "assistant", content: "subagent turn", agentName: "developer",
          isSubagent: true, subagentSessionId: "sub-ses-1" },
      ],
      toolCalls: [],
    });
    const out = await buildStructuredRecords("ses_1", prisma);
    expect((out.records[0] as Record<string, unknown>).parent_tool_use_id).toBeNull();
    expect((out.records[1] as Record<string, unknown>).parent_tool_use_id).toBe("sub-ses-1");
  });

  it("不同 subagentSessionId → 不同 scope(同 agent 两次独立 dispatch 不被合并)", async () => {
    // agentName 不能当 scope key:同 agent 多次 dispatch 会合并成一段。subagentSessionId 每次
    // dispatch 唯一,正确区分独立 run。
    const prisma = makeStructuredPrisma({
      turns: [
        { id: "a1", turnIndex: 0, role: "assistant", content: null, agentName: "dev",
          isSubagent: true, subagentSessionId: "sub-1" },
        { id: "a2", turnIndex: 1, role: "assistant", content: null, agentName: "dev",
          isSubagent: true, subagentSessionId: "sub-1" },
        { id: "b1", turnIndex: 2, role: "assistant", content: null, agentName: "dev",
          isSubagent: true, subagentSessionId: "sub-2" },
      ],
      toolCalls: [],
    });
    const out = await buildStructuredRecords("ses_1", prisma);
    const ptids = out.records.map((r) => (r as Record<string, unknown>).parent_tool_use_id);
    expect(ptids).toEqual(["sub-1", "sub-1", "sub-2"]);
  });

  it("子 agent turn 无 subagentSessionId → 回退 parentExecutionId(仍隔离出主 scope)", async () => {
    // 数据质量缺口:isSubagent=true 但 subagentSessionId 缺。用 parentExecutionId(每次 dispatch
    // 唯一)兜底,仍把该 run 隔离出主 scope,不退化回 flat。
    const prisma = makeStructuredPrisma({
      turns: [
        { id: "t1", turnIndex: 0, role: "assistant", content: null, agentName: "dev",
          isSubagent: true, subagentSessionId: null, parentExecutionId: "exec-1" },
      ],
      toolCalls: [],
    });
    const out = await buildStructuredRecords("ses_1", prisma);
    expect((out.records[0] as Record<string, unknown>).parent_tool_use_id).toBe("exec-1");
  });
});
