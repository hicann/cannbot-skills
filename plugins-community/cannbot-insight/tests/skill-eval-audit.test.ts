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
  recoverMainAgentWorkflowBody,
  buildAuditArgs,
  buildStructuredRecords,
  buildTranscriptArgs,
  auditKindsForEvents,
  getInflightAudit,
} from "@/lib/skill-eval-audit";

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
  it("拼出 skill-eval audit 的参数数组：位置参 skill 目录 + --db/--session/-o + --kind skill", () => {
    // --kind skill 必带（--db 路是 per-skill 对账）：不带会拿本 skill 的 SKILL.md 去对账整条
    // session（含别的 skill 跑的 turn）。skill-eval 靠 SKILL.md 的 name + --kind skill 切本 skill 的段。
    expect(
      buildAuditArgs({
        skillDir: "/tmp/x/myskill",
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
    ]);
  });
});

describe("buildTranscriptArgs", () => {
  it("kind=skill：拼出 --transcript 参数 + --kind skill", () => {
    expect(
      buildTranscriptArgs({
        skillDir: "/tmp/x/myskill",
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
    ]);
  });

  it("kind=agent：--kind agent（审被 dispatch 的 agent，按 agent 归属切）", () => {
    expect(
      buildTranscriptArgs({
        skillDir: "/tmp/x/developer",
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

describe("getInflightAudit", () => {
  // SkillEvalAuditDialog 的 POST 在 useEffect 里,dev 下 remount(HMR / Fast Refresh 回退全页刷新)
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
    expect(out.format).toBe("skill-eval-records");
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
    expect(out.format).toBe("skill-eval-records");
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
    // skill-eval 切片器(slice.py _scope_of)按 parent_tool_use_id 归作用域:None=主,其他值=
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
