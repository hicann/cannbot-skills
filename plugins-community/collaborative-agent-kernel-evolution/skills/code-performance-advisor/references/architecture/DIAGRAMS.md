# Code Performance Advisor - Architecture Diagrams

**Version**: 2.0
**Date**: 2026-02-25
**Related**: ARCHITECTURE.md

---

## 1. System Context Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User (Developer)                         │
│                                                                  │
│  Goals:                                                          │
│  - Optimize AscendC operator performance                        │
│  - Understand bottlenecks                                       │
│  - Learn optimization patterns                                  │
└─────────────────┬────────────────────────────────────────────────┘
                  │
                  │ CLI Commands
                  │ (advisor optimize, analyze, suggest...)
                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Code Performance Advisor                        │
│                                                                  │
│  Inputs:                                                         │
│  - AscendC source code (.cpp)                                   │
│  - Profiling data (CSV from msprof)                             │
│  - Performance goals (goal.md)                                  │
│                                                                  │
│  Outputs:                                                        │
│  - Optimization suggestions (ranked)                            │
│  - Modified code (validated)                                    │
│  - Performance reports                                          │
│  - New rules (knowledge capture)                                │
└─────┬───────────────────────────┬───────────────────────────────┘
      │                           │
      │ Compile & Install         │ Query/Update
      ↓                           ↓
┌──────────────────┐    ┌──────────────────────┐
│  CANN Toolchain  │    │   Rule Library       │
│  (External)      │    │   Case Library       │
│  - Compiler      │    │   (Knowledge Base)   │
│  - Runtime       │    │                      │
└──────────────────┘    └──────────────────────┘
```

---

## 2. Layered Architecture (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│                      L1: Interface Layer                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │  CLI         │   │  Python API  │   │  Web UI      │       │
│  │  (Current)   │   │  (Future)    │   │  (Future)    │       │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘       │
│         │                  │                  │                │
│         └──────────────────┴──────────────────┘                │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                      L2: Orchestration Layer                     │
├────────────────────────────┴────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────┐     │
│  │             PerformanceAdvisor (Main)                 │     │
│  │  - Workflow coordination                              │     │
│  │  - Mode management (interactive/auto/suggest-only)    │     │
│  └───┬──────────────────┬──────────────────┬────────────┘     │
│      │                  │                  │                   │
│  ┌───▼────────┐    ┌───▼────────┐    ┌───▼──────────┐        │
│  │ PhaseRouter│    │IterManager │    │StateManager  │        │
│  │            │    │            │    │              │        │
│  │ - Decide   │    │ - Control  │    │ - Persist    │        │
│  │   path     │    │   loop     │    │   state      │        │
│  │ - Routing  │    │ - Track    │    │ - Snapshot   │        │
│  │   logic    │    │   attempts │    │   baseline   │        │
│  └────────────┘    └────────────┘    └──────────────┘        │
│                                                                  │
└─────────────────────┬────────────────────────────────────────────┘
                      │
┌─────────────────────┼────────────────────────────────────────────┐
│                      L3: Analysis Layer                          │
├─────────────────────┴────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ RuleMatcher  │  │ Evidence     │  │ Suggestion   │         │
│  │              │  │ Extractor    │  │ Generator    │         │
│  │ - Tag-based  │  │              │  │              │         │
│  │   scoring    │  │ - Parse CSV  │  │ - Template   │         │
│  │ - Weighted   │  │ - Extract    │  │   rendering  │         │
│  │   Jaccard    │  │   metrics    │  │ - Code       │         │
│  │              │  │ - Identify   │  │   analysis   │         │
│  │              │  │   bottleneck │  │              │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                  │
│                           │                                     │
│                  ┌────────▼────────┐                           │
│                  │ Pattern         │   (LLM-powered)           │
│                  │ Recognizer      │   (Deep Path only)        │
│                  │                 │                            │
│                  │ - Novel pattern │                            │
│                  │   detection     │                            │
│                  └─────────────────┘                            │
│                                                                  │
└─────────────────────┬────────────────────────────────────────────┘
                      │
┌─────────────────────┼────────────────────────────────────────────┐
│                      L4: Execution Layer                         │
├─────────────────────┴────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Code         │  │ Operator     │  │ Optimization │         │
│  │ Transformer  │  │ Builder      │  │ Validator    │         │
│  │              │  │              │  │              │         │
│  │ - Apply      │  │ - Compile    │  │ - Correctness│         │
│  │   patches    │  │ - Install    │  │   check      │         │
│  │ - Syntax     │  │ - Error      │  │ - Performance│         │
│  │   validation │  │   handling   │  │   compare    │         │
│  │              │  │              │  │ - Goal check │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                  │
│                           │                                     │
└───────────────────────────┼─────────────────────────────────────┘
                            │
┌───────────────────────────┼─────────────────────────────────────┐
│                      L5: Knowledge Layer                         │
├───────────────────────────┴─────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Rule Library │  │ Case Library │  │ Performance  │         │
│  │              │  │              │  │ Model        │         │
│  │ - CRUD ops   │  │ - Store      │  │              │         │
│  │ - Indexing   │  │   validated  │  │ - Predict    │         │
│  │ - Metadata   │  │   examples   │  │   impact     │         │
│  │   tracking   │  │ - Regression │  │ - Learn from │         │
│  │              │  │   tests      │  │   history    │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  Storage: assets/rules/, assets/cases/                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Diagram (End-to-End Optimization)

```
User Input                  System Processing                     Output
────────                    ─────────────────                     ──────

┌──────────┐
│ Operator │
│ Code     │ ───┐
└──────────┘    │
                │
┌──────────┐    │
│ Profiling│    ├─→ [1. Data Loader] ─→ OperatorContext
│ CSV      │    │          │
└──────────┘    │          │
                │          ↓
┌──────────┐    │   [2. Tag Extractor] ─→ OperatorTags
│ Goal     │ ───┘          │
│ Config   │               │
└──────────┘               ↓
                    [3. Rule Matcher] ─→ RuleMatch[]
                           │                    │
                           │                    │
                           ↓                    │
                    [4. Router Decision]        │
                           │                    │
                  ┌────────┼────────┐          │
                  │        │        │          │
              Fast Path  Moderate Deep         │
                  │        │        │          │
                  ↓        ↓        ↓          │
              Rule      Deep    Pattern   ←────┘
              Based    Analysis  Recog.
                  │        │        │
                  └────────┴────────┘
                           │
                           ↓
                  [5. Suggestion Generator] ─→ Suggestion[]  ─┐
                           │                                  │
                           ↓                                  │
                  [User Approval?]                            │
                           │                                  │
                        Yes│                                  │
                           ↓                                  │
                  [6. Code Transformer] ─→ Modified Code      │
                           │                                  │
                           ↓                                  │
                  [7. Operator Builder] ─→ Build Result       │
                           │                                  │
                           ↓                                  │
                  [8. Profiler (Re-run)] ─→ New CSV           │
                           │                                  │
                           ↓                                  │
                  [9. Validator]                              │
                           │                                  │
                      ┌────┴────┐                            │
                  Improved?  No │                             │
                      │Yes       │                             │
                      ↓          ↓                             │
              [10. Capture]  [Retry]                          │
                      │          │                             │
                      ↓          └─────────────────────────────┘
              Rule Library
              Update
                      │
                      ↓
              ┌──────────────┐
              │ Final Report │ ────→ [User]
              └──────────────┘
```

---

## 4. State Transition Diagram

```
                   [IDLE]
                     │
                     │ advisor optimize <op>
                     ↓
              [INITIALIZING]
                     │
                     │ Load context, extract tags
                     ↓
              [ROUTING_DECISION]
                     │
                     │ Score rules, decide path
                     ↓
              [GENERATING_SUGGESTIONS]
                     │
                     │ Create suggestion objects
                     ↓
              [AWAITING_USER_APPROVAL]  (if interactive mode)
                     │
                     │ User confirms
                     ↓
              [APPLYING_TRANSFORMATION]
                     │
                     │ Modify code
                     ↓
              [BUILDING]
                     │
                     │ Compile + install
                     ↓
              [PROFILING]
                     │
                     │ Run performance test
                     ↓
              [VALIDATING]
                     │
            ┌────────┴────────┐
            │                 │
         Success          Failure
            │                 │
            ↓                 ↓
     [CAPTURING]      [RETRYING]
            │                 │
            │                 │ iteration++
            │                 │
            │                 └──→ [GENERATING_SUGGESTIONS]
            │                      (or COMPLETE if max reached)
            ↓
     [COMPLETE]
            │
            ↓
     Generate report
```

---

## 5. Module Dependency Graph

```
┌──────────────────────────────────────────────────────────────────┐
│                         advisor.py                                │
│                      (CLI Entry Point)                            │
└────────────────┬─────────────────────────────────────────────────┘
                 │
                 ↓
        ┌────────────────────┐
        │ cli.commands.*     │
        │ - analyze          │
        │ - suggest          │
        │ - optimize         │
        └────────┬───────────┘
                 │
                 ↓
    ┌────────────────────────────┐
    │ core.orchestration.advisor │  ←─┐
    └────────┬───────────────────┘    │
             │                         │
    ┌────────┴────────┐               │
    │                 │               │
    ↓                 ↓               │
┌─────────┐    ┌──────────────┐      │
│ router  │    │ state_manager│      │
└────┬────┘    └──────┬───────┘      │
     │                │               │
     │                │               │
     ↓                ↓               │
┌────────────────────────────┐       │
│ core.analysis.*            │       │
│  ┌──────────────┐          │       │
│  │ rule_matcher │──────────┼───────┘ (circular, via interfaces)
│  └──────┬───────┘          │
│         │                  │
│  ┌──────▼────────┐         │
│  │ suggestion_gen│         │
│  └───────────────┘         │
└────────┬───────────────────┘
         │
         ↓
┌────────────────────────────┐
│ core.execution.*           │
│  ┌──────────────┐          │
│  │ transformer  │          │
│  ├──────────────┤          │
│  │ builder      │          │
│  ├──────────────┤          │
│  │ validator    │          │
│  └──────────────┘          │
└────────┬───────────────────┘
         │
         ↓
┌────────────────────────────┐
│ core.knowledge.*           │
│  ┌──────────────┐          │
│  │ rule_library │          │
│  ├──────────────┤          │
│  │ case_library │          │
│  └──────────────┘          │
└────────────────────────────┘

All modules depend on:
└─→ core.common.* (data_models, config, utils)
```

---

## 6. Sequence Diagram: Fast Path Optimization

```
User    CLI    Advisor   Router   RuleMatcher   SuggestionGen   Executor   Validator   Knowledge

 │       │        │        │            │              │            │           │           │
 ├──────>│optimize                                                                         │
 │       │        │                                                                         │
 │       ├───────>│init()                                                                   │
 │       │        │                                                                         │
 │       │        ├───────>│decide()                                                        │
 │       │        │        │                                                                │
 │       │        │        ├───────────>│match()                                            │
 │       │        │        │            │                                                   │
 │       │        │        │<───────────┤ RuleMatch[]                                       │
 │       │        │        │            │                                                   │
 │       │        │<───────┤ RoutingDecision                                                │
 │       │        │        │  (Fast Path)                                                   │
 │       │        │        │                                                                │
 │       │        ├────────┼────────────┼─────────────>│generate()                         │
 │       │        │        │            │              │                                    │
 │       │        │<───────┼────────────┼──────────────┤ Suggestion                        │
 │       │        │        │            │              │                                    │
 │       │<───────┤ Display suggestion                                                      │
 │       │        │                                                                         │
 │<──────┤ "Apply? [Y/n]"                                                                   │
 │       │                                                                                  │
 ├──────>│ Y                                                                                │
 │       │                                                                                  │
 │       ├───────>│apply()                                                                  │
 │       │        │                                                                         │
 │       │        ├────────┼────────────┼──────────────┼───────────>│transform()           │
 │       │        │        │            │              │            │                      │
 │       │        │        │            │              │            ├──────────>│build()   │
 │       │        │        │            │              │            │           │          │
 │       │        │        │            │              │            │           ├────────>│profile() │
 │       │        │        │            │              │            │           │         │          │
 │       │        │        │            │              │            │<──────────┤validate()│          │
 │       │        │<───────┼────────────┼──────────────┼────────────┤Result    │          │          │
 │       │        │        │            │              │            │          │          │          │
 │       │        ├────────┼────────────┼──────────────┼────────────┼──────────┼──────────┼────────>│
 │       │        │        │            │              │            │          │          │  capture()
 │       │        │        │            │              │            │          │          │          │
 │       │<───────┤ Report                                                                             │
 │       │                                                                                              │
 │<──────┤ "Optimized! 44% faster"                                                                     │
```

---

## 7. Component Interaction Matrix

| Component | Reads From | Writes To | Calls | Called By |
|-----------|-----------|-----------|-------|-----------|
| CLI | User input | Terminal | Advisor | User |
| Advisor | State | State, Report | Router, Analysis, Executor | CLI |
| Router | ScoredResults | - | - | Advisor |
| RuleMatcher | RuleLibrary, Tags | - | - | Advisor, Router |
| SuggestionGen | Rule docs, Code, Profiling | Suggestion files | RuleMatcher, Evidence | Advisor |
| Evidence | Profiling CSV | - | - | SuggestionGen |
| Executor | Code, Suggestion | Modified code | Builder, Validator | Advisor |
| Builder | Code | Binary, Logs | CANN tools | Executor |
| Validator | Profiling (before/after) | ValidationResult | - | Executor |
| Knowledge | - | Rule/Case library | - | Advisor |

---

## 8. Deployment Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    Developer Machine                          │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Code Performance Advisor (Python 3.10+)               │   │
│  │                                                        │   │
│  │  - Core modules (pure Python)                         │   │
│  │  - CLI (argparse)                                     │   │
│  │  - Configuration (YAML)                               │   │
│  └───┬──────────────────────────────────────────────────┘   │
│      │                                                       │
│      │ Depends on                                            │
│      ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Python Dependencies                                   │   │
│  │  - PyYAML                                             │   │
│  │  - Jinja2                                             │   │
│  │  - (optional) Pydantic                                │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ CANN Toolchain (External)                            │   │
│  │  - Compiler                                           │   │
│  │  - Runtime                                            │   │
│  │  - msprof (profiling)                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ File System                                           │   │
│  │                                                        │   │
│  │  workspace/                                           │   │
│  │  ├── operators/{op}/                                  │   │
│  │  │   ├── input/                                       │   │
│  │  │   ├── baseline/                                    │   │
│  │  │   ├── iterations/                                  │   │
│  │  │   └── output/                                      │   │
│  │  │                                                     │   │
│  │  assets/                                              │   │
│  │  ├── rules/                                           │   │
│  │  └── cases/                                           │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘

Optional (Future):
┌───────────────────────────────────────────────────────────────┐
│                    Remote Server                              │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Advisor API Service                                   │   │
│  │  - REST API                                           │   │
│  │  - Shared rule library                                │   │
│  │  - Performance model training                         │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

---

## 9. Error Handling Flow

```
                        [Operation Start]
                              │
                              ↓
                    ┌──────────────────┐
                    │ Try Operation    │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                 Success          Exception
                    │                 │
                    ↓                 ↓
            [Return Result]   [Error Handler]
                    │                 │
                    │        ┌────────┴────────┐
                    │        │                 │
                    │   Known Error      Unknown Error
                    │        │                 │
                    │        ↓                 ↓
                    │  [Log + Format]    [Log + Traceback]
                    │        │                 │
                    │        ↓                 ↓
                    │  [User Message]    [Debug Info]
                    │        │                 │
                    │        ├─────────────────┘
                    │        │
                    │        ↓
                    │  [Recovery Suggestion]
                    │        │
                    │        ├→ "Check file permissions"
                    │        ├→ "Run: advisor config validate"
                    │        └→ "See logs: workspace/advisor.log"
                    │        │
                    └────────┴──→ [Exit with code 0/1]

Error Categories:
1. User Input Error (exit 2) - Wrong arguments
2. Data Error (exit 3) - Missing/corrupt files
3. Build Error (exit 4) - Compilation failure
4. Validation Error (exit 5) - Performance regression
5. System Error (exit 1) - Unexpected failure
```

---

## 10. Performance Optimization Strategy (Meta)

The system itself follows optimization principles:

```
┌──────────────────────────────────────────────────────────────┐
│ Performance Budget                                            │
├──────────────────────────────────────────────────────────────┤
│ Phase 0 (Analysis):        < 5 seconds                       │
│ Phase 1-3 (Suggestions):   < 10 seconds (Fast Path)          │
│ Phase 4 (Execution):       < 60 seconds (build + profile)    │
│ Total (single iteration):  < 90 seconds                      │
└──────────────────────────────────────────────────────────────┘

Optimization Techniques:
1. Rule index caching (avoid re-parsing rules)
2. Lazy loading (load modules on-demand)
3. Parallel profiling (if multiple operators)
4. Incremental builds (reuse unchanged components)
5. Memoization (cache expensive computations)
```

---

**END OF DIAGRAMS DOCUMENT**

This document complements ARCHITECTURE.md with visual representations of the system design.
