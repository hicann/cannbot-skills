# deadlock-triage — 同步死锁/死等分诊工作流

> 程序卡死/hang，怀疑同步导致。优先定位「永久阻塞」类问题。

## 分诊决策树

```
程序卡死/hang
  │
  ├─ 含 CrossCoreWaitFlag？
  │   ├─ 对应 SetFlag 是否所有路径都触发？→ SYNC-08（提前 return 跳过 SetFlag）
  │   ├─ Set/Wait 是否配对？→ SYNC-03 / SYNC-04
  │   ├─ 单核场景是否残留 CrossCore 同步？→ SYNC-03（该取消未取消 → WaitFlag 永久等待）
  │   └─ 是否与 Matmul 高阶 API 混用 flagId？→ SYNC-07（flagId 冲突）
  │
  ├─ 含 WaitFlag<HardEvent>？
  │   ├─ 是否 Wait 先于 Set？→ SYNC-01（死等）
  │   ├─ 同 EVENT_ID 是否有 Set？→ SYNC-04（无人 Set → 死等）
  │   └─ HardEvent 方向是否与数据流匹配？→ SYNC-05
  │
  ├─ 含 SyncAll？
  │   └─ 是否所有核都到达 SyncAll？分支不对称 → SYNC-12（部分核死等）
  │
  └─ 无显式同步但偶发错误？
      └─ 数据依赖缺同步 → SYNC-02（读到半成品，非卡死但结果错乱）
```

## 验证手段

1. 运行 `python3 scripts/sync_audit.py <目标文件+关联文件> --check all --format json`（决策树中 SYNC-05/SYNC-12 分支仅在 `--check all` 下执行，`--check pair` 不覆盖）
2. 若脚本无命中，按 [ascendc-crash-debug](skill:ascendc-crash-debug) 的 plog 解析定位卡死核
3. 临时加 `PipeBarrier<PIPE_ALL>()` 二分定位：若加在某处后结果正确，说明该处缺同步（SYNC-02）

> 决策树用于按症状定位条例；判定以脚本候选为准，LLM 不得否决脚本候选（见 [SKILL.md 执行规则 6](../SKILL.md)）。

## 输出
脚本输出的候选即为结果，编号展示修改方案（文件名+行号+diff），等用户选择后执行。
