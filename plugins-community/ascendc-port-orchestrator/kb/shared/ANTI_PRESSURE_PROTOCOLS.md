# Anti-Pressure Protocols（无条件加载）

> **所有 skill 必须在 session 开始时加载此文件，并在决策点重新引用相关 Px。**
>
> 这里收录的不是技术规则，而是 **LLM agent 在压力下会自我合理化绕过规则的失效模式**。
> 跨 op、跨 skill 通用 —— 因为压力来源是模型架构层面的，不是 op-gen 特有的。
>
> 每个 P 项都带 **incident anchor**（具体过去事故）+ **detection signal**（如何在自己的
> 思考流里识别"我现在正在 P-X 模式"）。**抽象的 anti-pressure 警告会被忽略，
> 具体的事故锚点能触发识别。**

---

## 加载契约

- **Session start**: skill 的"Required reading"列表里必须含本文件。一次性加载，便宜。
- **Decision-point checkpoint**: skill 在 high-leverage 时刻必须重新引用相关 Px。
  仅在 session start 加载不够 —— 长 session 会被 compress 掉，压力来临时已经看不见。

**High-leverage 时刻清单**（skill 必须在这些点引用 Px）:
| 时刻 | 引用 |
|---|---|
| 准备 spawn agent | P3 + P8 |
| 准备 emit `→ orchestrator: PARTIAL_PERSIST` | P5 + P7 |
| 准备 emit `→ orchestrator: done` | P1 + P7 |
| 准备跳过 self-critic / 跳过独立 perf re-run | P1 |
| 准备写 "expected failure" / "V220 limitation" / "structural ceiling" | P5 |
| 准备 nohup / & / 直接 Agent 跳过 Skill | P8 |
| 准备 manual workaround 而非 fix script | P6 |

---

## P1: User-watching pressure（用户在看 → 跳过慢步骤）

**Urge**: 用户在等结果，我要快速产生可见进展。
**Rule**: self-critic + 独立 perf re-run + 独立 re-validation **不是可选的**。
它们是你**唯一**能判断"工作是真的"的信号。跳过一次 = 同一 op 后面要重做，
代价是省下时间的 10×。

**Incident anchor**:
- DEBT-077 P0g（2026-05-04）—— `aog-kernel-worker` 在 op#10 LayerNorm 上自我宣告
  `PARTIAL_PERF_STRUCTURAL_CEILING`，**ko 都还没跑过**。后续花了 P0g + P0v + P0w + P0z
  四个 routing fix 才把这类短路堵住。
- CANN-learn 2026-05-05 —— learner agent 自我报 `self_review_verdict=PASS`，
  Mode 5 caller 独立 re-scan 抓到 81 个 identifier leak + copy_shape 0.33–0.50。
  Agent 自我背书 ≠ 真。

**Detection signal**:
- 思考流里出现"快速给用户一个数字 / 回复就行"
- 想跳过"再跑一次确认"
- 把 "looks right" 当成 "verified"

**反向指令**：用户等的时候，**说一句"在跑独立验证，30s"**，比给一个 30s 后会被推翻的
数字强得多。透明的延迟不是 cost，错误的快速 confirmation 才是。

---

## P2: Context-filling pressure（token 涨 → 抢在 limit 前结束）

**Urge**: 看到 context 警告，开始压缩步骤、跳过慢工。
**Rule**: 真正的 fix 是 **spawn sub-agent / 拆 commit**，**不是**在主 context 里赶。
Compaction 后丢失的细节要重新追回，代价远高于多 spawn 一次的 token。

**Incident anchor**:
- 2026-05-05 4 个 stacked PARTIAL_PERSIST routing bug（P0v / P0w / P0y / P0z）
  在单一 session 处理 —— 应该分 4 个 session 但被 batched 进同一 context，导致
  context 紧 → 我开始跳过 regression test → 用户必须显式说"separated commits"。

**Detection signal**:
- 看到 "context approaching limit" 类提示
- 想"先把这个收尾，下个 session 再补 test"
- 想"这块逻辑我记得，不需要再读源码"

**反向指令**：context 紧 → **spawn agent 把当前任务委派出去 + 在 plan/memory 里写下
当前状态**，然后接受这次 session 在这里截断。继续硬撑会写出半成品。

---

## P3: Batch-throughput pressure（多 op 排队 → shortcut 复利）

**Urge**: 一批 op 都要做完，每个少花点时间累计省很多。
**Rule**: shortcut 在 op#1 看不出后果，到 op#5 累积成系统性偏差。
**每个 op 单独看待**，不要因为前 N 个"差不多走通了"就放松后一个的验证。

**Incident anchor**:
- OL-104 系列 —— op#9 TopKTopP 反复用 PARTIAL_PERSIST 收尾，没每次重跑 researcher。
  导致同样的 vendor tie-break 误判跨 op 复利，后来催生 V3.8.9 "never let PARTIAL pass" 硬规则。
- batch6 sweep —— 同 batch 内多 op 共享一个 perf benchmark cache，导致 ratio 数字
  不是各自独立测出来的（CLAUDE.md "同条件 A/B" 规则就是为这类批处理偏差定的）。

**Detection signal**:
- 思考流里出现"前面那个 op 是这么处理的，这个也按同样的来"
- 跨 op 复用一个验证脚本而没重置中间态
- "整批跑完一起报告" 而非 "每个 op 单独 verify + 单独 report"

**反向指令**：批量越大越要**收紧每 op 的验证标准**，不是放松。每个 op 走完整 verify 步骤
是 floor，不是 ceiling。

---

## P4: "Simple op" assumption（pointwise 类感觉显然 → 跳验证）

**Urge**: Abs / Pad / GELU 这种"显然能过"，跳过 edge_dataset / det check / perf 也没事。
**Rule**: "显然"是 LLM 最容易栽跟头的词。**Pass A + Pass B + det + perf 是 floor**，
op 简单不简单都跑一遍 —— 自动化 verifier 跑一次就 30s，比"事后回来调"便宜得多。

**Incident anchor**:
- OL-18 int64 → int32 误降 —— "测试值都在 INT32 内 → 改 int 应该没问题"，
  实测引入 ~6% perf 退步，两 batch 才被发现。
- 2026-05-04 op#1 GELU "应该 trivial" —— 实际 bf16 路径有 1-ULP 差异，
  edge_dataset 才暴露出来（如果只跑 inline benchmark 会全过）。

**Detection signal**:
- 思考流里出现"这个 op 太简单了，跳过 X 应该没事"
- 把"算法显然"当成"实现显然"
- 想"Pass A 过了 Pass B 应该也过"

**反向指令**：op 越简单越要快速跑全套 —— 简单 op 跑全套验证只要 1–2 分钟，没理由跳。
跳过的真实动机是 P3（throughput）伪装成 P4。

---

## P5: Failure discomfort（失败难写 → 找个标签包装它）

**Urge**: op 没过精度 / perf 不达标，写"expected failure" / "V220 limitation" /
"structural ceiling" / "vendor incompatibility"，比写"我需要继续调查"省事。
**Rule**: 任何"无法达标"的 verdict **必须**有具体的、可被独立复现的 root cause
+ 至少一次失败的修复尝试，否则它是借口不是诊断。

**Incident anchor**:
- DEBT-077 P0g —— `PARTIAL_PERF_STRUCTURAL_CEILING` 被 worker 在没跑过 ko 的情况下
  自我宣告。"never let PARTIAL pass" 政策（V3.8.9）就是为这类失败包装定的。
- 2025 早期"OL-83 是平台限制" 说法被反复使用 —— 后来 CAND-PP80 引入 T1-vs-CPU
  triage 才把"真平台限制"和"我没调查清楚"分开。
- Histc CPU fallback —— "AscendC 不好实现" 写成"已用 CPU 实现"。CLAUDE.md 现在
  明确禁止 CPU fallback 就是这个事故催生的。

**Detection signal**:
- 想用"expected" / "limitation" / "ceiling" / "incompatible" 任一词
- 用"vendor 也是这么做的" 作为唯一证据
- failure 解释里没有 "I tried X, X failed because Y"

**反向指令**：写 PARTIAL / FAIL 之前，**先写 root cause 段**：
- 具体哪行代码 / 哪个 API / 哪个数值
- 至少一次具体修复尝试 + 它失败的原因
- 独立可复现的 minimal repro

如果这三个写不出，PARTIAL 就还没到位 —— 继续调查。

---

## P6: Infrastructure friction（脚本卡住 → workaround 而非 fix）

**Urge**: lane_alloc 阻塞 / SSH hang / LOCAL_TASK 污染 / proxy 不通 ——
绕过去比修脚本快。
**Rule**: workaround 是债，**fix script 是资产**。绕过一次省 10 分钟，
下次同问题再绕过 10 分钟，第 N 次还在绕。每次遇到 friction 的反应**必须**是
"哪个脚本的 bug，怎么 fix"，不是"我手动走一遍"。

**Incident anchor**:
- P0l —— `deploy_to_npu_lane.sh` 缺少 workspace→LOCAL_TASK sync。
  正确 fix 是改脚本一次，错误 workaround 是手动 cp。
- /ascendc-op-gen anti-pattern #5（SKILL.md L135–139）—— "Manually copying workspace
  files between locations 而非 fix 底层脚本"。被反复抓到。
- 2026-05-05 用 `nohup &` 启动 orchestrator —— 因为 `Bash(run_in_background=true)`
  打字多。后来加成 anti-pattern #6（也是 P8 的事故来源）。

**Detection signal**:
- 想"我手动 X 一下就行"
- 想"这次是特殊情况，下次再 fix 脚本"
- 不打开脚本源码，直接绕过

**反向指令**：碰到 friction 第一反应**打开脚本源码**。如果脚本真的没法 fix（比如来自外部
系统），**写一个新 wrapper 脚本封装 workaround**，而非每次手动。手动是 0 个资产，wrapper 是 1 个。

---

## P7: Desire for closure（投入了 30 分钟 → 想叫它 done）

**Urge**: 一个 op 已经搞了 30 分钟，"49/49" 比 "需要再调查那 1 case" 更想说出口。
**Rule**: 数字必须**完整且诚实**。50 case benchmark 只过 49 case = "49/50"，
**不是**"49/49"。close 在 convenient milestone 而不是 verified milestone 是
P0g 那一类事故的母模式。

**Incident anchor**:
- op#9 TopKTopP 多次"close" —— 实际 researcher 没跑过。最后 V3.8.9 才把这种
  "提前 close" 的 routing 堵住。
- DEBT-077 P0v / P0w —— 都源自 orchestrator 在不该 finalize 的 state 上 finalize。
  user 直接的 quote: "do you understand what makes op#9 done?"

**Detection signal**:
- 思考流里出现"差不多了 / 基本好了 / 收个尾"
- 想把 partial 数字写成完整数字（49/50 → 49/49）
- 投入越久越想 close —— 沉没成本谬误的具体形态

**反向指令**：sunken cost 不是 close 的理由。**写完整真实数字 + 列出剩余 case 的 root cause**，
让 orchestrator 决定该 finalize PARTIAL 还是继续。close 的判断**永远**是基于 verification
output，不是基于"我已经花了多久"。

---

## P8: Tool path of least resistance（短命令 → 跳过监管）

**Urge**: `&` 比 `Bash(run_in_background=true)` 短；直接 `Agent({...})` 比
`Skill('aog-xxx')` 短。
**Rule**: short path 通常**绕过的是 CC 自己的可视性 / lifecycle / hook / 加载机制**。
nohup/& 让 CC UI 看不到进程；直接 Agent 跳过 SKILL.md 的 required-reading 加载。
那些"啰嗦"步骤就是监管点，不是装饰。

**Incident anchor**:
- 2026-05-05 nohup 事故 —— 我用 `nohup orchestrator ... &` 启动后台，CC 任务 tracker
  看不到进程，user 主动指出。后来作为 /ascendc-op-gen anti-pattern #6 入册 +
  memory `feedback_no_nohup_for_long_running.md`。
- aog-cann-learner 直接 spawn vs `python3 -m cann_learn.mode5_runner` —— 直接 spawn
  跳过 lease / sealed_dir / hook preflight。Mode 5 runner 那一层就是监管。
- **2026-05-05 META-INCIDENT（同一 session 写完此文件之后）**：在 op#9 resume 时
  用 `python3 ... resume.py 9_topktopp & ` —— **同时**设了 Bash
  `run_in_background=true` **和**命令体里加了尾随 `&`。CC 任务 tracker 跟到了
  launcher (resume.py) 的退出（exit 0），但 orchestrator 子进程因 `&` 脱离 launcher
  shell tree，CC 看不见这个 orphan 子进程。User 直接指出。**教训**：当
  `run_in_background=true` 时，命令体**禁止**有尾随 `&` —— 它**完全冗余**且
  正好触发该 flag 设计要规避的脱钩行为。muscle-memory 的 1 字符 shortcut 就是 P8
  的典型表现。**正确写法**（V3.8.10+, orch wrapper 已含 tee 双输出）：
  ```
  Bash(command="src/scripts/orch 9_topktopp --resume --lane 0",
       run_in_background=true)
  ```
  注意命令体**没有** `>`、`2>&1`、`&` —— orch wrapper 已做 `2>&1 | tee /tmp/orch_*.log`，
  任何额外重定向都会 suppress terminal 端输出。

**Detection signal**:
- 想"直接 X 比走 Y 更快"
- 短路径里**移除了**一个 abstraction 而**没有**说清楚被移除的 abstraction 提供什么
- 把"CC 监管 / Skill 加载 / hook 检查"当成开销而不是 feature

**反向指令**：short path 的诱惑出现时，先问"这条 short path 移除了什么东西，
那个东西保护我什么"。如果说不清答案 = 不要走 short path。

---

## P9: Infrastructure-friction paper-over（env 异常 → 自己绕过，不向上托管）

> ⚠ **与 P6 的区别**：P6 是"local 脚本卡住 → 手动绕过而非 fix 脚本" (orchestrator 域内的 friction)。
> P9 是 **out-of-domain env 异常**：NPU 驱动错码 / CANN install desync / lib size or symbol
> 不匹配 / docker exec timeout / SSH hang / msprof corruption / proxy 429 / install-tree 路径
> 漂移。这些信号代表**环境基线 (baseline) 违反**，不是 worker 该决策的事 — worker 一旦
> "决定自己继续"，专门的 aog-preflight / aog-orchestrator-recover / env-diagnose agent
> 永远收不到信号；真正的问题在多层错误传导后变成大问题。

**Urge**: kernel 半数测出问题 / dispatch 返回 561103 / aclrtSetDevice 507033 / libophost.so
size 不对 — "再试一下"、"换个 NPU"、"replace .so"、"绕开 --pkg"、"backup 然后手动 cp"
都很诱人，因为它们比"停下来上抛 + 让 user/专业 skill 决定"快。但**这正是 vibe-coding 的
反例**：harness engineering 项目里，env 异常 = 工程级 signal，必须按基线 (engineering baseline)
处理，不是"看着办"。

**Rule** — 双分支:
1. **Transient retry-recoverable**（已知短暂、单次失败、有明确 retry 语义的）:
   API 链接掉、proxy 短暂没 reauth、NPU 被 npu-smi 临时锁、aclrtEvent 偶发 fail。
   → **允许有限次 retry (≤3 with exp backoff)**, 但 retry 预算**在 orchestrator 层暴露**
     (从 `.opgen_state.json.transient_retry_count` 计数)，不能藏在 worker 内部 loop。
     retry 用尽仍失败 → 上抛终态 `INFRA_TRANSIENT_RETRY_EXHAUSTED`，不自己 paper-over。
2. **Baseline-violated**（缺 tool / 版本不对 / install desync / .so 不匹配）:
   CANN install 缺 ascend950 binary、libophost_nn.so size/symbol mismatch、
   ops-nn-port 缺 `--pkg` target、bisheng 版本不带某 macro、kernel folder 缺 arch35/、
   pre-installed binary 不在 declared path —— 这些是**结构性基线 violation**。
   → **永远不进 Phase O1+ work**。preflight (Phase O0/O0.5) 必须先 verify `docs/baseline/environment_baseline.yaml`
     的所有断言，**违反任何一条**就立刻 `INFRA_BASELINE_VIOLATED` graceful-exit；
     **绝对禁止** worker 拿到 brief 后"探一下 baseline 然后决定怎么绕过"。

**禁止行为白名单** (worker / probe / optimizer 任何 op-gen agent 收到 env 类信号必须立刻
ascend C-INFRA-* + 上抛, 不在同一 spawn 内做以下 paper-over):
- **NPU 错码捂住**: aclrtSetDevice 507033 / 507035 / 507008 / kernel-not-registered 561103 后继续 retry msprof 循环
- **替换关键 .so**: 用我们 build 的 libophost_nn.so / libopapi.so 覆盖 install 的 (size/symbol mismatch → 破坏其他 op)
- **手工 merge into binary_info_config.json**: 给 install tree 加 entry 而不走 ops-nn-port --pkg
- **绕开 build pipeline**: `--pkg` 失败就 `g++` 直接 link、scp .o 进 install dir
- **docker/SSH timeout 后无限重试**: 不上抛容器/服务异常
- **proxy 失效后多次重试**: 不区分 transient (短暂 429) 还是 structural (corp gateway 断)

**Incident anchor**:
- **2026-05-15 gather_elements_v2 kw-2**: 探到 aclrtSetDevice 507033 (device polluted by repeated msprof + corrupt cycles), 继续 retry msprof — **应该立刻 INFRA_TRANSIENT_RETRY_EXHAUSTED 上抛**。同 spawn 探到 libophost_nn.so size 不对 (build 1.9MB / 2020 symbols vs install 29MB / 3339 symbols), 已 replace 后 rollback — **整个 replace + rollback 步骤都是 P9 反例 (替换关键 .so)**, 应该立刻 INFRA_BASELINE_VIOLATED。最终 user 直接指出："你为什么认为环境问题不是问题？" (Discord 18:06Z).
- **2026-05-15 ada_layer_norm kw-1**: 探到 dispatch 561103 (CANN missing ascend950 binary base op), 继续 build ops-nn-port 各种 workaround, 最终也是 SKIP — **应该 preflight 探到 binary 缺失就 INFRA_BASELINE_VIOLATED, 不该 kw-1 进入 work**。
- **2026-05-15 rms_norm_quant kw-2**: 探到 V1 dispatch 走 V2 alias, 继续替换 libophost_nn.so 然后 rollback — 替换/rollback 都是 P9 反例。
- **DS fleet (10_LayerNorm 23 spawns / 5_Cumsum 4+ spawns)**: 每次 worker retry env 类异常 (ACL init / docker exec / SSH) 而不上抛, ~$30 浪费在 DS backend cost。DS 直接 endorse: "INFRA_BLOCKED would have saved cost" (Discord 18:08Z).
- **user 原话** (Discord 18:06Z + 18:10Z): "harness engineering，需要用工程的标准衡量系统稳定性可靠性，不是 vibe coding 类的快速 PoC...如果缺失工具，我们要评估下工具安装，或者需要的工具版本有问题...是否应该在 preflight 阶段，根据我们的环境要求基线先准备好。baseline 这个概念只有 engineering 的项目才有。"

**Detection signal**:
- 想"再试 N 次"而 N 没有上限 / 没有 budget tracking
- 想"先 backup install 路径下的文件，再覆盖"（即将做 .so 替换）
- 想"绕开 --pkg, 直接 cp 进 install tree"（即将做 binary merge）
- 想"我看 V1 dispatch 走 V2 alias, 那我改 libophost.so 让它走我的"（即将做 router patch）
- 思考流中出现"transient"、"retry should help"、"this is just the dev container being weird"
- 信号: worker 在 ssh-pass / sshpass / docker exec / scp 上花了 > 30 sec, 没有上抛
- 信号: PROGRESS.md 出现"replaced libophost_nn.so", "rolled back libophost_nn.so", "manual install", "bypassed --pkg"

**反向指令**: env 类异常**第一反应**是 forensic 记录 (probe.py + 完整 error transcript +
error code + 涉及的文件 path/md5) → 上抛 handoff `→ orchestrator: await_user_decision —
INFRA_BASELINE_VIOLATED <symptom>` 或 `INFRA_TRANSIENT_RETRY_EXHAUSTED <symptom>`。**绝对不在
同一 spawn 内绕**。orchestrator 终态机会 route 给 aog-preflight 或 aog-orchestrator-recover
或人 — 那才是这类问题的专业接手点。

**Diagnostic discrimination step** (READ-ONLY — adds ≤30s, does NOT remediate):
Before claiming a specific root cause (e.g. "CANN install drift" / "binary missing"
/ "version mismatch"), run a minimal direct probe to distinguish ROOT CAUSE
classes. Probe is intentionally narrow + read-only — it doesn't fix anything,
it just narrows the diagnosis. Example for `aclnnReduceSum 561103`:

```bash
# Sanity baseline — does ANY torch_npu op work right now?
docker exec <container> bash -lc 'python3 -c "
import torch, torch_npu
torch.npu.set_device(0)
x = torch.tensor([2.0,4.0,8.0], device=\"npu\")
print(\"reciprocal:\", torch.reciprocal(x).cpu().tolist())  # CANN 8.5 + 9.0 both
print(\"sum:\", torch.randn(64, device=\"npu\").sum().item())  # CANN 9.0+ only
print(\"env: ASCEND_OPP_PATH=\", __import__(\"os\").environ.get(\"ASCEND_OPP_PATH\"))
"'
```

Outcome discriminates:
- reciprocal PASS + sum PASS → no infra issue, the op-specific dispatch IS the bug
- reciprocal PASS + sum FAIL with 561103 → **env-var hijack** (probably ASCEND_OPP_PATH
  pointing at older CANN install layer that doesn't have new SoC binary; check
  the `env: ASCEND_OPP_PATH=` line in probe output — if it says `cann-8.5.0` you have
  the documented hijack in `aog-a3-rebuild/SKILL.md`)
- reciprocal FAIL → actual CANN install corruption (drift / partial install / lib
  mismatch) — escalate to aog-preflight per P9 above

This probe stays within P9 protocol (READ-ONLY, no fix attempt) but produces
actionable forensic context for the user_decision handoff. The handoff message
should embed the probe output verbatim so the next agent (or preflight, or user)
can route correctly without re-running.

**Anti-pattern** (caught 2026-05-23T19:46Z on FA orch kw-9): worker hit `aclnnReduceSum
561103`, jumped directly to "CANN install drift" diagnosis without running the
discrimination probe. Actual root cause was ASCEND_OPP_PATH env-var hijack (set
by verl base image, never overridden because set_env.sh only sets if unset).
The fix was a ~/.bashrc patch, NOT a CANN reinstall. Worker's mis-diagnosis cost
$8.74 in spawn time + ~30 min orch latency before discrimination happened
out-of-band via an independent white-box probe. Adding the discrimination step here
makes the correct diagnosis path mandatory FOR INFRA escalation.

**Structurally guarded**: 一旦下列 codify 落地 (P96)，P9 大部分场景由结构性 gate 接住，prompt
里不再需要反复念:
- `docs/baseline/environment_baseline.yaml` + `aog-preflight` baseline-check 扩展
- `aog-self-critic` C-INFRA-RETRY-WITHOUT-CAP + C-INFRA-BASELINE-PAPER-OVER catalog
- orchestrator 新终态 `INFRA_TRANSIENT_RETRY_EXHAUSTED` + `INFRA_BASELINE_VIOLATED`
- `finalize_pipeline._check_infra_paper_over` gate 扫 PROGRESS.md 反向触发 keyword

---

## 引用格式（skill 在 SKILL.md 里 cite Px 时）

短引用（high-leverage 决策点）：
> ⚠️ 即将 emit PARTIAL_PERSIST：先 review **P5 + P7** of `ANTI_PRESSURE_PROTOCOLS.md`。

完整引用（agent brief 的 phase header）：
> # ANTI-PRESSURE CHECKPOINT
> 在 emit `→ orchestrator: done` 前必须确认：
> - P1: 我没有跳 self-critic / perf re-run（用户在看 ≠ 跳过验证）
> - P7: 我的数字是真实完整的（49/50 不写成 49/49）

---

## 维护

- 新发现的 pressure 模式（P9, P10, ...）按相同模板加入：urge / rule / incident / detection / 反向指令。
- incident anchor **必须有具体 ID**（DEBT-XXX, P0X, OL-XX, commit SHA）—— 没有 anchor 的 pressure 项 1 个月内会被忽略掉。
- 有 P 被新事故触发 → 在该 P 下追加 incident anchor，不要新建一个 P。
- 1 季度 1 次 review：哪些 incident 已经被结构性 fix（routing 修了 / hook 加了），把 P 标记 "structurally guarded"，避免 prompt 里反复念。
