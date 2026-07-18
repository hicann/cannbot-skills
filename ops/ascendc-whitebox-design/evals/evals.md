---
skill_name: ascendc-whitebox-design
eval_mode: text
---
# Case 1: 六步工作流概述

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-whitebox-design 技能的完整工作流程。请依次介绍以下 7 个方面（每个方面 1-2 句话即可）：
1. Step 1 输入收集（收集哪些参数、平台检测怎么做）
2. Step 2 源码分析（有哪些 Phase、分别做什么）
3. Step 3 交叉验证（验证什么、失败怎么处理）
4. Step 4 用户确认闸门（为什么需要、能否跳过）
5. Step 5 参数组合映射（low/high 两档位的区别）
6. Step 6 pytest 生成与执行（精度不达标怎么处理）
7. 可选 TTK CSV 模块（什么时候启用、产出什么）
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应覆盖 ascendc-whitebox-design 的完整工作流：
- Step 1：收集算子名称、路径、是否跳过闸门、是否启用 TTK；平台参数由 npu-arch skill 自动检测
- Step 2：Phase 0 文件侦察（Scout-T/Scout-K 并行）→ Phase 1 并行 Task A/B/C → Phase 2 参数推导 → Phase 3 测试设计
- Step 3：派独立子 agent 交叉验证，fail 回 Step 2 重分析（最多 3 轮）
- Step 4：安全闸门，展示摘要等待用户确认，可选择跳过
- Step 5：5a 路径覆盖 → 5b 网络用例 → 5c 合并 → 5d data_range 展开为 low/high 两档位
- Step 6：生成 pytest 脚本、执行测试、精度不达标标记 XFAIL、tilingkey 覆盖率分析
- TTK：可选模块，生成 CSV 文件和 golden_plugin.py

## Expectations

- [contains] Step 1
- [contains] Step 2
- [contains] pytest
- [contains] 闸门

---

# Case 2: Step 2 源码分析流程

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-whitebox-design 技能中 Step 2 源码分析的完整流程。请依次介绍以下 5 个方面（每个方面 1-2 句话即可）：
1. Phase 0 文件侦察（Scout-T 和 Scout-K 分别做什么、Scout-Verify 做什么）
2. Phase 1 并行任务（Task A 代码路径分析、Task B 接口分析、Task C 网络搜索分别做什么）
3. Phase 2 参数推导（Task D 做什么、输入是什么）
4. Phase 3 测试设计（Phase 3a/3b/3c 分别做什么）
5. 子 agent 并行派发规则（哪些任务并行、哪些串行）
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应覆盖 Step 2 的完整流程：
- Phase 0：Scout-T 分析 tiling 入口文件和分支可达性，Scout-K 分析 kernel dispatch 和 TILING_KEY_IS，Scout-Verify 交叉比对生成 file_manifest.json
- Phase 1：Task A 代码路径分析产 path_list.json，Task B 接口分析产 operator_model.json，Task C 网络搜索产 low_configs.json，三者并行
- Phase 2：Task D 参数推导产 param_def.json，串行在 Phase 1 之后
- Phase 3：3a 处理 disputed 路径、3b 生成 test_design.md、3c 生成 shape_mapping
- 并行规则：Phase 0 的 Scout-T/K 并行，Phase 1 的 A/B/C 并行，Phase 2 串行

## Expectations

- [contains] Phase 0
- [contains] Phase 1
- [contains] Task A
- [contains] tiling

---

# Case 3: Step 5 参数组合映射

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-whitebox-design 技能中 Step 5 参数组合映射的流程。请依次介绍以下 5 个方面（每个方面 1-2 句话即可）：
1. 5a 路径覆盖用例怎么生成（输入是什么、产出是什么）
2. 5b 网络用例怎么生成（shape_mapping 的作用）
3. 5c 合并和空 tensor 补全怎么做
4. 5d data_range 维度展开（low 和 high 两档位的区别）
5. shape_mapping 异常回退怎么处理
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应覆盖 Step 5 的完整流程：
- 5a：以 cases.json + operator_model.json 为输入，生成 case_mapper.py 和 mapped_cases_path.json
- 5b：读取 low_configs.json，通过 shape_mapping.role 将语义参数名映射为算子参数名
- 5c：过滤超大 tensor、合并路径+网络+空 tensor 补全，产出 mapped_cases_low.json
- 5d：low 档位全 normal，high 档位 one-hot + 全统一展开
- shape_mapping 异常：mapper 崩溃时回退重新派发 Task E 修正

## Expectations

- [contains] 5a
- [contains] shape_mapping
- [contains] low
- [contains] high

---

# Case 4: Step 6 pytest 生成与执行

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-whitebox-design 技能中 Step 6 pytest 生成与执行的流程。请依次介绍以下 4 个方面（每个方面 1-2 句话即可）：
1. 6a 生成 pytest 脚本（语法检查和收集检查怎么做）
2. 6b 执行 pytest（FAILED/XFAIL/ERROR 分别怎么处理、精度标准能否修改）
3. 6c tilingkey 覆盖率分析（plog 从哪来、覆盖率怎么算）
4. 无 NPU 环境怎么兼容处理
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应覆盖 Step 6 的完整流程：
- 6a：生成 S6_test_{op_name}.py，用 py_compile 语法检查 + pytest --collect-only 收集检查
- 6b：FAILED 转 XFAIL（精度不达标合法）、ERROR 修复代码、精度标准 rtol/atol 禁止修改
- 6c：从 plog 提取 tiling key 值，与 param_def.json 的 tiling_keys 期望集合对比，生成覆盖率报告
- 无 NPU 兼容：自动跳过硬件执行标记 SKIPPED，不会报错

## Expectations

- [contains] pytest
- [contains] XFAIL
- [contains] tiling
- [contains] SKIPPED

---

# Case 5: 执行约束与安全闸门

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-whitebox-design 技能的执行约束和安全闸门机制。请依次介绍以下 5 个方面（每个方面 1-2 句话即可）：
1. 全局顺序约束（禁止跳步、禁止抢跑、前置条件检查）
2. 子 agent 规则传递原则（子 agent 自读规则、主 Agent 只传上下文参数）
3. Step 4 安全闸门（什么时候触发、允许继续的条件、禁止抢跑规则）
4. JSON 完整性校验（什么时候校验、失败怎么处理）
5. 轮次耗尽协议（什么情况下触发、有哪些选项）
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应覆盖执行约束和安全闸门：
- 全局顺序：按 Step 编号顺序执行，禁止跳步和抢跑，前置条件未满足禁止启动
- 子 agent 规则：子 agent 自行 Read 源文件获取规则，主 Agent 只传上下文参数不转述规则
- Step 4 闸门：完成 Step 3 后必须停下展示摘要等待用户确认，用户确认前禁止生成 Step 5 产物
- JSON 校验：读取子 agent 产出 JSON 后立即验证可解析，失败重试最多 1 次
- 轮次耗尽：达到最大轮次仍未通过时向用户报告，提供强制继续/终止/手动修正三个选项

## Expectations

- [contains] 闸门
- [contains] 子 agent
- [contains] JSON
- [contains] 轮次

---

# Case 6: TTK CSV 可选模块

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-whitebox-design 技能的 TTK CSV 可选模块。请依次介绍以下 4 个方面（每个方面 1-2 句话即可）：
1. 启用条件（什么时候触发、Step 1 哪个参数控制）
2. 调用方式（方式 A 自动触发和方式 B 独立调用的区别）
3. 产出文件有哪些（CSV 文件、golden_plugin.py 等）
4. TTK 模块的限制（支持什么模式、前置条件）
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应覆盖 TTK CSV 模块：
- 启用条件：Step 1 输入选择"启用 TTK CSV 模块"时触发
- 调用方式：方式 A 在 Step 6 执行检查全部通过后自动触发，方式 B 可独立派发子 agent 生成
- 产出：ttk_extract_case_info.py、ttk_{op_name}_cases_low.csv、ttk_{op_name}_cases_full.csv、golden_plugin.py
- 限制：仅支持 kernel 模式（ttk kernel），不支持 e2e 和 aclnn 模式；需安装 numpy；需 ops-test-kit/ 目录

## Expectations

- [contains] TTK
- [contains] CSV
- [contains] kernel
- [contains] golden

---

# Case 7: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 200000
- Distractor skills: ascendc-ut-develop;ascendc-st-design;ascendc-code-review;ascendc-api-best-practices
- Ascend Platform: A2

## Prompt

请介绍 ascendc-whitebox-design 技能的核心功能和适用场景。请简要说明以下 3 个方面：
1. 这个技能做什么（核心能力）
2. 什么时候使用（触发关键词）
3. 与 ascendc-ut-develop 和 ascendc-st-design 的区别
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应说明 ascendc-whitebox-design 技能的核心能力：
- 核心功能：分析算子源码提取参数维度，自动枚举参数组合，生成白盒测试用例
- 触发场景：用户要求"为 X 算子生成白盒测试用例"、"算子白盒用例生成"、"路径覆盖"
- 与 UT/ST 的区别：白盒测试是基于源码分析自动枚举参数组合生成用例，UT 是手工编写单元测试，ST 是系统测试
回复应准确描述白盒测试的特点，即使不激活 skill 工具也应基于已有知识正确回答。

## Expectations

---

# Case 8: 信息不足时主动追问

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

帮我生成白盒测试用例

## Expected Output

回复应主动追问关键信息，而不是直接开始生成用例。应至少询问以下信息中的一项或多项：算子名称、算子路径、是否跳过 Step 4 闸门、是否启用 TTK CSV 模块。不应在缺乏算子规格的情况下直接生成测试用例。

## Expectations

- [not_contains] S5_mapped_cases
- [not_contains] S6_test
- [not_contains] pytest

---

# Case 9: 使用边界-不适用于 ST/UT 测试

## Config
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

我想用 ascendc-whitebox-design 这个技能来为我的算子编写系统测试（ST）用例，或者编写单元测试（UT）用例，这个技能适合吗？请详细说明这个技能的适用边界和不适用场景。

## Expected Output

回复应明确说明 ascendc-whitebox-design 不适用于 ST 或 UT 测试。应说明：
- 本技能的定位：白盒测试用例生成系统，基于源码分析自动枚举参数组合
- 适用场景：用户要求为算子生成白盒测试用例、路径覆盖用例
- 不适用场景：ST 系统测试（应使用 ascendc-st-design）、UT 单元测试（应使用 ascendc-ut-develop）
- 白盒测试与 ST/UT 的核心区别：白盒是自动化源码分析+参数枚举，ST 是系统级功能验证，UT 是手工编写单元测试

## Expectations

- [contains] 白盒
- [contains] 源码

---

# Case 10: 最终产物清单

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-whitebox-design 技能执行完成后会生成哪些产物文件。请依次介绍以下 4 个方面（每个方面 1-2 句话即可）：
1. Step 2 产物（S2P0/S2P1/S2P2/S2P3 分别是什么文件、做什么用）
2. Step 3 和 Step 5 产物（验证报告、映射用例文件）
3. Step 6 产物（pytest 脚本、tilingkey 覆盖率报告）
4. TTK 模块产物（CSV 文件、golden_plugin.py）
不需要执行任何操作，只需要介绍知识。

## Expected Output

回复应覆盖最终产物清单：
- Step 2：S2P0_scout_t.md（tiling 侦察）、S2P0_scout_k.md（kernel 侦察）、S2P1_path_list.json（路径清单）、S2P2_param_def.json（参数定义）、S2P3_test_design.md（测试设计）
- Step 3/5：S3_verification_report.json（交叉验证结果）、S5_mapped_cases_low.json（low 档位用例）、S5_mapped_cases_high.json（high 档位用例）
- Step 6：S6_test_{op_name}.py（pytest 脚本）、S6_tilingkey_coverage.json（tilingkey 覆盖率）
- TTK：ttk_{op_name}_cases_low.csv、ttk_{op_name}_cases_full.csv、golden_plugin.py
所有产物写入 {算子路径}/tests/whitebox/ 目录

## Expectations

- [contains] tests/whitebox
- [contains] param_def
- [contains] pytest
- [contains] tiling
