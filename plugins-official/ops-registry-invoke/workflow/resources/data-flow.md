# 算子开发流程数据流

> 各阶段输入输出文件说明

---

## 阶段表格

**轨道代号**：A1-Main (主线代码) | A1-P (穿刺验证) | A1-P-Retry (失败穿刺重试) | A2 (UT开发) | B (C++ ST测试) | C (PyTorch ST测试)

| 大阶段 | 子阶段 | 轨道 | 主要任务 | 输入文件 | 输出文件 | 输出位置 |
|--------|--------|------|----------|----------|----------|----------|
| **第一阶段：设计** | 1.1 开发准备 | - | 创建开发日志 | 用户描述 | LOG.md | `operators/{operator_name}/docs/` |
| | 1.2 需求分析 | - | 收集算子需求信息 | 用户描述 | REQUIREMENTS.md | `operators/{operator_name}/docs/` |
| | | | | | aclnn{OperatorName}.md | `operators/{operator_name}/docs/` |
| | **1.2.5 spec 生成** | - | 生成机器可校验的 L0 数学契约（11-stage 全 PASS） | REQUIREMENTS.md | spec.yaml | `operators/{operator_name}/docs/` |
| | 1.2.5R spec 评审 | - | SPEC 条款级评审（CP1.5 前置，由 spec-reviewer 独立执行） | REQUIREMENTS.md + spec.yaml | SPEC_REVIEW.md | `operators/{operator_name}/tmp/checks/` |
| | 1.3a 设计准备 | - | 路线决策 + 模板选型 + API 验证 | REQUIREMENTS.md + spec.yaml | DESIGN_PREP.md | `operators/{operator_name}/docs/` |
| | 1.3b-d 分段生成与组装 | - | 切片（主 Agent）→ 5 分段 Agent 并行生成 → 组装+校验（主 Agent） | spec.yaml + REQUIREMENTS.md + DESIGN_PREP.md | DESIGN.md + PLAN.md | `operators/{operator_name}/docs/` |
| | 1.3R 方案评审 | - | DESIGN 条款级评审（含 DESIGN-SPEC-1 与 spec.yaml 一致性条款） | REQUIREMENTS.md + spec.yaml + DESIGN.md | DESIGN_REVIEW.md | `operators/{operator_name}/tmp/checks/` |
| | 1.4 测试设计 | - | 设计测试用例（与1.3并行） | REQUIREMENTS.md + spec.yaml | TEST.md | `operators/{operator_name}/docs/` |
| | | | | | 测试用例.csv + 覆盖度报告 | `operators/{operator_name}/tests/st/testcases/` |
| | 1.4R 测试设计评审 | - | TEST 条款级评审（含 TEST-SPEC-1 与 spec.yaml 一致性条款，由 test-design-reviewer 独立执行） | REQUIREMENTS.md + spec.yaml + TEST.md + 测试用例 | TEST_REVIEW.md | `operators/{operator_name}/tmp/checks/` |
| **第二阶段：开发** | 2.1 初始化 | - | 创建目录 | - | - | `operators/{operator_name}/` |
| | Phase 1-3 | **A1-Main** | 主线代码开发 | DESIGN.md + spec.yaml | 算子代码 | `operators/{operator_name}/` |
| | Phase 1-2 | **A1-P** | 穿刺验证 | DESIGN.md + spec.yaml | 穿刺工程 + RESULT.md | `operators/{operator_name}/probe/` |
| | Phase 1-2 第二波 | **A1-P-Retry** | 失败穿刺重试 | PROBE_SUMMARY.md + 当前主线代码 | 更新的 RESULT.md + PROBE_SUMMARY.md（含重试次数） | `operators/{operator_name}/probe/` |
| | Phase 1-3 | **A2** | UT用例开发 | TEST.md + spec.yaml | UT测试代码 | `operators/{operator_name}/tests/ut/` |
| | Phase 1-3 | **B** | C++ ST测试开发 | TEST.md + spec.yaml | C++ ST测试代码 | `operators/{operator_name}/tests/st/` |
| **第二阶段：开发** | 汇合验证 | - | 开发联调 | UT + ST代码 | iter{N}-integration-report.md | `operators/{operator_name}/tests/reports/` |
| | 测试工程师验收 | - | 迭代验收 | 汇合验证报告 | iter{N}-acceptance-report.md | `operators/{operator_name}/tests/reports/` |
| **阶段二/三之间** | 白盒测试生成 | **W** | 白盒用例生成、ST 主验收、pytest 辅助结果和用例汇合 | 需求+spec+设计+实现+黑盒结果 | 白盒测试产物 + evidence_index | `operators/{operator_name}/tests/whitebox/`、`tests/reports/` |
| **阶段二/三之间** | PyTorch测试开发 | **C** | PyTorch ST测试开发（一次性完成L0+L1全量） | TEST.md + C++ ST + 白盒结果 | PyTorch ST测试代码 | `operators/{operator_name}/tests/st/torch/` |
| **第三阶段：验收** | 3.1 精度验收 | - | 执行精度验证 | PyTorch ST + 测试用例 | precision-report.md | `operators/{operator_name}/docs/` |
| | 测试分支合并执行 | - | 汇总最终黑盒、白盒与 PyTorch ST 执行证据 | 最终测试结果 | test-branches-merge-exec-report.md | `operators/{operator_name}/tests/reports/` |
| | 3.2 性能验收 | - | 性能分析（可选） | 算子二进制 | performance-report.md | `operators/{operator_name}/docs/` |
| **第四阶段：上库** | 4.1 文档与示例 | - | 生成文档示例 | 需求+设计+代码 | README.md + examples/ | `operators/{operator_name}/` |
| | 4.2 代码检视 | - | 主 Agent 直接调用 ascendc-code-review skill（file-review，自动探测设计文档并完成设计一致性检查） | 算子代码 + 设计文档 | {source_file}_review_summary.md（含设计一致性章节） | `operators/{operator_name}/tmp/checks/` |
| | 4.3 开发总结 | - | 总结输出文档 | 所有文档 | aclnn{OperatorName}.md（更新）+ LOG.md | `operators/{operator_name}/` |

---

## 测试用例分发

```
TEST.md
    │
    ├─→ L0 级别（门槛用例，≤50个）→ ST（核心功能直通）
    ├─→ L1 级别（功能/精度，按 ascendc-st-design 当前默认目标生成）→ ST（BC组合测试）
    └─→ L2 级别（异常用例，≤50个）→ UT/Host 或 ST 异常验证

固定门禁脚本读取所需机器证据并输出阶段是否通过；校验失败时，主 Agent 按脚本列出的差距调度对应 Subagent/任务修复，然后重跑门禁脚本。
```

---

## 文件路径速查表

**命名规范**：{operator_name} 使用 snake_case 风格（小写字母+下划线），例如：add_custom、matmul_v2、reduce_sum

| 阶段 | 文件类型 | 路径 |
|------|---------|------|
| **第一阶段** | 开发日志 | `operators/{operator_name}/docs/LOG.md` |
| | 需求文档 | `operators/{operator_name}/docs/REQUIREMENTS.md` |
| | aclnnAPI 接口文档 | `operators/{operator_name}/docs/aclnn{OperatorName}.md` |
| | 设计准备结论 | `operators/{operator_name}/docs/DESIGN_PREP.md` |
| | 设计文档 | `operators/{operator_name}/docs/DESIGN.md` |
| | 迭代计划 | `operators/{operator_name}/docs/PLAN.md` |
| | 测试设计文档 | `operators/{operator_name}/docs/TEST.md` |
| | spec 评审报告 | `operators/{operator_name}/tmp/checks/SPEC_REVIEW.md` |
| | 方案评审报告 | `operators/{operator_name}/tmp/checks/DESIGN_REVIEW.md` |
| | 测试设计评审报告 | `operators/{operator_name}/tmp/checks/TEST_REVIEW.md` |
| | 测试用例（L0/L1/L2） | `operators/{operator_name}/tests/st/testcases/` |
| **第二阶段** | 算子代码 | `operators/{operator_name}/` |
| | 图模式定义 | `operators/{operator_name}/op_graph/{operator_name}_proto.h` |
| | 问题记录 | `operators/{operator_name}/issues/issue_{YYYYMMDD}_{关键词}.md` |
| | UT 逐 case 机器报告 | `operators/{operator_name}/tests/ut/test-report.json` |
| | 黑盒 case 清单 | `operators/{operator_name}/tests/st/case_manifest.json` |
| | 黑盒开发期结果 | `operators/{operator_name}/tests/st/results/st_dev_result.json` |
| | 黑盒真实结果 | `operators/{operator_name}/tests/st/results/st_real_result.json` |
| | 汇合验证报告（迭代N） | `operators/{operator_name}/tests/reports/iter{N}-integration-report.md` |
| | 迭代验收报告（迭代N） | `operators/{operator_name}/tests/reports/iter{N}-acceptance-report.md` |
| **阶段二/三之间** | 白盒流程证明 | `operators/{operator_name}/tests/whitebox/WORKFLOW_PROVENANCE.json` |
| | 白盒参数与 tiling key 定义 | `operators/{operator_name}/tests/whitebox/S2P2_param_def.json` |
| | 白盒用例定义 | `operators/{operator_name}/tests/whitebox/S5_mapped_cases_low.json`、`S5_mapped_cases_high.json` |
| | 白盒 tiling key 覆盖 | `operators/{operator_name}/tests/whitebox/S6_tilingkey_coverage.json` |
| | 白盒 ST 主验收 | `operators/{operator_name}/tests/whitebox/results/st_result.json` |
| | 白盒 pytest 辅助证据 | `operators/{operator_name}/tests/whitebox/results/pytest_collect.json`、`pytest_result.json` |
| | 证据索引 | `operators/{operator_name}/tests/reports/evidence_index.json` |
| **第三阶段** | 最终精度验收报告 | `operators/{operator_name}/docs/precision-report.md` |
| | 测试分支合并执行报告 | `operators/{operator_name}/tests/reports/test-branches-merge-exec-report.md` |
| | 最终性能验收报告 | `operators/{operator_name}/docs/performance-report.md` |
| **第四阶段** | 算子 README | `operators/{operator_name}/README.md` |
| | 调用示例 | `operators/{operator_name}/examples/` |
| | 代码检视报告（含设计一致性结论） | `operators/{operator_name}/tmp/checks/{source_file}_review_summary.md` |
| | 代码检视概要分析 | `operators/{operator_name}/tmp/checks/code_summary.md` |
| | API 预研报告（如有） | `operators/{operator_name}/tmp/checks/api_prestudy.md` |

**报告命名规则**：
- `precision-report.md` / `performance-report.md` (小写+连字符) = 最终验收报告，放 `docs/`
- `{source_file}_review_summary.md`（含设计一致性章节） / `code_summary.md` / `api_prestudy.md` = 代码检视产物（全量检视 + 设计一致性 + 概要分析 + API预研），放 `tmp/checks/`（临时检查产物，与其他评审报告一致）
- `iter{N}-*-report.md` = 中间态报告，放 `tests/reports/`
- `integration` = 开发联调（汇合验证），侧重"ST在NPU上精度验证通过"，禁止仅编译通过或CPU Mock通过
- `acceptance` = 正式验收，侧重"功能是否达标"

---

## 算子代码目录结构

```
{operator_name}/
├── CMakeLists.txt
├── README.md
├── examples/                            # 调用示例
│   ├── test_aclnn_{operator_name}.cpp   # aclnn调用示例
│   └── test_geir_{operator_name}.cpp    # 图模式调用示例
├── op_graph/                            # 图模式适配
│   └── {operator_name}_proto.h          # 图模式算子定义（REG_OP）
├── op_host/
│   ├── {operator_name}_def.cpp
│   ├── {operator_name}_infershape.cpp
│   └── {operator_name}_tiling.cpp
├── op_kernel/
│   ├── {operator_name}.cpp
│   ├── {operator_name}.h
│   └── {operator_name}_tiling_data.h
├── tests/
│   ├── ut/
│   ├── st/
│   │   ├── CMakeLists.txt              # C++ 测试构建配置
│   │   ├── run.sh                      # 测试执行脚本（支持 --torch 选项）
│   │   ├── test_aclnn_{operator_name}.cpp  # C++ 测试主程序
│   │   ├── case_manifest.json
│   │   ├── results/st_dev_result.json
│   │   ├── results/st_real_result.json
│   │   ├── results/debug/              # 单 case debug/失败复现结果，不可覆盖 st_dev/st_real 主证据
│   │   ├── torch/                      # PyTorch 接入测试（可选）
│   │   │   ├── CMakeLists.txt          # PyTorch 适配层构建配置
│   │   │   ├── test.py                 # 测试入口（用例定义 + 调度）
│   │   │   ├── golden.py               # CPU golden 计算
│   │   │   ├── compare.py              # 精度比对逻辑
│   │   │   └── torch_adapter.cpp       # PyTorch 算子注册 + ACLNN 两段式封装
│   │   └── testcases/
│   │       ├── {operator_name}_l0_functional.csv
│   │       ├── {operator_name}_l1_functional.csv
│   │       ├── {operator_name}_l2_exception.csv
│   │       └── L*_coverage_report.yaml
│   ├── whitebox/
│   │   ├── WORKFLOW_PROVENANCE.json
│   │   ├── S2P2_param_def.json
│   │   ├── S5_mapped_cases_low.json
│   │   ├── S5_mapped_cases_high.json
│   │   ├── S6_test_{op_name}.py
│   │   ├── S6_tilingkey_coverage.json
│   │   └── results/
│   │       ├── st_result.json
│   │       ├── pytest_collect.json
│   │       └── pytest_result.json
│   └── reports/                           # 中间态报告
│       ├── iter{N}-integration-report.md  # 迭代 N 联调（N = 1..iteration_count）
│       ├── iter{N}-acceptance-report.md   # 迭代 N 验收（N = 1..iteration_count）
│       └── evidence_index.json
├── issues/                              # 问题解决记录
│   ├── issue_{YYYYMMDD}_{关键词}.md    # 单个问题记录
│   └── ...
├── tmp/                                # 临时检查产物
│   └── checks/
│       ├── SPEC_REVIEW.md
│       ├── DESIGN_REVIEW.md
│       ├── TEST_REVIEW.md
│       ├── code_summary.md               # 代码检视概要分析
│       ├── api_prestudy.md               # API 预研报告（条件生成）
│       └── {source_file}_review_summary.md     # 代码检视报告（含设计一致性章节）
└── docs/
    ├── LOG.md                             # 开发日志
    ├── REQUIREMENTS.md                    # 需求分析
    ├── aclnn{OperatorName}.md             # aclnnAPI 接口文档
    ├── DESIGN.md                          # 详细设计
    ├── TEST.md                            # 测试设计
    ├── PLAN.md                            # 迭代计划
    ├── precision-report.md                # 最终精度验收
    └── performance-report.md              # 最终性能验收（可选）
```

---

## 精度标准

精度标准统一从 `ops-precision-standard` 技能获取，根据算子类型和数据类型自动匹配对应的精度比对标准。

- 技能入口：`skills/ops-precision-standard/SKILL.md`
- 根据算子计算类型（浮点/整数/量化/随机数/非计算）和数据类型（FP16/FP32/BF16/INT 等）选择对应标准文档

---

## Git Checkpoint 对照表

> 集中式管理：主 Agent 统一执行 git commit + tag，Subagent 不操作 git
> 分支策略：每个算子在独立分支 `operators/{operator_name}` 上开发，1.1 阶段创建，4.3 完成后合回主线

### Tag 命名规范

| Tag | 触发时机 | 说明 |
|-----|---------|------|
| `operators/{operator_name}/requirements-approved` | CP1 确认通过 | 需求锁定 |
| `operators/{operator_name}/design-approved` | CP2 确认通过 | 设计锁定 |
| `operators/{operator_name}/iter{N}-passed` | 迭代 N 验收通过（N = 1..iteration_count） | 回滚锚点 |
| `operators/{operator_name}/precision-passed` | 精度验收通过 | |
| `operators/{operator_name}/performance-passed` | 性能验收通过 | 可选 |
| `operators/{operator_name}/done` | 4.3 上库完成 | 最终交付，合回主线 |

### Commit 消息规范

| 格式 | 示例 |
|------|------|
| `feat({operator_name}): {描述}` | `feat(abs_ex): 迭代一验收通过` |
| `fix({operator_name}): {描述}` | `fix(abs_ex): 修复 FP16 精度越界` |
| `test({operator_name}): {描述}` | `test(abs_ex): PyTorch ST测试开发完成` |
| `docs({operator_name}): {描述}` | `docs(abs_ex): 补充 README 调用示例` |
| `revert({operator_name}): 回退到 {tag}` | `revert(abs_ex): 回退到 iter2-passed` |
