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
| | **1.2.5 spec 生成** | - | 生成机器可校验的 L0 数学契约（9-stage 全 PASS） | REQUIREMENTS.md | spec.yaml | `operators/{operator_name}/docs/` |
| | 1.3 方案设计 | - | 制定技术方案 | REQUIREMENTS.md + spec.yaml | DESIGN.md | `operators/{operator_name}/docs/` |
| | 1.3R 方案评审 | - | DESIGN 条款级评审（含 DESIGN-SPEC-1 与 spec.yaml 一致性条款） | REQUIREMENTS.md + spec.yaml + DESIGN.md | DESIGN_REVIEW.md | `operators/{operator_name}/docs/` |
| | 1.4 测试设计 | - | 设计测试用例（与1.3并行） | REQUIREMENTS.md + spec.yaml | TEST.md | `operators/{operator_name}/docs/` |
| | | | | | 测试用例.csv + 覆盖度报告 | `operators/{operator_name}/tests/st/testcases/` |
| | 1.4R 测试设计评审 | - | TEST 条款级评审（含 TEST-SPEC-1 与 spec.yaml 一致性条款） | REQUIREMENTS.md + spec.yaml + TEST.md + 测试用例 | TEST_REVIEW.md | `operators/{operator_name}/docs/` |
| **第二阶段：开发** | 2.1 初始化 | - | 创建目录 | - | - | `operators/{operator_name}/` |
| | Phase 1-3 | **A1-Main** | 主线代码开发 | DESIGN.md + spec.yaml | 算子代码 | `operators/{operator_name}/` |
| | Phase 1-2 | **A1-P** | 穿刺验证 | DESIGN.md + spec.yaml | 穿刺工程 + RESULT.md | `operators/{operator_name}/probe/` |
| | Phase 1-2 第二波 | **A1-P-Retry** | 失败穿刺重试 | PROBE_SUMMARY.md + 当前主线代码 | 更新的 RESULT.md + PROBE_SUMMARY.md（含重试次数） | `operators/{operator_name}/probe/` |
| | Phase 1-3 | **A2** | UT用例开发 | TEST.md + spec.yaml | UT测试代码 | `operators/{operator_name}/tests/ut/` |
| | Phase 1-3 | **B** | C++ ST测试开发 | TEST.md + spec.yaml | C++ ST测试代码 | `operators/{operator_name}/tests/st/` |
| **第二阶段：开发** | 汇合验证 | - | 开发联调 | UT + ST代码 | iter{N}-integration-report.md | `operators/{operator_name}/tests/reports/` |
| | 测试工程师验收 | - | 迭代验收 | 汇合验证报告 | iter{N}-acceptance-report.md | `operators/{operator_name}/tests/reports/` |
| **阶段二/三之间** | 白盒测试生成 | **W** | 按白盒 skill 完成白盒测试生成与执行证据汇合 | 需求+spec+设计+实现+黑盒结果 | 白盒测试产物 + 证据汇总 | `operators/{operator_name}/tests/whitebox/`、`tests/reports/` |
| **阶段二/三之间** | PyTorch测试开发 | **C** | PyTorch ST测试开发（一次性完成L0+L1全量） | TEST.md + C++ ST + 白盒结果 | PyTorch ST测试代码 | `operators/{operator_name}/tests/st/torch/` |
| **第三阶段：验收** | 3.1 精度验收 | - | 执行精度验证 | PyTorch ST + 测试用例 | precision-report.md | `operators/{operator_name}/docs/` |
| | 3.2 性能验收 | - | 性能分析（可选） | 算子二进制 | performance-report.md | `operators/{operator_name}/docs/` |
| **第四阶段：上库** | 4.1 文档与示例 | - | 生成文档示例 | 需求+设计+代码 | README.md + examples/ | `operators/{operator_name}/` |
| | 4.2 代码检视 | - | 主 Agent 直接调用 ascendc-code-review skill（file-review + design-consistency） | 算子代码 + 设计文档 | {source_file}_review_summary.md + {source_file}_design_consistency_review.md | `operators/{operator_name}/docs/` |
| | 4.3 开发总结 | - | 总结输出文档 | 所有文档 | aclnn{OperatorName}.md（更新）+ LOG.md | `operators/{operator_name}/` |

---

## 测试用例分发

```
TEST.md
    │
    ├─→ L0 级别（门槛用例，≤50个）→ ST（核心功能直通）
    ├─→ L1 级别（功能/精度，按 ascendc-st-design 当前默认目标生成）→ ST（BC组合测试）
    └─→ L2 级别（异常用例，≤50个）→ UT/Host 或 ST 异常验证

测试证据由 workflow validator 对账。用例、执行结果、调试结果和汇总报告之间不一致，或只有自洽摘要但缺少可执行证据时不得进入后续验收。
```

---

## 文件路径速查表

**命名规范**：{operator_name} 使用 snake_case 风格（小写字母+下划线），例如：add_custom、matmul_v2、reduce_sum

| 阶段 | 文件类型 | 路径 |
|------|---------|------|
| **第一阶段** | 开发日志 | `operators/{operator_name}/docs/LOG.md` |
| | 需求文档 | `operators/{operator_name}/docs/REQUIREMENTS.md` |
| | aclnnAPI 接口文档 | `operators/{operator_name}/docs/aclnn{OperatorName}.md` |
| | 设计文档 | `operators/{operator_name}/docs/DESIGN.md` |
| | 测试设计文档 | `operators/{operator_name}/docs/TEST.md` |
| | 测试用例（L0/L1/L2） | `operators/{operator_name}/tests/st/testcases/` |
| **第二阶段** | 算子代码 | `operators/{operator_name}/` |
| | 图模式定义 | `operators/{operator_name}/op_graph/{operator_name}_proto.h` |
| | 问题记录 | `operators/{operator_name}/issues/issue_{YYYYMMDD}_{关键词}.md` |
| | UT 机器报告 | `operators/{operator_name}/tests/ut/` |
| | 黑盒测试证据 | `operators/{operator_name}/tests/st/` |
| | 汇合验证报告（迭代N） | `operators/{operator_name}/tests/reports/iter{N}-integration-report.md` |
| | 迭代验收报告（迭代N） | `operators/{operator_name}/tests/reports/iter{N}-acceptance-report.md` |
| **阶段二/三之间** | 白盒测试产物 | `operators/{operator_name}/tests/whitebox/` |
| | 测试证据汇总 | `operators/{operator_name}/tests/reports/` |
| **第三阶段** | 最终精度验收报告 | `operators/{operator_name}/docs/precision-report.md` |
| | 最终性能验收报告 | `operators/{operator_name}/docs/performance-report.md` |
| **第四阶段** | 算子 README | `operators/{operator_name}/README.md` |
| | 调用示例 | `operators/{operator_name}/examples/` |
| | 全量代码检视报告 | `operators/{operator_name}/docs/{source_file}_review_summary.md` |
| | 设计实现一致性报告 | `operators/{operator_name}/docs/{source_file}_design_consistency_review.md` |

**报告命名规则**：
- `*-report.md` (小写+连字符) = 最终交付报告，放 `docs/`
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
│   │   ├── results/                    # 测试执行证据；调试/复现结果不得覆盖主证据
│   │   ├── torch/                      # PyTorch 接入测试（可选）
│   │   │   ├── CMakeLists.txt          # PyTorch 适配层构建配置
│   │   │   ├── test.py                 # 测试入口（用例定义 + 调度）
│   │   │   ├── golden.py               # CPU golden 计算
│   │   │   ├── compare.py              # 精度比对逻辑
│   │   │   └── torch_adapter.cpp       # PyTorch 算子注册 + ACLNN 两段式封装
│   │   └── testcases/
│   │       ├── L0_test_cases.csv
│   │       ├── L1_test_cases.csv
│   │       ├── L2_test_cases.csv
│   │       └── L*_coverage_report.yaml
│   ├── whitebox/                       # 白盒 skill 产物，内部结构以 ascendc-whitebox-design 为准
│   └── reports/                           # 中间态报告
│       ├── iter1-integration-report.md    # 迭代一联调
│       ├── iter1-acceptance-report.md     # 迭代一验收
│       ├── iter2-integration-report.md
│       ├── iter2-acceptance-report.md
│       ├── iter3-integration-report.md
│       ├── iter3-acceptance-report.md
│       └── ...                         # validator 所需测试证据汇总
├── issues/                              # 问题解决记录
│   ├── issue_{YYYYMMDD}_{关键词}.md    # 单个问题记录
│   └── ...
└── docs/
    ├── LOG.md                             # 开发日志
    ├── REQUIREMENTS.md                    # 需求分析
    ├── aclnn{OperatorName}.md             # aclnnAPI 接口文档
    ├── DESIGN.md                          # 详细设计
    ├── TEST.md                            # 测试设计
    ├── PLAN.md                            # 迭代计划
    ├── precision-report.md                # 最终精度验收
    ├── performance-report.md              # 最终性能验收（可选）
    ├── {source_file}_review_summary.md     # 全量代码检视报告
    └── {source_file}_design_consistency_review.md  # 设计实现一致性报告
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
| `operators/{operator_name}/iter1-passed` | 迭代一验收通过 | 回滚锚点 |
| `operators/{operator_name}/iter2-passed` | 迭代二验收通过 | 回滚锚点 |
| `operators/{operator_name}/iter3-passed` | 迭代三验收通过 | 回滚锚点 |
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
