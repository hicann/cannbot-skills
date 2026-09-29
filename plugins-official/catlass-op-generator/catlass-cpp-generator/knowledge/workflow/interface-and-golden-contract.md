---
type: "workflow"
title: "接口与 Golden Contract"
description: "PR1069 的接口确认、唯一标杆、冲突裁决和 contract 冻结规则。"
tags: ["catlass-cpp", "workflow", "interface", "reference", "pr1069"]
status: "stable"
generated: {"by": "process:catlass-cpp-pr1069-extraction", "at": "2026-09-18T00:00:00Z"}
verified: [{"by": "process:content-equivalence-check", "at": "2026-09-18T00:00:00Z"}]
sources: [{"id": "pr1069-interface-reference", "resource": "git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/SKILL.md", "title": "PR1069 fixed source for Linear Attention 接口与 Golden Contract", "kind": "repository"}]
consumers: ["catlass-cpp-interface", "catlass-cpp-reference"]
---
# 接口与概念

本 concept 是 PR1069 固定来源的结构化入口；规范性技术正文完整保留在“PR1069 原始正文”。

## 算子算法

算法定义、公式、计算顺序和边界语义以原始正文及当前任务冻结的 operator/golden contract 为准。

## 分核策略与基本块切分

分核、任务组、基本块和尾块规则以原始正文为准，并需针对当前 shape、SoC 和资源重新推导。

## 数据路径与存储层级

GM、workspace、L1、UB、L0 的地址、生命周期和复用要求以原始正文为设计输入。

## 流水排布、同步关系与数值精度

Stage、slot、CrossCore/HardEvent、累加和转换规则以原始正文为准，实际结论必须通过验证。

# 用法

先通过 compact query 定位本 concept，再完整读取；不得把知识内容当作当前工程已验收的证据。

# 代码模式

原始正文中的伪代码和命令保持原义。具体 API 必须核对目标版本 CATLASS/CANN 源码。

# 约束

不得删减或放宽原始规则、阈值、失败条件、状态恢复或验收要求。

# 失败表现

若设计、代码或测试与原始约束不一致，按五阶段 workflow 的 issue_type/resume_from 矩阵恢复。

# 验证方法

对照 frontmatter `sources` 中的固定 PR SHA 和原文件逐段校验；阶段通过仍需实际构建、精度和性能证据。

# PR1069 原始正文

### 01 接口确认

输入为用户需求、参考资料或固定版本参考实现。先记录来源/版本、许可证约束、目标 SoC、
支持范围和性能目标，再分别记录两类约定：

- `operator_contract`：算子接口约定，包括输入、输出、张量形状（shape）、数据类型（dtype）、内存布局（layout）和递推状态（state）的表示；
- `golden_contract`：标杆计算约定，包括数学公式、状态更新、计算顺序、边界行为和误差容限。

以上内容写入 `docs/api.md`。接口经用户确认后，将 `operator_contract` 设为 `frozen`（已确认），
进入 02；`golden_contract` 保持 `provisional`（待确认），由 02 实际运行标杆后确认。
精度排查发现标杆计算约定有误时，可单独将 `golden_contract` 改为待确认，返回 02 修正。

接口说明至少覆盖调用入口、输入输出的顺序及必选/可选参数、shape、dtype、layout、state、
属性默认值、归一化、缩放、掩码（mask）和非法输入处理。支持范围应写清定长/变长序列
（fixed/varlen）、尾块（tail）、填充（padding）和不同 head 数量比例（head ratio）。
既有算子先列出改变项与保持不变项：公开接口或语义变化从 01 开始；只有标杆变化从 02 开始；
实现精度修复从 04 开始，性能优化从 05 开始；设计假设需要调整时回到 03。
已有专用流程按下方的问题恢复表更新状态。

材料冲突时按“公开接口/用户明确说明 → 测试和示例 → 参考实现 → 论文或相邻实现”的顺序列出
差异并请用户定夺。预留参数要记录名称、类型、当前处理方式、host 侧如何校验
以及后续启用时如何兼容。直调工程中的 `op_host`、TilingData、kernel 入口、host `main()`、运行脚本、
公开文档和测试中的名称、顺序、类型及默认值必须一致；同一公开接口在不同 SoC 上保持一致，
平台差异只留在内部实现。

### 02 标杆生成

`reference/reference.py` 是唯一可编辑的参考实现源码。

```text
reference.py -> generate_definition.py -> definition.json["reference"]
```

算子首次创建时，将 `{skill_dir}/templates/definition.template.json` 复制到
`{operator_dir}/reference/definition.template.json`，在模板中维护标杆描述等信息，保持 `reference` 为空。
该文件用于开发期生成和一致性校验，继续开发时需要保留；发布或部署包可以排除。

每次修改 `reference/reference.py` 后，通过以下命令生成 `definition.json.reference` 并校验一致性：

```bash
python "{skill_dir}/scripts/generate_definition.py" --source "{operator_dir}/reference/reference.py" \
  --template "{operator_dir}/reference/definition.template.json" --output "{operator_dir}/reference/definition.json"
python "{skill_dir}/scripts/validate_reference.py" --source "{operator_dir}/reference/reference.py" \
  --definition "{operator_dir}/reference/definition.json"
```

PyTorch 标杆源码独立于 NPU 实现，正式验收使用 CPU 运行结果。按 `precision-policy.md`
确定输入数值范围、参与比较的有效区域、需要分别检查的分区和混合容差；最大绝对误差上限
`max_abs_limit` 必须在本阶段按 dtype、输出值域和 shape 类别校准，并写入初始化时生成的
`reference/precision-policy.json`。`docs/workflow.json` 的 `precision_policy_path` 相对算子目录解析；
Agent 将解析出的完整路径显式传给比较脚本的 `--policy`，调用 `compare_case()` 时加载同一文件。
标杆计算规则的差异在 02 解决，记录验收结果后进入 03。

标杆实现使用设备无关的纯 PyTorch 算子，执行设备由输入张量或调用参数指定。
在 CPU 上以固定版本、固定随机种子和约定的高精度累加生成正式验收使用的标杆结果；同一
源码可在 GPU 上进行可选交叉验证。若用户提供已有 CPU 标杆，先验证其与 PyTorch 标杆
的接口、计算规则和结果一致，再将 PyTorch 标杆作为唯一源码。
固定随机种子、输入、dtype、layout、属性和代码版本，记录输入与输出的对应关系。精度比较统一使用
`precision-policy.md` 和 `scripts/compare_precision.py`，并将策略版本写入
`docs/workflow.json`。至少覆盖最小、
常用和非对齐 shape，定长/变长序列、多分块（chunk）、不满一块的尾部、填充、可选递推状态、head 数量比例
以及非法输入；每个输出记录有效区域、比较指标和混合容差。运行结果和来源记录写入
`docs/validation.md`。

02 的完成条件是至少取得一种实际运行结果：运行固定版本参考实现、比较用户提供的固定输入/预期
输出，或运行并验收用户提供的 CPU 结果。缺少运行证据时，`stage` 保持 `reference`，在
`docs/validation.md` 记录待补证据并完成上述运行验证。
参考实现因 Ascend 适配而修改时，必须记录修改内容，并确认公式、计算顺序、输入
输出和边界语义未被改变。

[^pr1069-interface-reference]: git:cann/cannbot-skills@4950cbd7c45e0d44cb8ec52598edf53cdc7f16ca:ops/catlass-linear-attention-workflow/SKILL.md
