# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Text writer - generates modified document based on evidence"""

import json
import re
from typing import Optional, Union

from llm.client import LLMClient
from src.logger import debug

from ..core.models import (
    ModificationResult,
    StyleExtraction,
    StyleSummary,
)
from ..style.loader import TemplateLoader


class TextWriter:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        template_loader: Optional[TemplateLoader] = None,
    ):
        self.llm = llm_client
        self.template_loader = template_loader

    def generate(
        self,
        original: Union[str, dict],
        evidences: dict = None,
        style_summary: StyleSummary = None,
        style_extraction: StyleExtraction = None,
        doc_part: str = "",
        sub_key: str = "",
        previous_issues: list = None,
        generated_doc: str = "",
        change_reason: Optional[dict] = None,
    ) -> ModificationResult:
        if self.llm is None:
            return ModificationResult(
                generated_text=original,
                evidence_usage=[],
                modification_summary={},
            )

        doc_part_label = self.template_loader.get_label(doc_part, sub_key)
        requirement = self.template_loader.get_requirement(doc_part, sub_key)
        style_template = self.template_loader.get_template_example(doc_part, sub_key)
        evidences_str = json.dumps(evidences, ensure_ascii=False, indent=2) if evidences else "无"

        change_reason_section = ""
        if change_reason and change_reason.get("keep"):
            reason_text = change_reason.get("reason", "")
            change_reason_section = f"""## 变更检测结论（代码变更 → 该章节被判定需要更新的理由）

**触发原因**：{reason_text}

> 写作指引：围绕上述理由组织变更，只调整与变更原因相关的部分；与理由无关的原文内容默认保留，不做无意义改写。

"""

        issues_section = ""
        if previous_issues:
            original = generated_doc
            issues_str = json.dumps(previous_issues, ensure_ascii=False, indent=2)
            issues_section = f"""
## 上次校验发现的问题（必须修正）

以下是上一轮生成后被校验器指出的问题，请依据上方证据逐一修正，避免引入新错误：

```json
{issues_str}
```
"""

        prompt = f"""你是一个增量文档更新助手。代码发生了变更，你需要用新证据更新旧文档。

## 一、原文（修改前的文档 — 基于旧代码编写）

以下是当前文档的已有内容，由旧版本的代码生成。其中的内容可能已过时，需要与证据比对后按需更新。

```markdown
{original}
```

## 二、证据（新代码的事实 — 最高数据优先级）

以下证据从**变更后的新代码**中提取，每条证据的 `content` 字段记录了新代码中某个属性的**真实值**。证据是所有事实数据的最终来源。

```json
{evidences_str}
```
{change_reason_section}{issues_section}
## 文档写作风格

### 原文整篇摘要（全局视角）
{style_summary}

### 文档风格约束
{style_extraction}

### 文档写作要求
1. 范围：只生成**当前文档部分**的内容（如：{doc_part_label}），而不是生成整个文档
2. **优先级规则**：**变更理由约束范围 > 证据值 > 模板格式 > 原文格式**
   - 变更理由：约束变更范围，只调整理由覆盖的部分，与理由无关的原文默认保留
   - 证据：提供所有事实数据（维度、数据类型、取值范围等）的最终真值
   - 模板：仅提供排版结构参考，其中的具体数据值是占位示例，不代表当前算子的真实值
   - 原文：证据未涉及的部分可保留，与证据冲突的部分必须用证据覆盖
3. 事实型证据（category=fact）的值必须在生成文档中**精确体现**
4. 描述型证据可自由组织语言
5. 保持与原文一致的写作风格
6. **内部链接保留**：原文中已有的内部锚点链接（如 `[表1 xxx](#table1)`）在对应内容仍存在时必须原样保留，不得丢弃或改写锚点
7. **文件路径以证据为准且必须完整显示**：文中引用的头文件/源文件路径必须与证据中的文件路径一致；原文中的路径可能是改名前的旧路径，与证据冲突时必须用证据覆盖，不得从原文其他章节照抄旧路径。显示时必须使用证据中的完整相对路径（如
`include/c_api/reg_compute/reg_vector.h`），不得缩写为裸文件名（如仅写 `reg_vector.h`）
8. **原型逐条有据**：函数原型章节中的每条函数签名必须逐字来自证据（原文档已有的原型，或证据代码片段中实际出现的签名）；禁止按已有重载的模式推测、补齐证据中不存在的重载——例如仅某一个数据类型新增了带新参数的重载时，
不得给其他数据类型补出同类重载
9. **仅限本接口**：不得引入原文档从未提及的其他接口名——证据上下文（整个头文件的兄弟接口）不构成提及依据，禁止据此泛化（如"大部分函数如 asc_xxx、asc_yyy 返回…"这类跨接口表述）；原约束条目除非证据表明已不成立，
否则必须保留。**例外**：变更检测结论明确指认的新增门面/关联接口（结论中出现"检测到新增门面接口 xxx"字样时）属于本文档 API 的关联接口，必须按结论要求将其纳入本章内容（功能说明写其功能、函数原型写其声明、返回值说明写其返回值）
，其语义按结论给出的转发实现代码推导；此例外仅限结论点名的接口，不开放给其他兄弟接口
10. **头文件声明位置**：接口所在头文件只在「功能说明」首段声明一次，句式固定为 `头文件路径为：`"<相对路径>"`。`（相对路径去掉 `include/` 前缀）；「函数原型」及其它章节禁止再写"头文件"行
11. **调用形式不得自造**：函数原型与调用示例中的每个接口调用，必须能在证据的 `include/` 头文件声明、或原文中逐字找到。实现文件（`impl/`）内的内部调用（如 `Reg::Gather(dstReg,
...)`）只能作为行为参考，**不得**直接写成对外接口原型，也不得为其补加模板参数（如 `<T, U>`）或命名空间限定。模板参数个数与顺序、参数列表必须与声明完全一致，禁止按"常见重载模式"推断补齐

## 模板参考（仅用于排版和结构）

除修改现有不符内容外，还需尽可能向标准模板靠拢。

### 模板样例
```markdown
{style_template}
```

### 模板具体要求
{requirement}

## 输出格式

### **evidence_conflict 判断标准**：
- 仅当证据间存在**明确且不可调和**的矛盾、无法同时满足所有证据时，才标记为 evidence_conflict
- 典型场景：
  - 部分证据已更新（如头文件中新增了 BF16 数据类型支持），但其他证据仍引用旧值（如参数校验代码中未包含 BF16），且差异无法用注释滞后等原因解释
  - 两个同等权威的证据对同一事实给出**明确相反**的描述（如一个明确支持 INT8，另一个明确排除了 INT8）
  - 证据要求的修改与文档中其他引用不一致，重新生成无法同时满足所有证据
- 不同来源的证据对同一事实给出不同的值（如参数名、数据类型支持列表、返回值等），仅当差异**明确且不可调和**、无法用表述详略/注释滞后/作用域不同等合理原因解释时，才标记为 evidence_conflict；
即使依据优先级规则可以选出"更权威"的一方，此类明确冲突仍须标记。因为这种不一致通常意味着代码修改不完整，文档无法确定应该采用哪个值
- **若冲突不能完全确定**（如差异可能仅因表述详略不同、注释未同步、不同文件关注点不同），**不要标记** evidence_conflict，视为正常证据差异，按优先级规则选用值即可
- 若判断为 evidence_conflict，need_change 仍可为 true（尝试生成），但需在 conflict_details 中说明冲突点
- conflict_details 中的每一条冲突描述都**必须**注明对应的源文件名（如 aclnn_abs.h），让读者能直接定位到具体文件

### 请严格按以下格式输出：

<analysis>
{{
    "need_change": true/false,
    "reason": "判断理由（如：证据与原文一致无需修改 / 证据间存在冲突无法确定正确内容 / 需要修改xxx）",
    "evidence_conflict": false,
    "conflict_details": []
}}
</analysis>

<changes>
（变更点列表，JSON数组。若 need_change 为 false，输出空数组 []）
[
  {{"type": "新增/删除/修改", "location": "位置", "content": "变更内容"}}
]
</changes>

<document>
（修改后的完整文档，纯markdown文本。若 need_change 为 false，直接输出原文）
</document>
"""

        debug(f"[Generating]\n{prompt}")
        raw = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是一个增量文档更新助手。你的核心任务是：代码发生了变更，原文档基于旧代码编写，"
                        "证据来自变更后的新代码。你必须用证据中的新事实覆盖原文中受影响的旧值。"
                        "证据中的事实值（维度、数据类型、取值范围、格式等）始终高于原文和模板中的旧值。"
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        )

        analysis_match = re.search(r"<analysis>(.*?)</analysis>", raw, re.DOTALL)
        doc_match = re.search(r"<document>(.*?)</document>", raw, re.DOTALL)
        changes_match = re.search(r"<changes>(.*?)</changes>", raw, re.DOTALL)

        generated_text = doc_match.group(1).strip() if doc_match else raw
        changes = []
        if changes_match:
            try:
                changes = json.loads(changes_match.group(1))
            except json.JSONDecodeError:
                changes = []

        analysis = {}
        if analysis_match:
            try:
                analysis = json.loads(analysis_match.group(1))
            except json.JSONDecodeError:
                analysis = {}

        return ModificationResult(
            generated_text=generated_text,
            evidence_usage=[],
            modification_summary={
                "summary": f"基于 {len(evidences) if evidences else 0} 个证据生成",
                "changes": changes,
                "analysis": analysis,
            },
        )
