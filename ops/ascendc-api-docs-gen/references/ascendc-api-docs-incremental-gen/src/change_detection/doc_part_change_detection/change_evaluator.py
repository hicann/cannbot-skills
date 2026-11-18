# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""LLM-based change evaluator to filter out trivial/small changes that don't require documentation updates."""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from src.generation_validation.config import format_doc_part
from src.llm.client import LLMClient
from src.logger import debug, error, info, warning

_TILING_FILE_PATTERN = re.compile(r"(/tiling/.*\.(cpp|h)|tiling.*\.(cpp|h)|tilling\.h)")


def _is_tiling_file(file_path: str) -> bool:
    return bool(_TILING_FILE_PATTERN.search(file_path.replace("\\", "/")))


SYSTEM_PROMPT = """你是一个专业的代码与文档关联性分析专家。

你的任务是判断用户修改代码后，是否需要同步更新对应文档部分（doc part/subpart）。

你需要基于以下信息做出判断：
1. 用户实际修改了哪些文件的哪些内容（git diff格式的变更详情）
2. 变更所影响的文档部分（doc_part）
3. 文件与文档部分的映射关系

## 重要：分析 diff 内容
- `type: added` 表示新增文件，diff 显示新增的内容
- `type: modified` 表示修改文件，diff 显示变更前后对比
- `type: deleted` 表示删除文件，diff 显示删除的内容

**必须根据 diff 的实际内容判断影响，不要仅凭文件名猜测。**

## 特别关注：产品支持情况（support）
以下变更会影响"产品支持情况"：
- 新增某个架构（如 npu_arch_2201、npu_arch_3510）的实现文件 → 该架构现在支持此 API
- 删除某个架构的实现文件 → 该架构不再支持此 API
- 修改架构条件编译（如 `#ifdef __NPU_ARCH__`）→ 可能影响支持情况

对于每个doc_part，你需要判断"是否有必要修改对应文档"，输出格式为JSON数组。

判断标准（默认 keep: true，仅以下明确情况才标记 keep: false）：

keep: false 的情况（仅限以下，其他一律 keep: true）：
- 仅修改了空白字符、缩进、换行
- 仅修改了代码注释内容
- 仅添加了调试日志（如 printf、LOG）
- 仅修改了函数体内部实现逻辑，且函数声明（函数名、参数名、参数类型、返回值）完全未变

keep: true 的情况（包括但不限于）：
- 函数名发生变化（如 aclnnXxx → aclnnXxx_new）
- 函数参数名发生变化（如 out → out_new）
- 函数参数类型或数量变化
- 返回值类型变化
- 新增或删除函数
- 头文件（.h）中任何声明级别的变化
- 约束条件、支持的数据类型、支持的格式等变化
- 新增或删除子函数
- 示例代码（example）文件内容变化
- 任何用户在文档中能看到的内容发生变化

## doc_part 代号与文档章节的对应关系
- "support" → "## 产品支持情况"（产品/芯片/架构支持矩阵）
- "summary" → "## 功能说明"（算子功能描述、算法说明）
- "impl_principle" → "## 实现原理"（实现原理说明）
- "function_proto" → "## 函数原型"（接口声明、函数签名）
- "para_clarify" → "## 参数说明"（参数描述与说明）
- "return_clarify" → "## 返回值说明"（返回值描述）
- "constraint" → "## 约束说明"（数据类型、shape、取值范围等约束条件）
- "example" → "## 调用示例"（调用示例代码）

## constraint（约束说明）章节的特殊判断规则
约束说明章节记录**除单参数自身约束外**的全部约束：单个参数自身的约束（取值范围、数据类型、默认值）就近写在参数说明表格，不写入约束章节；其余约束（多参数组合约束、不针对具体参数的接口级约束）都写入约束章节。判断代码变更是否影响
constraint：
- 单个参数自身的约束变化（如某参数自身的取值范围、数据类型、默认值变化）→ keep: false（就近写在参数说明表格，不动约束章节）
- 其余约束变化 → keep: true。典型场景：
  - 多参数/跨参数组合约束：
    - 源操作数与目的操作数的地址复用约束（如 dst 与 src 空间是否可复用）
    - 多个参数组合使用的限制（如某参数仅在另一参数为特定值/类型时才支持，例如 repeat 参数仅在 dst/src 为 float 类型时可用）
    - 输入shape的跨参数维度限制
  - 接口级约束（不针对具体参数）：
    - 操作数地址对齐要求
    - 调用后是否需要恢复环境（调用其他API）
    - 开发方式约束（如 --enable-simt 编译选项限制）
    - 实验特性/稳定性约束
    - 性能相关约束
    - 接口适用的 AI Core 类型及调用行为（如某函数仅 AIV 可用，在不支持的核心类型上返回0）
- 边界：芯片/产品级支持矩阵归「产品支持情况」章节（support），不在此判断；同芯片内核心类型（AIC/AIV）适用性及调用行为属接口级约束，归本章节

注意：只评估文档内容是否需要变更，不考虑代码正确性。如果不确定，标记为 keep: true。"""


USER_PROMPT_TEMPLATE = """请分析以下代码变更，判断每个affected doc_part是否需要更新对应文档。

## 文档内容
``` markdown
{doc_content}
```

## 变更详情（changed_detail）
{changed_detail}

## 受影响的doc_part列表
{affected_subparts}

## 文件与doc_part映射关系
{file_subpart_mapping}

请按照以下JSON格式返回结果：
```json
{{
  "evaluation": [
    {{
      "doc_part": "support",
      "keep": false,
      "reason": "只修改了注释和换行，不影响文档内容"
    }},
    {{
      "doc_part": "constraint",
      "keep": true,
      "reason": "函数参数从3个变为4个，约束条件发生变化"
    }}
  ]
}}
```

其中：
- "doc_part": 需要评估的doc_part名称
- "keep": true表示需要修改对应文档，false表示不需要
- "reason": 简要说明判断理由

如果某个doc_part没有受到任何实质性影响，保持其keep为false。"""


def _filter_mapping_for_prompt(
    file_subpart_mapping: Dict[str, List],
    affected_subparts: List[str],
) -> Dict[str, List[str]]:
    affected_set = set(affected_subparts)
    filtered = {}
    for doc_part, files in file_subpart_mapping.items():
        if doc_part not in affected_set:
            continue
        file_names = []
        for entry in files:
            if isinstance(entry, dict):
                file_names.append(entry.get("file_name", ""))
            else:
                file_names.append(entry)
        filtered[doc_part] = file_names
    return filtered


def _build_user_prompt(
    doc_content: str,
    changed_detail: Dict[str, Dict],
    affected_subparts: List[str],
    file_subpart_mapping: Dict[str, List],
) -> str:
    changed_detail_json = json.dumps(changed_detail, ensure_ascii=False, indent=2)
    affected_subparts_json = json.dumps(affected_subparts, ensure_ascii=False, indent=2)
    filtered_mapping = _filter_mapping_for_prompt(file_subpart_mapping, affected_subparts)
    file_subpart_json = json.dumps(filtered_mapping, ensure_ascii=False, indent=2)

    prompt = USER_PROMPT_TEMPLATE.format(
        doc_content=doc_content,
        changed_detail=changed_detail_json,
        affected_subparts=affected_subparts_json,
        file_subpart_mapping=file_subpart_json,
    )
    debug(f"LLM user prompt:\n{prompt}")
    return prompt


def evaluate_changes(
    doc_content: str,
    affected_subparts: List[str],
    changed_detail: Dict[str, Dict],
    file_subpart_mapping: Dict[str, List],
) -> Tuple[List[str], Dict[str, Dict]]:
    """
    使用LLM判断哪些affected_subparts真正需要更新文档。

    Args:
        affected_subparts: run_detection返回的doc_subpart_todo_list
        changed_detail: file_change_detector返回的详细变更信息
        file_subpart_mapping: 文件与doc_part的映射关系

    Returns:
        verified_todo_list: LLM验证后确认真实需要更新的doc_part列表
        evaluation_reasons: 所有doc_part的评估理由，格式为 {doc_part: {"keep": bool, "reason": str}}
    """
    llm_client = LLMClient()

    user_prompt = _build_user_prompt(doc_content, changed_detail, affected_subparts, file_subpart_mapping)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = llm_client.chat_with_json(messages)
        if not isinstance(response, dict):
            warning(f"LLM返回了非字典类型: {type(response)}")
            fallback_reasons = {p: {"keep": True, "reason": "LLM返回格式异常，默认保留"} for p in affected_subparts}
            return list(affected_subparts), fallback_reasons
        evaluation = response.get("evaluation", [])
        if evaluation is None:
            evaluation = []

        verified_todo_list = []
        evaluation_reasons = {}
        for item in evaluation:
            doc_part = item.get("doc_part", "")
            keep = item.get("keep", True)
            reason = item.get("reason", "")

            evaluation_reasons[doc_part] = {"keep": keep, "reason": reason, "source": "general"}
            if keep:
                verified_todo_list.append(doc_part)
                info(f"[保留] {format_doc_part(doc_part)}: {reason}")
            else:
                info(f"[跳过] {format_doc_part(doc_part)}: {reason}")

        for p in affected_subparts:
            if p not in evaluation_reasons:
                evaluation_reasons[p] = {"keep": True, "reason": "LLM未评估，默认保留"}
                verified_todo_list.append(p)

        info(
            f"LLM评估完成: 原始{len(affected_subparts)}个doc_part, "
            f"保留{len(verified_todo_list)}个, 过滤{len(affected_subparts) - len(verified_todo_list)}个"
        )
        return verified_todo_list, evaluation_reasons

    except Exception as e:
        import traceback

        error(f"LLM评估失败，返回全部affected_subparts: {e}\n{traceback.format_exc()}")
        fallback_reasons = {p: {"keep": True, "reason": f"LLM评估失败: {e}"} for p in affected_subparts}
        return list(affected_subparts), fallback_reasons


# 门面接口守卫：共享/聚合头文件中新增的对外接口，若其函数体转发调用
# {api_name}_impl（本文档 API 的底层实现，如 PR8 的
# asc_get_cluster_id() { return asc_get_core_id_impl(); }），
# 则它是本 API 的关联门面/变体，属于本文档范围，函数原型/功能说明/
# 返回值说明必须更新。LLM 按"函数名不同即无关"过滤会漏掉此类变更，
# 此处用确定性规则覆盖（按 diff 内容归因，不按文件名归因，
# 聚合文件中其他 API 的变更不会命中）。
FACADE_PARTS = ("function_proto", "summary", "return_clarify")


def _detect_facade_interfaces(added_text: str, api_name: str) -> Dict[str, str]:
    """在 diff 新增代码中找门面函数：新函数（≠api_name）的函数体调用 {api_name}_impl(。

    从每处 {api_name}_impl( 调用向前取窗口找最近的函数定义头 name(...) {...，
    定义头与调用之间不得出现 }（保证在同一函数体内，不误跨到前一个函数）。
    返回 {门面函数名: 定义所在行文本}。
    """
    call_pat = re.compile(rf"\b{re.escape(api_name)}_impl\s*\(")
    def_pat = re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{")
    lines = added_text.split("\n")
    facades: Dict[str, str] = {}
    for m in call_pat.finditer(added_text):
        window = added_text[max(0, m.start() - 500): m.start()]
        head = None
        for d in def_pat.finditer(window):
            head = d  # 取窗口内最后一个（最近的）定义头
        if head is None:
            continue
        if "}" in window[head.end():]:
            continue
        name = head.group(1)
        if name == api_name or name in facades:
            continue
        line_no = added_text[: head.start()].count("\n")
        facades[name] = lines[line_no].strip() if line_no < len(lines) else head.group(0)
    return facades


def _facade_guard(
    changed_detail: Dict[str, Dict],
    api_name: Optional[str],
    affected_subparts: List[str],
    verified_todo_list: List[str],
    all_reasons: Dict[str, Dict],
    console: Optional[Any],
) -> None:
    """确定性守卫：检测到门面接口时强制保留相关 doc_parts，覆盖 LLM 的 keep=false。"""
    if not api_name:
        return
    facades: Dict[str, str] = {}
    for file_path, detail in (changed_detail or {}).items():
        if not str(file_path).endswith((".h", ".hpp")):
            continue
        diff = (detail or {}).get("diff", "") or ""
        added_lines = [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        if not added_lines:
            continue
        for name, code in _detect_facade_interfaces("\n".join(added_lines), api_name).items():
            facades.setdefault(name, code)
    if not facades:
        return
    names = ", ".join(sorted(facades))
    info(f"门面接口守卫: 变更新增转发 {api_name}_impl 的对外接口 [{names}]，强制保留 {list(FACADE_PARTS)}")
    if console is not None:
        console.print(
            f"  [yellow]门面接口守卫[/yellow]: 新增门面接口 [{names}]"
            f"（函数体转发 {api_name}_impl），强制保留: {list(FACADE_PARTS)}"
        )
    # 定义行原样随 reason 下发：证据锚点通常只有声明列表（无语义），
    # writer 需要"转发调用本 API 底层实现"这行代码才能推导新接口的功能/返回值语义
    code_evidence = "；".join(f"`{c}`" for c in sorted(facades.values()))
    part_actions = {
        "function_proto": "函数原型需补充新接口的声明",
        "summary": "功能说明需补充新接口的功能描述（新接口转发调用本 API 的底层实现，行为语义与本 API 一致，可由接口名与该转发实现推导）",
        "return_clarify": "返回值说明需补充新接口的返回值描述",
    }
    for part in FACADE_PARTS:
        if part not in affected_subparts:
            continue
        reason = (
            f"检测到新增门面接口 {names}（定义于头文件、函数体转发调用 {api_name}_impl），"
            f"新增代码：{code_evidence}。该接口属于本文档 API 的关联接口，"
            f"{part_actions.get(part, '需将新接口纳入本章节')}"
        )
        entry = all_reasons.get(part)
        if entry and entry.get("keep"):
            entry["reason"] = f"{entry['reason']}；{reason}"
            continue
        all_reasons[part] = {"keep": True, "reason": reason, "source": "facade_guard"}
        if part not in verified_todo_list:
            verified_todo_list.append(part)


# ---------------------------------------------------------------------------
# 架构接线类型守卫：检测"新增 arch 门控接线 + 本 API 重载签名"的变更。
# 典型场景（PR5）：reg_vector.h 新增 __NPU_ARCH__==2201 分支 include 新文件
# reg_vector_impl.h，后者仅含 asc_abs(vector_float&...) 一个对外重载 —— 该 arch
# 下本 API 本次仅新增 float 类型能力（目录下存量实现如 asc_abs_impl.h 的 half
# 版本不在本次接线链上，不属于本次能力）。函数原型必须注明"xx型号的产品仅支持
# xxx类型"。LLM 评估常因"函数声明未变"判 keep=false 漏更新（PR5 实证），
# 此处用确定性规则覆盖。仅看 diff 新增行（变更增量），不扫仓库全量状态，
# 避免把存量实现误当本次新增。
# ---------------------------------------------------------------------------
_ARCH_PATH_RE = re.compile(r"(?:npu_arch_|dav_)(\d+)")

# __ubuf__ 底层指针类型 → 对外 vector 重载类型名（两种 impl 写法归一到对外口径）
_UBUF_TYPE_TO_VECTOR = {
    "int8_t": "vector_int8_t",
    "int16_t": "vector_int16_t",
    "int32_t": "vector_int32_t",
    "int64_t": "vector_int64_t",
    "half": "vector_half",
    "bfloat16_t": "vector_bfloat16_t",
    "float": "vector_float",
}


def _scan_arch_type_support(changed_detail: Dict[str, Dict], api_name: str) -> Dict[str, list]:
    """从 diff 新增内容提取 {arch: [本次新增支持的类型]}。

    只统计路径含 npu_arch_X / dav_X 的头文件中、diff 新增行里本 API 的重载签名，
    兼容三种写法并归一到对外 vector 类型口径：
      - 对外形态：asc_abs(vector_float& dst, ...)
      - impl 形态：asc_abs_impl(vector_int8_t& dst, ...)
      - 底层指针：asc_abs_impl(__ubuf__ half* dst, ...) → vector_half
    mask 参数（vector_bool）不属于数据类型维度，不计入。
    """
    if not api_name:
        return {}
    sig_pat = re.compile(rf"\b{re.escape(api_name)}(?:_impl)?\s*\(([^)]*)\)", re.S)
    result: Dict[str, Set[str]] = {}
    for file_path, detail in (changed_detail or {}).items():
        path_match = _ARCH_PATH_RE.search(str(file_path).replace("\\", "/"))
        if not path_match or not str(file_path).endswith((".h", ".hpp")):
            continue
        diff = (detail or {}).get("diff", "") or ""
        added = "\n".join(ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
        if not added:
            continue
        types = result.setdefault(path_match.group(1), set())
        for sig in sig_pat.findall(added):
            for v in re.findall(r"vector_(\w+)", sig):
                if v != "bool":
                    types.add(f"vector_{v}")
        for ubuf in re.findall(r"__ubuf__\s+(\w+)\*", added):
            mapped = _UBUF_TYPE_TO_VECTOR.get(ubuf)
            if mapped:
                types.add(mapped)
    return {arch: sorted(ts) for arch, ts in result.items() if ts}


def _resolve_arch_products(arch: str) -> list:
    """arch 号 → 产品名列表（resources/chip_map.yaml 的 npu_arch_mapping）。

    与支持矩阵同一事实源；解析失败返回空列表，守卫理由退化为 npu_arch_{arch} 表述。
    """
    try:
        import yaml

        chip_map_path = Path(__file__).resolve().parents[3] / "resources" / "chip_map.yaml"
        with open(chip_map_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        products = []
        for entry in raw.get("npu_arch_mapping", []) or []:
            if str(entry.get("arch", "")).strip() != arch:
                continue
            for p in re.split(r"[、，]", str(entry.get("product", ""))):
                p = p.strip()
                if p:
                    products.append(p)
        return sorted(set(products))
    except Exception as e:
        warning(f"解析 chip_map.yaml 失败，守卫理由退化为 arch 号表述: {e}")
        return []


def _arch_type_support_guard(
    changed_detail: Dict[str, Dict],
    api_name: Optional[str],
    affected_subparts: List[str],
    verified_todo_list: List[str],
    all_reasons: Dict[str, Dict],
    console: Optional[Any],
) -> None:
    """确定性守卫：本次变更新增某 arch 的本 API 重载接线时，强制保留 function_proto，
    并在 reason 中携带结构化的"型号 → 仅支持类型"信息（经 change_reason 通道进入
    writer prompt 的"变更检测结论"章节），覆盖 LLM 因"函数声明未变"给出的
    keep=false（PR5 场景实证漏更新）。
    """
    if not api_name:
        return
    arch_types = _scan_arch_type_support(changed_detail, api_name)
    if not arch_types:
        return
    parts = []
    for arch in sorted(arch_types):
        products = _resolve_arch_products(arch)
        product_names = "、".join(products) if products else f"npu_arch_{arch}"
        types = "、".join(arch_types[arch])
        parts.append(f"{product_names} 仅支持 {types} 类型")
    detail_text = "；".join(parts)
    reason = (
        f"检测到新增架构接线（头文件 __NPU_ARCH__ 门控 include 新实现文件）：{detail_text}。"
        f"该信息来自本次变更的新增代码，函数原型需在对应原型处注明型号支持差异"
        f"（格式如：xx型号的产品仅支持xxx类型），未涉及的原型不注记"
    )
    if console is not None:
        console.print(f"  [yellow]架构接线守卫[/yellow]: {detail_text}，强制保留 function_proto")
    info(f"架构接线类型守卫: {detail_text}，强制保留 function_proto")
    entry = all_reasons.get("function_proto")
    if entry and entry.get("keep"):
        entry["reason"] = f"{entry['reason']}；{reason}"
        return
    all_reasons["function_proto"] = {"keep": True, "reason": reason, "source": "arch_guard"}
    if "function_proto" not in verified_todo_list:
        verified_todo_list.append("function_proto")


TILING_SYSTEM_PROMPT = """你是一个专业的昇腾算子 tiling 代码变更分析专家。

你的任务是分析 tiling 文件的代码变更（git diff 格式），判断每处变更是否影响文档，以及影响哪些文档章节（doc_part）。

## 可用的 doc_part 列表
- "constraint": 约束说明（数据类型、shape、取值范围等约束条件）
- "sub_function_hash": 子函数列表/签名变化
- "support": 产品/架构支持情况
- "summary": 功能说明/算法描述
- "function_proto": 函数原型（接口声明）
- "example": 调用示例代码

## 总体原则
仅修改函数体内部实现逻辑且对外语义不变的变更（如重构、变量重命名、拆分函数、调整判断顺序），不影响文档。如果不确定，标记为 keep: true。

## 判断规则（严格按以下规则分类）

### 1. OP_TILING_CHECK / OP_LOGE 变更
- 约束条件变化（新增/修改/删除 check 条件、放宽/收紧约束）→ "constraint"
- 涉及到新增/修改错误码 → "sub_function_hash"
- 同时涉及约束变化和错误码 → "constraint" + "sub_function_hash"
- 修改 check 但约束语义不变（重构拆分、变量重命名、调整判断顺序）→ 不影响文档

### 2. 业务容器变更
承载业务语义的集合（*_MAP/*_map/*_SET/*_set/std::vector 白名单/std::array 配置/unordered_map 等）：
- 新增或删除条目 → 按容器类别影响对应 doc_part
- 仅重构容器结构但内容不变（如 vector 改 array、调整初始化方式）→ 不影响文档
容器类别与 doc_part 映射：
- 量化/数据类型相关（如 QUANT_MODE_MAP/DTYPE_MAP/SUPPORTED_DTYPE_SET 等）→ "sub_function_hash" + "constraint"
- 架构/产品相关（如 ARCH_MAP/SUPPORTED_ARCH_SET 等）→ "support"
- 功能/算法相关（如 OP_MODE_MAP/ALGO_SET/FEATURE_SET 等）→ "summary" + "constraint"
- 其他业务集合 → 不影响文档

### 3. CheckTensorShape* 变更
- 基本范围变更（取值区间、维度数限制）→ "sub_function_hash"
- 特定场景约束（如 quantMode=4 且 FP4 时 H 偶数）→ "constraint"
- 仅重构实现但检查逻辑不变 → 不影响文档

### 4. constexpr 常量变更
- 新增或修改 constexpr 常量，且该常量表示数值约束（如最大维度数、最小shape等）→ "constraint"
- 仅内部使用的常量 → 不影响文档

### 5. enum 变更
- 新增 enum 值，且对应新功能/新模式 → 不影响文档
- 新增 enum 值改变了已有约束语义 → "constraint"

### 6. 架构条件编译
- 新增或删除 #ifdef ASCEND_xxx / 架构条件编译分支 → "support"
- 仅调整已有分支内部实现逻辑 → 不影响文档

### 7. 仅内部重构
- 函数拆分、enum 重命名但语义不变、代码格式调整 → 不影响文档

## 输出要求
对每个受影响的 doc_part，输出 keep: true 并说明理由。如果某类变更不影响文档，输出 keep: false。
如果不确定，标记为 keep: true。"""

TILING_USER_PROMPT_TEMPLATE = """请分析以下 tiling 文件的代码变更，判断每处变更影响哪些文档章节。

## 变更的 tiling 文件及 diff
{tiling_diffs}

请按照以下JSON格式返回结果：
```json
{{
  "evaluation": [
    {{
      "doc_part": "constraint",
      "keep": true,
      "reason": "新增了 OP_TILING_CHECK 检查 inputDtype 必须为 FLOAT，新增约束条件"
    }},
    {{
      "doc_part": "support",
      "keep": false,
      "reason": "无架构条件编译变更"
    }}
  ]
}}
```"""


def analyze_tiling_changes(
    changed_detail: Dict[str, Dict],
    console: Optional[Any] = None,
) -> Tuple[Set[str], Dict[str, Dict]]:
    tiling_diffs = {}
    for file_path, detail in changed_detail.items():
        if _is_tiling_file(file_path):
            tiling_diffs[file_path] = detail

    if not tiling_diffs:
        info("无 tiling 文件变更，跳过 tiling 分析")
        return set(), {}

    info(f"检测到 {len(tiling_diffs)} 个 tiling 文件变更: {list(tiling_diffs.keys())}")

    diffs_text = ""
    for file_path, detail in tiling_diffs.items():
        diffs_text += f"\n### {file_path} (type: {detail.get('type', 'unknown')})\n"
        diffs_text += f"```diff\n{detail.get('diff', '(no diff)')}\n```\n"

    user_prompt = TILING_USER_PROMPT_TEMPLATE.format(tiling_diffs=diffs_text)
    messages = [
        {"role": "system", "content": TILING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    llm_client = LLMClient()
    try:
        response = llm_client.chat_with_json(messages)
        if not isinstance(response, dict):
            warning(f"Tiling LLM返回了非字典类型: {type(response)}")
            return set(), {}

        evaluation = response.get("evaluation", [])
        if evaluation is None:
            evaluation = []

        affected = set()
        all_reasons = {}
        for item in evaluation:
            doc_part = item.get("doc_part", "")
            keep = item.get("keep", False)
            reason = item.get("reason", "")
            if keep and doc_part:
                affected.add(doc_part)
                all_reasons[doc_part] = {"keep": keep, "reason": reason, "source": "tiling"}
                info(f"[TILING-保留] {format_doc_part(doc_part)}: {reason}")
            else:
                info(f"[TILING-跳过] {format_doc_part(doc_part)}: {reason}")

        info(f"Tiling 分析完成: {len(affected)} 个 doc_part 受影响: {affected}")
        return affected, all_reasons

    except Exception as e:
        import traceback

        error(f"Tiling LLM分析失败: {e}\n{traceback.format_exc()}")
        return set(), {}


# ===================== 架构作用域 & 语义变更守卫（确定性） =====================
#
# 这两类守卫针对业务反馈里“LLM 按 diff 字面判断 keep=false”的系统性漏判：
#   1. 架构作用域：一个 PR 只改 3510，却把 2201 的资料一并改写；
#   2. 语义变更：改了常量/取值范围、或“某个能力从不支持变为支持”，
#      在 diff 层面只是 impl 内部改动，但文档必须跟着改。

_DOC_LABEL_TO_PART = {
    "## 产品支持情况": "support",
    "## 功能说明": "summary",
    "## 实现原理": "impl_principle",
    "## 函数原型": "function_proto",
    "## 参数说明": "para_clarify",
    "## 返回值说明": "return_clarify",
    "## 约束说明": "constraint",
    "## 调用示例": "example",
}

_CONST_TOKEN = re.compile(r"2\^\d+|\b\d{3,}\b")
_CAMEL_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9_]{3,}\b")


def arch_scope_from_changes(changed_detail: Dict[str, Dict]) -> Dict[str, list]:
    """从改动文件路径提取本次变更涉及的架构 {'3510': [files], ...}。"""
    scope: Dict[str, list] = {}
    for f in changed_detail or {}:
        rel = str(f).replace("\\", "/")
        m = _ARCH_PATH_RE.search(rel)
        if m:
            scope.setdefault(m.group(1), []).append(rel)
    return scope


def _doc_arch_labels(api_name: Optional[str]) -> Optional[str]:
    """文档名里的架构归属（asc_copy_l0c2l1_arch_2201 → '2201'）。"""
    m = re.search(r"arch_(\d+)", api_name or "")
    return m.group(1) if m else None


def _iter_diff_lines(changed_detail: Dict[str, Dict]):
    added, removed = [], []
    for _, detail in (changed_detail or {}).items():
        diff = (detail or {}).get("diff", "") or ""
        for ln in diff.splitlines():
            if ln.startswith("+") and not ln.startswith("+++"):
                added.append(ln[1:])
            elif ln.startswith("-") and not ln.startswith("---"):
                removed.append(ln[1:])
    return added, removed


def _doc_sections(doc_content: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for label, part in _DOC_LABEL_TO_PART.items():
        m = re.search(
            rf"(?m)^{re.escape(label)}(?:<a[^>]*>\s*</a>)?[ \t]*$\n?(.*?)(?=^## |\Z)", doc_content or "", re.S
        )
        if m:
            out[part] = m.group(1)
    return out


def _force_keep(
    part: str,
    reason: str,
    tag: str,
    verified_todo_list: List[str],
    all_reasons: Dict[str, Dict],
    console: Optional[Any] = None,
) -> None:
    entry = all_reasons.get(part)
    if entry and entry.get("keep"):
        if reason not in entry.get("reason", ""):
            entry["reason"] = f"{entry.get('reason', '')}；{reason}"
        return
    all_reasons[part] = {"keep": True, "reason": reason, "source": tag}
    if part not in verified_todo_list:
        verified_todo_list.append(part)
    if console is not None:
        console.print(f"  [yellow]语义变更守卫[/yellow]: {part} ← {reason[:80]}")


def _semantic_change_guard(
    changed_detail: Dict[str, Dict],
    doc_content: str,
    verified_todo_list: List[str],
    all_reasons: Dict[str, Dict],
    console: Optional[Any] = None,
    extra_added_text: str = "",
    extra_removed_text: str = "",
) -> None:
    """常量/取值范围、能力否定表述、标识符重命名 三类语义变更 → 强制保留对应章节。

    这三类变化在 diff 里表现为“impl 内部改动”，LLM 常判 keep=false；
    但它们都会让文档里的某个数字或某句“不支持 X”失效。
    """
    sections = _doc_sections(doc_content)
    if not sections:
        return
    added, removed = _iter_diff_lines(changed_detail)
    added_text, removed_text = "\n".join(added), "\n".join(removed)
    # 叠加 PR 全量代码 diff：枚举/类型的改名常发生在文档快照未收录的头文件里
    if extra_added_text:
        added_text = f"{added_text}\n{extra_added_text}"
    if extra_removed_text:
        removed_text = f"{removed_text}\n{extra_removed_text}"

    # 1) 常量/取值范围：按“行骨架”配对删除/新增行，找出被改掉的数值。
    #    直接做文件级集合差会把同文件其它同名数值（如其它参数的 2^21）抵消掉，
    #    故先把数字抹成 # 得到骨架，同骨架的删除行与新增行才配对比较。
    doc_nums = set()
    for body in sections.values():
        doc_nums |= set(re.findall(r"\d{2,}", body))

    def _skel(s: str) -> str:
        return re.sub(r"\d+", "#", s.strip())

    added_by_skel: Dict[str, list] = {}
    for ln in added:
        added_by_skel.setdefault(_skel(ln), []).append(ln)

    for ln in removed:
        hit = set(re.findall(r"\d{2,}", ln)) & doc_nums
        if not hit:
            continue
        new_nums = set()
        for a in added_by_skel.get(_skel(ln), []):
            new_nums |= set(re.findall(r"\d{2,}", a)) - hit
        if not new_nums:
            continue
        for part, body in sections.items():
            if any(n in body for n in hit):
                _force_keep(
                    part,
                    f"代码中 {','.join(sorted(hit))} 本次改为 {','.join(sorted(new_nums))}，原文仍引用旧值",
                    "semantic_const",
                    verified_todo_list,
                    all_reasons,
                    console,
                )

    # 2) 能力否定表述：新增的支持类型/能力，若文档写着“不支持 X” → 该章节需更新
    gained = set(_CAMEL_TOKEN.findall(added_text)) - set(_CAMEL_TOKEN.findall(removed_text))
    for tok in sorted(gained):
        for part, body in sections.items():
            if re.search(rf"不支持\s*`?{re.escape(tok)}`?", body):
                _force_keep(
                    part,
                    f"本次变更使 {tok} 受支持，原文“不支持 {tok}”已失效",
                    "semantic_capability",
                    verified_todo_list,
                    all_reasons,
                    console,
                )

    # 3) 标识符重命名（NEAREST_ZERO → NEAREST_AWAY）：文档中引用旧名处需同步
    #    注意不能用“删除-新增”集合差：旧名常作为**兼容别名**保留在新增行里
    #    （如 `NEAREST_ZERO = NEAREST_AWAY // 兼容旧名`），集合差会把它抵消掉。
    #    判据改为：旧名出现在删除行 + 文档里已有该旧名 + 存在同前缀且文档里没有的新名。
    doc_tokens = set(_CAMEL_TOKEN.findall(doc_content))
    added_tokens = set(_CAMEL_TOKEN.findall(added_text))

    def _still_valid(tok: str) -> bool:
        """旧名若在新增代码里仍作为正常取值出现（非兼容说明），说明它没被改名。"""
        hit_lines = [ln for ln in added_text.splitlines() if re.search(rf"\b{re.escape(tok)}\b", ln)]
        if not hit_lines:
            return False
        alias_only = all(
            re.search(r"兼容|废弃|deprecated|compatible|equiv|alias|等同|旧名|不推荐", ln, re.I) for ln in hit_lines
        )
        return not alias_only

    lost = {t for t in doc_tokens if re.search(rf"\b{re.escape(t)}\b", removed_text) and not _still_valid(t)}
    for old in sorted(lost):
        prefix = old.split("_")[0]
        cands = sorted(t for t in added_tokens if t != old and t.split("_")[0] == prefix and t not in doc_content)
        if not cands:
            continue
        for part, body in sections.items():
            if old in body:
                _force_keep(
                    part,
                    f"代码中标识符 {old} 已更名为 {cands[0]}，文档引用需同步",
                    "semantic_rename",
                    verified_todo_list,
                    all_reasons,
                    console,
                )
                # 记录重命名对，供回填阶段的守卫区分“合法更新”与“编造/覆盖”
                all_reasons.setdefault(part, {}).setdefault("renames", {})[old] = cands[0]


def _arch_npu_labels(archs: list[str]) -> set[str]:
    """npu_arch 号 -> 文档里的产品块标签（950 / A3 / 910b…）。

    与 postprocess._arch_to_labels 同源（chip_map.yaml + templates.yaml）；
    缺映射时返回空集，调用方只输出 arch 名，不猜。
    """
    try:
        from src.md_fill.postprocess import _arch_to_labels
    except Exception:  # pragma: no cover - 环境缺依赖时不影响主流程
        return set()
    try:
        return set(_arch_to_labels(list(archs)))
    except Exception:  # pragma: no cover
        return set()


def _arch_scope_guard(
    changed_detail: Dict[str, Dict],
    api_name: Optional[str],
    verified_todo_list: List[str],
    all_reasons: Dict[str, Dict],
    console: Optional[Any] = None,
) -> None:
    """把“本次变更只涉及哪些架构”写进变更理由，约束生成不要改其它产品的段落。"""
    scope = arch_scope_from_changes(changed_detail)
    if not scope:
        return
    doc_arch = _doc_arch_labels(api_name)
    if doc_arch and doc_arch in scope:
        scope = {doc_arch: scope[doc_arch]}
    labels = "、".join(sorted(f"npu_arch_{a}" for a in scope))
    # 文档里的 npu 注释块是按**产品**标注的（950 / A3 / 910b…），只给 arch 号
    # 会让模型把“3510 对应的 950 块”当成别的产品，从而不敢改该改的约束（6512 实测）。
    npu_labels = sorted(_arch_npu_labels(list(scope)))
    scope_text = labels + (f"（对应文档中的产品块：{'、'.join(npu_labels)}）" if npu_labels else "")
    note = (
        f"本次代码变更仅涉及 {scope_text}；标注为「其它产品」的 npu 注释块"
        f"不得改动或新增内容。"
        f"上面列出的产品块属于本次变更范围，需要改就可以改。"
    )
    if npu_labels:
        # 部分支持/单个函数从不支持转为支持时，矩阵口径不翻转，差异写进约束说明。
        note += (
            "另：## 产品支持情况 只表达产品对该接口的总体支持与否，"
            "不因单个函数或单条限制的变化而翻转；本次新增实现带来的差异"
            "（含“原先不支持、现已支持”或仅部分场景支持）应写入 ## 约束说明，"
            "原已失效的限制条应删除或改写，不要原样保留。"
        )
    for part in list(verified_todo_list):
        entry = all_reasons.get(part)
        if entry and entry.get("keep"):
            entry["reason"] = f"{entry.get('reason', '')}；{note}"
    if console is not None and verified_todo_list:
        console.print(f"  [dim]架构作用域: {labels}（已注入变更理由）[/dim]")


_UNIT_HINT = re.compile(
    r"(?<![A-Za-z0-9])\d+\s*VL(?![A-Za-z0-9])"  # 2VL / 1VL
    r"|(?<![A-Za-z0-9])\d+\s*字节"  # 256字节
    r"|(?<![A-Za-z0-9])\d+\s*(?:KB|MB)(?![A-Za-z0-9])"  # 32KB
)
# 常见通用类型，不作为“新概念”去检索关联文档
_TOKEN_STOP = {
    "LocalTensor",
    "GlobalTensor",
    "RegTensor",
    "Tensor",
    "DataCopy",
    "InitValue",
    "HighLowPart",
    "DataCopyParams",
    "DataCopyMVType",
    "LoopModeParams",
    "PipeBarrier",
    "SetFlag",
    "WaitFlag",
}
_ENUM_ASSIGN = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\s*=\s*([0-9]+)\b")
_RANGE_SHIFT = re.compile(r"(?:\w+)\s*<\s*\(?\s*1(?:ul|u|ULL|ull)?\s*<<\s*(\d+)")
_CAMEL_TYPE = re.compile(r"\b(Reg[A-Z][A-Za-z0-9_]+|[A-Z][a-z]+[A-Z][A-Za-z0-9_]+)\b")

# 数据通路枚举：UB_TO_OUT / OUT_TO_UB / OUT_TO_L1 …（“源_到_目的”命名）
_DATAPATH_ENUM = re.compile(r"^([A-Z][A-Z0-9]*)_TO_([A-Z0-9]+)$")


def _buf_display(token: str) -> str:
    """枚举段名 -> 文档缓冲区叫法；映射表在 resources/doc_rules.yaml。

    未知段名原样返回（不猜）；规则文件缺失时同样原样返回。
    """
    try:
        from src.md_fill.postprocess import load_doc_rules

        names = load_doc_rules().get("buffer_names") or {}
    except Exception:  # pragma: no cover - 环境缺依赖时不影响主流程
        names = {}
    return str(names.get(token, token)) if isinstance(names, dict) else token


_MUST_COVER_HEADER = "\n\n## 必须覆盖的事实要点（来自代码/关联文档，生成时不得遗漏）\n"


_DOC_SCAN_CACHE: Dict[tuple, list] = {}


def _scan_related_doc_facts(repo_root: str, tokens: list[str], doc_content: str, max_files: int = 5000) -> list[str]:
    """从同仓其它文档中抽取与 token 相关的“数字+单位”事实句（如 2VL）。

    同一 PR 内多篇文档共享 repo_root 与 tokens，扫描结果按 (repo_root, tokens)
    缓存 —— 否则每篇文档都要读几千个 md（实测单次约 34s）。
    """
    if not repo_root or not tokens:
        return []
    key = (str(repo_root), tuple(sorted(tokens)[:6]))
    raw = _DOC_SCAN_CACHE.get(key)
    if raw is None:
        raw = _scan_raw_facts(repo_root, tokens, max_files)
        _DOC_SCAN_CACHE[key] = raw
    # VL 类事实（如“2VL 搬运量”）优先：它才是类型能力的关键派生事实，
    # 不能被 DataCopyPad/32KB 这类通用条目挤掉配额
    raw = sorted(raw, key=lambda t: 0 if "VL" in str(t[1]).upper() else 1)
    facts: list[str] = []
    for clause, unit, src in raw:
        if unit in doc_content:
            continue
        fact = f"{clause[:120]}（来源：{src}）"
        if fact not in facts:
            facts.append(fact)
        if len(facts) >= 3:
            break
    return facts


def _scan_raw_facts(repo_root: str, tokens: list[str], max_files: int = 5000) -> list:
    """扫描 docs/**/*.md，返回 [(句子, 单位, 来源文件)]（不带文档级过滤，可缓存）。"""
    docs_root = Path(repo_root) / "docs"
    if not docs_root.is_dir():
        return []
    found: list = []
    scanned = 0
    for md in sorted(docs_root.rglob("*.md")):
        scanned += 1
        if scanned > max_files or len(found) >= 40:
            break
        try:
            if md.stat().st_size > 300_000:
                continue
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if not any(t in line for t in tokens):
                continue
            if not _UNIT_HINT.search(line):
                continue
            for clause in re.split(r"[。；;]", line):
                if not any(t in clause for t in tokens):
                    continue
                m = _UNIT_HINT.search(clause)
                if not m:
                    continue
                found.append((clause.strip(), m.group(0).strip(), md.name))
                break
    info(f"关联文档扫描: {scanned} 个文件，候选 {len(found)} 条（tokens={tokens[:3]}）")
    return found


def _added_lines_code_only(changed_detail: Dict[str, Dict]) -> list[str]:
    """只取 include/ 与 impl/ 下改动的新增行（跳过测试/示例/文档，避免噪声）。"""
    lines: list[str] = []
    for f, detail in (changed_detail or {}).items():
        rel = str(f).replace("\\", "/")
        if not (rel.startswith("include/") or rel.startswith("impl/")):
            continue
        diff = (detail or {}).get("diff", "") or ""
        lines.extend(ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    return lines


def _must_cover_points(
    changed_detail: Dict[str, Dict], doc_content: str, repo_root: Optional[str] = None, extra_code_text: str = ""
) -> list[str]:
    """抽取“本次变更引入、但文档还没写”的结构化事实要点。

    两类来源：
      A. 代码 diff（仅 include/ 与 impl/）：新增/修改的枚举值与常量
         （``OUT_TO_L1 = 2``）、取值范围上限（``1ul << 22``）；
      B. 同仓关联文档：代码新增的类型名（如 RegTraitNumTwo）在其它文档里
         已有的“数字+单位”事实（如“支持扩展存储 2VL”）。

    只保留文档里尚不存在的要点，最多 6 条。
    """
    added_text = "\n".join(_added_lines_code_only(changed_detail))
    if extra_code_text:
        # PR 全量代码改动：本文档快照常只收录部分文件，叠加后可覆盖
        # “类型/枚举定义在别的文件里”的情况（如 RegTraitNumTwo）。
        added_text = f"{added_text}\n{extra_code_text}"

    points: list[str] = []
    for name, value in _ENUM_ASSIGN.findall(added_text):
        if name in doc_content:
            continue
        points.append(f"枚举/常量 {name} = {value} 为本次新增，需在取值说明中列出")
        if _DATAPATH_ENUM.match(name):
            # 评审口径（#6620 的 PR #7261）：枚举值本身就是一条数据通路，
            # 参考 UB_TO_OUT / OUT_TO_UB 并列补一行即可——不要新增说明段落，
            # 也不要往「具体支持的数据通路」清单里加条目。
            points[-1] += "；该枚举是一条数据通路，按并列枚举值补充一行即可，不要新增说明段落或数据通路清单条目"
    for shift in sorted(set(_RANGE_SHIFT.findall(added_text))):
        token = f"2^{shift}"
        if token in doc_content:
            continue
        points.append(f"取值范围上限本次改为 {token}，涉及该参数的说明需同步")

    # Trait/Reg 类新概念优先检索（2VL 这类事实常挂在 RegTrait* 上）
    tokens = sorted(
        {t for t in _CAMEL_TYPE.findall(added_text) if len(t) >= 8 and t not in _TOKEN_STOP},
        key=lambda t: (0 if ("Trait" in t or t.startswith("Reg")) else 1, t),
    )
    if tokens and repo_root:
        # 文档里常用类型的前缀写法（RegTraitNumTwo → RegTrait），一并作为检索词
        expanded = set(tokens)
        for t in tokens:
            parts = re.findall(r"[A-Z][a-z0-9]*", t)
            if len(parts) >= 2:
                prefix = "".join(parts[:2])
                if len(prefix) >= 6:
                    expanded.add(prefix)
        points.extend(_scan_related_doc_facts(repo_root, sorted(expanded), doc_content))

    seen, uniq = set(), []
    for p in points:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:6]


_POINT_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}|\d+\s*VL|2\^\d+")
_SRC_SUFFIX = re.compile(r"（来源：[^）]*）")


def _parts_for_points(points: list[str], doc_content: str) -> list[str]:
    """按事实类型把要点路由到章节；归不了类的再按关键词在原文章节里找。

    精确路由：枚举取值 -> 参数说明；数据通路 -> 功能说明；数值/断言/寄存器类型
    -> 约束说明。混在一起时各回各家，不互相串门（6620 的枚举要点原先会被
    关键词平铺，把整张取值表塞进功能说明）。
    归不了类的（关联文档抽到的能力事实、取值范围上限）沿用关键词定位。
    """
    if not points:
        return []
    sections = _doc_sections(doc_content)
    kinds = {_point_kind(p) for p in points} - {""}
    rest = [p for p in points if not _point_kind(p)]
    picked: list[str] = []
    for kind, part in (("data_path", "summary"), ("enum", "para_clarify"), ("constraint", "constraint")):
        if kind in kinds and part in sections and part not in picked:
            picked.append(part)
    if not rest:
        return picked[:2] or ["summary", "constraint"]
    # 关联文档抽到的事实（带“（来源：）”后缀）一律落在文字章节：
    # 这类“能力/派生事实”不该写进示例，示例整节回退时也会把它一并丢掉。
    if any("（来源：" in p for p in rest):
        _text_parts = [s for s in ("constraint", "para_clarify", "summary") if s in sections and s not in picked]
        picked.extend(_text_parts[:2])
        return picked[:2] or ["summary", "constraint"]
    hit: list[str] = []
    for p in rest:
        text = _SRC_SUFFIX.sub("", p)
        keys = {k for k in _POINT_KEY.findall(text) if len(k) >= 4}
        for part, body in sections.items():
            if any(k in body for k in keys) and part not in picked:
                hit.append(part)
    # 承载“新增能力/取值变化”的章节优先，最多两节，控制重生成成本
    order = ["constraint", "para_clarify", "summary", "function_proto", "example"]
    picked.extend([p for p in order if p in hit])
    return picked[:2] or ["summary", "constraint"]


def _point_kind(point: str) -> str:
    """要点归类：data_path / enum / constraint；归不了类返回 ""。"""
    if point.startswith("数据通路"):
        return "data_path"
    if point.startswith("枚举/常量"):
        return "enum"
    if ("static_assert" in point) or ("带单位" in point) or ("RegTraitNum" in point):
        return "constraint"
    return ""


def _part_kind(part: str) -> str:
    """章节 -> 要点归类（与 _point_kind 对应）。"""
    return {"summary": "data_path", "para_clarify": "enum", "constraint": "constraint"}.get(part, "")


def _attach_must_cover(
    all_reasons: Dict[str, Dict], verified: List[str], points: list[str], console: Optional[Any] = None
) -> None:
    """把必写要点注入每个待重写章节的变更理由（writer prompt 会一并带上）。

    已归类的要点只注入它自己的落点章节：枚举要点进功能说明会把整张取值表
    抄过去，数据通路要点进参数说明同理。
    """
    if not points:
        return
    for part in verified:
        entry = all_reasons.get(part)
        if not entry or not entry.get("keep"):
            continue
        _own = _part_kind(part)
        if _own:
            # 该章节有明确归属：只喂它自己的要点。否则“取值范围上限”这类
            # 无主要点会被塞进参数说明，模型顺手把整张表的 2^21 都改成 2^22。
            mine = [p for p in points if _point_kind(p) == _own]
        else:
            mine = [p for p in points if not _point_kind(p)]
        if not mine:
            continue
        block = _MUST_COVER_HEADER + "\n".join(f"- {p}" for p in mine)
        if _MUST_COVER_HEADER.strip() in entry.get("reason", ""):
            continue
        entry["reason"] = f"{entry.get('reason', '')}{block}"
        entry["must_cover"] = mine
    if console is not None:
        console.print(f"  [cyan]必写要点[/cyan]: {len(points)} 条（已注入变更理由）")


def _deterministic_targets(
    changed_detail: Dict[str, Dict], extra_added_text: str = "", extra_removed_text: str = ""
) -> Dict[str, list]:
    """按「变更类型 → 章节」的确定性规则，给出本次必须重生成的章节。

    不看 LLM 的 keep 判定，也不看 PR 的文档版本（生成期禁止读 head docs）。
    规则刻意保守：只认“公共头里的声明变化”和“精确的枚举/模板/断言/单位数值变化”，
    避免任何代码 diff 都命中，从而把无关章节也放进来。
    返回 {doc_part: [命中理由]}；空 dict 表示未命中。
    """
    added: list[str] = []
    removed: list[str] = []
    inc_files: list[str] = []
    for f, detail in (changed_detail or {}).items():
        rel = str(f).replace("\\", "/")
        diff = (detail or {}).get("diff", "") or ""
        if rel.startswith("include/"):
            inc_files.append(rel)
        for ln in diff.splitlines():
            if ln.startswith("+") and not ln.startswith("+++"):
                added.append(ln[1:])
            elif ln.startswith("-") and not ln.startswith("---"):
                removed.append(ln[1:])
    if extra_added_text:
        added.extend(extra_added_text.splitlines())
    if extra_removed_text:
        removed.extend(extra_removed_text.splitlines())

    add_txt, rem_txt = "\n".join(added), "\n".join(removed)
    both = add_txt + "\n" + rem_txt
    hits: Dict[str, list] = {}

    def hit(part: str, why: str) -> None:
        hits.setdefault(part, [])
        if why not in hits[part]:
            hits[part].append(why)

    # 1) 约束说明：断言增删 / 能力否定翻转 / 带单位数值 / 寄存器类型
    if re.search(r"\bstatic_assert\b", both):
        hit("constraint", "static_assert 变化")
    if re.search(r"不支持", rem_txt) or re.search(r"\bRegTraitNum(?:One|Two)\b", both):
        hit("constraint", "能力/类型约束变化")
    unit_nums = set(re.findall(r"\b\d+\s?(?:B|VL|KB|MB)\b", both))
    if unit_nums:
        hit("constraint", "带单位数值变化: " + ", ".join(sorted(unit_nums)[:4]))

    # 2) 参数说明：枚举取值 / 模板形参
    enum_rem = set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\s*=\s*\d+", rem_txt))
    enum_add = set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\s*=\s*\d+", add_txt))
    if enum_rem != enum_add:
        hit("para_clarify", "枚举/常量取值变化")
    if any("template<" in ln or "template <" in ln for ln in added + removed):
        hit("para_clarify", "模板形参变化")

    # 3) 函数原型：仅公共头（include/）里的声明行变化
    decl = re.compile(r"\b(?:inline|__aicore__|__simd_callee__)\b[^\n]*\(")
    decl_changed = bool(inc_files) and (any(decl.search(ln) for ln in added) or any(decl.search(ln) for ln in removed))
    if decl_changed:
        hit("function_proto", "公共头声明行变化")

    # 4) 调用示例：示例/测试文件变动，或代码里对接口的调用点变化
    if any(("examples/" in f) or ("tests/" in f) for f in (changed_detail or {})):
        hit("example", "示例/测试文件变动")
    if re.search(r"\basc_\w+\s*\(", both):
        hit("example", "接口调用点变化")

    # 5) 功能说明：公共头里新增声明行；枚举改名也会波及功能概述
    decl_add = [ln for ln in added if re.search(r"\basc_\w+\s*\(", ln)]
    decl_rem = [ln for ln in removed if re.search(r"\basc_\w+\s*\(", ln)]
    if decl_add and len(decl_add) != len(decl_rem):
        hit("summary", "新增/删除接口声明")
        hit("function_proto", "新增/删除接口声明")
    if enum_rem != enum_add:
        # 枚举改名影响：取值列表（参数说明）、调用示例、约束里的取值约束；
        # 不影响“功能概述”，故不命中 summary（否则引入无关改写）。
        hit("example", "枚举/常量改名波及调用示例")
        hit("constraint", "枚举/常量取值变化")

    return hits


def run_evaluation(
    evidence_dir: str,
    before_dir: str,
    after_dir: str,
    doc_content: str,
    fine_grained: bool = True,
    console: Optional[Any] = None,
    api_name: Optional[str] = None,
    extra_doc_parts: Optional[List[str]] = None,
    repo_root: Optional[str] = None,
    extra_code_text: str = "",
    extra_code_removed: str = "",
) -> Tuple[List[str], Dict[str, Dict], List[str]]:
    """
    完整的变更评估流程：运行检测 + LLM过滤。

    Args:
        evidence_dir: 证据目录路径
        before_dir: 修改前代码目录
        after_dir: 修改后代码目录
        fine_grained: 是否启用细粒度检测
        console: 可选，rich console 对象，用于输出结果

    Returns:
        verified_todo_list: LLM验证后的待处理doc_part列表
        evaluation_reasons: 所有doc_part的评估理由，格式为 {doc_part: {"keep": bool, "reason": str}}
        modified_file_dir_list: 有改动的文件的相对路径列表
    """
    from src.change_detection.doc_part_change_detection.docpart_file_extractor import DocpartFileExtractor
    from src.change_detection.doc_part_change_detection.subpart_mapper import run_detection

    extractor = DocpartFileExtractor(evidence_dir, fine_grained)
    if fine_grained:
        file_subpart_mapping = extractor.extract_fine_grained_mappings()
    else:
        file_subpart_mapping = extractor.extract_mappings()

    affected_subparts, changed_detail, mapped_files, unmatched_files = run_detection(
        evidence_dir, before_dir, after_dir, fine_grained=fine_grained, enable_llm_filter=True, console=console
    )

    # 快照补拷继承的 doc_part：并入受影响集合，避免“补了文件但没章节可评估”导致早退
    if extra_doc_parts:
        _extra = [p for p in extra_doc_parts if p not in affected_subparts]
        if _extra:
            affected_subparts = list(affected_subparts) + _extra
            info(f"快照补拷继承 doc_parts: {_extra}")

    if "example" in affected_subparts and "example" not in file_subpart_mapping:
        from src.change_detection.doc_part_change_detection.subpart_mapper import _is_example_file

        example_file_list = [f for f in list(changed_detail.keys()) + mapped_files if _is_example_file(f)]
        example_file_list = sorted(set(example_file_list))
        if fine_grained:
            file_subpart_mapping["example"] = [{"file_name": f, "line_range": None} for f in example_file_list]
        else:
            file_subpart_mapping["example"] = example_file_list

    modified_file_dir_list = [str(Path(after_dir) / f) for f in mapped_files]
    unmatched_file_dir_list = [str(Path(after_dir) / f) for f in unmatched_files if f.endswith((".h", ".cpp"))]
    unmatched_set = set(unmatched_files)
    matched_changed_detail = {f: d for f, d in changed_detail.items() if f not in unmatched_set}

    has_tiling = any(_is_tiling_file(f) for f in matched_changed_detail)

    # 证据 → 必写要点：放在早退之前计算，避免“没有 affected_subparts 就整段跳过”
    _points = _must_cover_points(changed_detail, doc_content, repo_root, extra_code_text)
    _point_parts = _parts_for_points(_points, doc_content)
    if _points:
        info(f"必写要点 {len(_points)} 条 -> 落点 {_point_parts}")
        if console is not None:
            console.print(f"  [cyan]必写要点[/cyan]: {len(_points)} 条 -> {_point_parts}")
            for _p in _points:
                console.print(f"    [dim]- {_p[:100]}[/dim]")
        for _p in _point_parts:
            if _p not in affected_subparts:
                affected_subparts = list(affected_subparts) + [_p]

    # 架构作用域（文档级）：文档名带 arch_X，但本次变更完全没碰 X → 不该改这篇文档
    _scope = arch_scope_from_changes(matched_changed_detail)
    _doc_arch = _doc_arch_labels(api_name)
    if _scope and _doc_arch and _doc_arch not in _scope:
        _touched = "、".join(sorted(_scope))
        if console is not None:
            console.print(
                f"  [yellow]架构作用域[/yellow]: 文档归属 npu_arch_{_doc_arch}，本次变更仅涉及 {_touched}，跳过该文档"
            )
        info(f"架构作用域跳过: 文档 arch_{_doc_arch}，变更涉及 {_touched}")
        return [], {}, modified_file_dir_list + unmatched_file_dir_list

    if not affected_subparts and not has_tiling:
        console.print("\n  [yellow]无受影响的 doc part 且无 tiling 变更，跳过 LLM 变更评估[/yellow]")
        info("affected_subparts 为空且无 tiling 变更，跳过 LLM 验证")
        return [], {}, modified_file_dir_list + unmatched_file_dir_list

    info(f"  [yellow]LLM 变更评估[/yellow]")
    info(f"  关联 doc parts: {affected_subparts}")
    info(f"  Tiling 变更: {has_tiling}")

    evaluate_result = []
    all_reasons = {}
    tiling_affected = set()
    tiling_reasons = {}

    task_labels = []
    if affected_subparts:
        task_labels.append("evaluate")
    if has_tiling:
        task_labels.append("tiling")

    prog_console = (
        console
        if (console is not None and getattr(console, "file", None) is not None and console.file.isatty())
        else Console()
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=prog_console,
        disable=not sys.stdout.isatty(),
    ) as progress:
        overall = progress.add_task(f"LLM 变更评估 - Tiling 变更: {has_tiling}", total=len(task_labels))

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            if "evaluate" in task_labels:
                futures[
                    executor.submit(
                        evaluate_changes,
                        doc_content=doc_content,
                        affected_subparts=affected_subparts,
                        changed_detail=matched_changed_detail,
                        file_subpart_mapping=file_subpart_mapping,
                    )
                ] = "evaluate"
            if "tiling" in task_labels:
                futures[
                    executor.submit(
                        analyze_tiling_changes,
                        changed_detail=matched_changed_detail,
                        console=console,
                    )
                ] = "tiling"

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    result = future.result()
                    if task_name == "evaluate":
                        evaluate_result, all_reasons = result
                    elif task_name == "tiling":
                        tiling_affected, tiling_reasons = result
                except Exception as e:
                    error(f"{task_name} 任务异常: {e}")
                progress.update(overall, advance=1, description=f"  LLM 变更评估 ({task_name} 完成)")

    # 门面守卫：在 LLM 评估结果上做确定性覆盖（合并 tiling 前执行）
    if "evaluate" in task_labels:
        _facade_guard(matched_changed_detail, api_name, affected_subparts, evaluate_result, all_reasons, console)
        _arch_type_support_guard(
            matched_changed_detail, api_name, affected_subparts, evaluate_result, all_reasons, console
        )
        # 语义守卫用**全量** changed_detail：取值/常量/能力变化常落在未映射到
        # doc_part 的 impl 文件里（如 loop2Size 上限在 dav_3510/..._impl.h），
        # 用 matched 子集会把这些信号整体丢掉。
        _semantic_change_guard(
            changed_detail,
            doc_content,
            evaluate_result,
            all_reasons,
            console,
            extra_added_text=extra_code_text,
            extra_removed_text=extra_code_removed,
        )
        _arch_scope_guard(matched_changed_detail, api_name, evaluate_result, all_reasons, console)
        # 补拷命中：这些章节必须重新生成（其证据文件本次有同族改动）
        for _p in extra_doc_parts or []:
            all_reasons.setdefault(
                _p,
                {
                    "keep": True,
                    "reason": "快照补拷：本次改动文件与证据同族（R1/R2/R3），需重新生成该章节",
                    "source": "snapshot_augment",
                },
            )
            if _p not in evaluate_result:
                evaluate_result.append(_p)

    # 必写要点：无论是否触发 LLM 评估都要落地（落点已按关键词定位）
    # 注意：必须**强制** keep=True —— 否则 LLM 判的 keep=false 会把该章节挡在生成队列外，
    # 事实要点就永远写不进文档（实测 5428 的 Store：constraint 被判无变化 → 整篇零改动）。
    for _p in _point_parts:
        if _p not in evaluate_result:
            evaluate_result.append(_p)
        _entry = all_reasons.get(_p)
        if not _entry:
            all_reasons[_p] = {
                "keep": True,
                "reason": "存在必须覆盖的事实要点（见下）",
                "source": "must_cover",
            }
        else:
            _entry["keep"] = True
            _entry["source"] = "must_cover"
            if "存在必须覆盖的事实要点" not in (_entry.get("reason") or ""):
                _entry["reason"] = (_entry.get("reason") or "") + "\n（存在必须覆盖的事实要点，已强制纳入生成）"
    _attach_must_cover(all_reasons, list(evaluate_result), _points, console)

    # 确定性裁决：LLM 的 keep 只作参考，最终由「变更类型→章节」映射 + 守卫覆盖 + 必写要点决定。
    # 动机：LLM 常把无关章节判成需要更新（实测 35 处「多改」，4970 占 15），
    # 而这些章节没有任何确定性证据表明受本次变更影响。
    if "evaluate" in task_labels:
        _targets = _deterministic_targets(changed_detail, extra_code_text, extra_code_removed)
        _forced = {p for p, e in (all_reasons or {}).items() if (e or {}).get("source") not in (None, "", "general")}
        # 安全网：该章节自身的证据文件本次确有改动（affected_subparts 给出映射），
        # 也算命中，避免映射规则不全时把“该改的章节”误剔除。
        # summary 最容易出现“看着沾边就整段重写”的无关生成，不纳入安全网，
        # 只允许通过映射规则（新增接口/重载等）或守卫强制进入。
        _evidence_backed = set(affected_subparts or []) - {"summary"}
        _allowed = set(_targets) | _forced | _evidence_backed
        if _allowed:
            _before = list(evaluate_result)
            evaluate_result[:] = [p for p in _before if p in _allowed]
            _dropped = [p for p in _before if p not in _allowed]
            if _dropped:
                info(f"确定性映射命中 {sorted(_targets)}；剔除 {_dropped}")
                if console is not None:
                    console.print(f"  [dim]确定性映射命中: {sorted(_targets)}[/dim]")
                    console.print(f"  [yellow]无确定性证据，剔除: {_dropped}[/yellow]")
        else:
            info("确定性映射未命中任何章节，回退 LLM 判定")

    final_set = set(evaluate_result) | tiling_affected
    verified_todo_list = sorted(final_set)

    conflicts = set(all_reasons.keys()) & set(tiling_reasons.keys())
    for dp in conflicts:
        ev_keep = all_reasons[dp]["keep"]
        if not ev_keep:
            info(f"冲突解决: {dp} evaluate=SKIP 但 tiling=KEEP → 最终 KEEP")

    merged_reasons = dict(all_reasons)
    for dp, detail in tiling_reasons.items():
        if dp in merged_reasons and merged_reasons[dp]["keep"] and detail["keep"]:
            merged_reason = f"【整体评估】{merged_reasons[dp]['reason']}\n【Tiling评估】{detail['reason']}"
            merged_reasons[dp] = {"keep": True, "reason": merged_reason, "source": "both"}
        elif dp not in merged_reasons or not merged_reasons[dp]["keep"]:
            merged_reasons[dp] = detail

    # Display the same final decisions and reasons that are returned for JSON output.
    table = Table(show_header=True, header_style="bold cyan", show_lines=True, expand=True)
    table.add_column("文档章节", width=14)
    table.add_column("是否影响", width=8, justify="center")
    table.add_column("评估理由", ratio=1, no_wrap=False, overflow="fold")
    for doc_part, detail in merged_reasons.items():
        tag = Text.from_markup("[red]是[/red]" if detail["keep"] else "[green]否[/green]")
        table.add_row(format_doc_part(doc_part), tag, detail["reason"])
    console.print(Panel(table, title="LLM 变更评估", border_style="cyan"))

    skipped = [p for p in affected_subparts if p not in final_set]
    if skipped:
        console.print(f"  [dim]已过滤 ({len(skipped)}):[/dim] {skipped}")
    console.print(f"  [bold green]最终保留 ({len(verified_todo_list)}):[/bold green] {verified_todo_list}")
    info(f"有改动的文件 ({len(modified_file_dir_list)}):")
    for f in modified_file_dir_list:
        info(f"  - {f}")
    if unmatched_file_dir_list:
        info(f"未匹配到 subpart 的变更文件 ({len(unmatched_file_dir_list)}):")
        for f in unmatched_file_dir_list:
            info(f"  - {f}")

    return verified_todo_list, merged_reasons, modified_file_dir_list + unmatched_file_dir_list


if __name__ == "__main__":
    from src.generation_validation import config

    evidence_dir = "scripts/cann-document-generation/data/outputs/snapshot/abs_back/base/anchor"
    before_dir = "scripts/cann-document-generation/data/test/vs_testdata/abs_back"
    after_dir = "scripts/cann-document-generation/data/test/vs_testdata/abs"

    verified_todo_list, evaluation_reasons, modified_file_dir_list = run_evaluation(
        str(evidence_dir),
        str(before_dir),
        str(after_dir),
        "",
        fine_grained=True,
        console=Console(),
    )
    info(f"verified_todo_list: {verified_todo_list}")
    info(f"evaluation_reasons: {evaluation_reasons}")
    info(f"modified_file_dir_list: {modified_file_dir_list}")
