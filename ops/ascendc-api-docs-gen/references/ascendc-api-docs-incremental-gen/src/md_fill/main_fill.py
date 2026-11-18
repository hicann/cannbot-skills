# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Markdown 文档章节替换工具：根据章节映射 dict 修改 md 文档的章节。"""

__all__ = ["update_md_with_dict", "wrap_product_lines_with_npu"]

import json
import re

import yaml

from generation_validation.config import DOC_PART_LABELS, TEMPLATES_YAML

try:
    from md_fill.postprocess import normalize_cjk_ascii_spacing
except ImportError:  # 以 src.md_fill.* 前缀导入时走相对导入
    from .postprocess import normalize_cjk_ascii_spacing

# 「产品支持情况」章节由模板生成、自带 npu 标签对，全文档包裹时跳过
_SUPPORT_TITLE = "## 产品支持情况"
# 原文自带的 npu 开/闭标签：<!-- npu="A3" id2 --> / <!-- end id2 -->
_NPU_OPEN_RE = re.compile(r'<!--\s*npu="[^"]*"\s+\S+\s*-->')
_NPU_CLOSE_RE = re.compile(r"<!--\s*end\s+\S+\s*-->")


def _load_data(data: dict[str, str] | str) -> dict[str, str]:
    """加载数据，支持 dict 或 JSON 文件路径。

    Args:
        data: dict 对象或 JSON 文件路径

    Returns:
        字典数据

    Raises:
        TypeError: data 类型错误或 JSON 内容不是 dict
        FileNotFoundError: JSON 文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        with open(data, "r", encoding="utf-8") as f:
            result = json.load(f)
        if not isinstance(result, dict):
            raise TypeError(f"JSON 文件内容类型错误，期望 dict，实际为 {type(result)}")
        return result
    raise TypeError(f"data 参数类型错误，期望 dict 或 str，实际为 {type(data)}")


def _is_empty_value(value: str | None) -> bool:
    """判断 value 是否为空（None、空字符串、纯空白字符串）。

    Args:
        value: dict 中某个 key 对应的 value

    Returns:
        True 表示为空，False 表示有内容

    Raises:
        TypeError: value 不是 str 或 None
    """
    if value is None:
        return True
    if not isinstance(value, str):
        raise TypeError(f"value 类型错误，期望 str 或 None，实际为 {type(value)}")
    return value.strip() == ""


def _read_md_file(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


def _split_sections(content: str) -> list[str]:
    """按 ## 章节标题分割文档，保持标题行作为独立元素。

    Args:
        content: 原始 markdown 内容

    Returns:
        分割后的列表，如 ['# 标题', '## 章节1\n内容1', '## 章节2\n内容2', ...]
    """
    return re.split(r"\n(?=## )", content)


def _find_section_index(parts: list[str], title: str) -> int | None:
    """查找指定标题的章节索引。

    Args:
        parts: 分割后的章节列表
        title: 要查找的标题（如 '## 产品支持情况'）

    Returns:
        章节索引，未找到返回 None
    """
    # title 后必须紧跟空白/锚点标签/行尾，避免 "## 功能说明" 误匹配 "## 功能说明补充"
    pattern = re.compile(re.escape(title) + r"(?=\s|<|$)")
    for i, part in enumerate(parts):
        if pattern.match(part.strip()):
            return i
    return None


def _replace_section_body(original_part: str, new_content: str) -> str:
    """保留原始标题行（含同行锚点等），用 new_content 的正文替换 original_part 的正文。"""
    original_title = original_part.split("\n", 1)[0]
    new_split = new_content.split("\n", 1)
    new_body = new_split[1] if len(new_split) > 1 else ""
    if new_body:
        cleaned = new_body.rstrip("\r\n")
        return original_title + "\n" + cleaned + "\n"
    return original_title


def _build_result(parts: list[str]) -> str:
    return "\n".join(parts)


def _load_npu_products() -> list[dict]:
    """从 templates.yaml 读产品表（doc_parts.support.products）。

    Returns:
        [{"npu": str, "id": str, "product": str}, ...]，按产品名长度降序
        （长名优先匹配，避免潜在子串误命中）。
    """
    with open(TEMPLATES_YAML, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    products = (raw.get("doc_parts", {}).get("support", {}) or {}).get("products", []) or []
    return sorted(
        (e for e in products if e.get("product") and e.get("npu") and e.get("id")),
        key=lambda e: len(e["product"]),
        reverse=True,
    )


# 多产品拆句时产品之间允许的连接段：仅顿号/逗号+空白（"和/与"及对比性表述
# 拆句后语义会崩，不拆）
_MULTI_CONNECTOR_RE = re.compile(r"^[、,，]\s*$")
# 后缀含对比性词汇时禁拆（"A2、A3的区别"拆成单产品句语义不成立）
_MULTI_CONTRAST_RE = re.compile(r"区别|差异|对比|比较|分别")


def _collect_product_spans(line: str, products: list[dict]) -> list[tuple[int, int, dict]]:
    """收集行内所有产品名出现区间 (start, end, entry)，按位置排序并去除嵌套重叠。"""
    spans: list[tuple[int, int, dict]] = []
    for e in products:
        start = 0
        while True:
            i = line.find(e["product"], start)
            if i < 0:
                break
            spans.append((i, i + len(e["product"]), e))
            start = i + len(e["product"])
    spans.sort(key=lambda s: s[0])
    filtered: list[tuple[int, int, dict]] = []
    pos = 0
    for s, t, e in spans:
        if s < pos:
            continue  # 与上一匹配重叠（嵌套名），外层已覆盖
        filtered.append((s, t, e))
        pos = t
    return filtered


def _split_multi_product_line(line: str, products: list[dict]) -> list[str] | None:
    """多产品行按产品拆成多句，每句行内整句包裹。

    "A、B仅支持X。" →
      <!-- npu="A" idX -->A仅支持X。<!-- end idX -->
      <!-- npu="B" idY -->B仅支持X。<!-- end idY -->

    拆分条件（不满足返回 None，调用方回退行内逐名包裹）：
      - 行内恰好多个产品、产品之间连接段为纯顿号/逗号
      - 后缀不含对比性词汇（区别/差异等，拆句后语义不成立）
    共同前缀/后缀复制到每个产品的句子中。
    """
    spans = _collect_product_spans(line, products)
    if len(spans) < 2:
        return None
    for (_, t1, _), (s2, _, _) in zip(spans, spans[1:]):
        seg = line[t1:s2].strip()
        if not seg or not _MULTI_CONNECTOR_RE.match(seg):
            return None
    suffix = line[spans[-1][1]:]
    if _MULTI_CONTRAST_RE.search(suffix):
        return None
    prefix = line[: spans[0][0]]
    return [
        f'{prefix}<!-- npu="{e["npu"]}" {e["id"]} -->{e["product"]}{suffix}<!-- end {e["id"]} -->' for _, _, e in spans
    ]


def _wrap_line_multi(line: str, products: list[dict]) -> str:
    """一行含多个不同产品时行内逐名包裹：每个产品名各自用其标签对包住。

    按行内出现位置从左到右拼接；长名优先（避免短名吞掉长名前缀），
    重叠区间（嵌套名被外层覆盖）跳过。
    """
    spans = _collect_product_spans(line, products)
    parts: list[str] = []
    pos = 0
    for s, t, e in spans:
        parts.append(line[pos:s])
        parts.append(f'<!-- npu="{e["npu"]}" {e["id"]} -->{e["product"]}<!-- end {e["id"]} -->')
        pos = t
    parts.append(line[pos:])
    return "".join(parts)


def _wrap_lines_with_npu(lines: list[str], products: list[dict]) -> list[str]:
    """对单个章节的行做 npu 标签对包裹（状态机，规则见 wrap_product_lines_with_npu）。"""
    out: list[str] = []
    pending: dict | None = None  # 当前已开启未闭合标签对的产品条目（整行包裹式）
    in_tag = False  # 原文自带的整行式 npu 标签对内部
    in_fence = False  # 代码围栏（``` / ~~~）内部

    def _close_pending() -> None:
        nonlocal pending
        if pending is not None:
            out.append(f"<!-- end {pending['id']} -->")
            pending = None

    for line in lines:
        stripped = line.lstrip()
        # 标签对状态更新：开/闭数量平衡的行（如行内逐名包裹行）不改变状态，
        # 避免行内成对的标签对把 in_tag 卡死在 True
        opens = len(_NPU_OPEN_RE.findall(line))
        closes = len(_NPU_CLOSE_RE.findall(line))
        if opens or closes:
            _close_pending()
            if opens > closes:
                in_tag = True
            elif closes > opens:
                in_tag = False
            out.append(line)
            continue
        if in_tag:
            out.append(line)
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            _close_pending()
            out.append(line)
            continue
        if in_fence:
            # 代码块内不放 npu 注释对（注释对会渲染为代码文本），跳过；
            # 函数原型的特殊说明由生成规范要求放在代码块外的正文行
            out.append(line)
            continue
        hits = {e["product"]: e for e in products if e["product"] in line}
        if not hits:
            _close_pending()
            out.append(line)
            continue
        if stripped.startswith("|"):
            # 表格行：不能拆行，行内逐名包裹（注释对在单元格文本内，结构安全）
            _close_pending()
            out.append(_wrap_line_multi(line, products))
            continue
        if len(hits) > 1:
            # 多产品行：按产品拆成多句、每句整句行内包裹（顿号并列场景）；
            # 连接复杂/对比性表述不可拆时，回退行内逐名包裹
            _close_pending()
            split = _split_multi_product_line(line, products)
            if split is not None:
                out.extend(split)
            else:
                out.append(_wrap_line_multi(line, products))
            continue
        entry = next(iter(hits.values()))
        if pending is not None and pending["product"] == entry["product"]:
            out.append(line)  # 连续同产品行，续用已开启的标签对
            continue
        _close_pending()
        out.append(f'<!-- npu="{entry["npu"]}" {entry["id"]} -->')
        out.append(line)
        pending = entry
    _close_pending()
    return out


def wrap_product_lines_with_npu(content: str, products: list[dict] | None = None) -> str:
    """全文档后处理：除「产品支持情况」章节外，凡出现产品名的行用 npu 注释对包裹。

    规范（asc-devkit docs/zh/asc_api_writing_guidelines.md 的「3 产品支持情况规范」，见 templates.yaml 的 requirement_from_spec）要求文档中
    具体产品型号内容使用注释对包裹（<!-- npu="A3" id2 --> ... <!-- end id2 -->），
    使文档平台可按产品过滤显示。「产品支持情况」章节由模板生成、自带标签对，跳过。

    包裹规则（文档中任何部分出现产品名称都必须被 npu 注释对包裹）：
      - 「产品支持情况」章节由模板生成、自带标签对，跳过
      - 普通文本/列表项行，单产品 → 注释对独立成行整行包裹（连续同产品行合并一对）
      - 多产品行 → 按产品拆成多句，每句整句行内包裹
        （<!-- npu="910b" id3 -->A仅支持X。<!-- end id3 -->），前缀/后缀复制到每句；
        产品间连接复杂（含"和/与"或对比性表述）时回退行内逐名包裹
      - 表格行 → 行内逐名包裹（不能拆行）
      - 代码块内不放 npu 注释对（会渲染为代码文本），跳过；
        特殊说明由生成规范（templates.yaml）要求放在代码块外的正文行
      - 原文已有标签对内的行不重复包裹（开/闭计数平衡的行不改变标签对状态）

    Args:
        content: 完整 markdown 文档
        products: 产品表，缺省时从 templates.yaml 加载

    Returns:
        包裹后的完整 markdown 文档
    """
    if products is None:
        products = _load_npu_products()
    if not products:
        return content
    parts = re.split(r"\n(?=## )", content)
    for i, part in enumerate(parts):
        title = part.split("\n", 1)[0]
        if title.strip().startswith(_SUPPORT_TITLE):
            continue
        parts[i] = "\n".join(_wrap_lines_with_npu(part.split("\n"), products))
    return "\n".join(parts)


def update_md_with_dict(
    data: dict[str, str] | str,
    md_path: str,
    out_file_path: str | None = None,
) -> str:
    """根据 dict 修改 md 文档。

    Args:
        data: 包含章节内容的字典，或 json 文件路径。
              key 为章节标识（如 "support"），value 为 markdown 完整内容（含 ## 标题行）。
        md_path: md 文档路径
        out_file_path: 输出文件路径，为 None 时不写文件

    Returns:
        修改后的完整 md 文档字符串

    替换规则:
        - dict 中的每个 key 对应 md 文档中的一个 ## 章节
        - 保留原始标题行（含同行锚点等），仅用 dict[key] 的正文替换对应章节的正文
        - dict 中不包含的 key，其对应章节保持原样
        - dict[key] 的 value 为空（None 或空/纯空白字符串）时，
          对应章节保留原内容不替换
    """
    data_dict = _load_data(data)
    md_content = _read_md_file(md_path)
    parts = _split_sections(md_content)

    for key, title in DOC_PART_LABELS.items():
        if key not in data_dict:
            continue
        if _is_empty_value(data_dict[key]):
            continue
        idx = _find_section_index(parts, title)
        if idx is not None:
            # 中英文边界多余空格属生成侧风格漂移，回填前统一收口（产品名行/代码块不动）
            body = normalize_cjk_ascii_spacing(data_dict[key])
            parts[idx] = _replace_section_body(parts[idx], body)

    ret = _build_result(parts)
    # 全文档后处理：非「产品支持情况」章节中出现产品名的行补 npu 标签对包裹
    ret = wrap_product_lines_with_npu(ret)

    if out_file_path:
        with open(out_file_path, "w", encoding="utf-8") as f:
            f.write(ret)

    return ret
