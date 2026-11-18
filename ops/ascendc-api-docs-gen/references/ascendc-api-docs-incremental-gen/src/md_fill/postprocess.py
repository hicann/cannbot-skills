# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""确定性后处理：中英文间距规范化 + 函数原型守卫。

这两类问题都源自“生成结果不可控”，这里用规则收口，不依赖模型能力：

1. 中英文之间多余空格
   生成正文常写出“接收 `pipe_t` 参数”“返回值类型为 void”这类
   中文与 ASCII / 行内代码之间的空格；原文风格是紧贴
   （如“该接口用于AI Core内部异步流水线之间的同步”）。
   规则：只在 CJK 与 ASCII 的**边界**去掉空格；
   代码围栏内、产品名行、行内代码段内部、列表符号一律不动。

2. 函数原型章节混入内部实现符号 / 无据签名
   section 锚点会把宏展开体整段带进证据，模型据此把内部符号
   ``asc_lock_impl`` 写成公开原型，并把 ``ascMutexExecuteMode``
   “规范化”成不存在的 ``asc_mutex_execute_mode``。
   规则：原型行出现 ``*_impl`` / ``*_detail`` 符号，或行内出现
   “证据 + 原文”语料中根本不存在的标识符，则整节回退原文
   （原型是代码事实，宁可不变也不写错）。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_YAML = PROJECT_ROOT / "resources" / "templates.yaml"

PROTO_HEADING = "## 函数原型"

# ---------------------------------------------------------------- 间距规范化

# CJK：汉字 + 中文标点（全角符号区）
_CJK = r"\u4e00-\u9fa5\u3000-\u303f\uff01-\uffef"
# ASCII：字母数字 + 行内代码常用符号（含反引号、路径分隔符、目录分隔符）；
# \x01 是行内代码段的占位符，必须算作 ASCII，边界才能对齐
_ASCII = r"A-Za-z0-9`_\\/()\[\]<>@%&+=~$\x01"

_SPACE_BEFORE_ASCII = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_ASCII}])")
_SPACE_AFTER_ASCII = re.compile(rf"(?<=[{_ASCII}])[ \t]+(?=[{_CJK}])")

# 行内代码 + 加粗标记：内部空格属原文格式（如 `**表** 矩阵信息说明`、`**表 1**`），
# 用占位符保护，不参与“中英文边界去空格”
# 行内代码 + 加粗标记：内部/尾部空格属原文格式（如 `**表** 矩阵信息说明`、`**表 1**`）。
# 两个占位符区分处理：
#   \x01 = 行内代码（参与边界去空格，故计入 _ASCII）
#   \x02 = 加粗标记（保留空格，故不计入 _ASCII）
_INLINE_CODE = re.compile(r"(`[^`\n]*`)|(\*\*[^*\n]+\*\*)")
_FENCE = re.compile(r"^\s*(```|~~~)")
# markdown 列表前缀：不能被当成“ASCII 与中文之间的空格”吃掉
_LIST_PREFIX = re.compile(r"^([ \t]*(?:[-*+][ \t]+|\d+\.[ \t]+)?)(.*)$", re.S)
# 产品名行：产品名内部的空格是规范写法，整行跳过
_PRODUCT_LINE = re.compile(r"系列产品|AI处理器|推理产品|SoC（")

_PLACEHOLDER = "\x01"

_products_cache: list[str] | None = None


def _load_product_names() -> list[str]:
    """从 templates.yaml 读产品名（chip_map 口径），用于整行跳过。"""
    global _products_cache
    if _products_cache is not None:
        return _products_cache
    names: list[str] = []
    try:
        raw = yaml.safe_load(TEMPLATES_YAML.read_text(encoding="utf-8")) or {}
        products = ((raw.get("doc_parts") or {}).get("support") or {}).get("products") or []
        names = [p.get("product") for p in products if p.get("product")]
    except (OSError, yaml.YAMLError):
        names = []
    _products_cache = names
    return names


def _fence_mask(lines: list[str]) -> list[bool]:
    """标记代码围栏内的行。

    对围栏**不成对**的文档（生成侧偶发多写一个 ```），只认前偶数个围栏标记，
    把落单的那个视为普通文本——否则围栏后的正文会全部被误判为代码而跳过。
    """
    idx = [i for i, ln in enumerate(lines) if _FENCE.match(ln)]
    if len(idx) % 2 == 1:
        idx = idx[:-1]
    mask = [False] * len(lines)
    for a, b in zip(idx[0::2], idx[1::2]):
        for i in range(a, b + 1):
            mask[i] = True
    return mask


def _normalize_line(line: str) -> str:
    prefix, body = _LIST_PREFIX.match(line).groups()
    spans: list[tuple[str, str]] = []

    def _keep(m: re.Match) -> str:
        ph = "\x01" if m.group(1) else "\x02"
        spans.append((ph, m.group(0)))
        return ph

    protected = _INLINE_CODE.sub(_keep, body)
    protected = _SPACE_BEFORE_ASCII.sub("", protected)
    protected = _SPACE_AFTER_ASCII.sub("", protected)
    for ph, text in spans:
        protected = protected.replace(ph, text, 1)
    return prefix + protected


def normalize_cjk_ascii_spacing(text: str, product_names: list[str] | None = None) -> str:
    """去掉正文中 CJK 与 ASCII 之间的多余空格。

    不改动：代码围栏内、产品名行、行内代码段内容、markdown 列表前缀。
    """
    if not text:
        return text
    products = _load_product_names() if product_names is None else product_names
    lines = text.split("\n")
    mask = _fence_mask(lines)
    for i, line in enumerate(lines):
        if mask[i]:
            continue
        if _PRODUCT_LINE.search(line) or any(p in line for p in products):
            continue
        lines[i] = _normalize_line(line)
    return "\n".join(lines)


# ---------------------------------------------------------------- 原型守卫

_CPP_KEYWORDS = {
    "void",
    "int",
    "bool",
    "char",
    "short",
    "long",
    "float",
    "double",
    "signed",
    "unsigned",
    "const",
    "constexpr",
    "static",
    "inline",
    "template",
    "typename",
    "class",
    "struct",
    "enum",
    "auto",
    "virtual",
    "explicit",
    "noexcept",
    "operator",
    "if",
    "else",
    "return",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "break",
    "size_t",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "half",
    "bfloat16_t",
    "float32_t",
    "dim3_t",
    "pipe_t",
}

_PROTO_HINT = re.compile(r"(__aicore__|__simd_callee__|\binline\b|\btemplate\b|\bvoid\b)")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SYMBOL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*[<(]")


def _extract_section(md: str, heading: str) -> str:
    """取 markdown 中某个 `## 标题` 章节的正文（到下一个 ## 或文末）。

    标题可能带锚点（``## 参数说明<a name="..."></a>``），需容忍尾部标记。
    """
    m = re.search(
        rf"(?m)^{re.escape(heading)}(?:<a[^>]*>\s*</a>)?[ \t]*$\n?(.*?)(?=^## |\Z)",
        md or "",
        re.S,
    )
    return m.group(1) if m else ""


def _iter_proto_lines(section: str):
    """产出函数原型代码围栏内、看起来是函数声明的行。"""
    lines = (section or "").split("\n")
    mask = _fence_mask(lines)
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not mask[i] or not line:
            continue
        if _FENCE.match(line):
            continue
        if line.startswith(("#", "//", "/*", "*")):
            continue
        if "(" not in line:
            continue
        if not _PROTO_HINT.search(line):
            continue
        yield line


def _identifiers(line: str) -> set[str]:
    """提取一行里的标识符，剔除关键字、编译器内建（__ 开头）与模板形参（T/U/V…）。"""
    return {
        t
        for t in _IDENT.findall(line)
        if t not in _CPP_KEYWORDS
        and not t.startswith("__")
        and not re.fullmatch(r"[A-Z][A-Z0-9]?", t)  # 模板形参：T / U / T1
    }


def guard_function_proto(
    generated_text: str,
    original_doc: str,
    evidence_text: str,
    heading: str = PROTO_HEADING,
) -> tuple[str, list[str]]:
    """校验生成的原型章节；命中内部符号或无据标识符则整节回退原文。

    Returns:
        (最终文本, 问题列表)。问题列表非空时最终文本为原文该章节。
    """
    issues: list[str] = []
    if not generated_text:
        return generated_text, issues

    corpus = f"{evidence_text or ''}\n{original_doc or ''}"
    known_set = set(_IDENT.findall(corpus))

    checked = 0
    for line in _iter_proto_lines(generated_text):
        checked += 1
        m = _SYMBOL.search(line)
        sym = m.group(1) if m else ""
        if sym and (sym.endswith("_impl") or sym.endswith("_detail")):
            issues.append(f"内部实现符号不应出现在函数原型: {sym}")
            continue
        unknown = sorted(t for t in _identifiers(line) if t not in known_set)
        if unknown:
            issues.append(f"原型中标识符在证据/原文中不存在: {', '.join(unknown)}")

    # 没有可校验的原型行时不动（避免误伤非代码原型文档）
    if issues and checked:
        orig_body = _extract_section(original_doc, heading).strip()
        if orig_body:
            return f"{heading}\n\n{orig_body}\n", issues

    return generated_text, issues


def _arch_to_labels(archs: list[str]) -> set[str]:
    """npu_arch 号 → 文档里的 npu 标签（950 / A3 / 910b ...）。

    两个词表不同源：chip_map.yaml 给 arch→产品名，templates.yaml 给产品名→npu 标签。
    任何一环缺失都返回空集合，调用方据此**跳过**校验（宁可放过，不可误伤）。
    """
    try:
        chips = yaml.safe_load((PROJECT_ROOT / "resources/chip_map.yaml").read_text(encoding="utf-8")) or {}
        tpl = yaml.safe_load(TEMPLATES_YAML.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return set()
    prods = ((tpl.get("doc_parts") or {}).get("support") or {}).get("products") or []
    name2npu = {p.get("product"): p.get("npu") for p in prods if p.get("product")}
    labels: set[str] = set()
    for entry in chips.get("npu_arch_mapping") or []:
        if str(entry.get("arch", "")).strip() not in {str(a) for a in archs}:
            continue
        for name in re.split(r"[、，]", str(entry.get("product", ""))):
            npu = name2npu.get(name.strip())
            if npu:
                labels.add(npu)
    return labels


def guard_arch_scope(
    generated_text: str,
    original_doc: str,
    allowed_archs: list[str] | None,
    evidence_text: str = "",
    skip_headings: tuple[str, ...] = ("## 产品支持情况",),
) -> tuple[str, list[str]]:
    """段落级架构作用域：只能改动“本次变更涉及架构”的 npu 标签块。

    文档用 ``<!-- npu="A3" id2 --> ... <!-- end id2 -->`` 承载多产品差异。
    若本次代码变更只涉及 npu_arch_3510，却改动了 npu="A3" 块的内容，
    说明生成的更新污染了不该动的产品段落 —— 整节回退原文。

    arch 号与 npu 标签是两套词表，映射不全时跳过校验（fail-open）。
    """
    issues: list[str] = []
    if not generated_text or not allowed_archs:
        return generated_text, issues

    allowed = _arch_to_labels(list(allowed_archs))
    if not allowed:
        return generated_text, issues

    heading = next((ln.strip() for ln in generated_text.splitlines() if ln.strip().startswith("## ")), "")
    if not heading:
        return generated_text, issues
    if heading in skip_headings:
        return generated_text, issues
    gen_body = "\n".join(generated_text.splitlines()[1:])
    orig_body = _extract_section(original_doc or "", heading)

    g_blocks = _npu_blocks(gen_body.split("\n"))
    o_blocks = _npu_blocks(orig_body.split("\n"))

    for label, lines in g_blocks.items():
        if not label or label in allowed:
            continue
        cur = [ln.strip() for ln in lines if ln.strip()]
        old = [ln.strip() for ln in o_blocks.get(label, []) if ln.strip()]
        if cur != old:
            issues.append(f'改动了本次变更之外的架构块 npu="{label}"')

    if issues:
        orig = orig_body.strip()
        if orig:
            return f"{heading}\n\n{orig}\n", issues
    return generated_text, issues


# ---------------------------------------------------------------- 头文件声明归一

_HEADER_LINE = re.compile(r"^\s*头文件(路径)?\s*[:：]")
_FUNC_HEADING = "## 功能说明"


def _norm_header_path(decl_file: str) -> str:
    """证据里的 include/xxx.h → 文档规范写法 xxx.h（去 include/ 前缀与引号）。"""
    p = (decl_file or "").strip().strip("`\"'")
    if p.startswith("include/"):
        p = p[len("include/"):]
    return p


def ensure_header_statement(doc_text: str, decl_file: str) -> tuple[str, list[str]]:
    """确保「功能说明」首段声明接口所在头文件，并清掉其它章节的头文件行。

    规范依据：同族文档（如 asc_disable_dma_atomic.md）在「功能说明」首段写
    ``头文件路径为：`"c_api/atomic/atomic.h"`。``；函数原型章节不重复声明。
    """
    issues: list[str] = []
    path = _norm_header_path(decl_file)
    if not doc_text or not path:
        return doc_text, issues

    parts = re.split(r"\n(?=## )", doc_text)
    cleaned: list[str] = []
    removed = 0
    for part in parts:
        title = part.split("\n", 1)[0].strip()
        if title.startswith(_FUNC_HEADING):
            cleaned.append(part)  # 功能说明单独处理
            continue
        kept = [ln for ln in part.split("\n") if not _HEADER_LINE.match(ln)]
        removed += len(part.split("\n")) - len(kept)
        cleaned.append(re.sub(r"\n{3,}", "\n\n", "\n".join(kept)))
    if removed:
        issues.append(f"移除非功能说明章节的头文件行 {removed} 行")

    for i, part in enumerate(cleaned):
        if not part.split("\n", 1)[0].strip().startswith(_FUNC_HEADING):
            continue
        head, _, body = part.partition("\n")
        # 已有任何形式的“头文件”声明就不再插入（原文可能用别的路径/句式）
        if "头文件" in body:
            break
        stmt = f'头文件路径为：`"{path}"`。'
        cleaned[i] = f"{head}\n\n{stmt}\n\n{body.lstrip(chr(10))}"
        issues.append(f"功能说明补入头文件声明: {path}")
        break

    return "\n".join(cleaned), issues


# ---------------------------------------------------------------- 章节级符号守卫

_ASC_API = re.compile(r"(?<![A-Za-z0-9_])asc_[A-Za-z0-9_]+(?![A-Za-z0-9_])")
# 驼峰型类型/接口名（如 LoadData3DParamsV1）：家族头里的兄弟接口符号，常被整段抄进不相干文档
_CAMEL_SYM = re.compile(r"(?<![A-Za-z0-9_])[A-Z][a-zA-Z0-9]{5,}(?:<[^>\n]{0,40}>)?(?![A-Za-z0-9_])")
_ASC_NS = re.compile(r"(?<![A-Za-z0-9_])AscendC::[A-Za-z_][A-Za-z0-9_:]*")
# 多级限定名（A::B::C，两级及以上命名空间）：原文没有就是编造
_QUALIFIED = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*){2,}")
_INTERNAL_SYM = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:_impl|_detail)\b")
_TEST_STUB = re.compile(r"\bEXPECT_[A-Z]+\b|\bASSERT_[A-Z]+\b|\b[A-Za-z0-9_]+_stub\b|\bTEST_F?\s*\(")

_NPU_OPEN = re.compile(r'<!--\s*npu="([^"]+)"\s+(\S+)\s*-->')
_NPU_CLOSE = re.compile(r"<!--\s*end\s+(\S+)\s*-->")


def _npu_blocks(lines: list[str]) -> dict[str, list[str]]:
    """按 npu 标签对切分正文行，返回 {产品标签: [内容行]}；块外归入 ""。"""
    blocks: dict[str, list[str]] = {"": []}
    cur = ""
    for ln in lines:
        m = _NPU_OPEN.search(ln)
        if m:
            cur = m.group(1)
            blocks.setdefault(cur, [])
            continue
        if _NPU_CLOSE.search(ln):
            cur = ""
            continue
        blocks.setdefault(cur, []).append(ln)
    return blocks


def _code_blocks(text: str) -> list[str]:
    """抽取 markdown 中所有围栏代码块的正文（已归一空白）。"""
    blocks, cur, inside = [], [], False
    for line in (text or "").split("\n"):
        if _FENCE.match(line.strip()):
            if inside:
                norm = re.sub(r"\s+", " ", "\n".join(cur)).strip()
                if norm:
                    blocks.append(norm)
                cur = []
            inside = not inside
            continue
        if inside:
            cur.append(line)
    return blocks


_COMPAT_HINT = re.compile(r"兼容|废弃|deprecated|等同|旧名|不推荐")


def apply_renames(text: str, renames: dict[str, str] | None) -> tuple[str, list[str]]:
    """把产物里的旧标识符替换为本次重命名后的新名（确定性）。

    只替换“纯引用”的旧名；出现在兼容/废弃说明里的旧名原样保留
    （如“本接口兼容枚举值 `NEAREST_ZERO`，实际等同 `NEAREST_AWAY`”）。
    """
    issues: list[str] = []
    if not text or not renames:
        return text, issues
    out_lines = []
    for line in text.split("\n"):
        if _COMPAT_HINT.search(line):
            out_lines.append(line)
            continue
        new_line = line
        for old, new in renames.items():
            new_line = re.sub(rf"\b{re.escape(old)}\b", new, new_line)
        if new_line != line:
            issues.append(f"按重命名替换: {line.strip()[:40]} → 新名")
        out_lines.append(new_line)
    return "\n".join(out_lines), issues


# 领域规则一律来自 resources/doc_rules.yaml（见该文件注释）；
# 代码里不写死具体接口/参数的特殊处理。
_RULES_YAML = PROJECT_ROOT / "resources" / "doc_rules.yaml"
_rules_cache: dict | None = None


def load_doc_rules() -> dict:
    """读取可配置规则；缺失或格式错误时返回空 dict（对应处理自动跳过）。"""
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache
    try:
        data = yaml.safe_load(_RULES_YAML.read_text(encoding="utf-8")) or {}
        _rules_cache = data if isinstance(data, dict) else {}
    except (OSError, yaml.YAMLError):
        _rules_cache = {}
    return _rules_cache


_NPU_SEG = re.compile(r'(<!--\s*npu="[^"]+"\s+\S+\s*-->)(.*?)(<!--\s*end\s+\S+\s*-->)', re.S)


def _semantics_rules() -> list[dict]:
    """param_semantics 规则（每条：param / param_hint / source_text / append）。"""
    rules = load_doc_rules().get("param_semantics") or []
    return [r for r in rules if isinstance(r, dict) and r.get("param") and r.get("source_text") and r.get("append")]


def _add_fixed_offset(block: str, rule: dict) -> tuple[str, int]:
    """在“来源开关”句后补读取起点；块内无显式地址参数行则不动。"""
    append = rule["append"]
    if not block or append in block:
        return block, 0
    hint = rule.get("param_hint") or "地址"
    addr_row = re.compile(rf"^\s*\|\s*{re.escape(rule['param'])}\s*\|[^\n]*?(?:{hint})[^\n]*?$", re.M)
    if not addr_row.search(block):
        return block, 0
    sent = re.compile(rf"{re.escape(rule['source_text'])}(。)?")
    return sent.subn(lambda m: f"{rule['source_text']}{append}{m.group(1) or ''}", block)


def apply_buffer_source_offset_clause(text: str) -> tuple[str, list[str]]:
    """给“没有地址参数的来源开关”补上读取起点（确定性，不经模型）。

    规则来自 resources/doc_rules.yaml 的 param_semantics：同一 npu 块内既有
    显式地址参数行、又有来源开关句时，在开关句后补 append 片段。
    没拆出地址重载的架构表不该被补，因此按块判定。
    """
    issues: list[str] = []
    rules = _semantics_rules()
    if not text or not rules:
        return text, issues

    total = 0
    for rule in rules:
        if rule["source_text"] not in text:
            continue
        if _NPU_SEG.search(text):

            def _seg(m: "re.Match[str]", _r: dict = rule) -> str:
                nonlocal total
                body, n = _add_fixed_offset(m.group(2), _r)
                total += n
                return f"{m.group(1)}{body}{m.group(3)}"

            text = _NPU_SEG.sub(_seg, text)
        else:
            text, n = _add_fixed_offset(text, rule)
            total += n

    if total:
        issues.append(f"补充来源开关的读取起点：{total} 处")
    return text, issues


# 二级条目两种写法：`GM -> VECIN`（成对）与 `OUT_TO_L1`（光秃符号，模型凑的）
_TPOS_ARROW = re.compile(
    r"^\s{2,}[-*+]\s*[A-Za-z][A-Za-z0-9_]*(?:\s*/\s*[A-Za-z][A-Za-z0-9_]*)*"
    r"\s*->\s*[A-Za-z][A-Za-z0-9_]*(?:\s*/\s*[A-Za-z][A-Za-z0-9_]*)*\s*$"
)
_TPOS_BARE = re.compile(r"^\s{2,}[-*+]\s*[A-Za-z][A-Za-z0-9_]*(?:\s*/\s*[A-Za-z][A-Za-z0-9_]*)*\s*$")


def _is_tpos_bullet(line: str) -> bool:
    """是否是「数据通路」清单的二级条目（缩进 + 纯 ASCII 标识符）。"""
    return bool(_TPOS_ARROW.match(line or "") or _TPOS_BARE.match(line or ""))


def _norm_sub_line(line: str) -> str:
    """二级条目归一：去掉列表符号、压缩空白，便于与原文比对。"""
    body = re.sub(r"^\s*[-*+]\s*", "", line or "")
    return re.sub(r"\s+", " ", body).strip()


def strip_unsupported_tpos_lines(text: str, original_doc: str = "", code_text: str = "") -> tuple[str, list[str]]:
    """删掉“原文与本次改动都没有”的 TPosition 二级条目（确定性）。

    「数据通路」清单形如::

        - Global Memory -> UB
            - GM -> VECIN

    TPosition（VECIN/A1/B1…）属于领域约定，代码证据里通常没有。模型补新通路时
    会顺手编一条二级条目（6620: OUT_TO_L1 想补 `GM -> A1/B1`）。这里删条目、
    保留一级通路名 —— 宁可少一行，也不写没证据的逻辑位置。
    """
    issues: list[str] = []
    if not text:
        return text, issues
    # 只认原文功能说明里已有的二级条目：TPosition 换算是领域约定，代码证据里没有，
    # 模型会拿枚举名或“源 -> 目的”凑一条（6620: `- OUT_TO_L1` / `- OUT -> L1`）。
    orig_sec = _extract_section(original_doc or "", "## 功能说明")
    known = {_norm_sub_line(ln) for ln in (orig_sec or "").split("\n") if _is_tpos_bullet(ln)}
    out: list[str] = []
    for line in text.split("\n"):
        if _is_tpos_bullet(line) and _norm_sub_line(line) not in known:
            issues.append(f"删除无据的 TPosition 条目: {line.strip()}")
            continue
        out.append(line)
    return "\n".join(out), issues


# ---------------------------------------------------------------------------
# 业务评审意见的生成后收口（确定性）
# ---------------------------------------------------------------------------

# 必写要点的原句漏进正文（#7261 评审指出）：这些短语只应出现在提示词里
_PROMPT_LEAK = re.compile(
    r"为本次新增[，,]\s*需|清单中补一条|必写要点|需在取值说明中列出"
    r"|涉及该参数的说明需同步|不要新增$|需要补一条"
)

# 术语替换表来自 resources/doc_rules.yaml 的 terms（仓内统一叫法）


def _term_fix() -> dict:
    terms = load_doc_rules().get("terms") or {}
    return {str(k): str(v) for k, v in terms.items()} if isinstance(terms, dict) else {}


_PROTO_OPEN = re.compile(r"^\s*(__aicore__|__simd_callee__)\b")


def strip_prompt_leak(text: str) -> tuple[str, list[str]]:
    """删除漏进正文的提示词原句（整行命中即删）。

    命中后该节若无正文只剩标题，调用方应回退原文——由 apply_review_fixes 处理。
    """
    issues: list[str] = []
    if not text:
        return text, issues
    out: list[str] = []
    for line in text.split("\n"):
        if _PROMPT_LEAK.search(line):
            issues.append(f"删除泄漏的提示词: {line.strip()[:70]}")
            continue
        out.append(line)
    return "\n".join(out), issues


def fix_terms(text: str) -> tuple[str, list[str]]:
    """术语替换（跳过围栏代码与行内代码）。"""
    issues: list[str] = []
    if not text:
        return text, issues
    out: list[str] = []
    inside = False
    for line in text.split("\n"):
        if _FENCE.match(line.strip()):
            inside = not inside
            out.append(line)
            continue
        _fix = _term_fix()
        if inside or not any(k in line for k in _fix):
            out.append(line)
            continue
        parts = line.split("`")
        for i in range(0, len(parts), 2):  # 偶数段在行内代码之外
            for old, new in _fix.items():
                if old in parts[i]:
                    parts[i] = parts[i].replace(old, new)
                    issues.append(f"术语替换 {old}→{new}")
        out.append("`".join(parts))
    return "\n".join(out), issues


def normalize_single_item_list(text: str) -> tuple[str, list[str]]:
    """整节只有一个顶层列表项时去掉列表符号（#7316 评审意见）。"""
    issues: list[str] = []
    if not text:
        return text, issues
    lines = text.split("\n")
    items = [i for i, ln in enumerate(lines) if re.match(r"^-\s+\S", ln)]
    if len(items) != 1:
        return text, issues
    idx = items[0]
    lines[idx] = re.sub(r"^-\s+", "", lines[idx])
    issues.append("单条内容的章节去掉列表符号")
    return "\n".join(lines), issues


def format_proto_one_param_per_line(text: str) -> tuple[str, list[str]]:
    """函数原型：一行一个参数，续行与首参对齐（#7256 评审意见）。

    只处理「单行、只有一个左括号、>=2 个逗号、长度>80」的声明，避免误伤
    已经合规的多行原型或模板声明。
    """
    issues: list[str] = []
    if not text:
        return text, issues
    out: list[str] = []
    inside = False
    for line in text.split("\n"):
        if _FENCE.match(line.strip()):
            inside = not inside
            out.append(line)
            continue
        if (
            inside
            and _PROTO_OPEN.match(line)
            and line.count("(") == 1
            and line.count(")") == 1
            and line.count(",") >= 2
            and len(line) > 80
        ):
            head, rest = line.split("(", 1)
            params = rest.rsplit(")", 1)[0]
            tail = rest.rsplit(")", 1)[1]
            pad = " " * (len(head) + 1)
            out.append(head + "(")
            pieces = _split_top_level(params)
            for n, p in enumerate(pieces):
                suffix = "," if n < len(pieces) - 1 else ")"
                out.append(pad + p.strip() + suffix + (tail if n == len(pieces) - 1 else ""))
            issues.append("函数原型改为每行一个参数")
            continue
        out.append(line)
    return "\n".join(out), issues


def _split_top_level(params: str) -> list[str]:
    """按顶层逗号切参数（跳过 <> () [] 内部）。"""
    parts, depth, cur = [], 0, []
    for ch in params:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def apply_review_fixes(doc_part: str, text: str, original_doc: str = "") -> tuple[str, list[str]]:
    """按章节套用评审口径的确定性修正；返回 (新文本, 说明)。"""
    issues: list[str] = []
    if not text:
        return text, issues

    text, i1 = strip_prompt_leak(text)
    issues += i1
    if i1:
        # 清完只剩标题 → 该节等于没产出，回退原文
        body = [ln for ln in text.split("\n")[1:] if ln.strip()]
        if not body and original_doc:
            head = next((ln.strip() for ln in text.split("\n") if ln.strip()), "")
            orig = _extract_section(original_doc, head) if head else ""
            if orig:
                issues.append("清除泄漏内容后章节为空，回退原文")
                return orig, issues

    text, i2 = fix_terms(text)
    issues += i2
    if doc_part == "constraint":
        text, i3 = normalize_single_item_list(text)
        issues += i3
    if doc_part == "function_proto":
        text, i4 = format_proto_one_param_per_line(text)
        issues += i4
    return text, issues


def guard_section(
    doc_part: str,
    generated_text: str,
    original_doc: str,
    evidence_text: str,
    api_stem: str = "",
    renames: dict[str, str] | None = None,
    code_text: str = "",
) -> tuple[str, list[str]]:
    """章节级符号守卫：生成的正文不得出现“本文档/证据都不支持”的符号。

    针对业务反馈的三类越界（不依赖模型能力，命中即整节回退原文）：

    1. 内部实现符号（``asc_lock_impl`` 等）出现在任何章节；
    2. 引言/正文提及其它接口（``asc_mmad_mx`` 写进 ``asc_mmad`` 文档）——
       ``asc_*`` 名必须已在本篇原文出现过，或就是本文档自身的接口；
    3. 编造限定的符号形式（``AscendC::Reg::Gather<T, U>``）——
       ``AscendC::`` 限定名必须能在“证据 + 原文”里找到；
    4. 调用示例里出现测试桩（``EXPECT_EQ`` / ``*_stub`` / ``TEST_F``）。

    ``function_proto`` 章节另行走 :func:`guard_function_proto`（原型行逐条校验）。
    """
    issues: list[str] = []
    if not generated_text:
        return generated_text, issues

    heading = next((ln.strip() for ln in generated_text.splitlines() if ln.strip().startswith("## ")), "")
    corpus = f"{evidence_text or ''}\n{original_doc or ''}"
    # 白名单按**章节**取：同一份文档别处提到兄弟接口是允许的，
    # 但本章节原文没有出现的接口名，属于“凭空扩写”，应拦截。
    orig_section = _extract_section(original_doc or "", heading) if heading else ""
    known_asc = set(_ASC_API.findall(orig_section or ""))
    if api_stem:
        known_asc.add(api_stem)
    known_ns = set(_ASC_NS.findall(corpus))

    body = "\n".join(generated_text.splitlines()[1:])

    # ② 头文件信息保全：本节的“头文件”行按规范应写在功能说明首段，
    #    这里只记录，由 ensure_header_statement() 在回填后统一补到功能说明。
    if heading and not heading.startswith(_FUNC_HEADING):
        if any(_HEADER_LINE.match(ln) for ln in body.splitlines()):
            issues.append("头文件行不在功能说明章节：由功能说明统一声明")

    bad_internal = sorted(set(_INTERNAL_SYM.findall(body)))
    if bad_internal:
        issues.append("正文出现内部实现符号: " + ", ".join(bad_internal))

    # 证据缺失时不做“无据”判定：语料不完整会把合法新增误判为编造，
    # 只保留不依赖语料的内部符号检查。
    if evidence_text and evidence_text.strip():
        rename_new = set((renames or {}).values())

        def _is_renamed_enum(name: str) -> bool:
            """PR 重命名后的枚举值写法（ns::Type::NEW_VALUE）视为合法。"""
            segs = name.split("::")
            return len(segs) >= 2 and segs[-1] in rename_new and segs[-2] in (original_doc or "")

        bad_asc = sorted({n for n in _ASC_API.findall(body) if n not in known_asc})
        if bad_asc:
            issues.append("提及本篇未出现的接口: " + ", ".join(bad_asc))

        bad_ns = sorted({n for n in _ASC_NS.findall(body) if n not in known_ns and not _is_renamed_enum(n)})
        if bad_ns:
            issues.append("编造的限定符号: " + ", ".join(bad_ns))

        # 多级限定名（如 AscendC::Reg::Gather）必须已出现在原文中：
        # 实现文件里的内部调用（Reg::Gather(...)）不构成对外接口原型依据。
        # 例外：PR 把枚举值重命名了（NEAREST_ZERO → NEAREST_AWAY）时，
        #       `命名空间::类型::新值` 是合法写法，只要该类型名在原文出现过。
        bad_qualified = sorted(
            {
                q
                for q in _QUALIFIED.findall(body)
                if q not in (original_doc or "")
                and not (
                    len(q.split("::")) >= 2
                    and q.split("::")[-1] in rename_new
                    and q.split("::")[-2] in (original_doc or "")
                )
            }
        )
        if bad_qualified:
            issues.append("编造的多级限定名: " + ", ".join(bad_qualified))

        # 证据串味：家族头会把兄弟接口的约束/参数聚合进同一份 anchor，
        # 模型容易整段抄过来。判据：驼峰符号必须出现在「原文」或「本次改动的代码」里。
        _allow_sym = (original_doc or "") + "\n" + (code_text or "")
        _stray = sorted({t for t in _CAMEL_SYM.findall(body) if t not in _allow_sym and len(t) >= 12})
        if _stray:
            issues.append("疑似串味符号（原文与本次改动均无）: " + ", ".join(_stray[:6]))

    if doc_part == "example":
        stub = sorted(set(_TEST_STUB.findall(body)))
        if stub:
            issues.append("调用示例混入测试桩: " + ", ".join(stub))
        # 示例不得“覆盖”原文示例：原文的每个代码块都必须还在（允许追加）。
        # 例外：块内只发生了 PR 的重命名（NEAREST_ZERO→NEAREST_AWAY）时，
        #       视为“合法更新”而非“覆盖”——否则会把该更新的示例退回过期写法。
        orig_example = _extract_section(original_doc or "", heading) if heading else ""
        old_blocks = _code_blocks(orig_example)
        # 本次存在标识符重命名时，示例内容本就应该跟着变：不再用“原文代码块必须在”
        # 判覆盖，否则会把合法更新退回过期写法（5081 的 NEAREST_ZERO 就是这样回来的）。
        if old_blocks and not renames:
            new_blocks = _code_blocks(body)

            def _renamed(text: str) -> str:
                for old, new in (renames or {}).items():
                    text = re.sub(rf"\b{re.escape(old)}\b", new, text)
                return text

            missing = [b for b in old_blocks if b not in new_blocks and _renamed(b) not in new_blocks]
            if missing:
                issues.append(f"示例覆盖了原文内容：丢失原文代码块 {len(missing)}/{len(old_blocks)} 个")

    # 原型章节：在通用符号检查之外，再做逐条原型校验（内部符号/无据标识符）
    if doc_part == "function_proto":
        _, proto_issues = guard_function_proto(generated_text, original_doc, evidence_text)
        issues.extend(proto_issues)

    if issues and heading:
        orig_body = _extract_section(original_doc, heading).strip()
        if orig_body:
            return f"{heading}\n\n{orig_body}\n", issues

    return generated_text, issues
