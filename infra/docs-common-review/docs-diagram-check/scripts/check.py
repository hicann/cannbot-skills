#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""docs-diagram-check 图形规范性检查执行器

检查对象:
  *.drawio.svg  drawio"包含副本"导出的图形（解析内嵌mxfile，检查设计参数）
  *.md          其中的 ```mermaid 代码块（源码文本级检查，不渲染）

用法:
  python check.py --scan <目录>                # 目录/仓库全扫
  python check.py --files <清单文件>           # PR变更文件模式（每行一个路径）
  python check.py --scan <目录> --json         # JSON输出（供文档检查工具集成）
  python check.py --scan <目录> --rules <目录> # 指定规则目录

报告: 文件 | 位置 | 规则ID | 级别(must/recommended/info) | 期望 | 实际
"""

import argparse
import base64
import collections
import html
import json
import os
import re
import sys
import time as _t
import zlib

try:
    from urllib.parse import unquote
except ImportError:
    unquote = None

RULES_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "rules"
)

# 成员环境多为 GBK 控制台：不可编码字符以 ? 替代输出，避免 print 直接崩溃
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass


def load_rules(rules_dir):
    with open(os.path.join(rules_dir, "drawio_rules.json"), encoding="utf-8") as f:
        drawio = json.load(f)
    with open(os.path.join(rules_dir, "mermaid_rules.json"), encoding="utf-8") as f:
        mermaid = json.load(f)
    return drawio, mermaid


def palette_hex_set(drawio_rules):
    """25色白名单: 返回大写hex集合(不含#)"""
    p = drawio_rules["palette"]
    s = set()
    for group in ("default", "neutral", "highlight", "stencil"):
        for item in p.get(group, []):
            s.add(item["hex"] if isinstance(item, dict) else item)
    return {c.upper() for c in s}


def palette_rgb_set(drawio_rules):
    """25色的 rgb(r,g,b) 形式白名单"""
    p = drawio_rules["palette"]
    out = set()
    for group in ("default", "neutral", "highlight", "stencil"):
        for item in p.get(group, []):
            hx = (item["hex"] if isinstance(item, dict) else item).upper()
            r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            out.add("rgb(%d, %d, %d)" % (r, g, b))
            out.add("rgb(%d,%d,%d)" % (r, g, b))
    return out


def norm_color(v):
    """颜色归一：drawio 的 light-dark(亮,暗) 双态取亮色（暗色变体不计入规范比对），
    'default' 表示继承默认（视为合规，返回 None 交由调用方豁免）。
    """
    if not v:
        return v
    m = re.match(r"light-dark\(\s*([^,()]+?)\s*,", v)
    return m.group(1) if m else v


def rgb_to_hex(v):
    """rgb(r, g, b) 字符串归一为大写 hex；非法返回 None"""
    m = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", v)
    if not m:
        return None
    return "%02X%02X%02X" % (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def hex6(v):
    """3/6 位 hex 归一为大写 6 位；非法返回 None"""
    v = v.strip().lstrip("#").upper()
    if len(v) == 3:
        v = "".join(ch * 2 for ch in v)
    return v if re.fullmatch(r"[0-9A-F]{6}", v) else None


def parse_style(style_str):
    d = {}
    for part in (style_str or "").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, _, v = part.partition("=")
            d[k.strip()] = v.strip()
        elif "shape" not in d:
            d["shape"] = part
    return d


def find_files(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".drawio.svg") or fn.endswith(".md"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


# ---------------- drawio 检查 ----------------


def decode_diagram_content(content):
    """解码<diagram>内容: drawio压缩(base64+deflate+encodeURIComponent)或明文XML"""
    model = None
    for attempt in ("direct", "unquote"):
        try:
            c = (
                content
                if attempt == "direct"
                else (unquote(content) if unquote else content)
            )
            model = zlib.decompress(base64.b64decode(c), -15).decode("utf-8")
            break
        except Exception:
            pass
    if model is None and content.lstrip().startswith("<"):
        model = content
    if model is not None and not model.lstrip().startswith("<") and "%" in model:
        try:
            model = unquote(model)
        except Exception:
            pass
    return model


def extract_pages(svg_text):
    """从.drawio.svg中提取图形数据; 返回 [(page_name, model_xml)] 或 None(无副本)

    支持两种drawio导出形式:
    1. <mxfile>子元素（diagram内容为压缩串）
    2. svg根的content属性（HTML转义的mxfile，drawio"包含副本"导出）
    """
    pages = None
    m = re.search(r"<mxfile\b[^>]*>(.*?)</mxfile>", svg_text, re.S)
    if m:
        pages = []
        for dm in re.finditer(r"<diagram\b([^>]*)>(.*?)</diagram>", m.group(1), re.S):
            attrs, content = dm.group(1), dm.group(2).strip()
            nm = re.search(r'name="([^"]*)"', attrs)
            name = nm.group(1) if nm else "page%d" % (len(pages) + 1)
            model = decode_diagram_content(content)
            if model:
                pages.append((name, model))
    if not pages:
        # 形式2: svg根的content属性（drawio"包含副本"导出的mxfile副本，HTML转义）
        m2 = re.search(r'<svg[^>]*\bcontent="([^"]*)"', svg_text)
        if m2:
            mxfile_xml = html.unescape(m2.group(1))
            pages = []
            for dm in re.finditer(
                r"<diagram\b([^>]*)>(.*?)</diagram>", mxfile_xml, re.S
            ):
                attrs, content = dm.group(1), dm.group(2).strip()
                nm = re.search(r'name="([^"]*)"', attrs)
                name = nm.group(1) if nm else "page%d" % (len(pages) + 1)
                model = decode_diagram_content(content)
                if model:
                    pages.append((name, model))
        if not pages:
            pages = None
    return pages


def parse_cells(model_xml):
    """解析mxGraphModel中的mxCell: 返回 [{id,parent,style_str,style,value,vertex,edge,geo}]
    UserObject/object 包裹的 mxCell 无 id 属性（id 在包装元素上，vsdx 导入常见），
    此处继承包装元素的 id；完全无 id 的兜底为 _noid_N。
    """
    cells = []
    # 包裹区间 (open_pos, close_pos, id)
    intervals = []
    for w in re.finditer(
        r'<(UserObject|object)\b[^>]*?id="([^"]*)"[^>]*?(/?)>', model_xml
    ):
        if w.group(0).endswith("/>"):
            continue  # 自闭合包裹内不含 mxCell
        end = w.end()
        cm = re.search(r"</%s\s*>" % w.group(1), model_xml[end:])
        if cm:
            intervals.append((w.start(), w.end() + cm.end(), w.group(2)))

    def wrapper_id_at(pos):
        for a, b, wid in intervals:
            if a <= pos < b:
                return wid
        return None

    for m in re.finditer(
        r"<mxCell\b([^>]*?)(?:/>|>((?:(?!<mxCell\b).)*?)</mxCell>)", model_xml, re.S
    ):
        attrs = m.group(1)
        body = m.group(2) or ""
        cell = {"attrs_raw": attrs, "body": body}
        for k, v in re.findall(r'(\w+)="([^"]*)"', attrs):
            cell[k] = v
        if "id" not in cell:
            wid = wrapper_id_at(m.start())
            cell["id"] = wid if wid else "_noid_%d" % len(cells)
        style_str = cell.get("style", "")
        cell["style_str"] = style_str
        cell["style"] = parse_style(style_str)
        cell["vertex"] = cell.get("vertex") == "1"
        cell["edge"] = cell.get("edge") == "1"
        g = re.search(r"<mxGeometry([^>]*)", body or "")
        geo = {}
        if g:
            for k, v in re.findall(r'(\w+)="([-\d.]+)"', g.group(1)):
                try:
                    geo[k] = float(v)
                except ValueError:
                    pass
        cell["geo"] = geo
        cells.append(cell)
    return cells


def bbox_of(cells, ids):
    xs, ys, xe, ye = [], [], [], []
    for c in cells:
        if c["id"] in ids and c["geo"]:
            xs.append(c["geo"].get("x", 0))
            ys.append(c["geo"].get("y", 0))
            xe.append(c["geo"].get("x", 0) + c["geo"].get("width", 0))
            ye.append(c["geo"].get("y", 0) + c["geo"].get("height", 0))
    if not xs:
        return None
    return (min(xs), min(ys), max(xe), max(ye))


def check_drawio_svg(path, rules, findings):
    d_rules = rules
    palette = palette_hex_set(d_rules)
    rgb_whitelist = palette_rgb_set(d_rules)
    font_whitelist = {c.upper() for c in d_rules.get("font_color_whitelist", [])}
    container_fills = {c.upper() for c in d_rules.get("container_fill_colors", [])}
    neutral_fills = {c.upper() for c in d_rules.get("neutral_fill_colors", [])}
    fs_min = d_rules["font_size"]["min"]
    hl_set = set()
    for i in d_rules["palette"]["highlight"]:
        hl_set.add((i["hex"] if isinstance(i, dict) else i).upper())

    text = open(path, encoding="utf-8", errors="replace").read()
    pages = extract_pages(text)

    if pages is None:
        findings.append(
            {
                "file": path,
                "type": "drawio",
                "rule": "INFO-EXPORT",
                "level": "info",
                "expected": "内嵌mxfile副本",
                "actual": "缺失",
                "message": '未包含drawio副本，设计参数检查降级（请用drawio桌面版勾选"包含副本"重新导出）',
            }
        )
        return

    for page_name, model in pages:
        cells = parse_cells(model)
        for c in cells:
            c["page"] = page_name
        check_page(
            path,
            page_name,
            cells,
            d_rules,
            palette,
            rgb_whitelist,
            font_whitelist,
            container_fills,
            neutral_fills,
            fs_min,
            findings,
        )


def check_page(
    path,
    page,
    cells,
    d_rules,
    palette,
    rgb_wl,
    font_wl,
    container_fills,
    neutral_fills,
    fs_min,
    findings,
):
    def add(cell, rule, level, expected, actual, message=""):
        findings.append(
            {
                "file": path,
                "page": page,
                "cell": cell.get("id", "?"),
                "rule": rule,
                "level": level,
                "expected": expected,
                "actual": actual[:120],
                "message": message,
            }
        )

    def is_exempt(cell):
        st = cell.get("style_str", "")
        return "vsdxID=" in st or "shape=stencil(" in st

    # 预解析富文本片段
    for c in cells:
        c["rt_fs"] = re.findall(r"font-size:\s*([0-9.]+)\s*px", c.get("value", ""))
        c["rt_color"] = re.findall(
            r"color:\s*(rgb\(\s*[0-9, ]+\s*\)|#[0-9A-Fa-f]{3,6})", c.get("value", "")
        )

    for c in cells:
        st = c["style"]
        exempt_art = is_exempt(c)
        # 结构单元格（页面/图层标记：无 vertex/edge 属性，如 value="页-1"/图层名）
        # 不参与渲染，跳过全部图元级检查
        if not c["vertex"] and not c["edge"]:
            continue

        # C01 颜色白名单（light-dark 双态归一为亮色后比对；3 位 hex 展开）
        for k in ("fillColor", "strokeColor"):
            v = norm_color(st.get(k, ""))
            vv = hex6(v)
            if vv and vv not in palette:
                add(c, "C01", "must", "∈25色白名单", v, "颜色不在色板白名单内")
        # 富文本颜色
        if not exempt_art:
            for col in c["rt_color"]:
                norm = re.sub(r"\s+", "", col)
                if norm.startswith("rgb("):
                    h6 = rgb_to_hex(norm)
                else:
                    h6 = hex6(norm)
                ok = h6 is not None and (h6 in font_wl or h6 in hl_set)
                if not ok:
                    add(
                        c,
                        "F03",
                        "must",
                        "文字色∈白名单rgb/hex",
                        col,
                        "label富文本文字色不在白名单",
                    )

        if exempt_art:
            continue  # Visio装饰图形：跳过字体/字号/几何检查（颜色已查）

        # F01 字体（有文字内容的图元）
        ff = st.get("fontFamily", "")
        has_text = (c.get("value", "") or "").strip() != "" or "text;" in c.get(
            "style_str", ""
        )
        if has_text and (
            not ff
            or d_rules.get("font_family", "HarmonyOS Sans").lower() not in ff.lower()
        ):
            add(c, "F01", "must", "HarmonyOS Sans", ff, "字体不符合规范")

        # F02 字号（属性）
        fs = st.get("fontSize", "")
        if fs:
            if not re.match(r"^\d+$", fs):
                add(c, "F02", "must", "整数字号且>=10px", fs, "字号为非整数或异常")
            elif int(fs) < fs_min:
                add(c, "F02", "must", ">=10px", fs, "字号低于下限")
            elif int(fs) > d_rules["font_size"]["highlight"]:
                add(
                    c,
                    "F02",
                    "recommended",
                    "<=%gpx(突出)" % d_rules["font_size"]["highlight"],
                    fs,
                    "字号超过突出文本规范值",
                )
        # F02 富文本字号
        for v in c["rt_fs"]:
            if "." in v:
                add(
                    c, "F02", "must", "整数字号且>=10px", v + "px", "富文本字号为非整数"
                )
            elif float(v) < fs_min:
                add(c, "F02", "must", ">=10px", v + "px", "富文本字号低于下限")
            elif float(v) > d_rules["font_size"]["highlight"]:
                add(
                    c,
                    "F02",
                    "recommended",
                    "<=%gpx(突出)" % d_rules["font_size"]["highlight"],
                    v + "px",
                    "富文本字号超过突出文本规范值",
                )

        # F03 fontColor 属性：先 norm_color（light-dark 亮色归一），rgb 归一为 hex，
        # 放行集合收敛为 font_wl ∪ highlight（与 expected 一致，修「声明与实际校验不一致」漏报）
        fc = norm_color(st.get("fontColor", ""))
        if fc and fc.lower() != "default":
            h6 = hex6(fc) if not fc.lower().startswith("rgb(") else rgb_to_hex(fc)
            ok = h6 is not None and (h6 in font_wl or h6 in hl_set)
            if not ok:
                add(
                    c,
                    "F03",
                    "must",
                    "∈{#202329,#FFFFFF,高亮色}",
                    fc,
                    "文字颜色不在白名单",
                )

        # N01/N03 节点几何（带彩色填充的矩形vertex）
        fill_raw = norm_color(st.get("fillColor", ""))
        fill = fill_raw.lstrip("#").upper()
        shape_name = st.get("shape", "")
        # 画布背景框：白底/默认色填充 + 无边框 + 大尺寸顶点（edraw-to-drawio 插入的
        # 画布框与组件库模板背景框同口径），不按节点几何检查
        is_canvas_bg = (
            c["vertex"]
            and fill in ("FFFFFF", "DEFAULT")
            and st.get("strokeColor", "") == "none"
            and (c["geo"].get("width") or 0) >= 300
            and (c["geo"].get("height") or 0) >= 100
        )
        # 非节点豁免：无填充(text/说明)、中性填充(次级/分组/背景)、判断菱形、
        # 超大图块(h≥100 分区/生命线)、细条(h≤20 图例色带)
        is_node = (
            c["vertex"]
            and fill
            and fill != "NONE"
            and fill not in container_fills
            and fill not in neutral_fills
            and fill != "DEFAULT"
            and shape_name not in ("rhombus", "text")
            and "ellipse" not in shape_name
            and "image" not in shape_name
            and not is_canvas_bg
            and 20 <= (c["geo"].get("height") or 0) <= 100
        )
        # F04 文字居中（仅色块节点；text 说明/中性填充/背景框豁免，与节点口径一致）
        if is_node and c.get("value", "").strip():
            al = st.get("align", "center")
            va = st.get("verticalAlign", "middle")
            if al != "center":
                add(c, "F04", "must", "align=center", al, "文字水平未居中")
            if va != "middle":
                add(c, "F04", "must", "verticalAlign=middle", va, "文字垂直未居中")

        # N01/N03 节点几何
        if is_node:
            h = c["geo"].get("height")
            if h is not None and abs(h - d_rules["node"]["height"]) > 0.01:
                add(
                    c,
                    "N01",
                    "recommended",
                    "height=32px",
                    "height=%g" % h,
                    "节点高度不符（建议改）",
                )
            if st.get("rounded") != "1":
                add(
                    c,
                    "N03",
                    "must",
                    "rounded=1(圆角4)",
                    "rounded=" + st.get("rounded", "absent"),
                    "节点未设置圆角",
                )

        # N02 开始/结束椭圆
        if c["vertex"] and "ellipse" in st.get("shape", ""):
            w = c["geo"].get("width")
            h = c["geo"].get("height")
            dia = d_rules["start_end"]["diameter"]
            if (
                w is not None
                and h is not None
                and (abs(w - dia) > 0.01 or abs(h - dia) > 0.01)
            ):
                add(
                    c,
                    "N02",
                    "must",
                    "width==height==%gpt" % dia,
                    "w=%g h=%g" % (w, h),
                    "开始/结束节点直径不符",
                )

        # S02 容器宽度
        if c["vertex"] and fill in container_fills:
            w = c["geo"].get("width")
            if w is not None and w > d_rules["container"]["max_width"]:
                add(c, "S02", "recommended", "<=1000px", "width=%g" % w, "容器宽度超限")

    # C02 单图内容色≤3（排除中性色：文字202329/边框686D75/中性填充/背景FFFFFF，
    # 继承默认DEFAULT，及流程开始&结束节点（椭圆）的填充色）
    neutral_all = {n.upper() for n in neutral_fills} | {"202329", "686D75", "DEFAULT"}
    page_fills = collections.defaultdict(set)
    for c in cells:
        f = norm_color(c["style"].get("fillColor", "")).lstrip("#").upper()
        if not f or f == "NONE":
            continue
        if c["vertex"] and "ellipse" in c["style"].get("shape", ""):
            continue  # 开始/结束节点
        if f not in neutral_all:
            page_fills[c["page"]].add(f)
    for pg, fs_set in page_fills.items():
        content_colors = {
            f
            for f in fs_set
            if f not in {s.upper() for s in d_rules["palette"]["stencil"]}
        }
        if len(content_colors) > 3:
            findings.append(
                {
                    "file": path,
                    "page": pg,
                    "rule": "C02",
                    "level": "recommended",
                    "expected": "默认色<=3种",
                    "actual": "%d种(%s)"
                    % (len(content_colors), ",".join(sorted(content_colors))),
                    "message": "单图颜色超过3种",
                }
            )
    # C03 高亮色提示（light-dark 归一后比对）
    page_hl = collections.defaultdict(int)
    for c in cells:
        f = norm_color(c["style"].get("fillColor", "")).lstrip("#").upper()
        if f in hl_set:
            page_hl[c["page"]] += 1
    for pg, n in page_hl.items():
        findings.append(
            {
                "file": path,
                "page": pg,
                "rule": "C03",
                "level": "recommended",
                "expected": "高亮色非必要不使用",
                "actual": "高亮色图元%d处" % n,
                "message": "图中有高亮色，请确认非必要使用且避免大面积",
            }
        )

    # S03 容器上下留白 >=40px（子图元坐标为相对容器原点的坐标）
    for c in cells:
        fill = c["style"].get("fillColor", "").lstrip("#").upper()
        if c["vertex"] and fill in container_fills:
            kids = [k for k in cells if k.get("parent") == c["id"]]
            bb = bbox_of(kids, {k["id"] for k in kids})
            geo = c["geo"]
            ch = geo.get("height")
            if bb and ch is not None:
                top_gap = bb[1]
                bottom_gap = ch - bb[3]
                if (
                    top_gap < d_rules["container"]["padding_vertical_min"]
                    or bottom_gap < d_rules["container"]["padding_vertical_min"]
                ):
                    add(
                        c,
                        "S03",
                        "recommended",
                        "上下留白>=40px",
                        "top=%g bottom=%g" % (round(top_gap, 1), round(bottom_gap, 1)),
                        "容器上下留白不足",
                    )
    # S04 图片容器
    for c in cells:
        w = c["geo"].get("width")
        if (
            c["vertex"]
            and w is not None
            and w >= 300
            and (
                "shape=image" in c.get("style_str", "")
                or c["style"].get("shape", "") == "image"
            )
        ):
            if int(round(w)) not in d_rules["image_container_widths"]:
                add(
                    c,
                    "S04",
                    "recommended",
                    "width∈{920,600,300}px",
                    "width=%g" % w,
                    "图片容器宽度不在三档内",
                )


# ---------------- mermaid 检查 ----------------

MERMAID_BLOCK = re.compile(r"```mermaid\s*\n(.*?)```", re.S)


def strip_init(block):
    m = re.search(r"%%\{init:\s*(\{.*?\})\s*\}%%", block, re.S)
    init = None
    if m:
        try:
            init = json.loads(m.group(1))
        except Exception:
            try:
                init = json.loads(m.group(1).replace("\n", ""))
            except Exception:
                init = None
        start, end = m.start(), m.end()
        block = block[:start] + block[end:]
    return block, init


def check_mermaid_block(path, idx, block, m_rules, findings):
    btype = (
        "flowchart"
        if re.search(r"^\s*(flowchart|graph)\b", block, re.M)
        else (
            "sequence"
            if re.search(r"^\s*sequenceDiagram\b", block, re.M | re.I)
            else "other"
        )
    )

    def add(rule, level, expected, actual, message):
        findings.append(
            {
                "file": path,
                "type": "mermaid",
                "block": idx,
                "rule": rule,
                "level": level,
                "expected": expected,
                "actual": actual[:160],
                "message": message,
            }
        )

    body, init = strip_init(block)

    # 仅 flowchart / sequence 在 mermaid_rules.json 的 scope 内；
    # 其他类型（pie/stateDiagram 等）跳过检查，避免按错误模板比对产生 must 级误报
    if btype not in m_rules.get("scope", {}):
        return

    # M02 init比对
    if init is None:
        add(
            "M02",
            "must",
            "存在%%{init:...}%%规范配置",
            "缺失",
            "缺少mermaid init配置或JSON格式错误",
        )
    else:
        base = (
            m_rules["flowchart_init"]
            if btype == "flowchart"
            else m_rules["sequence_init"]
        )
        tv = init.get("themeVariables", {})
        fc = init.get("flowchart", {})
        exp_tv = base.get("themeVariables", {})
        exp_fc = base.get("flowchart", {})
        optional = set(k.lower() for k in base.get("themeVariables_optional", []))
        alts = m_rules.get("font_family_alternatives", [])
        for k, exp_v in exp_tv.items():
            act_v = tv.get(k)
            if act_v is None:
                if k.lower() in optional:
                    continue
                add(
                    "M02",
                    "must",
                    "%s=%s" % (k, exp_v),
                    "缺失",
                    "init.themeVariables缺少规范配置项 " + k,
                )
            elif k == "fontFamily":
                if not any(act_v == a for a in alts) and act_v != exp_v:
                    add(
                        "M02",
                        "must",
                        "%s=%s" % (k, exp_v),
                        act_v,
                        "init.fontFamily与模板规范值不一致",
                    )
            elif str(act_v).lower() != str(exp_v).lower():
                add(
                    "M02",
                    "must",
                    "%s=%s" % (k, exp_v),
                    str(act_v),
                    "init.themeVariables与模板规范值不一致: " + k,
                )
        if btype == "flowchart":
            for k, exp_v in exp_fc.items():
                act_v = fc.get(k)
                if act_v is None:
                    add(
                        "M02",
                        "must",
                        "flowchart.%s=%s" % (k, exp_v),
                        "缺失",
                        "init.flowchart缺少规范配置项 " + k,
                    )
                elif str(act_v).lower() != str(exp_v).lower():
                    add(
                        "M02",
                        "must",
                        "flowchart.%s=%s" % (k, exp_v),
                        str(act_v),
                        "init.flowchart配置与模板规范值不一致: " + k,
                    )

    if btype == "flowchart":
        # M01 判断菱形
        if re.search(r"[\w\]\)\"]\s*\{", body):
            m = re.search(r"[\w\]\)\"]\s*\{[^}]*\}", body)
            add(
                "M01",
                "must",
                "无判断菱形节点",
                m.group(0)[:60] if m else "",
                "flowchart包含判断菱形节点及分支，mermaid不适用，请改用drawio流程图",
            )
        # M03 classDef颜色
        wl = {c.upper() for c in m_rules.get("classdef_fill_whitelist", [])}
        for line in body.splitlines():
            if line.strip().startswith("classDef"):
                for cm in re.finditer(r"fill:#([0-9A-Fa-f]{3,6})", line):
                    if hex6(cm.group(1)) not in wl:
                        add(
                            "M03",
                            "must",
                            "classDef填充色∈白名单",
                            "#" + cm.group(1).upper(),
                            "classDef填充色不在白名单",
                        )
                for cm in re.finditer(r"color:#([0-9A-Fa-f]{3,6})", line):
                    if hex6(cm.group(1)) not in wl:
                        add(
                            "M03",
                            "must",
                            "classDef文字色∈白名单",
                            "#" + cm.group(1).upper(),
                            "classDef文字色不在白名单",
                        )
        # M04 节点div样式
        divs = re.findall(r"<div style=\'([^\']*)\'>", body)
        for dv in divs:
            low = re.sub(r"\s+", "", dv.lower())
            if "width:40px" in low:
                for frag in m_rules["node_div_styles"]["start_end"][
                    "required_fragments"
                ]:
                    if frag.lower() not in low:
                        add(
                            "M04",
                            "must",
                            frag,
                            dv[:80],
                            "开始/结束节点div缺少必需样式片段: " + frag,
                        )
            elif "width:148px" in low:
                for frag in m_rules["node_div_styles"]["content"]["required_fragments"]:
                    if frag.lower() not in low:
                        add(
                            "M04",
                            "must",
                            frag,
                            dv[:80],
                            "内容节点div缺少必需样式片段: " + frag,
                        )
            # 颜色白名单
            for cm in re.finditer(r"color:\s*(#[0-9A-Fa-f]{3,6})", dv):
                if hex6(cm.group(1)) not in wl:
                    add(
                        "M03",
                        "must",
                        "div文字色∈白名单",
                        cm.group(1),
                        "节点div文字色不在白名单",
                    )
        # M05 连线标签（recommended提示）
        for m in re.finditer(r"-->[|]([^|]+)[|]", body):
            add(
                "M05",
                "recommended",
                "确认非分支语义",
                m.group(1),
                "带标签连线：请确认为普通标注而非分支",
            )


def check_mermaid_md(path, m_rules, findings):
    text = open(path, encoding="utf-8", errors="replace").read()
    idx = 0
    for m in MERMAID_BLOCK.finditer(text):
        idx += 1
        line_no = text[: m.start()].count("\n") + 1
        findings_ctx = len(findings)
        check_mermaid_block(path, idx, m.group(1), m_rules, findings)
        for i in range(findings_ctx, len(findings)):
            findings[i]["file"] = path + "#L%d(mermaid块%d)" % (line_no, idx)


# ---------------- 主流程 ----------------


def write_spec_report(
    rpt_dir, scope, rules_dir, files_checked, findings, rule_desc=None
):
    """生成图形规范性检查报告 spec_check_report.md（写入检查目录），结构与
    edraw-to-drawio 的 conversion_report.md 同风格：汇总（单表合并规则分布，含规则含义）+
    按文件分组明细。
    """
    from collections import Counter as _C

    rule_desc = rule_desc or {}
    rule_cnt = _C((f.get("level", "?"), f.get("rule", "?")) for f in findings)
    by_file = {}
    for f in findings:
        by_file.setdefault(f.get("file", "?"), []).append(f)
    lines = ["# 图形规范性检查报告", ""]
    lines.append("- 生成时间：%s" % _t.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("- 检查范围：%s" % scope)
    lines.append("- 规则基线：%s" % os.path.abspath(rules_dir))
    lines.append("")
    lines.append("## 一、汇总")
    lines.append("")
    lines.append("共检查 **%d** 个文件，规范性问题按规则汇总如下：" % files_checked)
    lines.append("")
    lines.append("| 项目 | 级别 | 数量 | 说明 |")
    lines.append("| :-- | :-- | :-- | :-- |")
    if not rule_cnt:
        lines.append("| 全部通过 | - | - | 未发现规范性问题 |")
    else:
        for (lv, rule), n in sorted(rule_cnt.items(), key=lambda x: (x[0][0], x[0][1])):
            if lv == "info":
                continue  # info 环境提示单独成行（附解决方案），避免重复
            desc = rule_desc.get(rule, "")
            lines.append("| %s | %s | %d | %s |" % (rule, lv, n, desc))
    # info 级环境提示：报因 + 解决方案单独成行
    info_solutions = {
        "INFO-EXPORT": '环境信息：图形未包含内嵌mxfile副本（drawio导出时未勾选"包含副本"或非桌面版导出），'
        '设计参数无法检查；解决：用 drawio 桌面版勾选"包含副本"重新导出后重查',
        "INFO-NOTFOUND": "环境信息：清单中的文件不存在，已跳过；解决：核对清单/目录中的文件路径后重查",
    }
    for (lv, rule), n in sorted(rule_cnt.items()):
        if lv == "info" and rule in info_solutions:
            lines.append("| %s | info | %d | %s |" % (rule, n, info_solutions[rule]))
    lines.append("")
    lines.append("## 二、问题明细（按文件）")
    lines.append("")
    if not findings:
        lines.append(
            "无规范性问题。info 级为环境提示（内嵌副本缺失/清单未找到），非规范性问题。"
        )
    else:
        for fpath in sorted(by_file):
            fs = by_file[fpath]
            cnt = _C(f["level"] for f in fs)
            lines.append(
                "### %s（must: %d, recommended: %d, info: %d）"
                % (
                    fpath,
                    cnt.get("must", 0),
                    cnt.get("recommended", 0),
                    cnt.get("info", 0),
                )
            )
            lines.append("")
            lines.append("| 规则 | 级别 | 期望 | 实际 | 处数 | 说明 |")
            lines.append("| :-- | :-- | :-- | :-- | :-- | :-- |")
            # 同文件内按 (规则/级别/期望/实际/说明) 聚合，"处数"为出现次数
            agg = _C(
                (
                    f.get("rule", ""),
                    f.get("level", ""),
                    f.get("expected", ""),
                    f.get("actual", ""),
                    f.get("message", ""),
                )
                for f in fs
            )
            for (rule, level, expected, actual, message), n in sorted(agg.items()):
                lines.append(
                    "| %s | %s | %s | %s | %d | %s |"
                    % (rule, level, expected, actual, n, message)
                )
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "info 级为环境提示（INFO-EXPORT 内嵌副本缺失 / INFO-NOTFOUND 清单未找到），非规范性问题；"
        "修复 must/recommended 项后重跑检查，直至全部通过。"
    )
    rpt_dir = os.path.abspath(rpt_dir)
    os.makedirs(rpt_dir, exist_ok=True)
    rp = os.path.join(rpt_dir, "spec_check_report.md")
    with open(rp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return rp


def main():
    ap = argparse.ArgumentParser(description="图形规范性检查")
    ap.add_argument("--scan", help="目录/仓库全扫")
    ap.add_argument("--files", help="变更文件清单（每行一个路径）或逗号分隔列表")
    ap.add_argument("--rules", default=RULES_DIR_DEFAULT, help="规则目录")
    ap.add_argument("--json", action="store_true", help="JSON输出")
    args = ap.parse_args()

    d_rules, m_rules = load_rules(args.rules)

    # 规则含义映射（报告汇总表直接呈现，免查 rules.json）
    rule_desc = {}
    for r in d_rules.get("rules", []) + m_rules.get("rules", []):
        if isinstance(r, dict) and r.get("id"):
            rule_desc[r["id"]] = r.get("desc", "")
    rule_desc.setdefault(
        "INFO-EXPORT", '未包含内嵌mxfile副本（drawio桌面版需勾选"包含副本"导出）'
    )
    rule_desc.setdefault("INFO-NOTFOUND", "清单文件不存在，已跳过")

    targets = []
    if args.scan:
        targets = find_files(args.scan)
    elif args.files:
        src = args.files
        if os.path.isfile(src):
            with open(src, encoding="utf-8") as fh:
                targets = [row.strip() for row in fh if row.strip()]
        else:
            targets = [s.strip() for s in src.split(",") if s.strip()]
        targets = [t for t in targets if t.endswith(".drawio.svg") or t.endswith(".md")]
    else:
        ap.error("需要 --scan 或 --files")

    findings = []
    files_checked = 0
    for t in targets:
        if not os.path.isfile(t):
            findings.append(
                {
                    "file": t,
                    "rule": "INFO-NOTFOUND",
                    "level": "info",
                    "expected": "文件存在",
                    "actual": "不存在",
                    "message": "文件不存在，跳过",
                }
            )
            continue
        files_checked += 1
        if t.endswith(".drawio.svg"):
            check_drawio_svg(t, d_rules, findings)
        elif t.endswith(".md"):
            check_mermaid_md(t, m_rules, findings)

    must_n = sum(1 for f in findings if f["level"] == "must")
    rec_n = sum(1 for f in findings if f["level"] == "recommended")
    info_n = sum(1 for f in findings if f["level"] == "info")

    if args.json:
        print(
            json.dumps(
                {
                    "summary": {
                        "files": files_checked,
                        "must": must_n,
                        "recommended": rec_n,
                        "info": info_n,
                        "total": len(findings),
                    },
                    "findings": findings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("===== 图形规范性检查报告 =====")
        print(
            "检查文件: %d | must问题: %d | recommended问题: %d | info: %d"
            % (files_checked, must_n, rec_n, info_n)
        )
        if findings:
            print()
            for f in findings:
                loc = f.get("file", "") + (("#" + f["cell"]) if f.get("cell") else "")
                print("[%s][%s] %s" % (f["level"].upper(), f["rule"], loc))
                print(
                    "   期望: %s | 实际: %s"
                    % (f.get("expected", ""), f.get("actual", ""))
                )
                if f.get("message"):
                    print("   %s" % f["message"])
        else:
            print("全部通过")

    # 生成报告文件（写入检查目录；JSON 模式静默生成，不污染 stdout）
    if args.scan:
        rpt_dir = args.scan
        scope = args.scan
    else:
        existing = [t for t in targets if os.path.isfile(t)]
        rpt_dir = (
            os.path.dirname(os.path.abspath(existing[0])) if existing else os.getcwd()
        )
        scope = args.files
    try:
        rp = write_spec_report(
            rpt_dir, scope, args.rules, files_checked, findings, rule_desc=rule_desc
        )
        if not args.json:
            print("\n报告: %s" % rp)
    except OSError as e:
        if not args.json:
            print("\n[warn] 报告写入失败: %s" % e)
    sys.exit(1 if must_n > 0 else 0)


if __name__ == "__main__":
    main()
