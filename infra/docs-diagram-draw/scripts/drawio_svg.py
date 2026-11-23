#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
drawio_svg.py — CANN 文档配图工具链（docs-diagram-draw skill）

子命令:
  render  调用 drawio 官方 CLI 导出 .drawio.svg（drawio -x -f svg -e）：
          嵌入绘图副本（content 属性）、嵌入图片、嵌入字体（官方默认行为：
          SVG 中 text 固化成 image，跨环境字面一致）——全部为 drawio 官方行为，
          本脚本不实现渲染。
  extract 调用 drawio 官方 CLI 从 .drawio.svg 提取 drawio XML（drawio -x -f xml）。

render/extract 依赖 drawio 桌面版 CLI（Linux 无显示环境需 xvfb-run，本脚本自动包装）；
edit 仅依赖标准库（render 前自动获取 HarmonyOS Sans 装入系统字体，供 drawio 固化文字）。
"""

import argparse
import base64
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
import zlib
import xml.etree.ElementTree as ET

FONT_URL = (
    "https://alliance-communityfile-drcn.dbankcdn.com/FileServer/getFile/cmtyManage/011/111/1"
    "11/0000000000011111111.20260627152129.89276966309836366526585265125586:50001231000000:28"
    "00:A0161E048334FE0271F9F5ECBBD5070D17381C7846125F4D7109DBC7B532C715.zip?needInitFileName"
    "=true"
)


_NO_DOWNLOAD = os.environ.get("DOCS_DIAGRAM_NO_DOWNLOAD") == "1"


def locate_fonts():
    """定位/下载 HarmonyOS Sans 双字体（拉丁+简中），返回已就绪 ttf 路径列表。"""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "docs-diagram-draw")
    targets = ("HarmonyOS_Sans.ttf", "HarmonyOS_Sans_SC.ttf")
    out = [
        os.path.join(cache_dir, t)
        for t in targets
        if os.path.isfile(os.path.join(cache_dir, t))
    ]
    if len(out) == len(targets):
        return out
    if _NO_DOWNLOAD:
        sys.stderr.write("[warn] DOCS_DIAGRAM_NO_DOWNLOAD=1，跳过字体下载\n")
        return out
    try:
        import tempfile

        sys.stderr.write("[info] 本地 HarmonyOS Sans 字体不全，从官方渠道下载...\n")
        req = urllib.request.Request(FONT_URL, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        with tempfile.TemporaryDirectory() as td:
            zpath = os.path.join(td, "f.zip")
            with open(zpath, "wb") as f:
                f.write(data)
            import zipfile

            os.makedirs(cache_dir, exist_ok=True)
            with zipfile.ZipFile(zpath) as zf:
                names = zf.namelist()
                for t in targets:
                    hit = next((n for n in names if n.endswith(t)), None)
                    dest = os.path.join(cache_dir, t)
                    if hit and not os.path.isfile(dest):
                        with zf.open(hit) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
                    if os.path.isfile(dest) and dest not in out:
                        out.append(dest)
    except Exception as e:
        sys.stderr.write("[warn] 字体下载失败: %s\n" % e)
    return out


def _inflate(data):
    """drawio 压缩 diagram 文本（raw deflate + base64url）→ XML 字符串。"""
    pad = "=" * (-len(data) % 4)
    raw = base64.urlsafe_b64decode(data + pad)
    return zlib.decompress(raw, -15).decode("utf-8")


# ---------------------------------------------------------------------------
# drawio 官方 CLI 包装（render / extract）
#
# 版本固定 32.4.1（行为一致性保证）：
# - Linux：缺 drawio 或版本 ≠ 32.4.1 时自动下载安装 32.4.1（AppImage 缓存于
#   ~/.cache/docs-diagram-draw/drawio/，无 fuse 环境自动解压运行，无需 root）
# - Windows/macOS：要求用户安装 32.4.1（打印下载指引），自动发现常见安装路径；
#   检测到非 32.4.1 时告警
# - DRAWIO_BIN 显式指定的可执行文件不校验版本（用户自担）
# ---------------------------------------------------------------------------
DRAWIO_VERSION = "32.4.1"
DRAWIO_DL_BASE = (
    "https://github.com/jgraph/drawio-desktop/releases/download/v"
    + DRAWIO_VERSION
    + "/"
)
_DRAWIO_LINUX_ASSET = {
    "x86_64": "drawio-x86_64-%s.AppImage",
    "aarch64": "drawio-arm64-%s.AppImage",
}
_DRAWIO_LINUX_SYS = [
    "drawio",
    "/opt/drawio/drawio",
    "/usr/bin/drawio",
    "/usr/local/bin/drawio",
    "/snap/bin/drawio",
]
_DRAWIO_WINDOWS_SYS = [
    r"C:\Program Files\draw.io\draw.io.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\draw.io\draw.io.exe"),
]
_DRAWIO_MAC_SYS = ["/Applications/draw.io.app/Contents/MacOS/draw.io"]


def _drawio_cache_dir():
    return os.path.join(
        os.path.expanduser("~"), ".cache", "docs-diagram-draw", "drawio"
    )


def _cached_drawio():
    """缓存中的 32.4.1：已解压 AppRun 优先（无 fuse 环境），其次 AppImage 本体。"""
    d = _drawio_cache_dir()
    apprun = os.path.join(d, "squashfs-root", "AppRun")
    if os.path.isfile(apprun) and os.access(apprun, os.X_OK):
        return apprun
    asset = _DRAWIO_LINUX_ASSET.get(platform.machine())
    if asset:
        appimage = os.path.join(d, asset % DRAWIO_VERSION)
        if os.path.isfile(appimage) and os.access(appimage, os.X_OK):
            return appimage
    return None


def _wrap_xvfb(cmd):
    """Linux 无显示环境且无可用缓存路径时用 xvfb-run 包装。"""
    if (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and shutil.which("xvfb-run")
    ):
        return ["xvfb-run", "-a"] + cmd
    return cmd


def _drawio_version(bin_):
    """跑 --version 解析版本号（失败返回 None）。"""
    try:
        r = subprocess.run(
            _wrap_xvfb([bin_, "--version"]), capture_output=True, text=True, timeout=30
        )
        m = re.search(r"(\d+\.\d+\.\d+)", (r.stdout or "") + (r.stderr or ""))
        return m.group(1) if m else None
    except Exception:
        return None


def _install_drawio_linux():
    """Linux 自动下载安装 32.4.1 AppImage 到缓存目录（无需 root）。"""
    d = _drawio_cache_dir()
    os.makedirs(d, exist_ok=True)
    asset = _DRAWIO_LINUX_ASSET.get(platform.machine())
    if not asset:
        sys.stderr.write(
            "[错误] 不支持的架构 %s（支持 x86_64/aarch64）\n" % platform.machine()
        )
        return None
    url = DRAWIO_DL_BASE + asset % DRAWIO_VERSION
    appimage = os.path.join(d, asset % DRAWIO_VERSION)
    if not os.path.isfile(appimage):
        if _NO_DOWNLOAD:
            sys.stderr.write(
                "[错误] DOCS_DIAGRAM_NO_DOWNLOAD=1，跳过 drawio 自动下载；"
                "请预装 %s 或取消该环境变量\n" % DRAWIO_VERSION
            )
            return None
        sys.stderr.write("[info] 下载 drawio %s（%s）...\n" % (DRAWIO_VERSION, url))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with (
                urllib.request.urlopen(req, timeout=180) as resp,
                open(appimage + ".part", "wb") as f,
            ):
                total = int(resp.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        sys.stderr.write("\r  %d/%d MB" % (got >> 20, total >> 20))
                    else:
                        sys.stderr.write("\r  %d MB" % (got >> 20))
                sys.stderr.write("\n")
        except Exception as e:
            if os.path.isfile(appimage + ".part"):
                os.remove(appimage + ".part")
            sys.stderr.write("[错误] drawio 下载失败: %s\n" % e)
            return None
        os.rename(appimage + ".part", appimage)
    os.chmod(appimage, 0o755)
    # 试运行；无 fuse 环境解压一次后用 AppRun
    if _drawio_version(appimage):
        return appimage
    sys.stderr.write("[info] 无 fuse 环境，解压 AppImage 一次性运行...\n")
    subprocess.run(
        [appimage, "--appimage-extract"], cwd=d, capture_output=True, timeout=120
    )
    apprun = os.path.join(d, "squashfs-root", "AppRun")
    if os.path.isfile(apprun) and _drawio_version(apprun):
        return apprun
    sys.stderr.write(
        "[错误] drawio %s 安装失败（可删除 %s 后重试）\n" % (DRAWIO_VERSION, d)
    )
    return None


def _drawio_bin():
    """定位 drawio CLI；固定版本 32.4.1。

    Linux：缓存 32.4.1 → 系统 drawio（版本相符才用，否则触发自动安装）；
    Windows/macOS：系统安装路径（不自动下载，缺失时打印安装指引）。"""
    env = os.environ.get("DRAWIO_BIN")
    if env and os.path.isfile(env):
        return env, None
    if sys.platform.startswith("linux"):
        cached = _cached_drawio()
        if cached:
            return cached, None
        for c in _DRAWIO_LINUX_SYS:
            p = shutil.which(c) or (c if os.path.isfile(c) else None)
            if p:
                if _drawio_version(p) == DRAWIO_VERSION:
                    return p, None
                sys.stderr.write(
                    "[warn] 系统 drawio 版本 %s，本技能固定 %s，"
                    "自动下载安装固定版本...\n"
                    % (_drawio_version(p) or "未知", DRAWIO_VERSION)
                )
                break
        installed = _install_drawio_linux()
        if installed:
            return installed, None
        return None, "linux"
    cand = _DRAWIO_WINDOWS_SYS if os.name == "nt" else _DRAWIO_MAC_SYS
    for c in cand:
        if os.path.isfile(c):
            v = _drawio_version(c)
            if v and v != DRAWIO_VERSION:
                sys.stderr.write(
                    "[warn] 检测到 drawio %s（本技能固定 %s，"
                    "建议安装固定版本）\n" % (v, DRAWIO_VERSION)
                )
            return c, None
    return None, "manual"


def _manual_install_hint():
    plat = "Windows" if os.name == "nt" else "macOS"
    exe = (
        "draw.io-32.4.1-windows-installer.exe / draw.io-32.4.1.msi"
        if os.name == "nt"
        else "draw.io-x64-32.4.1.dmg / draw.io-universal-32.4.1.dmg"
    )
    sys.stderr.write(
        "[错误] 未找到 drawio CLI。%s 环境需要自行安装 drawio 桌面版（固定版本 %s）：\n"
        "  下载: https://github.com/jgraph/drawio-desktop/releases/tag/v%s\n"
        "  安装包: %s\n"
        "  安装后自动发现；非默认路径请设置 DRAWIO_BIN 环境变量指向 draw.io 可执行文件。\n"
        % (plat, DRAWIO_VERSION, DRAWIO_VERSION, exe)
    )


def _run_drawio(cli_args, out_path):
    """执行 drawio CLI；Linux 无显示环境自动用 xvfb-run 包装。

    drawio CLI 在容器内会向 stderr 打印 dbus 报错（无害噪音），成功判定以
    returncode 与输出文件为准。Electron 首次启动偶发一次性初始化失败
    （新解压的 AppImage 尤甚），失败自动重试一次。"""
    bin_, miss = _drawio_bin()
    if not bin_:
        if miss == "linux":
            sys.stderr.write("[错误] drawio %s 自动安装失败\n" % DRAWIO_VERSION)
        else:
            _manual_install_hint()
        return 1
    cmd = _wrap_xvfb([bin_] + cli_args)
    last_err = ""
    for attempt in (1, 2):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            sys.stderr.write("[错误] drawio CLI 超时（300s）\n")
            return 1
        if (
            r.returncode == 0
            and os.path.isfile(out_path)
            and os.path.getsize(out_path) > 0
        ):
            return 0
        last_err = "exit %d" % r.returncode
        if attempt == 1:
            sys.stderr.write("[warn] drawio CLI 失败（%s），重试一次...\n" % last_err)
    sys.stderr.write("[错误] drawio CLI 失败（%s）\n" % last_err)
    tail = (r.stderr or "").strip().splitlines()[-5:]
    for line in tail:
        sys.stderr.write("  " + line + "\n")
    return 1


def _font_installed_win_mac():
    """Windows/macOS 检测 HarmonyOS Sans 是否已装（按字体目录文件名近似判定）。"""
    home = os.path.expanduser("~")
    dirs = []
    if os.name == "nt":
        dirs += [
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
            os.path.join(
                os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Windows", "Fonts"
            ),
        ]
    else:  # macOS
        dirs += [
            os.path.join(home, "Library", "Fonts"),
            "/Library/Fonts",
            "/System/Library/Fonts",
        ]
    for d in dirs:
        try:
            for name in os.listdir(d):
                if "harmonyos" in name.lower():
                    return True
        except OSError:
            continue
    return False


def _ensure_system_font():
    """确保系统有 HarmonyOS Sans（drawio CLI 固化文字用系统字体渲染位图）。按平台分策略，
    与 drawio 依赖一致（Windows 用户自装、Linux 全自动）：

    Linux：fc-list 检测，无则（下载后）装入 ~/.fonts 并 fc-cache——全自动；
    Windows/macOS：按字体目录检测，已装静默；未装则自动把 TTF 下载到缓存目录，
    打印安装指引（Windows 双击安装 / macOS 拷入 ~/Library/Fonts），不代写注册表。"""
    if sys.platform.startswith("linux"):
        try:
            r = subprocess.run(["fc-list"], capture_output=True, text=True, timeout=30)
            if "HarmonyOS Sans" in (r.stdout or ""):
                return
        except Exception:
            return  # 无 fontconfig 的环境，跳过（drawio 自行回退）
        ttfs = locate_fonts()
        if not ttfs:
            sys.stderr.write(
                "[warn] 系统未安装 HarmonyOS Sans，且字体获取失败——drawio 导出的文字位图"
                "将使用回退字体，不符合'中英文统一 HarmonyOS Sans'规范\n"
            )
            return
        dest_dir = os.path.join(os.path.expanduser("~"), ".fonts")
        os.makedirs(dest_dir, exist_ok=True)
        for ttf in ttfs:
            dest = os.path.join(
                dest_dir,
                os.path.basename(ttf).replace(".ttf", "_docs-diagram-draw.ttf"),
            )
            if os.path.abspath(ttf) != os.path.abspath(dest):
                shutil.copyfile(ttf, dest)
        if shutil.which("fc-cache"):
            subprocess.run(
                ["fc-cache", "-f", dest_dir], capture_output=True, timeout=120
            )
        sys.stderr.write(
            "[info] HarmonyOS Sans（拉丁+简中）已装入 %s（drawio 文字固化可用）\n"
            % dest_dir
        )
        return
    # Windows / macOS
    if _font_installed_win_mac():
        return
    ttfs = locate_fonts()
    if not ttfs:
        sys.stderr.write(
            "[warn] 系统未安装 HarmonyOS Sans，且字体下载失败——drawio 导出的文字位图"
            "将使用回退字体\n"
        )
        return
    for ttf in ttfs:
        if os.name == "nt":
            sys.stderr.write(
                "[提示] 系统未安装 HarmonyOS Sans，字体已下载到：\n  %s\n"
                '  双击该文件并选择"安装"（仅当前用户即可）。\n' % ttf
            )
        else:
            sys.stderr.write(
                "[提示] 系统未安装 HarmonyOS Sans，字体已下载到：\n  %s\n"
                "  将该文件拷入 ~/Library/Fonts 即完成安装（macOS 无需注册）。\n" % ttf
            )
    if os.name == "nt":
        sys.stderr.write(
            "  官方发布页：https://developer.huawei.com/consumer/cn/design/harmonyos-symbol/\n"
        )


def cmd_render(args):
    """渲染 = drawio 官方 CLI：-x -f svg -e（嵌入绘图副本/图片/字体，官方默认行为）。"""
    _ensure_system_font()
    out = args.output or (re.sub(r"\.(drawio|xml)$", "", args.input) + ".drawio.svg")
    rc = _run_drawio(["-x", "-f", "svg", "-e", args.input, "-o", out], out)
    if rc == 0:
        print("已生成: %s（%.1f KB）" % (out, os.path.getsize(out) / 1024.0))
    return rc


def cmd_extract(args):
    """提取 = drawio 官方 CLI：-x -f xml（从 .drawio.svg 取回 drawio XML）。"""
    if not args.output:
        sys.stderr.write("[错误] extract 需要 -o 指定输出 .drawio 文件\n")
        return 1
    rc = _run_drawio(["-x", "-f", "xml", args.input, "-o", args.output], args.output)
    if rc == 0:
        print("已提取: %s" % args.output)
    return rc


# ---------------------------------------------------------------------------
# edit 子命令：按指令修改 drawio XML（改图工作流的确定性执行器）
#
# 白名单 = 改图范围矩阵中标 ✅/🔶 的键；白名单外一律拒绝（把矩阵从文档约束
# 升级为代码强制）。原子性：全部指令校验通过才执行，任一非法则整体拒绝。
# ---------------------------------------------------------------------------
EDIT_STYLE_KEYS = {
    # 顶点/连线颜色（矩阵 3.1）
    "fillColor",
    "strokeColor",
    "fontColor",
    "labelBackgroundColor",
    # 字体（矩阵 2.1.2 / 3.1.2）
    "fontFamily",
    "fontSize",
    # 内容节点圆角三键（矩阵 2.1.3 白名单）
    "rounded",
    "arcSize",
    "absoluteArcSize",
    # 连线视觉（矩阵 3.4）
    "strokeWidth",
    "dashed",
    "dashPattern",
    "startArrow",
    "endArrow",
    "startSize",
    "endSize",
    "startFill",
    "endFill",
    # 连线路由键（矩阵 2.3 边属性 🔶 点名可改：改线走线/锚点，仅点名场景）
    "edgeStyle",
    "curved",
    "bezier",
    "exitX",
    "exitY",
    "exitDx",
    "exitDy",
    "exitPerimeter",
    "entryX",
    "entryY",
    "entryDx",
    "entryDy",
    "entryPerimeter",
}
EDIT_GEO_KEYS = {"x", "y", "width", "height"}  # 矩阵 2.2（点名放行，仅顶点）


def _edit_load_model(path):
    """读 drawio 文件返回 (tree, model_el)。压缩 diagram 解包为明文子元素形态。"""
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag == "mxfile":
        diagram = root.find("diagram")
        if diagram is None:
            raise ValueError("mxfile 内未找到 diagram")
        model = diagram.find("mxGraphModel")
        if model is None:
            inner = _inflate((diagram.text or "").strip())
            model = ET.fromstring(inner)
            diagram.text = None
            diagram.append(model)
        return tree, model
    if root.tag == "mxGraphModel":
        return tree, root
    raise ValueError("无法识别的 drawio 文件结构（根元素 %s）" % root.tag)


def _edit_cells(model):
    """root 下的 mxCell 列表，附带 UserObject 包装信息。"""
    root = model.find("root")
    if root is None:
        raise ValueError("mxGraphModel 内未找到 root")
    out = []
    for owner in list(root):
        if owner.tag == "UserObject":
            mx = owner.find("mxCell")
            if mx is not None:
                out.append((mx, owner))
        elif owner.tag == "mxCell":
            out.append((owner, owner))
    return out


def _style_put(style, key, value):
    """style 串中替换/追加一个键值（保持其余段原样，最小触碰）。"""
    if not style:
        return key + "=" + value
    parts = style.split(";")
    out = []
    hit = False
    for p in parts:
        if p.strip() == key or p.strip().startswith(key + "="):
            out.append(key + "=" + value)
            hit = True
        else:
            out.append(p)
    if not hit:
        while out and out[-1] == "":
            out.pop()
        out.append(key + "=" + value)
    return ";".join(out)


def _style_get(style, key):
    for p in (style or "").split(";"):
        if p.strip().startswith(key + "="):
            return p.split("=", 1)[1]
    return None


def cmd_edit(args):
    """按 --set 指令修改存量元素；新增节点/连线仍由工作流手写 XML。"""
    specs = []
    for raw in args.set:
        seg = raw.split(",", 1)  # id=X,KEY=VALUE（VALUE 可含逗号）
        if len(seg) < 2 or not seg[0].startswith("id="):
            sys.stderr.write("[错误] 指令格式: id=<元素id>,<键>=<值>，收到: %s\n" % raw)
            return 1
        cid = seg[0][3:]
        kv = seg[1].split("=", 1)
        if len(kv) != 2 or not kv[0]:
            sys.stderr.write("[错误] 指令缺少 <键>=<值>: %s\n" % raw)
            return 1
        key, value = kv[0].strip(), kv[1]
        if key in EDIT_STYLE_KEYS and ";" in value:
            sys.stderr.write(
                "[错误] 样式键 %s 的值不得包含 ';'（会注入矩阵外样式）: %s\n"
                % (key, value)
            )
            return 1
        if key == "value" or key in EDIT_STYLE_KEYS or key in EDIT_GEO_KEYS:
            specs.append((cid, key, value))
        else:
            sys.stderr.write(
                "[错误] 键 %s 不在可改白名单（改图范围矩阵外属性一律保留原值）\n"
                "  白名单: value / %s / %s（几何仅顶点）\n"
                % (
                    key,
                    " ".join(sorted(EDIT_STYLE_KEYS)),
                    " ".join(sorted(EDIT_GEO_KEYS)),
                )
            )
            return 1
    tree, model = _edit_load_model(args.input)
    cells = _edit_cells(model)
    by_id = {}
    for mx, owner in cells:
        # UserObject 包装的元素 id 在包装层（mxCell 通常无 id）
        by_id[owner.get("id") or mx.get("id")] = (mx, owner)

    # ---- 第一遍：全量校验（原子性）+ 快照 ----
    plan = []
    for cid, key, value in specs:
        if cid not in by_id:
            sys.stderr.write("[错误] 元素 id 不存在: %s\n" % cid)
            return 1
        mx, owner = by_id[cid]
        if key in EDIT_GEO_KEYS:
            if mx.get("vertex") != "1":
                sys.stderr.write(
                    "[错误] 几何键 %s 仅适用于顶点（id=%s）\n" % (key, cid)
                )
                return 1
            g = mx.find("mxGeometry")
            if g is None:
                sys.stderr.write("[错误] 元素无几何（id=%s）\n" % cid)
                return 1
            old = g.get(key)
        elif key == "value":
            old = (
                owner.get("label")
                if owner is not mx and owner.get("label")
                else mx.get("value")
            )
        else:
            old = _style_get(mx.get("style"), key)
        plan.append((mx, owner, cid, key, value, old))

    # ---- 第二遍：执行 ----
    for mx, owner, cid, key, value, old in plan:
        if key in EDIT_GEO_KEYS:
            mx.find("mxGeometry").set(key, value)
        elif key == "value":
            mx.set("value", value)
            if owner is not mx and owner.get("label") is not None:
                owner.set("label", value)
        else:
            mx.set("style", _style_put(mx.get("style"), key, value))
        print(
            "[edit] %s: %s  %s -> %s"
            % (cid, key, old if old not in (None, "") else "(未设置)", value)
        )

    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    n_el = len(set(p[2] for p in plan))
    print("已修改 %d 处（%d 个元素），保存: %s" % (len(plan), n_el, args.output))
    return 0


def main():
    ap = argparse.ArgumentParser(description="CANN 文档配图工具链（docs-diagram-draw）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "render", help="渲染 drawio XML 为 .drawio.svg（调用 drawio 官方 CLI）"
    )
    p.add_argument("input", help="输入 .drawio/.xml 文件")
    p.add_argument("-o", "--output", help="输出路径（默认 输入名.drawio.svg）")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser(
        "extract", help="从 .drawio.svg 提取 drawio XML（调用 drawio 官方 CLI）"
    )
    p.add_argument("input", help="输入 .drawio.svg / .drawio / .xml")
    p.add_argument("-o", "--output", required=True, help="输出 .drawio 文件")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser(
        "edit", help="按指令修改 drawio XML（白名单键校验+快照+diff 摘要）"
    )
    p.add_argument("input", help="输入 .drawio/.xml 文件")
    p.add_argument(
        "-o", "--output", required=True, help="输出 .drawio 文件（另存，不就地覆盖）"
    )
    p.add_argument(
        "--set",
        action="append",
        required=True,
        metavar="id=<id>,<键>=<值>",
        help="修改指令，可重复；键白名单：value、颜色/字体/圆角/线型/箭头键、几何键 x/y/width/height（仅顶点）；白名单外拒绝执行",
    )
    p.set_defaults(func=cmd_edit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
