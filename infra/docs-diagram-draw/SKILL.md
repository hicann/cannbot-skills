---
name: docs-diagram-draw
description: CANN 文档配图自动绘图/改图工具。为 CANN 开源仓文档（如算子开发指南、API 参考、教程）生成、修改符合《绘图规范》的技术配图（流程图、时序图、架构图、组网图、原理图等），产出 drawio.svg 格式（含绘图副本、嵌入图片、文字固化为 image 位图——对齐 drawio 官方默认"嵌入字体"行为，可直接入库、可在 drawio 中二次编辑）。当用户提到以下场景时必须使用本技能：画图、绘图、作图、生成配图/插图/示意图、流程图、时序图、架构图、组网图、原理图、改图、修改图、调整图、drawio、drawio.svg、.drawio 文件、按绘图规范出图、文档配图，即使用户没有明说"绘图规范"。注意：一般性的图表/数据可视化（柱状图/折线图等）不属于本技能范围。
---

# docs-diagram-draw（CANN 文档自动绘图改图）

为 CANN 产出符合《绘图规范》的技术配图（规范源为 gitcode.com/cann/docs 仓，版本指纹见 `references/sources.manifest`）。
产出物为 **`.drawio.svg`**（渲染由 drawio 官方 CLI 完成）：SVG 画面 + 根元素 `content` 属性
内嵌绘图副本（drawio 可打开二次编辑）+ 图片内嵌 + **文字固化成 `<image>` 位图**
（drawio 官方默认"嵌入字体"行为：导出 SVG 时 text 变成 image；跨环境字面 100% 一致，
代价是文字不可选不可搜索）。

**技能范围**：只做"画新图"（工作流一）与"改图"（工作流二）。改图的可修改键白名单与逐键规则
内置 `references/xml-property-matrix.md`（矩阵外属性一律保留原值），改图属性以矩阵为准。

**三条铁律：**

1. **永远不手写 SVG 画面**。流程是"编写 drawio XML → 脚本渲染出 .drawio.svg"，手写 SVG 无法保证内嵌绘图副本、可再编辑与规范一致性。
2. **样式串一律从 `references/style-library.md` 原样复制**，只改文字、id、几何与连接，不自行拼凑样式、不发明新颜色。
3. **改图前先查 `references/xml-property-matrix.md` 改图范围矩阵**：可修改键清单（白名单），逐键标注权限（规范化/自定义两场景两态）与修改规则；矩阵外/无条款的属性一律保留原值（兜底）。规范源唯一二源（https://gitcode.com/cann/docs/blob/master/specification/drawing_specs.md 条款 + https://gitcode.com/cann/docs/blob/master/writing_template/drawing_template_drawio.xml 取值，版本指纹 `references/sources.manifest`）。

## 工具

脚本均位于本 skill 目录 `scripts/` 下（相对路径按 skill 根解析）。

```bash
# 渲染：调用 drawio 官方 CLI（drawio -x -f svg -e），嵌入绘图副本/图片/字体全为官方行为
python3 scripts/drawio_svg.py render <输入.drawio> [-o 输出.drawio.svg]

# 提取：调用 drawio 官方 CLI（drawio -x -f xml），从 .drawio.svg 取回 drawio XML（改图第一步）
python3 scripts/drawio_svg.py extract <输入.drawio.svg> -o <输出.drawio>

# 改图执行器：按指令修改存量元素（白名单键校验+快照+diff 摘要；不直接手改 XML）
python3 scripts/drawio_svg.py edit <提取稿.drawio> -o <改后稿.drawio> \
    --set id=<元素id>,<键>=<值>   # 可重复；白名单=矩阵 ✅ 键；白名单外/不存在 id 整体拒绝

# 规范源漂移检查：源仓最新 commit 对比 references/sources.manifest 指纹（发版前跑；0=一致 1=漂移 2=离线）
bash scripts/check_sources.sh


```

render/extract 是 drawio 桌面版 CLI 的薄包装（渲染本体不在本技能），**版本固定 32.4.1**：

**渲染依赖不可用时的降级约定（强制）**：当 drawio 自动安装失败、下载超时快速失败或 `DRAWIO_BIN` 无效时——**立即明确告知用户**缺失了什么依赖、对应安装方法（Windows/macOS 官方发布页自装、`DRAWIO_BIN` 显式指定）、本次无法交付渲染图；**禁止**产出半成品 .drawio.svg、禁止手写 SVG 顶替、禁止反复重试下载或长时间挂起等待（受限网络可设 `DOCS_DIAGRAM_NO_DOWNLOAD=1` 快速失败）。
- **Windows**：用户自行安装 drawio 32.4.1（https://github.com/jgraph/drawio-desktop/releases/tag/v32.4.1，
  exe/msi 安装包），安装后自动发现常见路径；非默认路径用 `DRAWIO_BIN` 指向 draw.io 可执行文件
  （显式指定不校验版本）。
- **Linux**：缺 drawio 或版本不符时**自动下载安装** 32.4.1（AppImage 缓存于
  `~/.cache/docs-diagram-draw/drawio/`，无 fuse 环境自动解压运行，无需 root）；
  无显示环境自动经 xvfb-run 调用。
edit 仅依赖 Python3 标准库。

**HarmonyOS Sans**（运行时依赖，双字体：拉丁 `HarmonyOS_Sans.ttf` 族名 "HarmonyOS Sans" 使
`fontFamily=harmonyOS sans` 精确命中 + 简中 `HarmonyOS_Sans_SC.ttf` 承接中文字形回退）：首次使用
自动从华为官方渠道下载，缓存到 `~/.cache/docs-diagram-draw/`；Linux render 前自动装入 `~/.fonts/`
并 fc-cache（Windows/macOS 下载后打印安装指引，不代装；linux 系统已装则跳过）。

## 规范速查

完整规范见 https://gitcode.com/cann/docs/blob/master/specification/drawing_specs.md（版本指纹 `references/sources.manifest`，发版前跑 `scripts/check_sources.sh` 校验漂移；速查与规范原文不一致时以规范原文为准），速查如下：

| 项 | 规则 |
| --- | --- |
| 字体 | 中英文统一 HarmonyOS Sans；一般文本 14px，突出文本 16px，下限 10px；文字方向从左到右、字头朝上 |
| 用色 | 色板三组（优先级从高到低）：默认色 `#C2D5FE`>`#B0F0E2`>`#B9E9FC`>`#E0E2E6`>`#FEE2BA`警示>`#FAB9B6`警示；高亮色 `#5478FB`>`#03B5A5`>`#2AA6F3`>`#FB8F2B`>`#CD0000`（强调，白字，非必要不用、避免大面积）；中性色 文字`#202329`/边框`#686D75`/填充`#E0E2E6`+`#EDEFF2`+`#F4F5F7`+`#FFFFFF`。画新图内容色 ≤3 种（不计中性色）；改图（工作流二）不受 ≤3 约束——有彩色按矩阵附录二优先级取色，保类型区分不归一 |
| 线条 | 线宽 0.75pt（style 写 `strokeWidth=0.75`，渲染输出为 1px，浏览器可见），颜色 `#686D75`；虚线 `dashed=1;dashPattern=4 2` |
| 节点 | 高 32pt；默认宽 148pt，宽度档位取模板实际值（44/64/72/80/108/148/180），超出档位随场景设定、保持整体和谐；圆角 4px；文字 14px 居中，默认色 `#202329`；开始/结束为 52×52 圆 `#686D75` 白字；判断菱形 148×72 |
| 间距 | 节点间距 8 的倍数（8/16/24/32…），最小 4pt；保证整体和谐 |
| 容器 | 画布宽度三档 920/600/300pt，高度自适应（一般 ≤1000pt 宽）；四周留白参考左右 80pt、上下 40pt |
| 命名 | 交付图片文件名用英文小写+下划线（snake_case），不含中文/连字符/空格；改图另存时加英文下划线后缀（如 `_normalized`、`_recolor`），不得覆盖原图。如果批量处理图片时，指定了新图目录，不覆盖原图的情况下可以和原图重名。 |
| 层级 | 单层级：无填充 + `#686D75` 实线边框（虚线仅用于弱化的可选分组）；多层级：最小层级 `#EDEFF2` 填充无边框，其余无填充实线框 |
| 换行 | 同组数字不跨行；英文单词不拆行；各行长度尽量接近 |

单位说明：规范中几何/线宽标 pt，drawio XML 数值与 pt **同值直写**（148pt → `width="148"`，渲染 148px，禁止做 1pt≈1.33px 倍率换算）；字号/圆角规范即用 px，同样按原值写入。


## 工作流一：画新图

### 1. 明确要素

从用户需求中列出：图类型（流程/时序/架构/组网/原理）、节点及文字、流向与分支、分组/层级、主次关系（谁用主色、谁次要、谁强调）。
文字过长时先精简（节点文字建议单行 ≤9 个汉字，两行 ≤18；放不下按宽度档位加宽节点，或依赖 whiteSpace=wrap 自动换行）。

### 2. 布局计算（先算坐标再写 XML）

- 选画布档位：单列流程图 300pt，双列/时序图 600pt，复杂架构图 920pt。
- 纵向单列流程：节点 x=76（300pt 画布水平居中，中心线 150）；行间距 24pt；开始圆与相邻节点间距 24pt。
- 多列：列间距 = 节点宽 + 40（如 148+40=188 列距）。
- 画布高 = 顶留白 40 + 内容总高 + 底留白 40。
- 所有间距取 8 的倍数；分支边文字（是/否）用 `labelBackgroundColor=#FFFFFF`。

### 3. 编写 drawio XML

骨架（纵向流程图，已验证可渲染；样式串从 `references/style-library.md` 复制）：

```xml
<mxfile host="app.diagrams.net" type="device">
  <diagram id="page-1" name="Page-1">
    <mxGraphModel dx="800" dy="600" grid="0" page="1" pageWidth="920" pageHeight="600">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        <mxCell id="bg" value="" style="（画布容器样式串）" vertex="1" parent="1">
          <mxGeometry x="0" y="0" width="300" height="466" as="geometry"/>
        </mxCell>
        <mxCell id="start" value="开始" style="（开始圆样式串）" vertex="1" parent="1">
          <mxGeometry x="124" y="40" width="52" height="52" as="geometry"/>
        </mxCell>
        <mxCell id="n1" value="环境准备" style="（默认节点样式串）" vertex="1" parent="1">
          <mxGeometry x="76" y="116" width="148" height="32" as="geometry"/>
        </mxCell>
        <mxCell id="e1" style="（实线箭头样式串）exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="start" target="n1">
          <mxGeometry relative="1" as="geometry"/>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

编写规则：

- 每个元素都要有唯一 `id`；元素顺序：画布 → 容器 → 节点 → 边。
- 边必须写 `source`/`target` 与锚点（`exitX/exitY` 出边、`entryX/entryY` 入边，取值 0~1）；
  上下流向用 `exitX=0.5;exitY=1` → `entryX=0.5;entryY=0`。
- 层级嵌套用分组框样式或 `group` 逻辑容器（子元素 `parent` 指向容器 id，坐标相对容器）。
- 文本换行交给 `whiteSpace=wrap`（drawio 自动断行，数字组/英文单词不拆）。

### 4. 渲染（必做）

```bash
python3 scripts/drawio_svg.py render <图名>.drawio
```

渲染后核对：

- 输出文件存在且体积与图内元素量相称。
- 文件名为英文小写+下划线（snake_case，如 `ascend_c_dev_flow.drawio.svg`）；不合适时先重命名再交付。
- 交付时说明：文件路径 + 一句话图内容。
- 输出目录只保留本次交付的 .drawio.svg；历史尝试的中间 SVG 一律删除，避免交付混入旧版文件。

## 工作流二：改图

1. **提取**：`extract 原图.drawio.svg -o 编辑稿.drawio`（无 content 属性的纯 SVG 无法提取，需告知用户无法程序化改图）。
2. **定界**：改哪个属性前先查 `references/xml-property-matrix.md`（改图范围矩阵：可修改键白名单，逐键四列——属性/含义/权限（规范化与自定义两场景分别标注）/规则）。
   - 可修改集 = edit 白名单键；⛔ 一律保留原值，点名也不执行（告知依据+替代建议）。
   - 同为可修改键，动不动由模式决定：规范化只执行有规则目标值的键；自定义只执行用户点名的键。
3. **呈现模式选择，用户定夺改动范围**——两个模式：

   **模式 A：规范化调整（推荐）**——按通用规范批量执行，用户只需做三个选择：
   - **颜色**：① 规范色板自动映射（一对一入板，附录二判定链，不改变业务含义）【推荐】；② 保持原色不动（跳过颜色类）
   - **字体**：① 规范默认（统一 HarmonyOS Sans）【推荐】；② 保持原字体不动（仅颜色/容器规范化）；③ 自定义字体。字号规范化暂不支持——需改字号走模式 B 点名
   - **容器**：① 画布档位规格化（按内容宽自动归 300/600/920 档，见下节规则）【推荐】；② 保持原容器尺寸
   其余类别按矩阵规则默认执行。
   用户确认选择后：全图扫描 → 按规则推导每个元素目标值 → 生成 edit 指令集批量执行（另存，原图不动，不满意可重来）。

   **模式 B：自定义调整**（通用规范无法覆盖的特殊场景）——用户点名具体修改：
   呈现**元素级决策单**（对象×属性×现值→目标值×依据；依据列=矩阵规则，用户可确认可反驳；联动项显式列出不隐式改。用户确认后生成 edit 指令。
   - 点名 ⛔ 不可修改键：不执行，告知依据与替代建议（用可修改键等效实现，或走工作流一重画）
   - 模糊诉求（"太乱/好看点"）反问澄清分流到两个模式，不猜不擅动
4. **最小改动（edit 执行）**：把用户确认的改动翻译成 edit 指令交脚本执行，不直接手改 XML
   （白名单即范围矩阵的代码化，越界键自动拒绝；指令输出天然是改动审计记录）：
   `edit 提取稿.drawio -o 改后稿.drawio --set id=n2,fillColor=#5478FB --set id=n1,value=数据搬运`
   新增节点/连线不在 edit 能力内，仍按工作流一规则写 XML。
5. **渲染交付**：`render` 另存命名 `原名_修改点.drawio.svg`（修改点用英文小写单词，如 `vec_add_normalized.drawio.svg`，避免覆盖用户原图）。如果用户指定新图的目录，则可以保持原图名称不变。
6. **核对**：跑 diff 校验提取稿与改后稿（仅作校验，不生成/留存 diff 文件），未勾选项的 value/style/几何必须零变化。

**画布档位规格化（受限场景，仅用户明示触发）**：内容 bbox 与全部元素坐标不动，仅将白底容器宽度归 300/600/920 档（内容 >920 按内容宽+留白）、四周留白左右 80/上下 40、水平居中包内容。白底画布识别：全图白底无边框顶点中宽高最大者且位于最底层（挂默认图层）。
**图中无白底画布容器时，用户选"归档"= 按工作流一画布规则新建白底容器**（内容坐标不动、水平居中包内容、宽取内容档位）——"结构性硬约束禁止增删元素"不禁止此操作（新建画布容器与新增节点同理，走工作流一规则）。禁止为归档而迁移内容坐标。

## 常见图型构图模式

**流程图**：开始圆 → 节点列 → 判断菱形（分支出边带 是/否 标签）→ 结束圆；返回线走列外侧（waypoints 绕行，虚线）。

**时序图**（600/920 宽）：顶部参与者矩形 148×32 横排（间距 8 倍数）→ 各参与者下垂虚线生命线到底部 →
水平消息箭头（实线=调用，虚线=返回）→ 激活条为生命线上的窄竖条（宽 17pt，主色填充，无边框圆角 4）。
自消息用 waypoints 折返。

**架构图/分层图**（920 宽）：外层白画布 → 各层用层级框（实线边框，标签在左上）→ 层内模块网格排列；
核心模块主色、外围模块次要色、关键路径强调色（强调节点无边框白字）；层间用无箭头连线或箭头表达依赖。

**组网图**：设备节点（矩形或图片节点）+ 实线（物理连接）/虚线（逻辑连接）；终端/服务器/网络分层摆放。

## 禁止事项

- 禁止手写/直接编辑 .drawio.svg 的 SVG 画面部分（只能改 content 属性对应的 XML 后重新渲染）。
- 禁止使用色板外颜色、自造灰色；禁止字号 <10 或 >16（无特殊说明）。
- 禁止画新图用超过 3 种内容色（改图按用户勾选项执行，点名改色按矩阵一对一规则建议）。
- 数据可视化图表（柱状图/折线图/饼图等）不在本技能范围，说明后不做处理。
