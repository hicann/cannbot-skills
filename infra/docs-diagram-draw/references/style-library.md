# 样式库（drawio style 速查）

本样式库从官方绘图模板 https://gitcode.com/cann/docs/blob/master/writing_template/drawing_template_drawio.xml（版本指纹 `sources.manifest`，核对用 `scripts/check_sources.sh`）提炼并标准化。
**画图时 style 串原样复制本文档中的字符串**，只改 `value`（文字）、`id`、几何坐标和连接关系，不要自行拼凑样式。

与模板原始串的差异（标准化处理，视觉等效）：
- 模板节点串 fontSize 混用 12/14（166/116 例），实际字号多由 HTML value 内 `<font style="font-size:14px">` 承载；本库统一为 `fontSize=14` + 纯文本 value，视觉等效且更易维护。
- 模板开始/结束用 `shape=stencil(...)` 圆形；本库用等效的 `ellipse`（52×52 圆，同模板 stencil 等效），渲染结果一致。
- 文字色默认 `#202329`（规范节点文字色值），强调节点用白字 `#FFFFFF`。

## 色板（绘图规范）

| 语义 | 色值 | 用途 |
| --- | --- | --- |
| 主要内容 | `#C2D5FE` | 图中主体节点（默认首选） |
| 次要内容 | `#E0E2E6` | 次要/背景节点；2 色方案中的次要色 |
| 备选色 | `#B0F0E2` | 需要第 3 种内容色时选用的浅青绿 |
| 数据 | `#B9E9FC` | 数据类型流程色块（平行四边形） |
| 判断 | `#FEE2BA` | 判断/条件流程色块（菱形） |
| 强调色 1 | `#5478FB` | 特别强调的节点（白字） |
| 强调色 2 | `#03B5A5` | 特别强调的节点（白字） |
| 层级填充 | `#EDEFF2` | 多层级嵌套时最小层级的填充（无边框） |
| 浅底 | `#F4F5F7` | 分组浅色背景（模板示例使用） |
| 线条/边框/开始结束 | `#686D75` | 所有连线、边框、开始/结束圆 |
| 文字 | `#202329` | 默认节点文字色 |
| 画布 | `#FFFFFF` | 白色画布容器 |

规则：一张图内容用色 **≤3 种**（不计文字/线条/边框/开始结束等中性色）；非必要不用强调色，避免大面积高亮。

## 节点样式

统一几何：单位 pt，drawio XML 数值同值直写、不做倍率换算（详见 SKILL.md 速查表"单位说明"）。高 `32`pt，默认宽 `148`pt（模板主流取值，148×32 出现 70 次），
宽度档位取模板实际值（44/64/72/80/108/148/180），超出档位随场景设定、保持整体和谐，
节点间距用 8 的倍数（8/16/24/32…），最小可 4pt。圆角 4px。

**默认节点（主要内容 #C2D5FE）**

```
whiteSpace=wrap;html=1;rounded=1;strokeColor=#C2D5FE;strokeWidth=0.75;align=center;verticalAlign=middle;arcSize=4;absoluteArcSize=1;fontFamily=harmonyOS sans;fontSize=14;fontColor=#202329;fillColor=#C2D5FE;
```

**次要节点（#E0E2E6）**：上串中 3 处色值换为 `#E0E2E6`（strokeColor/fillColor）。

**备选色节点（#B0F0E2）**：3 处色值换为 `#B0F0E2`。

**强调节点（#5478FB，白字，无边框）**

```
whiteSpace=wrap;html=1;rounded=1;strokeColor=none;strokeWidth=0.75;align=center;verticalAlign=middle;arcSize=4;absoluteArcSize=1;fontFamily=harmonyOS sans;fontSize=14;fontColor=#FFFFFF;fillColor=#5478FB;
```

**强调节点 2（#03B5A5，白字，无边框）**：`fillColor` 换为 `#03B5A5`。

**开始/结束（圆，52×52，#686D75 白字）**

```
ellipse;whiteSpace=wrap;html=1;fillColor=#686D75;strokeColor=none;fontFamily=harmonyOS sans;fontSize=14;fontColor=#FFFFFF;
```

**判断节点（菱形，148×72，#FEE2BA）**

```
rhombus;whiteSpace=wrap;html=1;shapeInside=1;rounded=1;strokeColor=#FEE2BA;strokeWidth=0.75;align=center;verticalAlign=middle;arcSize=4;fontFamily=harmonyOS sans;fontSize=14;fillColor=#FEE2BA;fontColor=#202329;
```

**数据节点（平行四边形，148×32，#B9E9FC）**

```
shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;shapeInside=1;fixedSize=1;rounded=1;strokeColor=#B9E9FC;strokeWidth=0.75;align=center;verticalAlign=middle;arcSize=4;fontFamily=harmonyOS sans;fontSize=14;fontColor=#202329;fillColor=#B9E9FC;roundedPerimeter=1;
```

## 线条与箭头

线宽一律 `strokeWidth=0.75`（drawio 单位 0.75pt），颜色 `#686D75`。边（edge）统一挂在 `parent="1"`，`source`/`target` 引用节点 id；
建议显式写 `exitX/exitY/entryX/entryY`（0~1）固定出边/入边锚点，避免歧义。

**实线箭头（默认）**

```
edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#686D75;strokeWidth=0.75;fontFamily=harmonyOS sans;fontSize=14;startArrow=none;startSize=9;endArrow=block;endSize=2;
```

**虚线箭头**：上串追加 `dashed=1;dashPattern=4 2;`

**无箭头连线**：上串中 `endArrow=block;endSize=2` 换为 `endArrow=none;`

**带文字的边（如判断分支 是/否）**：上串追加 `labelBackgroundColor=#FFFFFF;`，文字写在边的 `value`。

**折线/绕行**：在 `mxGeometry` 内加 `<Array as="points"><mxPoint x="…" y="…"/></Array>` 途经点。

## 容器与层级

**白色画布容器**（图的底板，宽 920/600/300 三档，高自适应）

```
rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=none;fontFamily=harmonyOS sans;
```

**层级/分组框**（无填充实线边框，默认首选）

```
rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#686D75;strokeWidth=0.75;fontFamily=harmonyOS sans;fontSize=14;align=left;verticalAlign=top;spacingLeft=8;spacingTop=8;fontColor=#202329;
```

需要弱化分组边界（如流程图中的可选分组、回路圈定）时可改用虚线：追加 `dashed=1;dashPattern=4 2;`。

层级规则：一个层级框 → 无填充 + `#686D75` 实线边框；多于一个层级 → 最小层级用 `#EDEFF2` 填充无边框（`fillColor=#EDEFF2;strokeColor=none;`），其余层级同上实线框。

**分组 group**（纯逻辑容器，不渲染）：`style="group;fontFamily=harmonyOS sans;"`，子元素 `parent` 指向 group id、坐标相对 group。

## 图片节点

```
shape=image;html=1;verticalLabelPosition=bottom;labelBackgroundColor=none;verticalAlign=top;aspect=fixed;image=data:image/png,BASE64数据;
```

- drawio 数据 URI 格式为 `data:image/png,xxxx`（逗号后直接 base64，无 `;base64` 标记）。
- 外链 http(s) 图片会被渲染脚本自动下载转内嵌（规范要求嵌入图片）。

## 字体

- 全部元素（含边、容器）显式带 `fontFamily=harmonyOS sans;`
- 一般文本 14px；突出文本 16px；下限 10px
- 中文/英文统一 HarmonyOS Sans；导出 .drawio.svg 时按需子集化内嵌字体

## 模板原始文件

完整模板（含图标库、七类完整图例）见规范源仓 https://gitcode.com/cann/docs/blob/master/writing_template/drawing_template_drawio.xml，
可在 drawio 中"文件 → 打开库"导入（或 raw 地址下载后导入）。模板图例仅供参考，以实际业务场景为准。模板是否过期用 `scripts/check_sources.sh` 校验。
