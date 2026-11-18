---
name: docs-diagram-check
description: |
  对 drawio 图形（.drawio.svg，需以"包含副本"方式导出）与 .md 中的 mermaid 代码块进行绘图规范符合性检查，适用于文档开发与评审场景的自检。检查项定义见 rules/：
  - 25 色白名单（fillColor/strokeColor）
  - HarmonyOS Sans 字体
  - 节点尺寸 32px/52px 直径、圆角 4（arcSize）
  - 文字居中、字号整数且≥10px
  - mermaid 禁判断分支、init 配置合规
  支持目录全扫与 PR 变更文件清单两种模式，自动生成报告 spec_check_report.md（汇总单表含规则含义 + 按文件聚合明细）。存量亿图转换产物与新绘制的 drawio 图形均适用；当用户在文档开发/评审中提到以下任一场景时使用本技能：图形规范性检查 / drawio 规范检查 / drawio.svg 检查 / 图形质检报告。
version: 1.0.0
metadata:
  author: Huawei Technologies Co., Ltd.
  category: utility-doc
---

# Skill: docs-diagram-check（图形规范性检查）

## 适用范围

| 图形类别 | 检查对象 | 检查方式 |
| --- | --- | --- |
| drawio绘图（流程图/组网图/原理图/时序图/架构图） | `*.drawio.svg`（drawio桌面版勾选"包含副本"导出） | 解析内嵌mxfile源 → 设计参数逐项比对 |
| 简单流程图/时序图（代码模板） | `.md` 中的 ```mermaid 代码块 | 源码文本级比对（init配置/classDef/内联样式），不渲染 |
| 截图类位图（png/jpg/gif） | 不检查 | 无规范性要求 |

- **导出要求**：`.drawio.svg` 必须由 **drawio 桌面版**导出并勾选 **"包含副本"**——否则无内嵌 mxfile，设计参数无法检查，报告将给出 INFO-EXPORT 提示。
- **与 edraw-to-drawio 配合**：存量亿图 vsdx 先经其转换为 drawio 格式；`.md` 文件的图形引用统一指向 drawio 图片。

## 执行方式（Agent 执行参考）

Agent 执行检查时调用 `scripts/check.py`，命令行参数如下：

```bash
# 目录/仓库全扫（报告写入被检查目录 spec_check_report.md）
python scripts/check.py --scan <目录>

# PR变更文件模式（清单文件：每行一个路径；报告写入首个文件所在目录）
python scripts/check.py --files <清单文件>

# JSON输出（供文档检查工具集成；报告文件仍会生成，stdout 为纯 JSON）
python scripts/check.py --scan <目录> --json

# 指定规则目录（默认为skill内rules/）
python scripts/check.py --scan <目录> --rules <规则目录>
```

退出码：存在 must 问题为 1（可用于 CI），否则 0。

用户触发示例（对 Agent 说一句话即可）：

- 「用 docs-diagram-check skill 检查 `<目录>` 下的 drawio.svg 是否符合图形规范」
- 「检查 `<目录>` 里的 drawio.svg，把不符合规范的项按规则汇总给我」

## 报告（spec_check_report.md）

- **一、汇总**：单表——每行一条规则（规则 ID / 级别 / 数量 / 说明，说明取自 rules.json 的 desc，无需查表）；info 环境提示单独成行并附解决方案；
- **二、问题明细（按文件）**：每个文件一节，同类问题按（规则/级别/期望/实际/说明）聚合，"处数"为出现次数；
- 检查出的 must/recommended 问题由使用者按需修复后重查，直至全部通过。

## 规则分级

- **must（error，必改）**：字体HarmonyOS Sans、字号整数且≥10px、文字色白名单、文字居中、开始/结束直径52px、圆角4px、颜色∈25色白名单、mermaid禁判断菱形/分支、mermaid init配置合规
- **recommended（warn，建议改）**：节点高32px、单图默认色≤3（不含文字/箭头/线条/边框/开始&结束等中性色）、高亮色提示、容器≤1000px、上下留白≥40px、图片容器三档、字号∈{14,16}
- **info（环境提示，非规范性问题）**：仅两类——内嵌mxfile副本缺失（INFO-EXPORT）、清单文件未找到（INFO-NOTFOUND）

## 规则基线

- 色板：`rules/drawio_rules.json` palette（17色权威色板 ∪ 8种stencil图形色 = 25色；17色权威色板来源于 drawio文稿设计规范.pptx，8种stencil图形色为官方 Visio 导入图形的固有配色）
- mermaid init/classDef 基线：`rules/mermaid_rules.json`（来源：drawing_template_mermaid.md 模板，基线随模板同步维护）
- 组件库：drawing_template_drawio.xml（即"计算产品资料绘图组件drawio版"；规则与库同源维护；N01 节点高度为 recommended 级提示，库内节点以设计值为准）

## 解析兼容口径（vsdx 转换产物 / 组件库）

1. `UserObject/object` 包裹的 mxCell（id 在包装元素上，vsdx 导入常见）按包装 id 解析
2. `fillColor=#FFFFFF/default + strokeColor=none + 宽≥300` 的顶点视为**画布背景框**（edraw-to-drawio 插入），不按节点几何检查
3. 页面/图层标记等**结构单元格**（无 vertex/edge 属性）不参与图元级检查
4. 颜色解析支持 `light-dark(亮,暗)` 双态（取亮色比对，与 edraw-to-drawio 的 svg 归一同口径）；`fontColor=default` 为继承默认值，视为合规
5. **N01 节点判定口径（recommended）**：仅"彩色填充（node_fill_colors）的矩形/平行四边形 vertex，高度 20~100px"按节点高32px提示；纯文本(fill=none)、判断菱形(rhombus)、中性填充（E0E2E6/EDEFF2/F4F5F7/FFFFFF，次级说明/分组/背景性质）、大块（h≥100 分区/生命线）、细条（h≤20 图例色带）不参与 N01

## 常见问题

- **单位口径（pt 与 px，防误解）**：drawio 界面的单位标注不统一——**尺寸面板（宽/高）的 "pt" 实为渲染像素**（数值 1:1，如 300pt 渲染 300px，pt 为历史标签习惯）；**线宽面板的 "pt" 是真 pt**（0.75pt 显示 = 1px 渲染/存储）；字体为 px。本 skill 与检查器按同数值（px）直接比对，**无需任何 pt→px 换算**；仅线宽按 0.75pt=1px 固定对应（已内建）。
- **C01 颜色不合规怎么改**：报告"实际"列给出了违规色值，对照 25 色规范色板（见 `rules/drawio_rules.json` palette）修复后重查；
- **深色系统下颜色比对**：`light-dark(亮,暗)` 双态按亮色比对，与 edraw-to-drawio 的归一口径一致；
- **图元怎么定位**：报告已按规则聚合处数，如需逐个定位可在 drawio 的 Extras → 编辑绘图源码 中检索违规色值；
- **must 与 recommended 的区别**：must 必须修复，recommended 为建议项（视图形重要性取舍）。

## 参考

- `rules/drawio_rules.json` / `rules/mermaid_rules.json`：规则与阈值定义。
