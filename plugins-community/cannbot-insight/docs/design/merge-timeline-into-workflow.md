# Workflow Tab 合并 Timeline 需求

## 目标

将 Timeline Tab 合并到 Workflow Tab，减少 Tab 数量，提供统一的工作流分析入口。

## Tab 结构变更

```
合并前 (8个Tab):
[Overview] [Turns] [Workflow ✦] [Timeline] [Subagents] [Skills] [Interactions] [Workflow (AI)]

合并后 (7个Tab):
[Overview] [Turns] [Workflow ✦] [Subagents] [Skills] [Interactions] [Workflow (AI)]
```

## Workflow Tab 内部结构

### 顶部汇总条（保留）
```
3 Phases │ 27 Steps │ 3 Checkpoints │ Active: 1.5h │ Wait: 6.5h │ 18.8%
```

### 子视图切换
```
[🌳 树形]  [📊 时间线]  [🔄 流程图]
```

### 子视图 1：树形视图（现有 WorkflowTreeView）
- 保留现有功能不变
- Phase → Step → Checkpoint 树形展示

### 子视图 2：时间线视图（合并自 TimelineGantt）
在现有 Gantt 图基础上增强：

**增强项：**

1. **Phase 背景色块**
   - 时间轴顶部用半透明色块标记 Phase 边界
   - Phase 名称标注在色块上方
   - 颜色：Phase1=蓝, Phase2=绿, Phase3=橙

2. **Checkpoint 标记**
   - 在时间轴上用 ⛔/⚪ 图标标记 CP 位置
   - 等待时间用灰色虚线条表示
   - hover 显示等待时长

3. **按 Agent Type 分行**（现有）
   - build / general / architect / developer 各占一行

4. **执行条增强**
   - Hover tooltip：subagent name, duration, tokens, status
   - 点击展开详情面板（inline）
   - 颜色按 subagent type 区分
   - 并行执行用虚线框包围

5. **时间轴刻度增强**
   - 增加绝对时间刻度（10:28, 11:00, 12:00...）
   - 保留相对时间刻度（0min, 60min, 120min...）
   - 可切换显示模式

6. **图例**
   - 右上角小图例说明颜色含义

### 子视图 3：流程图视图（可选，后续）
- Mermaid DAG 图
- 显示阶段 → 步骤的依赖关系

## 联动交互

- 树形视图点击 Step → 切换到时间线视图并高亮对应执行条
- 时间线视图点击执行条 → 切换到树形视图并展开对应 Step
- 两个视图共享选中状态

## 删除内容

- 删除独立的 Timeline Tab
- 删除 `TimelineGantt` 组件（逻辑合并到 `WorkflowTimelineView`）
- page.tsx 中 TabKey 去掉 "timeline"

## 新增文件

| 文件 | 说明 |
|------|------|
| `WorkflowTimelineView.tsx` | 增强版时间线（基于 TimelineGantt 重构） |
| `WorkflowViewSwitcher.tsx` | 子视图切换控件 |

## 修改文件

| 文件 | 说明 |
|------|------|
| `WorkflowTreeView.tsx` | 添加联动逻辑 |
| `page.tsx` | 删除 Timeline Tab，Workflow 内嵌子视图切换 |
| `TimelineGantt.tsx` | 删除或标记废弃 |

## 预估工时

- 子视图切换控件：1h
- 时间线增强（phase色块 + CP标记 + tooltip + 颜色）：4h
- 树形 ↔ 时间线联动：2h
- 删除旧 Timeline Tab + 清理：0.5h
- 测试：1h
- **总计：~8.5h**
