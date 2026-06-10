# 批量 Issue 创建指南

批量创建 Issue 的完整指南，包含合并策略、文件命名规范、自动填充规则。

---

## 合并策略

### 智能合并判断

根据问题数量自动决定合并策略，减少用户交互：

| 问题数量 | 策略 | 用户交互 |
|:-------:|------|---------|
| ≥ 10 | **自动合并** | 仅提示用户，不询问 |
| 3-9 | **询问合并** | 展示选项，让用户选择 |
| 1-2 | **不合并** | 直接创建，不询问 |

### 合并维度

| 维度 | 说明 | 示例 |
|------|------|------|
| **问题类型** | 按 issue_type 分组 | README缺失、UT缺失、CMake错误 |
| **仓库** | 按仓库分组 | ops-math、ops-nn、ops-cv |
| **问题级别** | 可选，按严重程度分组 | 高优先级、中优先级、低优先级 |

### 合并规则配置

```yaml
# .opencode/config/merge_rules.yaml
merge_threshold:
  auto_merge: 10      # 自动合并阈值
  ask_merge: 3        # 询问合并阈值
  no_merge: 1         # 不合并阈值

merge_dimensions:
  - issue_type        # 问题类型（默认）
  - repo              # 仓库（默认）
  # - severity        # 问题级别（可选）
```

---

## Issue 文件命名规范

### 目录结构

```
reports/
└── {YYYYMMDD}/                               # 日期目录
    └── {repo}/                                # 仓库目录
        ├── {command}_report_{HHMMSS}.md       # 扫描报告
        └── issues/                            # Issue 文件目录
            ├── {issue_type}_merged_issue_{time}.md  # 合并 Issue
            ├── {op_name}_{issue_type}_issue_{time}.md  # 单算子 Issue
            └── his/                           # 历史归档（可选）
```

### 文件命名格式

| 模式 | 格式 | 示例 |
|------|------|------|
| **合并模式** | `{issue_type}_merged_issue_{HHMMSS}.md` | `readme_missing_merged_issue_143058.md` |
| **单算子模式** | `{op_name}_{issue_type}_issue_{HHMMSS}.md` | `add_readme_missing_issue_143058.md` |
| **优先级模式** | `{issue_type}_{priority}_issue_{HHMMSS}.md` | `cmake_error_high_issue_143058.md` |

### 时间戳生成

```bash
# Shell
date_str=$(date +"%Y%m%d")
time_str=$(date +"%H%M%S")

# Python
from datetime import datetime
date_str = datetime.now().strftime("%Y%m%d")
time_str = datetime.now().strftime("%H%M%S")
```

---

## 自动填充规则

### 问题类型 → 模板字段映射

| 问题类型 | 模板 | 自动填充字段 |
|---------|------|-------------|
| **README缺失** | Documentation | doc_link（算子路径）、issues_section（缺失说明） |
| **aclnn文档缺失** | Documentation | doc_link（算子路径）、issues_section（缺失说明） |
| **CMake配置错误** | Bug-Report | description（错误详情）、environment（仓库环境）、steps（复现命令） |
| **UT缺失** | Bug-Report | description（UT类型统计）、environment（仓库环境） |
| **UT测试失败** | Bug-Report | description（失败详情）、steps（测试命令）、logs（错误日志） |
| **Examples缺失** | Requirement | background（缺失说明）、origin（扫描分析） |
| **Examples失败** | Bug-Report | description（失败详情）、steps（测试命令）、logs（错误日志） |
| **断链问题** | Documentation | doc_link（问题链接）、issues_section（链接列表） |

### 环境信息自动提取

| 字段 | 来源 | 提取方式 |
|------|------|---------|
| `CANN 版本` | 扫描报告 | `scan_report.cann_version` |
| `操作系统` | 扫描报告 | `scan_report.os_version` |
| `NPU 型号` | 扫描报告 | `scan_report.npu_type` |
| `仓库` | 输入参数 | `repo` |
| `问题类型` | 扫描报告 | `issue_type` |
| `问题文件数` | 扫描报告 | `len(problem_list)` |

### 正文内容生成模板

```python
# Bug-Report 正文生成
def generate_bug_report_body(problem_data):
    description = f"{problem_data.issue_type}: {problem_data.summary}"
    environment = f"""
**软件环境**:
- CANN 版本: {problem_data.cann_version}
- 操作系统: {problem_data.os_version}

**硬件环境**:
- NPU 型号: {problem_data.npu_type}

**问题环境**:
- 仓库: {problem_data.repo}
- 问题类型: {problem_data.issue_type}
- 问题文件数: {problem_data.count}
"""
    steps = f"""
1. 克隆仓库: git clone https://gitcode.com/cann/{problem_data.repo}.git
2. 进入目录: cd {problem_data.repo}
3. 执行扫描: {problem_data.scan_command}
4. 查看报告: {problem_data.report_path}
"""
    expected = f"所有 {problem_data.issue_type} 问题已修复"
    
    return BUG_REPORT_TEMPLATE.format(
        description=description,
        environment=environment,
        steps=steps,
        expected=expected,
        logs=problem_data.logs or "无",
        notes=""
    )

# Documentation 正文生成
def generate_documentation_body(problem_data):
    doc_link = f"https://gitcode.com/cann/{problem_data.repo}/blob/main/{problem_data.doc_path}"
    issues_section = f"""
{problem_data.issue_type} 问题描述：
- 算子路径: {problem_data.op_path}
- 问题类型: {problem_data.issue_type}
"""
    existing_issues = f"{problem_data.summary}"
    
    return DOCUMENTATION_TEMPLATE.format(
        doc_link=doc_link,
        issues_section=issues_section,
        existing_issues=existing_issues
    )
```

---

## 合并 Issue 内容结构

### 合并 Issue 必需章节

```markdown
# [模板类型]: [AI 识别] {repo} {问题类型}（{n}个算子）

**标签**: `{标签}`
**生成时间**: {timestamp}
**涉及算子数**: {n}

---

## Missing Operators List（问题算子列表）

| 序号 | 算子名称 | 算子路径 | 状态 |
|:---:|---------|---------|:---:|
| 1 | {op_name_1} | {repo}/{op_class}/{op_name_1}/ | ❌ {问题描述} |
| ... | ... | ... | ... |

## Existing Issues（存在的问题）

{问题描述详情}

## Suggested Fix（修复建议）

{修复建议}

---

**提交地址**: https://gitcode.com/cann/{repo}/issues/new
```

### 合并 Issue 表格字段

| 字段 | 必填 | 说明 |
|------|:---:|------|
| `序号` | ✅ | 1-n 的序号 |
| `算子名称` | ✅ | 算子名称（op_name） |
| `算子路径` | ✅ | 相对路径（{repo}/{op_class}/{op_name}/） |
| `状态` | ✅ | ❌ + 简短问题描述 |

---

## 批量创建流程

### 标准流程（ops-qa-suite）

```
Step 1: 扫描完成 → 提取问题
Step 2: 问题分类 → 按问题类型+仓库分组
Step 3: 合并判断 → 应用智能合并策略
Step 4: 模板选择 → 问题类型 → 模板
Step 5: 内容生成 → 自动填充模板字段
Step 6: 文件生成 → 生成 Issue MD 文件
Step 7: 询问提交 → 展示 Issue 列表，询问用户
Step 8: 执行提交 → API 创建或手动提交
```

### 通用流程（非扫描驱动）

```
Step 1: 接收问题列表 → 用户提供 JSON/CSV/手动输入
Step 2: 问题分类 → 按 issue_type 分组
Step 3: 合并判断 → 应用智能合并策略
Step 4: 模板选择 → 问题类型 → 模板
Step 5: 内容生成 → 用户提供字段或自动填充
Step 6: 文件生成 → 生成 Issue MD 文件
Step 7: 询问提交 → 展示 Issue 列表，询问用户
Step 8: 执行提交 → API 创建或手动提交
```

---

## 批量创建 API 调用优化

### 速率限制处理

```python
import time

def create_issue_with_retry(project_id, title, description, token, labels, max_retries=3):
    for retry in range(max_retries):
        try:
            response = create_issue(project_id, title, description, token, labels)
            return response
        except RateLimitError:
            wait_time = min(60, 2 ** retry)  # 指数退避
            print(f"速率限制，等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
        except PermissionError:
            # 403 权限不足，不重试
            raise
```

### 并发创建控制

```python
import concurrent.futures

def batch_create_issues(issues, max_workers=5):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for issue in issues:
            future = executor.submit(create_issue_with_retry, **issue)
            futures.append(future)
        
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append({"status": "success", "issue": result})
            except Exception as e:
                results.append({"status": "failed", "error": str(e)})
        
        return results
```

---

## 失败处理与降级

### API 失败降级策略

| 状态码 | 原因 | 降级方案 |
|--------|------|---------|
| 403 | Token 无写权限 | 生成 Issue MD 文件，提供手动提交链接 |
| 401 | Token 无效或过期 | 提示用户提供新 token，保留 Issue 文件 |
| 404 | 项目不存在 | 确认仓库路径，检查仓库名拼写 |
| 422 | 参数验证失败 | 检查 title/description 格式，修复后重试 |
| 429 | 速率限制 | 指数退避重试，最多 3 次 |

### 手动提交降级

```markdown
Issue 文件已生成，请手动提交：

| Issue | 提交链接 |
|-------|---------|
| Issue 1 | https://gitcode.com/cann/{repo}/issues/new |
| Issue 2 | https://gitcode.com/cann/{repo}/issues/new |

文件路径：
- reports/{date}/{repo}/issues/{issue_file}.md

操作步骤：
1. 点击提交链接
2. 复制 Issue 文件内容
3. 粘贴到 Description 区域
4. 填写 Title（已在文件中预设）
5. 点击 Submit
```