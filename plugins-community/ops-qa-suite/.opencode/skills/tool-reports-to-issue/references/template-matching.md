# 模板匹配规则

根据问题类型自动选择合适的 Issue 模板，支持仓库模板查询和预设模板 fallback。

---

## 模板查询流程

### 优先级顺序

| 步骤 | 检查路径 | 说明 |
|:---:|---------|------|
| 1 | `.gitcode/ISSUE_TEMPLATE/*.zh-CN.yml` | GitCode 中文模板（优先） |
| 2 | `.gitcode/ISSUE_TEMPLATE/*.yml` | GitCode 默认 YAML 表单模板 |
| 3 | `.gitcode/ISSUE_TEMPLATE/*.md` | GitCode Markdown 模板 |
| 4 | `.github/ISSUE_TEMPLATE/*.yml` | GitHub YAML 表单模板（兼容） |
| 5 | `.github/ISSUE_TEMPLATE/*.md` | GitHub Markdown 模板（兼容） |
| 6 | **预设模板** | 仓库无模板时使用 references/issue-templates.md |

### API 查询命令

```bash
# 查询模板目录列表
owner="cann"
repo="ops-math"
token="${GITCODE_TOKEN}"

# Step 1: 检查 .gitcode/ISSUE_TEMPLATE/
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE?access_token=${token}"

# Step 2: 检查 .github/ISSUE_TEMPLATE/
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.github/ISSUE_TEMPLATE?access_token=${token}"

# 获取单个模板内容（Base64 编码）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE/bug_report.yml?access_token=${token}"

# 解码模板内容
echo "${content}" | base64 -d
```

### 模板内容解析

#### YAML 表单模板解析

```python
import yaml
import base64

def parse_yaml_template(content_base64):
    content = base64.b64decode(content_base64).decode('utf-8')
    template = yaml.safe_load(content)
    
    return {
        "name": template.get("name"),
        "description": template.get("description"),
        "title": template.get("title", ""),
        "labels": template.get("labels", []),
        "assignees": template.get("assignees", []),
        "body": template.get("body", []),
        "type": "yaml"
    }
```

#### Markdown front-matter 解析

```python
def parse_markdown_template(content_base64):
    content = base64.b64decode(content_base64).decode('utf-8')
    
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            front_matter = yaml.safe_load(parts[1])
            body = parts[2].strip()
            
            return {
                "name": front_matter.get("name"),
                "description": front_matter.get("about", ""),
                "title": front_matter.get("title", ""),
                "labels": front_matter.get("labels", []),
                "assignees": front_matter.get("assignees", []),
                "body": body,
                "type": "markdown"
            }
    
    return None
```

---

## 问题类型 → 模板选择

### ops-qa-suite 问题类型映射

| 问题类型 | 推荐模板 | 标签 | 标题前缀 |
|---------|---------|------|---------|
| **README缺失** | Documentation | documentation | `[Documentation|文档反馈]:` |
| **aclnn文档缺失** | Documentation | documentation | `[Documentation|文档反馈]:` |
| **CMake配置错误** | Bug-Report | bug-report | `[Bug-Report|缺陷反馈]:` |
| **UT缺失** | Bug-Report | bug-report | `[Bug-Report|缺陷反馈]:` |
| **UT测试失败** | Bug-Report | bug-report | `[Bug-Report|缺陷反馈]:` |
| **Examples缺失** | Requirement | requirement | `[Requirement|需求建议]:` |
| **Examples失败** | Bug-Report | bug-report | `[Bug-Report|缺陷反馈]:` |
| **断链问题** | Documentation | documentation | `[Documentation|文档反馈]:` |
| **文档错误** | Documentation | documentation | `[Documentation|文档反馈]:` |

### 通用问题类型映射

| 用户意图关键词 | 推荐模板 | 标签 |
|--------------|---------|------|
| **发现 bug/缺陷/问题/错误/失败** | Bug-Report | bug-report |
| **文档有错/文档问题/文档缺失** | Documentation | documentation |
| **需要新功能/需求建议/需求** | Requirement | requirement |
| **咨询问题/讨论/疑问** | Question | question |
| **其他** | Blank | 无 |

### 模板不存在时的降级顺序

```
首选模板不存在时，按以下顺序降级：
1. requirement → feature-request → bug-report → documentation
2. feature-request → requirement → bug-report
3. bug-report → documentation
4. documentation → bug-report
5. question → documentation
```

---

## 模板自动匹配算法

### 匹配逻辑

```python
def select_template(issue_type, repo_templates, preset_templates):
    """
    选择合适的 Issue 模板
    
    Args:
        issue_type: 问题类型（如 README缺失、UT缺失）
        repo_templates: 仓库模板列表（从 API 查询）
        preset_templates: 预设模板列表（references/issue-templates.md）
    
    Returns:
        选中的模板信息
    """
    # Step 1: 根据问题类型确定推荐模板名称
    recommended_name = get_recommended_template_name(issue_type)
    
    # Step 2: 查找仓库模板
    for template in repo_templates:
        if template["name"] == recommended_name:
            return template
    
    # Step 3: 降级查找仓库模板
    fallback_names = get_fallback_template_names(recommended_name)
    for fallback_name in fallback_names:
        for template in repo_templates:
            if template["name"] == fallback_name:
                return template
    
    # Step 4: 使用预设模板
    return preset_templates.get(recommended_name) or preset_templates.get("blank")

def get_recommended_template_name(issue_type):
    """问题类型 → 模板名称映射"""
    mapping = {
        "README缺失": "Documentation|文档反馈",
        "aclnn文档缺失": "Documentation|文档反馈",
        "CMake配置错误": "Bug-Report|缺陷反馈",
        "UT缺失": "Bug-Report|缺陷反馈",
        "UT测试失败": "Bug-Report|缺陷反馈",
        "Examples缺失": "Requirement|需求建议",
        "Examples失败": "Bug-Report|缺陷反馈",
        "断链问题": "Documentation|文档反馈",
        "文档错误": "Documentation|文档反馈",
        "bug": "Bug-Report|缺陷反馈",
        "documentation": "Documentation|文档反馈",
        "requirement": "Requirement|需求建议",
        "question": "Question|问题咨询",
    }
    return mapping.get(issue_type, "Blank")

def get_fallback_template_names(primary_name):
    """降级模板名称列表"""
    fallback_map = {
        "Requirement|需求建议": ["Feature Request", "Bug-Report|缺陷反馈"],
        "Feature Request": ["Requirement|需求建议", "Bug-Report|缺陷反馈"],
        "Bug-Report|缺陷反馈": ["Documentation|文档反馈"],
        "Documentation|文档反馈": ["Bug-Report|缺陷反馈"],
        "Question|问题咨询": ["Documentation|文档反馈"],
    }
    return fallback_map.get(primary_name, [])
```

---

## 模板字段自动填充

### Bug-Report 字段映射

| 字段 | 来源 | 自动填充规则 |
|------|------|-------------|
| `问题描述` | 扫描报告 | `f"{issue_type}: {summary}"` |
| `环境信息` | 扫描报告 | 从 `scan_report.environment` 提取 |
| `重现步骤` | 扫描报告 | 生成标准复现命令序列 |
| `预期结果` | 默认 | `f"所有 {issue_type} 问题已修复"` |
| `日志/截图` | 扫描报告 | 从 `scan_report.logs` 提取 |

### Documentation 字段映射

| 字段 | 来源 | 自动填充规则 |
|------|------|-------------|
| `文档链接` | 扫描报告 | `f"https://gitcode.com/cann/{repo}/blob/main/{doc_path}"` |
| `问题文档片段` | 扫描报告 | 生成算子路径和问题描述 |
| `存在的问题` | 扫描报告 | 从 `scan_report.summary` 提取 |

### Requirement 字段映射

| 字段 | 来源 | 自动填充规则 |
|------|------|-------------|
| `背景信息` | 扫描报告 | `f"{issue_type}: {summary}"` |
| `信息来源` | 扫描报告 | `"扫描报告分析结果"` |
| `价值/作用` | 默认 | `f"提升{算子}的可维护性"` |
| `设计方案` | 默认 | `"参考同类算子实现"` |

---

## 模板缓存优化

### 缓存策略

```python
import json
import time
from pathlib import Path

TEMPLATE_CACHE_FILE = "/tmp/gitcode_template_cache.json"
CACHE_EXPIRY_SECONDS = 3600  # 1 小时

def get_cached_templates(owner, repo):
    """获取缓存的模板列表"""
    cache_key = f"{owner}/{repo}"
    
    try:
        with open(TEMPLATE_CACHE_FILE, 'r') as f:
            cache = json.load(f)
        
        if cache_key in cache:
            cached_data = cache[cache_key]
            if time.time() - cached_data["timestamp"] < CACHE_EXPIRY_SECONDS:
                return cached_data["templates"]
    except FileNotFoundError:
        pass
    
    return None

def cache_templates(owner, repo, templates):
    """缓存模板列表"""
    cache_key = f"{owner}/{repo}"
    
    try:
        with open(TEMPLATE_CACHE_FILE, 'r') as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}
    
    cache[cache_key] = {
        "templates": templates,
        "timestamp": time.time()
    }
    
    with open(TEMPLATE_CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def query_templates_with_cache(owner, repo, token):
    """带缓存的模板查询"""
    # Step 1: 检查缓存
    cached = get_cached_templates(owner, repo)
    if cached:
        return cached
    
    # Step 2: API 查询
    templates = query_templates_from_api(owner, repo, token)
    
    # Step 3: 缓存结果
    cache_templates(owner, repo, templates)
    
    return templates
```

---

## 模板查询脚本示例

```python
#!/usr/bin/env python3
"""查询仓库 Issue 模板"""

import argparse
import requests
import base64
import yaml
import subprocess

API_BASE = "https://api.gitcode.com/api/v5"

def get_token():
    """从 git credential store 获取 token"""
    result = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=gitcode.com\n\n",
        capture_output=True,
        text=True
    )
    for line in result.stdout.split("\n"):
        if line.startswith("password="):
            return line.split("=", 1)[1]
    return None

def query_template_directory(owner, repo, path, token):
    """查询模板目录"""
    url = f"{API_BASE}/repos/{owner}/{repo}/contents/{path}"
    resp = requests.get(url, params={"access_token": token})
    
    if resp.status_code == 200:
        return resp.json()
    return None

def get_template_content(owner, repo, file_path, token):
    """获取模板文件内容"""
    url = f"{API_BASE}/repos/{owner}/{repo}/contents/{file_path}"
    resp = requests.get(url, params={"access_token": token})
    
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content
    return None

def parse_template(content, filename):
    """解析模板内容"""
    if filename.endswith(".yml") or filename.endswith(".yaml"):
        template = yaml.safe_load(content)
        return {
            "name": template.get("name"),
            "description": template.get("description"),
            "title": template.get("title", ""),
            "labels": template.get("labels", []),
            "type": "yaml"
        }
    elif filename.endswith(".md"):
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                front_matter = yaml.safe_load(parts[1])
                return {
                    "name": front_matter.get("name"),
                    "description": front_matter.get("about", ""),
                    "title": front_matter.get("title", ""),
                    "labels": front_matter.get("labels", []),
                    "type": "markdown"
                }
    return None

def main():
    parser = argparse.ArgumentParser(description="查询仓库 Issue 模板")
    parser.add_argument("--owner", required=True, help="仓库 owner")
    parser.add_argument("--repo", required=True, help="仓库名称")
    parser.add_argument("--token", help="GitCode token（可选，默认从 git credential 获取）")
    
    args = parser.parse_args()
    
    token = args.token or get_token()
    if not token:
        print("错误: 无法获取 GitCode token")
        return
    
    # 查询路径列表
    paths = [
        ".gitcode/ISSUE_TEMPLATE",
        ".github/ISSUE_TEMPLATE"
    ]
    
    templates = []
    
    for path in paths:
        files = query_template_directory(args.owner, args.repo, path, token)
        if files and isinstance(files, list):
            for file in files:
                if file["name"].endswith(".yml") or file["name"].endswith(".yaml") or file["name"].endswith(".md"):
                    content = get_template_content(args.owner, args.repo, file["path"], token)
                    template = parse_template(content, file["name"])
                    if template:
                        templates.append(template)
    
    if templates:
        print(f"检测到 {len(templates)} 个 Issue 模板:")
        for i, template in enumerate(templates, 1):
            print(f"{i}. {template['name']} - {template['description']}")
            print(f"   标签: {template['labels']}")
            print(f"   类型: {template['type']}")
    else:
        print("未检测到 Issue 模板，将使用预设模板")

if __name__ == "__main__":
    main()
```