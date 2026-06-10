# UT 缺失检测方法

## 一、目录结构对比法

### 步骤

1. 扫描算子目录结构（包括 arch* 子目录）
2. 检查 tests/ut 目录（包括 arch* 子目录）
3. 对比源文件与 UT 文件
4. 输出缺失报告

### Bash 命令示例（含 arch* 目录 + opapi 两种写法）

```bash
# 进入算子目录
cd {算子目录}

# 检查 op_host 源文件（含 arch* 子目录）
echo "=== op_host 源文件 ==="
ls op_host/*.cpp 2>/dev/null | grep -E "(infershape|tiling)"
ls op_host/arch20/*.cpp 2>/dev/null
ls op_host/arch32/*.cpp 2>/dev/null
ls op_host/arch35/*.cpp 2>/dev/null

# 检查 op_kernel 源文件（含 arch* 子目录）
echo "=== op_kernel 源文件 ==="
ls op_kernel/*.cpp 2>/dev/null
ls op_kernel/arch35/*.cpp 2>/dev/null

# 检查 op_api 源文件（标准写法 + 老写法）
echo "=== op_api 源文件 ==="
ls op_api/*.cpp 2>/dev/null | grep -E "aclnn"
ls op_host/op_api/*.cpp 2>/dev/null | grep -E "aclnn"  # 老写法

# 检查已有 UT（含 arch* 子目录 + opapi 老写法 UT）
echo "=== 已有 UT ==="
ls tests/ut/op_host/*.cpp 2>/dev/null
ls tests/ut/op_host/arch20/*.cpp 2>/dev/null
ls tests/ut/op_host/arch32/*.cpp 2>/dev/null
ls tests/ut/op_host/arch35/*.cpp 2>/dev/null
ls tests/ut/op_kernel/*.cpp 2>/dev/null
ls tests/ut/op_kernel/arch35/*.cpp 2>/dev/null
ls tests/ut/op_api/*.cpp 2>/dev/null
ls tests/ut/op_host/op_api/*.cpp 2>/dev/null  # opapi UT 老写法位置
```

### 判断规则（含 arch* 目录 + opapi 两种写法）

| 源文件 | 应有 UT | 检查位置 |
|--------|--------|---------|
| `op_host/*_infershape.cpp` | `test_{op}_infershape.cpp` | `tests/ut/op_host/` |
| `op_host/arch*/*_infershape.cpp` | `test_*_infershape*.cpp` | `tests/ut/op_host/arch*/` |
| `op_host/*_tiling.cpp` | `test_{op}_tiling.cpp` | `tests/ut/op_host/` |
| `op_host/arch*/*_tiling*.cpp` | `test_*_tiling*.cpp` | `tests/ut/op_host/arch*/` |
| `op_kernel/*.cpp` | `test_{op}.cpp` | `tests/ut/op_kernel/` |
| `op_kernel/arch*/*.cpp` | `test_*.cpp` | `tests/ut/op_kernel/arch*/` |
| `op_api/aclnn_*.cpp` (标准写法) | `test_aclnn_*.cpp` | `tests/ut/op_api/` (必须) |
| `op_host/op_api/aclnn_*.cpp` (老写法) | `test_aclnn_*.cpp` | `tests/ut/op_api/` 或 `tests/ut/op_host/op_api/` |

> **opapi 特殊说明**：
> - 标准写法：源文件在 `op_api/aclnn_*.cpp`，UT **必须**在 `tests/ut/op_api/`
> - 老写法：源文件在 `op_host/op_api/aclnn_*.cpp`，UT 可在 `tests/ut/op_api/` 或 `tests/ut/op_host/op_api/`
> - 只有 `aclnn_*.cpp` 文件才需要 UT，其他如 `{op}.cpp` 不需要

### 源文件与 UT 文件匹配规则

对于 `arch*` 目录下的文件，匹配规则：

| 源文件命名 | UT 文件命名 | 匹配方法 |
|-----------|-------------|---------|
| `{op}_tiling_arch35.cpp` | `test_{op}_tiling_arch35.cpp` | 完全对应 |
| `{op}_tiling.cpp` | `test_{op}_tiling.cpp` | 无后缀对应 |
| `{op}_tiling_simt.cpp` | `test_{op}_tiling_simt.cpp` | 变体对应 |

**注意**：UT 文件名应与源文件名保持一致（除前缀 `test_`）。

---

## 二、覆盖率报告分析法

### 获取覆盖率报告

```bash
# 编译并运行 UT，获取覆盖率
cd {repo}
bash build.sh -u --ophost --ops='{op}' --soc='ascend910b' --cov
bash build.sh -u --opkernel --ops='{op}' --soc='ascend910b' --cov
bash build.sh -u --opapi --ops='{op}' --soc='ascend910b' --cov
```

### 分析覆盖率报告

覆盖率报告通常包含：
- 未覆盖的源文件列表
- 未覆盖的代码行号
- 分支覆盖率统计

### 根据未覆盖代码设计 UT

| 未覆盖类型 | 补测方法 |
|-----------|---------|
| dtype 分支 | 新增对应 dtype 测试用例 |
| shape 条件 | 构造满足条件的 shape |
| SOC 版本分支 | 设置对应 SOC 版本 |
| 特殊参数分支 | 提供必要参数 |

---

## 三、Python 扫描脚本（含 arch* 目录）

### 完整扫描脚本（含 opapi 老写法）

```python
#!/usr/bin/env python3
import os
import glob
from pathlib import Path

# 支持的架构列表
ARCH_LIST = ['arch20', 'arch32', 'arch35']

def analyze_op_ut_status(op_dir):
    """分析算子 UT 状态（含 arch* 目录 + opapi 老写法）"""
    op_name = os.path.basename(op_dir)
    result = {
        'op_name': op_name,
        'op_dir': op_dir,
        'sources': {},
        'tests': {},
        'missing': []
    }
    
    # 检查源文件（含 arch* 目录）
    result['sources']['infershape'] = glob.glob(f"{op_dir}/op_host/*_infershape.cpp")
    result['sources']['tiling'] = glob.glob(f"{op_dir}/op_host/*_tiling*.cpp")
    result['sources']['kernel'] = glob.glob(f"{op_dir}/op_kernel/*.cpp")
    # opapi 标准写法
    result['sources']['api'] = glob.glob(f"{op_dir}/op_api/aclnn_*.cpp")
    # opapi 老写法
    result['sources']['api_legacy'] = glob.glob(f"{op_dir}/op_host/op_api/aclnn_*.cpp")
    
    # 检查 arch* 目录下的源文件
    for arch in ARCH_LIST:
        result['sources'][f'infershape_{arch}'] = glob.glob(f"{op_dir}/op_host/{arch}/*_infershape*.cpp")
        result['sources'][f'tiling_{arch}'] = glob.glob(f"{op_dir}/op_host/{arch}/*_tiling*.cpp")
        result['sources'][f'kernel_{arch}'] = glob.glob(f"{op_dir}/op_kernel/{arch}/*.cpp")
    
    # 检查 UT 文件（含 arch* 目录）
    result['tests']['infershape'] = glob.glob(f"{op_dir}/tests/ut/op_host/*_infershape*.cpp")
    result['tests']['tiling'] = glob.glob(f"{op_dir}/tests/ut/op_host/*_tiling*.cpp")
    result['tests']['kernel'] = glob.glob(f"{op_dir}/tests/ut/op_kernel/*.cpp")
    # opapi 标准位置
    result['tests']['api'] = glob.glob(f"{op_dir}/tests/ut/op_api/*.cpp")
    # opapi 老写法 UT 位置
    result['tests']['api_legacy'] = glob.glob(f"{op_dir}/tests/ut/op_host/op_api/*.cpp")
    
    # 检查 arch* 目录下的 UT 文件
    for arch in ARCH_LIST:
        result['tests'][f'infershape_{arch}'] = glob.glob(f"{op_dir}/tests/ut/op_host/{arch}/*_infershape*.cpp")
        result['tests'][f'tiling_{arch}'] = glob.glob(f"{op_dir}/tests/ut/op_host/{arch}/*_tiling*.cpp")
        result['tests'][f'kernel_{arch}'] = glob.glob(f"{op_dir}/tests/ut/op_kernel/{arch}/*.cpp")
    
    # 对比分析缺失
    for key in ['infershape', 'tiling', 'kernel']:
        if result['sources'][key] and not result['tests'][key]:
            result['missing'].append(f'{key} UT')
    
    # opapi 缺失判断（标准写法 + 老写法）
    if result['sources']['api'] and not result['tests']['api']:
        result['missing'].append(f'api UT')
    if result['sources']['api_legacy'] and not (result['tests']['api'] or result['tests']['api_legacy']):
        result['missing'].append(f'api UT (老写法)')
    
    # 对比 arch* 目录下的缺失
    for arch in ARCH_LIST:
        for key in ['infershape', 'tiling', 'kernel']:
            arch_key = f'{key}_{arch}'
            if result['sources'][arch_key] and not result['tests'][arch_key]:
                result['missing'].append(f'{key} UT ({arch})')
    
    return result

def print_report(result):
    """打印分析报告（含 opapi 老写法）"""
    print(f"\n{'='*60}")
    print(f"算子: {result['op_name']}")
    print(f"目录: {result['op_dir']}")
    print(f"{'='*60}")
    
    print("\n源文件:")
    for type_name, files in result['sources'].items():
        if files:
            # 特殊处理 api_legacy
            if type_name == 'api_legacy':
                print(f"  api (老写法): {len(files)} 个")
            elif type_name == 'api':
                print(f"  api (标准写法): {len(files)} 个")
            else:
                arch_info = f" [{type_name.split('_')[-1]}]" if '_' in type_name and type_name.split('_')[-1] in ARCH_LIST else ""
                print(f"  {type_name.split('_')[0]}{arch_info}: {len(files)} 个")
            for f in files:
                print(f"    - {os.path.basename(f)}")
    
    print("\nUT 文件:")
    for type_name, files in result['tests'].items():
        if files:
            # 特殊处理 api_legacy
            if type_name == 'api_legacy':
                print(f"  api (老写法 UT): {len(files)} 个")
            elif type_name == 'api':
                print(f"  api: {len(files)} 个")
            else:
                arch_info = f" [{type_name.split('_')[-1]}]" if '_' in type_name and type_name.split('_')[-1] in ARCH_LIST else ""
                print(f"  {type_name.split('_')[0]}{arch_info}: {len(files)} 个")
            for f in files:
                print(f"    - {os.path.basename(f)}")
    
    print("\n缺失 UT:")
    if result['missing']:
        for m in result['missing']:
            print(f"  ❌ {m}")
    else:
        print("  ✅ 无缺失")
    
    return result

def scan_repo(repo_path, category=None):
    """扫描仓库所有算子"""
    results = []
    
    # 扫算算子目录
    if category:
        patterns = [f"{repo_path}/{category}/*/*/"]
    else:
        patterns = [
            f"{repo_path}/math/*/",
            f"{repo_path}/conversion/*/",
            f"{repo_path}/random/*/",
            f"{repo_path}/experimental/math/*/",
            f"{repo_path}/experimental/conversion/*/",
        ]
    
    op_dirs = []
    for pattern in patterns:
        op_dirs.extend(glob.glob(pattern))
    
    op_dirs = [d for d in op_dirs if os.path.isdir(d) and (
        os.path.exists(f"{d}/op_host") or 
        os.path.exists(f"{d}/op_kernel") or 
        os.path.exists(f"{d}/op_api"))]
    
    for op_dir in op_dirs:
        result = analyze_op_ut_status(op_dir)
        results.append(result)
        print_report(result)
    
    # 统计缺失
    missing_count = sum(1 for r in results if r['missing'])
    print(f"\n{'='*60}")
    print(f"扫描结果: {len(results)} 个算子, {missing_count} 个有 UT 缺失")
    print(f"{'='*60}")
    
    return results

# 使用示例
if __name__ == "__main__":
    repo_path = "/path/to/ops-math"
    results = scan_repo(repo_path)
```

---

## 四、批量检测命令（含 arch* 目录）

### 检测所有 infershape UT 缺失

```bash
# 找有 infershape 源文件但无 UT 的算子（含 arch* 目录）
for dir in */*/; do
    # 检查通用目录
    if ls $dir/op_host/*_infershape.cpp 2>/dev/null; then
        if ! ls $dir/tests/ut/op_host/*_infershape*.cpp 2>/dev/null; then
            echo "缺失 infershape UT: $dir"
        fi
    fi
    # 检查 arch* 目录
    for arch in arch20 arch32 arch35; do
        if ls $dir/op_host/$arch/*_infershape*.cpp 2>/dev/null; then
            if ! ls $dir/tests/ut/op_host/$arch/*_infershape*.cpp 2>/dev/null; then
                echo "缺失 infershape UT ($arch): $dir"
            fi
        fi
    done
done
```

### 检测所有 tiling UT 缺失

```bash
# 找有 tiling 源文件但无 UT 的算子（含 arch* 目录）
for dir in */*/; do
    # 检查通用目录
    if ls $dir/op_host/*_tiling*.cpp 2>/dev/null; then
        if ! ls $dir/tests/ut/op_host/*_tiling*.cpp 2>/dev/null; then
            echo "缺失 tiling UT: $dir"
        fi
    fi
    # 检查 arch* 目录
    for arch in arch20 arch32 arch35; do
        if ls $dir/op_host/$arch/*_tiling*.cpp 2>/dev/null; then
            if ! ls $dir/tests/ut/op_host/$arch/*_tiling*.cpp 2>/dev/null; then
                echo "缺失 tiling UT ($arch): $dir"
            fi
        fi
    done
done
```

### 检测所有 kernel UT 缺失

```bash
# 找有 kernel 源文件但无 UT 的算子（含 arch* 目录）
for dir in */*/; do
    # 检查通用目录（排除 tiling 相关文件）
    kernel_files=$(ls $dir/op_kernel/*.cpp 2>/dev/null | grep -v tiling)
    if [ -n "$kernel_files" ]; then
        if ! ls $dir/tests/ut/op_kernel/*.cpp 2>/dev/null; then
            echo "缺失 kernel UT: $dir"
        fi
    fi
    # 检查 arch* 目录
    for arch in arch20 arch32 arch35; do
        if ls $dir/op_kernel/$arch/*.cpp 2>/dev/null; then
            if ! ls $dir/tests/ut/op_kernel/$arch/*.cpp 2>/dev/null; then
                echo "缺失 kernel UT ($arch): $dir"
            fi
        fi
    done
done
```

### 检测所有 api UT 缺失（含老写法）

```bash
# opapi 有两种源文件位置和两种 UT 位置
for dir in */*/; do
    # 检查标准写法：op_api/aclnn_*.cpp
    if ls $dir/op_api/aclnn_*.cpp 2>/dev/null; then
        if ! ls $dir/tests/ut/op_api/*.cpp 2>/dev/null; then
            echo "缺失 api UT (标准写法): $dir"
        fi
    fi
    # 检查老写法：op_host/op_api/aclnn_*.cpp
    if ls $dir/op_host/op_api/aclnn_*.cpp 2>/dev/null; then
        # UT 可在 tests/ut/op_api/ 或 tests/ut/op_host/op_api/
        if ! ls $dir/tests/ut/op_api/*.cpp 2>/dev/null && ! ls $dir/tests/ut/op_host/op_api/*.cpp 2>/dev/null; then
            echo "缺失 api UT (老写法): $dir"
        fi
    fi
done
```

### 一键检测所有 UT 缺失（含 opapi 老写法）

```bash
# 综合检测脚本
for dir in */*/; do
    missing=""
    
    # infershape
    if ls $dir/op_host/*_infershape*.cpp 2>/dev/null; then
        if ! ls $dir/tests/ut/op_host/*_infershape*.cpp 2>/dev/null; then
            missing+=" infershape"
        fi
    fi
    for arch in arch20 arch32 arch35; do
        if ls $dir/op_host/$arch/*_infershape*.cpp 2>/dev/null; then
            if ! ls $dir/tests/ut/op_host/$arch/*_infershape*.cpp 2>/dev/null; then
                missing+=" infershape($arch)"
            fi
        fi
    done
    
    # tiling
    if ls $dir/op_host/*_tiling*.cpp 2>/dev/null; then
        if ! ls $dir/tests/ut/op_host/*_tiling*.cpp 2>/dev/null; then
            missing+=" tiling"
        fi
    fi
    for arch in arch20 arch32 arch35; do
        if ls $dir/op_host/$arch/*_tiling*.cpp 2>/dev/null; then
            if ! ls $dir/tests/ut/op_host/$arch/*_tiling*.cpp 2>/dev/null; then
                missing+=" tiling($arch)"
            fi
        fi
    done
    
    # kernel
    kernel_files=$(ls $dir/op_kernel/*.cpp 2>/dev/null | grep -v tiling)
    if [ -n "$kernel_files" ]; then
        if ! ls $dir/tests/ut/op_kernel/*.cpp 2>/dev/null; then
            missing+=" kernel"
        fi
    fi
    for arch in arch20 arch32 arch35; do
        if ls $dir/op_kernel/$arch/*.cpp 2>/dev/null; then
            if ! ls $dir/tests/ut/op_kernel/$arch/*.cpp 2>/dev/null; then
                missing+=" kernel($arch)"
            fi
        fi
    done
    
    # api (标准写法)
    if ls $dir/op_api/aclnn_*.cpp 2>/dev/null; then
        if ! ls $dir/tests/ut/op_api/*.cpp 2>/dev/null; then
            missing+=" api"
        fi
    fi
    # api (老写法)
    if ls $dir/op_host/op_api/aclnn_*.cpp 2>/dev/null; then
        if ! ls $dir/tests/ut/op_api/*.cpp 2>/dev/null && ! ls $dir/tests/ut/op_host/op_api/*.cpp 2>/dev/null; then
            missing+=" api(老写法)"
        fi
    fi
    
    if [ -n "$missing" ]; then
        echo "$dir: 缺失$missing"
    fi
done
```

---

## 五、UT 缺失优先级判断

### 高优先级

| 类型 | 原因 |
|-----|------|
| op_api UT | 暴露给用户的最外层接口，必须验证 |
| op_kernel UT | 核心计算逻辑，影响精度 |

### 中优先级

| 类型 | 原因 |
|-----|------|
| tiling UT | 影响性能和正确执行 |
| infershape UT | 影响 shape 推导 |

### 低优先级

| 类型 | 原因 |
|-----|------|
| 特殊场景 UT | 边界条件、特殊值处理 |

---

## 六、UT 缺失报告模板（含 arch* 目录）

```markdown
# UT 缺失分析报告

## 基本信息
- 算子名：{op_name}
- 目录：{op_dir}
- 分析时间：{timestamp}

## 源文件与 UT 对应表

### 通用实现

| 源文件 | 类型 | UT 文件 | 状态 |
|--------|------|---------|------|
| op_host/{op}_infershape.cpp | infershape | tests/ut/op_host/test_{op}_infershape.cpp | ✅/❌ |
| op_host/{op}_tiling.cpp | tiling | tests/ut/op_host/test_{op}_tiling.cpp | ✅/❌ |
| op_kernel/{op}.cpp | kernel | tests/ut/op_kernel/test_{op}.cpp | ✅/❌ |
| op_api/aclnn_{op}.cpp (标准写法) | api | tests/ut/op_api/test_aclnn_{op}.cpp | ✅/❌ |
| op_host/op_api/aclnn_{op}.cpp (老写法) | api | tests/ut/op_api/test_aclnn_{op}.cpp 或 tests/ut/op_host/op_api/ | ✅/❌ |

### 架构特定实现（如有）

| 源文件 | 类型 | 架构 | UT 文件 | 状态 |
|--------|------|------|---------|------|
| op_host/arch35/{op}_tiling_arch35.cpp | tiling | arch35 | tests/ut/op_host/arch35/test_{op}_tiling_arch35.cpp | ✅/❌ |
| op_host/arch35/{op}_tiling.cpp | tiling | arch35 | tests/ut/op_host/arch35/test_{op}_tiling.cpp | ✅/❌ |
| op_kernel/arch35/{op}.cpp | kernel | arch35 | tests/ut/op_kernel/arch35/test_{op}.cpp | ✅/❌ |

### opapi 两种写法说明

| 写法类型 | 源文件位置 | UT 位置要求 |
|---------|-----------|------------|
| 标准写法 | op_api/aclnn_*.cpp | **必须** tests/ut/op_api/ |
| 老写法 | op_host/op_api/aclnn_*.cpp | tests/ut/op_api/ 或 tests/ut/op_host/op_api/ |

## 缺失 UT 详情

### 通用 infershape UT
- 状态：缺失
- 建议测试场景：静态 shape、动态 shape、空 tensor

### 通用 tiling UT
- 状态：缺失
- 建议测试场景：FP16/FP32/BF16、不同 SOC 版本

### arch35 tiling UT
- 状态：缺失
- 源文件：op_host/arch35/{op}_tiling_arch35.cpp
- 建议测试场景：arch35 特定 dtype、arch35 特定 shape

### arch35 kernel UT
- 状态：缺失
- 源文件：op_kernel/arch35/{op}.cpp
- 建议测试场景：arch35 特定计算精度

### kernel UT
- 状态：缺失
- 建议测试场景：不同 dtype 计算精度

### api UT
- 状态：缺失
- 建议测试场景：dtype 支持、参数校验、SOC 兼容

## 补测优先级

| 优先级 | UT 类型 | 原因 |
|--------|---------|------|
| 高 | api | 用户接口层 |
| 高 | kernel | 核心计算 |
| 中 | tiling | 性能影响 |
| 中 | infershape | shape 推导 |

## 建议的测试用例数量

| UT 类型 | 建议用例数 | 覆盖重点 |
|---------|-----------|---------|
| infershape | 3-5 | 静态/动态/空 tensor |
| tiling | 6-10 | 各 dtype/SOC 版本 |
| kernel | 3-5 | 各 dtype 计算精度 |
| api | 10-15 | dtype/参数校验/边界 |
```