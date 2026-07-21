---
name: skill-trace
description: >
  Track skill invocations, durations, and outcomes during operator generation.
  Records which skills were called, their inputs/outputs, and correlates with final results.
  MUST use when: (1) Starting any skill step, (2) Completing any skill step,
  (3) Final summary to correlate skills with outcomes.
allowed-tools: Read, Write, Edit, Bash
---

# Skill Trace — 技能调用追踪

记录算子生成和性能优化过程中各 skill 的调用信息，用于分析不同 skill 对最终生成结果的影响。

## 文件位置

`skill_trace.json` 保存在任务输出目录下：
- **cake 模式**: `output/{op_name}/skill_trace.json`
- **cake-evo 模式**: `output/{op_name}_evo_{timestamp}/skill_trace.json`
- **cake-partial 模式**: `output/{op_name}_evo_{timestamp}/round_{r}/parallel_{p}/skill_trace.json`

## 数据结构

```json
{
  "version": "1.0",
  "op_name": "FastGELU",
  "mode": "cake | cake-evo | cake-partial",
  "agent_id": "cake-evo",
  "variant_id": "round_1/parallel_0",
  "started_at": "2026-03-31T10:00:00Z",
  "completed_at": "2026-03-31T10:30:00Z",
  "total_duration_s": 1800,
  "skills": [
    {
      "skill_name": "op-desc-generation",
      "stage": "stage_1",
      "started_at": "2026-03-31T10:00:00Z",
      "completed_at": "2026-03-31T10:02:30Z",
      "duration_s": 150,
      "status": "success",
      "retry_count": 0,
      "input_summary": "op_name=FastGELU, description='Compute x * sigmoid(1.702 * x)'",
      "output_summary": "Generated FastGELU_op_desc.json with 5 test cases",
      "output_files": ["FastGELU_op_desc.json"],
      "error_message": null
    },
    {
      "skill_name": "dsl-lowering",
      "stage": "stage_7",
      "started_at": "2026-03-31T10:10:00Z",
      "completed_at": "2026-03-31T10:18:00Z",
      "duration_s": 480,
      "status": "success",
      "retry_count": 1,
      "input_summary": "DSL file: FastGELU_dsl.py, strategies: [perf_01, perf_02, acc_01]",
      "output_summary": "Generated AscendC kernel, compiled successfully after 1 retry",
      "output_files": ["FastGELUCustom/op_kernel/FastGELU_custom.cpp"],
      "error_message": null,
      "metadata": {
        "strategies_applied": ["perf_01_double_buffer", "perf_02_adaptive_tiling", "acc_01_fp32_intermediate"],
        "compilation_attempts": 2
      }
    }
  ],
  "final_result": {
    "compilation_success": true,
    "precision_passed": true,
    "speedup": 2.3,
    "base_time_ms": 1.42,
    "gen_time_ms": 0.62
  },
  "skill_impact_summary": {
    "total_skills_called": 8,
    "total_duration_s": 1200,
    "failed_skills": [],
    "retried_skills": ["dsl-lowering"],
    "critical_path": ["op-desc-generation", "reference-generation", "functional-conversion", "ascend-call-generation", "dsl-baseline-generation", "dsl-lowering", "cake-code-review", "ascendc-evaluation"],
    "optimization_skills": ["code-performance-advisor", "dsl-optimization"],
    "debug_skills_used": ["ascendc-op-debug"]
  }
}
```

## 操作规则

### TRACE-INIT: 初始化 trace 文件

在任务开始时（cake 阶段 0 / cake-evo 步骤 3 / cake-partial 前置阶段）创建 `skill_trace.json`：

```bash
TRACE_FILE="{output_dir}/skill_trace.json"
cat > ${TRACE_FILE} << 'TRACE_EOF'
{
  "version": "1.0",
  "op_name": "{op_name}",
  "mode": "{mode}",
  "agent_id": "{agent_id}",
  "variant_id": "{variant_id}",
  "started_at": "{timestamp_iso}",
  "completed_at": null,
  "total_duration_s": null,
  "skills": [],
  "final_result": null,
  "skill_impact_summary": null
}
TRACE_EOF
```

其中:
- `mode`: `"cake"` / `"cake-evo"` / `"cake-partial"`
- `agent_id`: agent 名称（如 `"cake-evo"`）
- `variant_id`: 变体标识（cake-partial 用 `"round_1/parallel_0"`，其余为 `"main"`）

### TRACE-START: 记录 skill 开始

在每个 skill 执行**之前**，追加一条记录到 `skills` 数组：

```python
# 概念性 Python 伪代码 — 实际用 bash/jq 或直接在 JSON 中追加
trace_entry = {
    "skill_name": "{skill_name}",
    "stage": "{stage_id}",
    "started_at": "{now_iso}",
    "completed_at": None,
    "duration_s": None,
    "status": "running",
    "retry_count": 0,
    "input_summary": "{简要描述输入}",
    "output_summary": None,
    "output_files": [],
    "error_message": None
}
```

**实际执行方式**（使用 python3 单行脚本操作 JSON）：

```bash
python3 -c "
import json, datetime
with open('${TRACE_FILE}') as f: data = json.load(f)
data['skills'].append({
    'skill_name': '${SKILL_NAME}',
    'stage': '${STAGE_ID}',
    'started_at': datetime.datetime.now().isoformat() + 'Z',
    'completed_at': None,
    'duration_s': None,
    'status': 'running',
    'retry_count': 0,
    'input_summary': '${INPUT_SUMMARY}',
    'output_summary': None,
    'output_files': [],
    'error_message': None
})
with open('${TRACE_FILE}', 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False)
print(f'TRACE: {\"${SKILL_NAME}\"} started')
"
```

### TRACE-END: 记录 skill 完成

在每个 skill 执行**之后**，更新最后一条记录：

```bash
python3 -c "
import json, datetime
with open('${TRACE_FILE}') as f: data = json.load(f)
entry = data['skills'][-1]
assert entry['skill_name'] == '${SKILL_NAME}', f'Trace mismatch: expected ${SKILL_NAME}, got {entry[\"skill_name\"]}'
now = datetime.datetime.now()
started = datetime.datetime.fromisoformat(entry['started_at'].rstrip('Z'))
entry['completed_at'] = now.isoformat() + 'Z'
entry['duration_s'] = round((now - started).total_seconds(), 1)
entry['status'] = '${STATUS}'  # success / failed / skipped
entry['retry_count'] = ${RETRY_COUNT}
entry['output_summary'] = '${OUTPUT_SUMMARY}'
entry['output_files'] = ${OUTPUT_FILES_JSON}  # e.g. [\"file1.json\", \"file2.py\"]
entry['error_message'] = ${ERROR_MSG}  # null or '\"error text\"'
with open('${TRACE_FILE}', 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False)
print(f'TRACE: {\"${SKILL_NAME}\"} completed ({entry[\"duration_s\"]}s, {\"${STATUS}\"})')
"
```

### TRACE-META: 附加元数据（可选）

对于需要记录额外信息的 skill（如 dsl-lowering 的策略选择），追加 metadata 字段：

```bash
python3 -c "
import json
with open('${TRACE_FILE}') as f: data = json.load(f)
entry = data['skills'][-1]
entry.setdefault('metadata', {})
entry['metadata']['strategies_applied'] = ${STRATEGIES_JSON}
with open('${TRACE_FILE}', 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False)
"
```

### TRACE-FINALIZE: 任务完成时写入最终结果

在任务最后（cake 阶段 11 / cake-evo 步骤 5 / cake-partial 阶段 5）：

```bash
python3 -c "
import json, datetime
with open('${TRACE_FILE}') as f: data = json.load(f)

# 写入完成时间
now = datetime.datetime.now()
data['completed_at'] = now.isoformat() + 'Z'
started = datetime.datetime.fromisoformat(data['started_at'].rstrip('Z'))
data['total_duration_s'] = round((now - started).total_seconds(), 1)

# 写入最终结果（从 evaluation_results.json 读取）
try:
    eval_path = '${OUTPUT_DIR}/evaluation_results.json'
    with open(eval_path) as ef: eval_data = json.load(ef)
    data['final_result'] = {
        'compilation_success': eval_data.get('compilation_success', False),
        'precision_passed': eval_data.get('precision_passed', False),
        'speedup': eval_data.get('speedup', 0),
        'base_time_ms': eval_data.get('base_time_ms', 0),
        'gen_time_ms': eval_data.get('gen_time_ms', 0)
    }
except FileNotFoundError:
    data['final_result'] = {'compilation_success': False, 'precision_passed': False, 'speedup': 0}

# 生成影响摘要
skills = data['skills']
data['skill_impact_summary'] = {
    'total_skills_called': len(skills),
    'total_duration_s': round(sum(s.get('duration_s', 0) or 0 for s in skills), 1),
    'failed_skills': [s['skill_name'] for s in skills if s.get('status') == 'failed'],
    'retried_skills': [s['skill_name'] for s in skills if (s.get('retry_count', 0) or 0) > 0],
    'critical_path': [s['skill_name'] for s in skills if s.get('status') == 'success'],
    'optimization_skills': [s['skill_name'] for s in skills if s['skill_name'] in ('code-performance-advisor', 'dsl-optimization')],
    'debug_skills_used': [s['skill_name'] for s in skills if s['skill_name'] in ('ascendc-op-debug',)]
}

with open('${TRACE_FILE}', 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False)
print('TRACE: finalized skill_trace.json')
print(f'  Total skills: {len(skills)}')
print(f'  Total time: {data[\"total_duration_s\"]}s')
print(f'  Failed: {data[\"skill_impact_summary\"][\"failed_skills\"]}')
print(f'  Final speedup: {data[\"final_result\"].get(\"speedup\", \"N/A\")}x')
"
```

### TRACE-AGGREGATE: 聚合多变体 trace（仅 cake-evo 使用）

在 cake-evo 步骤 5（最终结果）中，聚合所有变体的 trace 生成全局分析：

```bash
python3 -c "
import json, os, glob

evo_dir = '${EVO_DIR}'
traces = []
for trace_file in sorted(glob.glob(os.path.join(evo_dir, 'round_*/parallel_*/skill_trace.json'))):
    with open(trace_file) as f:
        traces.append(json.load(f))

# 聚合分析
aggregate = {
    'version': '1.0',
    'op_name': '${OP_NAME}',
    'total_variants': len(traces),
    'variants': [],
    'skill_frequency': {},
    'skill_avg_duration': {},
    'skill_success_rate': {},
    'correlation_analysis': {
        'best_variant': None,
        'best_speedup': 0,
        'best_variant_skills': [],
        'worst_variant': None,
        'worst_speedup': float('inf'),
        'worst_variant_skills': []
    }
}

for t in traces:
    variant_info = {
        'variant_id': t.get('variant_id', 'unknown'),
        'total_duration_s': t.get('total_duration_s', 0),
        'speedup': t.get('final_result', {}).get('speedup', 0),
        'precision_passed': t.get('final_result', {}).get('precision_passed', False),
        'skills_called': [s['skill_name'] for s in t.get('skills', [])],
        'failed_skills': [s['skill_name'] for s in t.get('skills', []) if s.get('status') == 'failed'],
        'strategies': []
    }
    # 提取策略信息
    for s in t.get('skills', []):
        if s.get('metadata', {}).get('strategies_applied'):
            variant_info['strategies'] = s['metadata']['strategies_applied']
    aggregate['variants'].append(variant_info)

    # 统计技能频率和时长
    for s in t.get('skills', []):
        name = s['skill_name']
        aggregate['skill_frequency'][name] = aggregate['skill_frequency'].get(name, 0) + 1
        if name not in aggregate['skill_avg_duration']:
            aggregate['skill_avg_duration'][name] = []
        if s.get('duration_s'):
            aggregate['skill_avg_duration'][name].append(s['duration_s'])
        if name not in aggregate['skill_success_rate']:
            aggregate['skill_success_rate'][name] = {'success': 0, 'total': 0}
        aggregate['skill_success_rate'][name]['total'] += 1
        if s.get('status') == 'success':
            aggregate['skill_success_rate'][name]['success'] += 1

    # 找最佳/最差变体
    speedup = variant_info['speedup'] or 0
    if speedup > aggregate['correlation_analysis']['best_speedup']:
        aggregate['correlation_analysis']['best_speedup'] = speedup
        aggregate['correlation_analysis']['best_variant'] = variant_info['variant_id']
        aggregate['correlation_analysis']['best_variant_skills'] = variant_info['skills_called']
    if 0 < speedup < aggregate['correlation_analysis']['worst_speedup']:
        aggregate['correlation_analysis']['worst_speedup'] = speedup
        aggregate['correlation_analysis']['worst_variant'] = variant_info['variant_id']
        aggregate['correlation_analysis']['worst_variant_skills'] = variant_info['skills_called']

# 计算平均时长
for name in aggregate['skill_avg_duration']:
    durations = aggregate['skill_avg_duration'][name]
    aggregate['skill_avg_duration'][name] = round(sum(durations) / len(durations), 1) if durations else 0

# 计算成功率
for name in aggregate['skill_success_rate']:
    info = aggregate['skill_success_rate'][name]
    info['rate'] = round(info['success'] / info['total'], 2) if info['total'] > 0 else 0

output_path = os.path.join(evo_dir, 'skill_trace_aggregate.json')
with open(output_path, 'w') as f:
    json.dump(aggregate, f, indent=2, ensure_ascii=False)

print(f'TRACE-AGGREGATE: saved to {output_path}')
print(f'  Variants: {len(traces)}')
print(f'  Best: {aggregate[\"correlation_analysis\"][\"best_variant\"]} ({aggregate[\"correlation_analysis\"][\"best_speedup\"]}x)')
print(f'  Skill frequency: {dict(sorted(aggregate[\"skill_frequency\"].items(), key=lambda x: -x[1]))}')
"
```

## 输出展示

### cake-evo 步骤 5 最终结果中的 Skill Trace 摘要

在最终结果展示中追加：

```
Skill 调用追踪:
  总技能调用: {total_skills_called}
  总耗时: {total_duration_s}s
  失败技能: {failed_skills}
  重试技能: {retried_skills}

  各技能平均耗时:
    op-desc-generation:     {avg_duration}s
    reference-generation:   {avg_duration}s
    dsl-lowering:           {avg_duration}s (成功率: {success_rate}%)
    ascendc-evaluation:     {avg_duration}s

  最佳变体 ({best_variant}, {best_speedup}x) 使用的技能:
    {skills_list}
    策略: {strategies_list}

  最差变体 ({worst_variant}, {worst_speedup}x) 差异:
    {diff_skills}

  详细数据: {evo_dir}/skill_trace_aggregate.json
```

### cake 阶段 11 总结中的 Skill Trace 摘要

```
Skill 调用追踪:
  总技能调用: {total_skills_called}
  总耗时: {total_duration_s}s
  关键路径: {critical_path}
  失败技能: {failed_skills}
  调试技能: {debug_skills_used}
  详细数据: output/{op_name}/skill_trace.json
```
