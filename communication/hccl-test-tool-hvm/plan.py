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

"""检查并串行执行已审查的测试计划；不构建、不安装、不自动重试。"""
import argparse
from datetime import datetime, timezone
import json
import itertools
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import evidence
from parameters import validate_dtypes

OPS = {'allgather', 'allreduce', 'reduce_scatter', 'reduce', 'scatter', 'broadcast', 'alltoall', 'alltoallv'}
MODES = {'AI_CPU', 'CCU_MS', 'CCU_SCHED'}


def path_value(base, value):
    p = Path(value)
    return str((base / p).resolve())


def read_plan(path):
    path = Path(path).resolve()
    plan = evidence.load(path)
    if plan.get('repair_origin'):
        origin = plan['repair_origin']
        root = Path(origin['run'])
        if (evidence.digest(root / 'manifest.json') != origin['manifest_sha256'] or
                evidence.digest(root / 'results.jsonl') != origin['results_sha256']):
            raise ValueError('补测计划来源已变化，请重新核对原始记录')
    if plan.get('schema_version') != 1 or not plan.get('batches'):
        raise ValueError('计划须为 schema_version=1 且包含非空 batches')
    for key in ('install_dir', 'cann', 'cluster'):
        if not isinstance(plan.get(key), str) or not plan[key]:
            raise ValueError('缺少计划配置: ' + key)
    ids = set()
    for batch in plan['batches']:
        bid = batch.get('id', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', bid) or bid in ids:
            raise ValueError('批次 id 非法或重复')
        ids.add(bid)
        role = batch.get('role')
        if role not in evidence.ROLES:
            raise ValueError(bid + ': 必须指定固定 role')
        if not isinstance(batch.get('covers'), list) or not batch['covers'] or not all(
                isinstance(x, str) and x.strip() for x in batch['covers']):
            raise ValueError(bid + ': covers 须说明覆盖目标')
        for key in ('modes', 'ops', 'dtypes', 'sizes', 'comms'):
            values = batch.get(key)
            if not isinstance(values, list) or not values or len(set(map(str, values))) != len(values):
                raise ValueError(bid + ': 非空且无重复的列表必填: ' + key)
            if key == 'sizes':
                if any(type(x) is not int or x <= 0 for x in values):
                    raise ValueError(bid + ': sizes 只接受正整数字节数，避免范围展开歧义')
            elif any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9_]+', x) for x in values):
                raise ValueError(bid + ': 非法列表值: ' + key)
        if not set(batch['modes']) <= MODES or not set(batch['ops']) <= OPS:
            raise ValueError(bid + ': 未知引擎或算子')
        test_bin_dir = path_value(path.parent, plan['test_bin_dir']) if plan.get('test_bin_dir') else None
        validate_dtypes(batch['dtypes'], test_bin_dir)
        env = batch.get('env', {})
        if not isinstance(env, dict) or any(not re.fullmatch(r'HCCL_[A-Z0-9_]+', k)
                                          or not isinstance(v, str) for k, v in env.items()):
            raise ValueError(bid + ': env 仅接受显式 HCCL_* 字符串配置')
        if set(env) & {'HCCL_VM_INSTALL_DIR', 'HCCL_OP_EXPANSION_MODE', 'HCCL_ENABLE_OPEN_CCU'}:
            raise ValueError(bid + ': 不得覆盖工具管理的安装路径或引擎变量')
        if role in ('baseline', 'regression'):
            if not batch.get('artifact') or env.get('HCCL_ALGO'):
                raise ValueError(bid + ': 基线/回归须关联 artifact 且不得设置 HCCL_ALGO')
        if role == 'directed' and (not batch.get('expect_algo') or not env.get('HCCL_ALGO')):
            raise ValueError(bid + ': 定向批次须指定 DSL 和 expect_algo')
        if int(batch.get('timeout', 180)) < 1:
            raise ValueError(bid + ': timeout 必须为正')
        if (type(batch.get('iterations', 1)) is not int or batch.get('iterations', 1) < 1 or
                type(batch.get('warmup', 0)) is not int or batch.get('warmup', 0) < 0 or
                type(batch.get('runner', False)) is not bool):
            raise ValueError(bid + ': iterations/warmup/runner 类型或范围无效')
        for assertion in batch.get('assertions', []):
            if not assertion.get('name') or not assertion.get('basis'):
                raise ValueError(bid + ': 日志断言须有 name 和源码/公式依据 basis')
            pattern = re.compile(assertion['pattern'])
            if pattern.groups != 1 or type(assertion.get('minimum')) is not int:
                raise ValueError(bid + ': 日志断言须有一个数值捕获组和整数 minimum')
    return path.parent, plan


def command(base, plan, batch, output, dry_run=False):
    args = ['bash', str(Path(__file__).with_name('run.sh')),
            '--install-dir', path_value(base, plan['install_dir']),
            '--cann', path_value(base, plan['cann']), '--cluster', plan['cluster'],
            '--log-dir', str(output), '--role', batch['role'], '--phase', batch['id'],
            '--timeout', str(batch.get('timeout', 180))]
    args += ['-n', str(batch.get('iterations', 1)), '-w', str(batch.get('warmup', 0))]
    if batch.get('runner', False):
        args.append('--runner')
    for key, flag in (('modes', '-m'), ('ops', '-o'), ('dtypes', '-d'), ('sizes', '-s'), ('comms', '-t')):
        args += [flag, ','.join(map(str, batch[key]))]
    if plan.get('test_bin_dir'):
        args += ['--test-bin-dir', path_value(base, plan['test_bin_dir'])]
    if batch.get('artifact'):
        args += ['--artifact', path_value(base, batch['artifact'])]
    if batch.get('expect_algo'):
        args += ['--expect-algo', batch['expect_algo']]
    batch_env = {'HCCL_ALGO': '', 'HCCL_USE_NEW_SELECTOR': '0', **batch.get('env', {})}
    for k, v in sorted(batch_env.items()):
        args += ['--env', k + '=' + v]
    if dry_run:
        args.append('--dry-run')
    return args


def assertions_for(batch, rows):
    results = []
    for row in rows:
        text = Path(row['raw_log']).read_text(errors='replace')
        if evidence.digest(row['raw_log']) != row['raw_log_sha256']:
            raise ValueError('原始日志被修改')
        for assertion in batch.get('assertions', []):
            matches = re.findall(assertion['pattern'], text)
            values = [int(v) for v in matches if re.fullmatch(r'\d+', v)]
            results.append(dict(case_id=row['case_id'], name=assertion['name'],
                                values=values, minimum=assertion['minimum'],
                                passed=bool(values) and max(values) >= assertion['minimum']))
    return results


def validate_batch(batch, runs, rows):
    if len(runs) != 1:
        raise ValueError('批次缺少唯一运行目录')
    manifest = evidence.load(runs[0] / 'manifest.json')
    expected = [
        evidence.case_id(*c)
        for c in itertools.product(
            batch['modes'], batch['ops'], batch['dtypes'],
            batch['sizes'], batch['comms']
        )
    ]
    keys = [r['case_id'] for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError('实际用例重复或与计划集合不一致')
    if manifest.get('role') != batch['role'] or set(manifest['expected_cases']) != set(expected) or manifest['dry_run']:
        raise ValueError('实际运行角色或计划身份不匹配')
    if batch['role'] in ('baseline', 'regression'):
        evidence.validate(runs[0], require_pass=True, require_role=batch['role'])
    for row in rows:
        raw = Path(row['raw_log'])
        parsed = evidence.parse_log(raw.read_text(errors='replace'), batch.get('expect_algo', ''), row['timed_out'])
        if parsed['verdict'] != 'PASS':
            raise ValueError('原始日志未通过 Checker 或算法命中断言: ' + row['case_id'])


def execute(a):
    base, plan = read_plan(a.plan)
    batches = [b for b in plan['batches'] if b['role'] == a.role]
    if not batches:
        raise ValueError('计划内没有该角色的批次')
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    evidence.write_json(out / 'plan.json', plan)
    summary = dict(started_at=datetime.now(timezone.utc).isoformat(), role=a.role,
                   dry_run=a.dry_run, status='RUNNING', batches=[])
    plan_start = time.monotonic()
    evidence.write_json(out / 'summary.json', summary)
    # 每批从同一显式计划构造环境，防止调用者或上一批遗留 ALGO/BUFFSIZE/selector。
    env = {k: v for k, v in os.environ.items() if not k.startswith('HCCL_')}
    env['HWLOC_COMPONENTS'] = '-gl,-opencl'
    failed = False
    for batch in batches:
        batch_out = out / batch['id']
        batch_out.mkdir()
        args = command(base, plan, batch, batch_out, a.dry_run)
        result = dict(id=batch['id'], covers=batch['covers'], command=args,
                      started_at=datetime.now(timezone.utc).isoformat())
        start = time.monotonic()
        print('RUN ' + batch['id'], flush=True)
        with (batch_out / 'console.log').open('w') as log:
            completed = subprocess.run(args, env=env, stdout=log, stderr=subprocess.STDOUT)
        result.update(exit_code=completed.returncode, wall_seconds=round(time.monotonic() - start, 3))
        runs = list(batch_out.glob('run_*'))
        rows = []
        if len(runs) == 1 and (runs[0] / 'results.jsonl').is_file():
            rows = [json.loads(line) for line in (runs[0] / 'results.jsonl').read_text().splitlines() if line.strip()]
        expected_count = 1
        for key in ('modes', 'ops', 'dtypes', 'sizes', 'comms'):
            expected_count *= len(batch[key])
        assertion_error = None
        try:
            if not a.dry_run:
                validate_batch(batch, runs, rows)
            assertions = assertions_for(batch, rows) if not a.dry_run else []
        except (ValueError, OSError, KeyError) as e:
            assertions, assertion_error = [], str(e)
        result.update(expected_cases=expected_count, executed_cases=len(rows),
                      passed=sum(r['verdict'] == 'PASS' for r in rows),
                      failed=sum(r['verdict'] != 'PASS' for r in rows),
                      case_seconds=sum(r['duration_seconds'] for r in rows),
                      assertions=assertions, assertion_error=assertion_error)
        failed = bool(assertion_error) or completed.returncode != 0 or (not a.dry_run and (
            len(rows) != expected_count or result['failed'] or
            any(not r['passed'] for r in result['assertions'])))
        summary['batches'].append(result)
        summary['status'] = 'FAIL' if failed else 'RUNNING'
        evidence.write_json(out / 'summary.json', summary)
        if failed:
            break  # 保留原失败；诊断后另建计划/运行目录，不循环直到 PASS。
    summary.update(status='FAIL' if failed else 'DRY_RUN' if a.dry_run else 'PASS',
                   finished_at=datetime.now(timezone.utc).isoformat(),
                   pending_batches=len(batches) - len(summary['batches']))
    recorded = [] if a.dry_run else summary['batches']
    summary['totals'] = dict(
        attempted_batches=len(summary['batches']),
        executed_batches=sum(b['executed_cases'] > 0 for b in recorded),
        dry_run_batches=len(summary['batches']) if a.dry_run else 0,
        **{key: sum(b[key] for b in recorded)
           for key in ('executed_cases', 'passed', 'failed', 'case_seconds')},
        batch_wall_seconds=round(sum(b['wall_seconds'] for b in summary['batches']), 3),
        wall_seconds=round(time.monotonic() - plan_start, 3))
    evidence.write_json(out / 'summary.json', summary)
    print(str(out / 'summary.json'))
    return int(failed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check')
    check.add_argument('plan')
    run = sub.add_parser('run')
    run.add_argument('plan')
    run.add_argument('--role', choices=evidence.ROLES, required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--dry-run', action='store_true')
    loop = sub.add_parser('loop-size', help='使用已核验的 executor 计数单位，计算刚超过一轮的对齐测试大小')
    loop.add_argument('--max-count-per-loop', type=int, required=True)
    loop.add_argument('--test-bytes-per-count', type=int, required=True)
    loop.add_argument('--alignment-count', type=int, default=1)
    a = p.parse_args()
    try:
        if a.command == 'run':
            return execute(a)
        if a.command == 'check':
            _, plan = read_plan(a.plan)
            print('计划格式有效: %d 批；不代表覆盖充分或环境可用' % len(plan['batches']))
        else:
            if min(a.max_count_per_loop, a.test_bytes_per_count, a.alignment_count) <= 0:
                raise ValueError('计数、换算比例和对齐均须为正')
            count = (a.max_count_per_loop // a.alignment_count + 1) * a.alignment_count
            print(json.dumps(dict(executor_count=count, test_bytes=count * a.test_bytes_per_count,
                                  requires_log_assertion=True)))
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
