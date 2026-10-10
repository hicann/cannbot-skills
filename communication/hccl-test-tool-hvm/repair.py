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

"""生成失败/缺失用例补测计划，合并原始证据，检查切离旧包前的基线；不执行测试。"""
import argparse
import copy
import json
from pathlib import Path
import sys

import evidence
import plan

CONFIG_KEYS = ('role', 'source_id', 'declared_artifact', 'environment', 'cluster', 'iterations', 'warmup', 'runner')


def read_run(root):
    root = Path(root).resolve()
    m = evidence.load(root / 'manifest.json')
    if m['dry_run']:
        raise ValueError('不能补测或合并 dry-run')
    if m.get('role') not in evidence.ROLES:
        raise ValueError('缺少明确 role')
    if m.get('declared_artifact') or m['role'] in ('baseline', 'regression'):
        evidence.check_identity(m)
    for source in m.get('evidence_sources', []):
        origin = Path(source['path'])
        if (evidence.digest(origin / 'manifest.json') != source['manifest_sha256'] or
                evidence.digest(origin / 'results.jsonl') != source['results_sha256']):
            raise ValueError('合并来源证据已变化')
    rows = {}
    expected = m['expected_cases']
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('原计划为空或重复')
    for line in (root / 'results.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row['case_id']
        if key in rows or key not in expected:
            raise ValueError('实际用例重复或不在原计划内')
        for attempt in [row] + row.get('prior_attempts', []):
            log = Path(attempt['raw_log'])
            if evidence.digest(log) != attempt['raw_log_sha256']:
                raise ValueError('原始日志被修改: ' + str(log))
            parsed = evidence.parse_log(log.read_text(errors='replace'), attempt.get('expected_algorithm', ''),
                                        attempt.get('timed_out', False))
            if parsed['verdict'] != attempt['verdict']:
                raise ValueError('结果与原始日志不一致: ' + key)
        rows[key] = row
    return root, m, rows


def retry_plan(a):
    root, manifest, rows = read_run(a.run)
    missing = [k for k in manifest['expected_cases'] if k not in rows or rows[k]['verdict'] != 'PASS']
    if not missing:
        raise ValueError('没有失败或缺失用例；不会重跑已通过矩阵')
    if any(rows.get(k, {}).get('prior_attempts') for k in missing):
        raise ValueError('已有补测历史，停止自动生成下一轮；先定位根因')
    replay = manifest.get('replay', {})
    fields = {}
    for key in ('cann', 'install_dir', 'test_bin_dir', 'artifact'):
        fields[key] = getattr(a, key, None) or replay.get(key, '')
    for key in ('cann', 'install_dir', 'test_bin_dir'):
        if not fields[key]:
            raise ValueError('旧记录缺 replay，请明确提供 --' + key.replace('_', '-'))
        fields[key] = str(Path(fields[key]).resolve())
    if manifest.get('declared_artifact'):
        if not fields['artifact'] or evidence.load(fields['artifact']) != manifest['declared_artifact']:
            raise ValueError('补测必须使用同一归档身份；请提供对应 --artifact')
        fields['artifact'] = str(Path(fields['artifact']).resolve())
    batches = []
    for i, key in enumerate(missing, 1):
        mode, op, dtype, size, comm = key.split('|')
        batch = dict(id='repair-%03d' % i, role=manifest['role'], modes=[mode], ops=[op], dtypes=[dtype],
                     sizes=[int(size)], comms=[comm], covers=['补测原用例 ' + key],
                     env={k: v for k, v in manifest['environment'].items() if k.startswith('HCCL_')},
                     iterations=manifest['iterations'], warmup=manifest['warmup'], runner=bool(manifest['runner']))
        if fields['artifact']:
            batch['artifact'] = fields['artifact']
        expected = rows.get(key, {}).get('expected_algorithm') or replay.get('expected_algorithm')
        if expected:
            batch['expect_algo'] = expected
        batches.append(batch)
    result = dict(schema_version=1, install_dir=fields['install_dir'], cann=fields['cann'],
                  test_bin_dir=fields['test_bin_dir'], cluster=manifest['cluster'], batches=batches,
                  repair_origin=dict(run=str(root), manifest_sha256=evidence.digest(root / 'manifest.json'),
                                     results_sha256=evidence.digest(root / 'results.jsonl')))
    out = Path(a.output)
    with out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    try:
        plan.read_plan(out)
    except (ValueError, OSError, KeyError, TypeError):
        out.unlink()
        raise
    return {'plan': str(out.resolve()), 'cases': len(missing), 'note': '仅诊断补测；原FAIL不会自动豁免'}


def supplement(a):
    base, manifest, rows = read_run(a.base)
    merged = copy.deepcopy(rows)
    sources = [base]
    for run in a.rerun:
        root, other, additions = read_run(run)
        if any(manifest.get(k) != other.get(k) for k in CONFIG_KEYS):
            raise ValueError('补测包身份、角色或配置不同，不能合并')
        for key, row in additions.items():
            if row.get('prior_attempts'):
                raise ValueError('补测输入已有历史，不能覆盖或折叠诊断尝试')
            if key not in manifest['expected_cases']:
                raise ValueError('补测包含原矩阵以外的用例')
            # 同一算子的测试程序不能在补缺项时悄然更换。
            op = key.split('|')[1]
            known_bins = {r.get('test_binary_sha256') for k, r in merged.items() if k.split('|')[1] == op}
            if known_bins and known_bins != {row.get('test_binary_sha256')}:
                raise ValueError('补测测试二进制身份不同')
            old = merged.get(key)
            if old and (old['verdict'] == 'PASS' or old.get('prior_attempts')):
                raise ValueError('禁止替换已通过用例或重复追加诊断尝试')
            if old:
                for field in ('test_binary_sha256', 'case_environment', 'expected_algorithm'):
                    if old.get(field) != row.get(field):
                        raise ValueError('补测执行条件不同: ' + field)
            new = copy.deepcopy(row)
            if old:
                new['prior_attempts'] = [old]
                new['outcome'] = 'PASS_AFTER_RETRY' if new['verdict'] == 'PASS' else 'FAIL'
            merged[key] = new
        sources.append(root)
    out = Path(a.output).resolve()
    # 补测可分批合并；未补齐保持不完整，不伪造剩余行。
    manifest = copy.deepcopy(manifest)
    manifest.update(run_id=out.name, evidence_sources=[dict(path=str(root),
        manifest_sha256=evidence.digest(root / 'manifest.json'), results_sha256=evidence.digest(root / 'results.jsonl'))
        for root in sources])
    out.mkdir(parents=True, exist_ok=False)
    evidence.write_json(out / 'manifest.json', manifest)
    with (out / 'results.jsonl').open('x') as stream:
        for key in manifest['expected_cases']:
            if key in merged:
                stream.write(json.dumps(merged[key], ensure_ascii=False) + '\n')
    return dict(output=str(out), recorded=len(merged), expected=len(manifest['expected_cases']),
                recovered=sum(r.get('outcome') == 'PASS_AFTER_RETRY' for r in merged.values()),
                note='复跑恢复仅供诊断；check --require-pass / compare 仍拦截未解决失败历史')


def ready(a):
    selectors, artifact = set(), None
    for root in a.baseline:
        manifest, rows = evidence.validate(root, require_pass=True, require_role='baseline')
        if not rows:
            raise ValueError('基线为空')
        if artifact is not None and artifact != manifest['declared_artifact']:
            raise ValueError('两组基线使用了不同包')
        artifact = manifest['declared_artifact']
        selectors.add(manifest['environment'].get('HCCL_USE_NEW_SELECTOR', '0'))
    if not ({'0', '1'} if a.require_new_selector else {'0'}) <= selectors:
        raise ValueError('缺少所需 selector 基线')
    return dict(ready=True, note='基线完整且通过，可以按已有授权切包；此工具不执行安装')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('plan')
    r.add_argument('--run', required=True)
    r.add_argument('--output', required=True)
    for flag in ('cann', 'install-dir', 'test-bin-dir', 'artifact'):
        r.add_argument('--' + flag)
    m = sub.add_parser('merge')
    m.add_argument('--base', required=True)
    m.add_argument('--rerun', action='append', required=True)
    m.add_argument('--output', required=True)
    gate = sub.add_parser('ready')
    gate.add_argument('--baseline', action='append', required=True)
    gate.add_argument('--require-new-selector', action='store_true')
    a = p.parse_args()
    try:
        result = {'plan': retry_plan, 'merge': supplement, 'ready': ready}[a.command](a)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
