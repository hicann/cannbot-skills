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

"""Checker 证据记录、完整性检查与基线比较；不启动测试或构建。"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

ROLES = ('ordinary', 'baseline', 'directed', 'regression')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def case_id(mode, op, dtype, size, comm):
    return '|'.join([mode, op, dtype, str(size), comm])


def check_identity(manifest):
    artifact = manifest.get('declared_artifact')
    if not isinstance(artifact, dict):
        raise ValueError('缺少归档安装包身份')
    for key in ('source_id', 'environment_id', 'package_sha256', 'build_log_sha256', 'build_command', 'package'):
        if not artifact.get(key):
            raise ValueError('归档缺少身份字段: ' + key)
    if artifact.get('schema_version') != 1 or artifact.get('build_exit_code') != 0:
        raise ValueError('归档 schema 或构建退出码无效')
    if not manifest.get('source_id') or manifest['source_id'] != artifact['source_id']:
        raise ValueError('源码身份与归档不一致')
    expected = artifact.get('expected_libraries', {})
    actual = {p['kind']: p['sha256'] for p in manifest['installed_libraries']}
    if not expected.get('host') or not expected.get('device') or expected != actual:
        raise ValueError('已安装 host/device 库缺失或与归档构建库哈希不一致')


def initialize(a):
    root = Path(a.run_dir)
    libs = [Path(a.cann_home) / 'lib64/libhccl.so',
            Path(a.install_dir) / 'lib/aarch64/libscatter_aicpu_kernel.so']
    artifact = load(a.artifact) if a.artifact else None
    role = getattr(a, 'role', 'ordinary')
    if role not in ROLES:
        raise ValueError('未知测试角色')
    if a.phase.startswith(('baseline', 'regression')) and role == 'ordinary':
        raise ValueError('phase 只是标签；基线/回归必须显式设置 --role')
    if role in ('baseline', 'regression') and not artifact:
        raise ValueError('基线/回归运行须先提供 --artifact，避免测试后才发现缺少包身份')
    if role == 'directed' and not getattr(a, 'expected', ''):
        raise ValueError('定向测试须提供 --expect-algo')
    if artifact:
        package = Path(a.artifact).resolve().parent / artifact['package']
        if digest(package) != artifact['package_sha256']:
            raise ValueError('归档安装包哈希不匹配')
    plan = [
        case_id(*c)
        for c in itertools.product(
            a.modes.split(','), a.ops.split(','), a.dtypes.split(','),
            a.sizes.split(','), a.comms.split(',')
        )
    ]
    if len(set(plan)) != len(plan):
        raise ValueError('测试计划包含重复用例')
    env = {k: v for k, v in os.environ.items() if k.startswith('HCCL_') or k == 'HWLOC_COMPONENTS'}
    env.pop('HCCL_VM_INSTALL_DIR', None)
    # 这两个变量由 run.sh 在每个 case 设置，不记录调用者可能残留的值。
    env.pop('HCCL_OP_EXPANSION_MODE', None)
    env.pop('HCCL_ENABLE_OPEN_CCU', None)
    env['HCCL_USE_NEW_SELECTOR'] = env.get('HCCL_USE_NEW_SELECTOR', '0')
    env['HCCL_ALGO'] = env.get('HCCL_ALGO', '')
    if role == 'directed' and not env['HCCL_ALGO']:
        raise ValueError('定向测试须提供非空 HCCL_ALGO 配置')
    if role in ('baseline', 'regression') and env['HCCL_ALGO']:
        raise ValueError('基线/回归不得设置 HCCL_ALGO')
    source_id = a.source_id or (artifact['source_id'] if artifact else '')
    if artifact and source_id != artifact['source_id']:
        raise ValueError('source-id 与归档包不一致')
    manifest = dict(schema_version=1, run_id=root.name, role=role, phase=a.phase, source_id=source_id,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    declared_artifact=artifact, expected_cases=plan, environment=env,
                    cluster=a.cluster, iterations=a.iterations, warmup=a.warmup,
                    runner=a.runner, dry_run=a.dry_run,
                    replay=dict(install_dir=str(Path(a.install_dir).resolve()),
                                cann=getattr(a, 'cann_script', ''),
                                test_bin_dir=getattr(a, 'test_bin_dir', ''),
                                artifact=str(Path(a.artifact).resolve()) if a.artifact else '',
                                expected_algorithm=getattr(a, 'expected', '')),
                    installed_libraries=[dict(kind=kind, path=str(p.resolve()),
                                               sha256=digest(p) if p.is_file() else None)
                                         for kind, p in zip(('host', 'device'), libs)])
    if artifact:
        check_identity(manifest)  # 在昂贵测试前检查，避免结束后才发现包身份不完整。
    write_json(root / 'manifest.json', manifest)
    (root / 'results.jsonl').touch(exist_ok=False)


def parse_log(text, expected='', timed_out=False):
    text = re.sub(r'\x1b\[[0-9;]*m', '', text)
    algorithms = re.findall(r'the selected algo type is\s+([A-Za-z][A-Za-z0-9]*)', text)
    configs = re.findall(r'\[FilterCmByHcclAlgo\] use algo config: \[(.*?)\]', text)
    exits = [int(x) for x in re.findall(r'\[CMD DONE\] mpirun \(exit=(\d+)\)', text)]
    summaries = re.findall(r'\[CHECKER_RUN_SUMMARY\]\s*([^\n]+)', text)
    success = bool(summaries) and all(re.search(
        r'All Success\s*\(Total Op:\s*[1-9]\d*,\s*Big Graph:\s*[1-9]\d*\)', s) for s in summaries)
    reasons = []
    if timed_out:
        reasons.append('timeout')
    if len(exits) != 1 or exits[0] != 0:
        reasons.append('mpirun 未完整结束或退出码非零（单会话须一次 mpirun）')
    if not success:
        reasons.append('缺少完整 Checker 全成功汇总或存在失败汇总')
    if not algorithms:
        reasons.append('缺少选中算法证据')
    if expected and set(algorithms) != {expected}:
        reasons.append('算法命中不一致')
    failure_stage = ('timeout' if timed_out else
                     'initialization' if 'InitHvmCommEnv failed' in text and not algorithms else
                     'selection' if not algorithms or (expected and set(algorithms) != {expected}) else
                     'execution_or_checker' if reasons else None)
    return dict(verdict='FAIL' if reasons else 'PASS', failure_stage=failure_stage,
                detail='; '.join(reasons) if reasons else 'Checker Success',
                algo_first=algorithms[0] if algorithms else None,
                algo_last=algorithms[-1] if algorithms else None,
                observed_algorithms=sorted(set(algorithms)), effective_algo_configs=sorted(set(configs)),
                algorithm_counts=dict(Counter(algorithms)),
                mpirun_exit_codes=exits, checker_summaries=summaries,
                single_op_checker='success' if success else 'unconfirmed',
                big_graph_checker='success' if success else 'unconfirmed', timed_out=timed_out)


def record(a):
    root = Path(a.run_dir)
    manifest = load(root / 'manifest.json')
    cid = case_id(a.mode, a.op, a.dtype, a.size, a.comm)
    if cid not in manifest['expected_cases']:
        raise ValueError('用例不在预先记录的计划内')
    log = Path(a.log).resolve()
    result = parse_log(log.read_text(errors='replace'), a.expected, a.timed_out)
    result.update(case_id=cid, run_id=manifest['run_id'], duration_seconds=a.duration,
                  case_environment={k: os.environ.get(k, '') for k in
                                    ('HCCL_OP_EXPANSION_MODE', 'HCCL_ENABLE_OPEN_CCU')},
                  expected_algorithm=a.expected, raw_log=str(log), raw_log_sha256=digest(log),
                  test_binary_sha256=digest(a.test_binary) if Path(a.test_binary).is_file() else None)
    with (root / 'results.jsonl').open('a') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(result['verdict'] + '\t' + result['detail'])


def validate(root, require_pass=False, require_role=None):
    root = Path(root)
    manifest = load(root / 'manifest.json')
    rows = [json.loads(line) for line in (root / 'results.jsonl').read_text().splitlines() if line.strip()]
    issues = []
    if manifest['dry_run']:
        raise ValueError('dry-run 不能作为基线')
    if require_role and manifest.get('role') != require_role:
        raise ValueError('测试角色不匹配或旧记录缺少 role: 需要 ' + require_role)
    check_identity(manifest)
    for source in manifest.get('evidence_sources', []):
        origin = Path(source['path'])
        if (digest(origin / 'manifest.json') != source['manifest_sha256'] or
                digest(origin / 'results.jsonl') != source['results_sha256']):
            raise ValueError('合并来源证据已变化')
    if manifest.get('role') in ('baseline', 'regression'):
        if manifest['environment'].get('HCCL_ALGO') or any(r['effective_algo_configs'] for r in rows):
            raise ValueError('默认基线/回归存在 HCCL_ALGO 或通信域算法覆盖，不能使用这批证据')
    keys = [r['case_id'] for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(manifest['expected_cases']):
        issues.append('用例重复、遗漏或不在计划内')
    for r in rows:
        p = Path(r['raw_log'])
        if not p.is_file() or digest(p) != r['raw_log_sha256']:
            issues.append(r['case_id'] + ': 原始日志丢失或被修改')
        if not r['observed_algorithms'] or not r['test_binary_sha256']:
            issues.append(r['case_id'] + ': 缺算法名或测试二进制身份')
        if not r['checker_summaries'] or len(r['mpirun_exit_codes']) != 1 or r['timed_out']:
            issues.append(r['case_id'] + ': 测试未完整结束，不能用作已完成基线')
        if require_pass and r['verdict'] != 'PASS':
            issues.append(r['case_id'] + ': Checker 未通过')
        if r.get('prior_attempts'):
            for prior in r['prior_attempts']:
                p = Path(prior['raw_log'])
                if not p.is_file() or digest(p) != prior['raw_log_sha256']:
                    issues.append(r['case_id'] + ': 历史尝试日志缺失或被修改')
            if require_pass and any(p['verdict'] != 'PASS' for p in r['prior_attempts']):
                issues.append(r['case_id'] + ': 复跑恢复仍有未解决失败历史，不能自动豁免')
    if issues:
        raise ValueError('\n'.join(issues))
    return manifest, {r['case_id']: r for r in rows}


def compare(base, candidate):
    bm, br = validate(base, require_role='baseline')
    cm, cr = validate(candidate, require_role='regression')
    issues = []
    if bm['declared_artifact'].get('environment_id') != cm['declared_artifact'].get('environment_id'):
        issues.append('归档包的构建环境身份不同')
    for key in ('environment', 'cluster', 'iterations', 'warmup', 'runner'):
        if bm[key] != cm[key]:
            issues.append('测试配置不同: ' + key)
    if set(br) != set(cr):
        issues.append('基线与回归用例集合不同')
    for cid in br.keys() & cr.keys():
        b, c = br[cid], cr[cid]
        if b['observed_algorithms'] != c['observed_algorithms']:
            issues.append(cid + ': 默认选路变化')
        if b['algorithm_counts'] != c['algorithm_counts']:
            issues.append(cid + ': 算法调用证据数量不同')
        if b['case_environment'] != c['case_environment']:
            issues.append(cid + ': 用例执行环境不同')
        if b['test_binary_sha256'] != c['test_binary_sha256']:
            issues.append(cid + ': 测试程序变化，需要匹配的基线')
        if b['verdict'] != 'PASS' or c['verdict'] != 'PASS':
            issues.append(cid + ': 存在 FAIL，不能以相同失败豁免门禁')
        if any(p['verdict'] != 'PASS' for r in (b, c) for p in r.get('prior_attempts', [])):
            issues.append(cid + ': 存在复跑前失败，不能自动豁免门禁')
        if b['effective_algo_configs'] or c['effective_algo_configs']:
            issues.append(cid + ': 默认回归存在通信域算法覆盖')
    if bm['environment'].get('HCCL_ALGO') or cm['environment'].get('HCCL_ALGO'):
        issues.append('默认回归不能设置 HCCL_ALGO')
    if issues:
        raise ValueError('\n'.join(issues))
    print('PASS: 默认选路一致，双侧 Checker 全通过，共 %d 例' % len(br))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('init')
    for flag in ('run-dir', 'install-dir', 'cann-home', 'modes', 'ops', 'dtypes', 'sizes', 'comms', 'cluster'):
        i.add_argument('--' + flag, required=True)
    for flag in ('phase', 'source-id', 'artifact'):
        i.add_argument('--' + flag, default='')
    i.add_argument('--role', choices=ROLES, default='ordinary')
    i.add_argument('--expected', default='')
    i.add_argument('--cann-script', default='')
    i.add_argument('--test-bin-dir', default='')
    for flag in ('iterations', 'warmup', 'runner', 'dry-run'):
        i.add_argument('--' + flag, type=int, default=0)
    r = sub.add_parser('record')
    for flag in ('run-dir', 'mode', 'op', 'dtype', 'size', 'comm', 'log', 'test-binary'):
        r.add_argument('--' + flag, required=True)
    r.add_argument('--duration', type=int, required=True)
    r.add_argument('--expected', default='')
    r.add_argument('--timed-out', action='store_true')
    c = sub.add_parser('check')
    c.add_argument('run_dir')
    c.add_argument('--require-pass', action='store_true')
    c.add_argument('--require-role', choices=ROLES)
    c = sub.add_parser('compare')
    c.add_argument('baseline')
    c.add_argument('candidate')
    a = p.parse_args()
    try:
        if a.command == 'init':
            initialize(a)
        elif a.command == 'record':
            record(a)
        elif a.command == 'compare':
            compare(a.baseline, a.candidate)
        else:
            _, rows = validate(a.run_dir, a.require_pass, a.require_role)
            print('证据完整: %d 例，FAIL=%d（完整不等于通过）' %
                  (len(rows), sum(r['verdict'] != 'PASS' for r in rows.values())))
    except (ValueError, OSError, KeyError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
