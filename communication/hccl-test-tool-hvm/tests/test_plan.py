# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""计划编排离线验证：仅使用 dry-run 或模拟进程，不启动 Checker。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence
import plan


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'install/bin').mkdir(parents=True)
        (self.root / 'env.sh').write_text('export ASCEND_HOME_PATH=' + str(self.root) + '\n')
        self.batch = dict(id='sole', role='ordinary', covers=['单级冒烟'], modes=['AI_CPU'],
                          ops=['reduce_scatter'], dtypes=['int32'], sizes=[64], comms=['112'])
        self.doc = dict(schema_version=1, install_dir='install', cann='env.sh', cluster='cluster.yaml',
                        batches=[self.batch])
        self.path = self.root / 'plan.json'

    def save(self):
        evidence.write_json(self.path, self.doc)

    def args(self, dry_run=False):
        return argparse.Namespace(plan=str(self.path), role='ordinary', output=str(self.root / 'out'), dry_run=dry_run)

    def test_role_and_directed_requirements_precede_execution(self):
        for role in ('preinstall', 'baseline', 'regression', 'directed'):
            self.batch['role'] = role
            self.save()
            with self.assertRaises(ValueError):
                plan.read_plan(self.path)
        self.batch.update(role='directed', expect_algo='RegisteredName', env={'HCCL_ALGO': 'verified-dsl'})
        self.save()
        plan.read_plan(self.path)

    def test_dry_run_records_no_pass_and_clears_inherited_algo(self):
        self.save()
        with patch.dict(os.environ, {'HCCL_ALGO': 'stale', 'HCCL_BUFFSIZE': '512'}):
            self.assertEqual(plan.execute(self.args(True)), 0)
        result = evidence.load(self.root / 'out/summary.json')
        self.assertEqual(result['status'], 'DRY_RUN')
        self.assertEqual(result['batches'][0]['executed_cases'], 0)
        self.assertEqual(result['totals']['dry_run_batches'], 1)
        self.assertEqual(result['totals']['executed_batches'], 0)
        self.assertEqual(result['totals']['executed_cases'], 0)
        manifest = evidence.load(next((self.root / 'out/sole').glob('run_*/manifest.json')))
        self.assertEqual(manifest['environment']['HCCL_ALGO'], '')
        self.assertNotIn('HCCL_BUFFSIZE', manifest['environment'])
        self.assertEqual(manifest['environment']['HWLOC_COMPONENTS'], '-gl,-opencl')
        with self.assertRaises(FileExistsError):
            plan.execute(self.args(True))

    def test_stops_after_failed_batch_and_keeps_pending_count(self):
        self.doc['batches'].append(dict(self.batch, id='second'))
        self.save()
        with patch.object(plan.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)) as run:
            self.assertEqual(plan.execute(self.args()), 1)
            self.assertEqual(run.call_count, 1)
        result = evidence.load(self.root / 'out/summary.json')
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['pending_batches'], 1)
        self.assertEqual(result['totals']['attempted_batches'], 1)
        self.assertEqual(result['totals']['executed_cases'], 0)

    def test_totals_include_failures_but_not_pending_cases(self):
        self.batch['sizes'] = [64, 128]
        self.doc['batches'] += [dict(self.batch, id='second'), dict(self.batch, id='pending')]
        self.save()

        def execute(args, **kwargs):
            directory = Path(args[args.index('--log-dir') + 1])
            run = directory / 'run_fixture'
            run.mkdir()
            rows = [dict(verdict='PASS', duration_seconds=3),
                    dict(verdict='FAIL' if directory.name == 'second' else 'PASS', duration_seconds=5)]
            for i, row in enumerate(rows):
                log = run / ('case-%d.log' % i)
                log.write_text('synthetic runner result\n')
                row.update(case_id=str(i), raw_log=str(log), raw_log_sha256=evidence.digest(log))
            (run / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            return subprocess.CompletedProcess(args, 0)
        # Evidence validation has separate identity/case tests; simulate only the
        # runner here to exercise accumulation and stop-on-failure together.
        with patch.object(plan.subprocess, 'run', side_effect=execute), patch.object(plan, 'validate_batch'):
            self.assertEqual(plan.execute(self.args()), 1)
        result = evidence.load(self.root / 'out/summary.json')
        self.assertEqual(result['pending_batches'], 1)
        totals = result['totals']
        self.assertEqual((totals['executed_batches'], totals['executed_cases'], totals['passed'],
                          totals['failed'], totals['case_seconds']), (2, 4, 3, 1, 16))
        self.assertGreaterEqual(totals['wall_seconds'], totals['batch_wall_seconds'] - 0.002)

    def test_zero_exit_with_missing_cases_is_not_pass(self):
        self.save()
        with patch.object(plan.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)):
            self.assertEqual(plan.execute(self.args()), 1)

    def test_duplicate_case_cannot_replace_missing_case(self):
        self.batch['sizes'] = [64, 128]
        root = self.root / 'fake-run'
        root.mkdir()
        evidence.write_json(root / 'manifest.json', dict(role='ordinary', dry_run=False,
            expected_cases=['AI_CPU|reduce_scatter|int32|64|112', 'AI_CPU|reduce_scatter|int32|128|112']))
        rows = [dict(case_id='AI_CPU|reduce_scatter|int32|64|112')] * 2
        with self.assertRaisesRegex(ValueError, '重复'):
            plan.validate_batch(self.batch, [root], rows)

    def test_managed_environment_cannot_change_recorded_install_identity(self):
        for key in ('HCCL_VM_INSTALL_DIR', 'HCCL_OP_EXPANSION_MODE', 'HCCL_ENABLE_OPEN_CCU'):
            self.batch['env'] = {key: 'wrong'}
            self.save()
            with self.assertRaisesRegex(ValueError, '工具管理'):
                plan.read_plan(self.path)

    def test_loop_assertion_requires_trace_from_each_case(self):
        self.batch['assertions'] = [dict(name='outer_loop', basis='executor::Orchestrate loop trace',
                                         pattern=r'Executor loop\[(\d+)\]', minimum=1)]
        log = self.root / 'case.log'
        log.write_text('Executor loop[0]\nOther repeat[8]\n')
        row = dict(case_id='case', raw_log=str(log), raw_log_sha256=evidence.digest(log))
        self.assertFalse(plan.assertions_for(self.batch, [row])[0]['passed'])
        log.write_text('Executor loop[0]\nExecutor loop[1]\n')
        row['raw_log_sha256'] = evidence.digest(log)
        self.assertTrue(plan.assertions_for(self.batch, [row])[0]['passed'])
        log.write_text('overwritten')
        with self.assertRaisesRegex(ValueError, '日志被修改'):
            plan.assertions_for(self.batch, [row])

    def test_loop_size_uses_supplied_count_unit_and_alignment(self):
        cmd = [sys.executable, str(Path(plan.__file__)), 'loop-size', '--max-count-per-loop', '100',
               '--test-bytes-per-count', '32', '--alignment-count', '16']
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        values = json.loads(result.stdout)
        self.assertEqual(values['executor_count'], 112)
        self.assertEqual(values['test_bytes'], 3584)


if __name__ == '__main__':
    unittest.main()
