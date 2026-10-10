# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Repair evidence offline: no builds, installs, MPI, or Checker invocation."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evidence
import repair

GOOD = ('the selected algo type is AicpuReduceScatterSoleNHR\n'
        '[CMD DONE] mpirun (exit=0)\n'
        '[CHECKER_RUN_SUMMARY] All Success (Total Op: 2, Big Graph: 1)\n')
BAD = GOOD.replace('exit=0', 'exit=1')
KEYS = ['AI_CPU|reduce_scatter|int32|%s|112' % size for size in (32, 64, 128)]


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.artifact = dict(schema_version=1, source_id='commit+snapshot', package_sha256='package',
            environment_id='test-env', build_log_sha256='log', build_command='build', package='pkg.run',
            build_exit_code=0, expected_libraries={'host': 'host', 'device': 'device'})
        self.artifact_path = self.root / 'artifact.json'
        evidence.write_json(self.artifact_path, self.artifact)

    def run_record(self, name, expected, results, role='baseline'):
        root = self.root / name
        root.mkdir()
        manifest = dict(role=role, dry_run=0, source_id='commit+snapshot', declared_artifact=self.artifact,
            installed_libraries=[dict(kind='host', sha256='host'), dict(kind='device', sha256='device')],
            expected_cases=expected, environment={'HCCL_ALGO': '', 'HCCL_USE_NEW_SELECTOR': '0',
            'HWLOC_COMPONENTS': '-gl,-opencl'}, cluster='cluster', iterations=1, warmup=0, runner=0,
            replay=dict(cann=str(self.root / 'set_env.sh'), install_dir=str(self.root / 'install'),
                        test_bin_dir=str(self.root / 'bin'), artifact=str(self.artifact_path), expected_algorithm=''))
        evidence.write_json(root / 'manifest.json', manifest)
        rows = []
        for index, (key, text) in enumerate(results):
            log = root / ('case-%d.log' % index)
            log.write_text(text)
            row = evidence.parse_log(text)
            row.update(case_id=key, raw_log=str(log), raw_log_sha256=evidence.digest(log),
                       test_binary_sha256='test-bin', expected_algorithm='',
                       case_environment={'HCCL_OP_EXPANSION_MODE': 'AI_CPU', 'HCCL_ENABLE_OPEN_CCU': ''})
            rows.append(row)
        (root / 'results.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return root

    def merge(self, base, reruns, name='merged'):
        output = self.root / name
        repair.supplement(argparse.Namespace(base=str(base), rerun=list(map(str, reruns)), output=str(output)))
        return output

    def plan(self, run, name='plan.json'):
        output = self.root / name
        result = repair.retry_plan(argparse.Namespace(run=str(run), output=str(output), cann=None,
            install_dir=None, test_bin_dir=None, artifact=None))
        return result, evidence.load(output)

    def test_retry_plan_contains_only_failed_and_missing(self):
        base = self.run_record('base', KEYS, [(KEYS[0], GOOD), (KEYS[1], BAD)])
        result, plan = self.plan(base)
        self.assertEqual(result['cases'], 2)
        self.assertEqual([batch['sizes'] for batch in plan['batches']], [[64], [128]])
        self.assertEqual(plan['repair_origin']['results_sha256'], evidence.digest(base / 'results.jsonl'))
        self.assertTrue(all(batch['artifact'] == str(self.artifact_path) for batch in plan['batches']))
        self.assertTrue(all(batch['env']['HCCL_USE_NEW_SELECTOR'] == '0' for batch in plan['batches']))

    def test_all_pass_plan_is_refused(self):
        base = self.run_record('base', [KEYS[0]], [(KEYS[0], GOOD)])
        with self.assertRaisesRegex(ValueError, '没有失败或缺失'):
            self.plan(base)

    def test_merge_missing_preserves_complete_matrix_and_original_logs(self):
        base = self.run_record('base', KEYS, [(KEYS[0], GOOD)])
        extra = self.run_record('extra', KEYS[1:], [(key, GOOD) for key in KEYS[1:]])
        merged = self.merge(base, [extra])
        manifest, rows = evidence.validate(merged, require_pass=True, require_role='baseline')
        self.assertEqual(set(rows), set(KEYS))
        self.assertEqual(len(manifest['evidence_sources']), 2)
        self.assertEqual(Path(rows[KEYS[0]]['raw_log']).parent, base)
        self.assertFalse(any(row.get('prior_attempts') for row in rows.values()))
        self.assertTrue(repair.ready(argparse.Namespace(baseline=[str(merged)], require_new_selector=False))['ready'])

    def test_fail_then_pass_history_never_becomes_strict_pass(self):
        base = self.run_record('base', [KEYS[0]], [(KEYS[0], BAD)])
        extra = self.run_record('extra', [KEYS[0]], [(KEYS[0], GOOD)])
        merged = self.merge(base, [extra])
        _, rows = evidence.validate(merged)
        row = rows[KEYS[0]]
        self.assertEqual(row['verdict'], 'PASS')
        self.assertEqual(row['outcome'], 'PASS_AFTER_RETRY')
        self.assertEqual(row['prior_attempts'][0]['verdict'], 'FAIL')
        with self.assertRaises(ValueError):
            evidence.validate(merged, require_pass=True)
        candidate = self.run_record('candidate', [KEYS[0]], [(KEYS[0], GOOD)], role='regression')
        with self.assertRaises(ValueError):
            evidence.compare(merged, candidate)
        with self.assertRaises(ValueError):
            repair.ready(argparse.Namespace(baseline=[str(merged)], require_new_selector=False))

    def test_config_change_is_rejected(self):
        base = self.run_record('base', KEYS[:2], [(KEYS[0], GOOD)])
        extra = self.run_record('extra', [KEYS[1]], [(KEYS[1], GOOD)])
        manifest = evidence.load(extra / 'manifest.json')
        manifest['environment']['HCCL_USE_NEW_SELECTOR'] = '1'
        evidence.write_json(extra / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, '配置不同'):
            self.merge(base, [extra])

    def test_second_retry_and_replacing_pass_are_rejected(self):
        base = self.run_record('base', [KEYS[0]], [(KEYS[0], BAD)])
        extra = self.run_record('extra', [KEYS[0]], [(KEYS[0], BAD)])
        merged = self.merge(base, [extra])
        with self.assertRaisesRegex(ValueError, '已有补测历史'):
            self.plan(merged)
        with self.assertRaisesRegex(ValueError, '重复追加'):
            self.merge(merged, [extra], 'second')
        passed = self.run_record('passed', [KEYS[0]], [(KEYS[0], GOOD)])
        with self.assertRaisesRegex(ValueError, '已通过'):
            self.merge(passed, [extra], 'replace-pass')

    def test_original_and_historical_log_tampering_are_rejected(self):
        base = self.run_record('base', [KEYS[0]], [(KEYS[0], BAD)])
        extra = self.run_record('extra', [KEYS[0]], [(KEYS[0], GOOD)])
        merged = self.merge(base, [extra])
        (base / 'case-0.log').write_text(GOOD)
        with self.assertRaisesRegex(ValueError, '日志被修改'):
            self.plan(base)
        with self.assertRaisesRegex(ValueError, '日志被修改'):
            repair.read_run(merged)

    def test_duplicate_rerun_in_single_merge_is_rejected(self):
        base = self.run_record('base', KEYS[:2], [(KEYS[0], GOOD)])
        extra = self.run_record('extra', [KEYS[1]], [(KEYS[1], GOOD)])
        with self.assertRaisesRegex(ValueError, '已通过'):
            self.merge(base, [extra, extra])


if __name__ == '__main__':
    unittest.main()
