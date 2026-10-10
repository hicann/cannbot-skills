# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""离线工具回归；不执行 HCCL ST/UT、构建或 Checker。"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

VM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VM))
import evidence

spec = importlib.util.spec_from_file_location(
    'archive_package', VM.parent / 'hccl-aicpu-best-practice/scripts/archive_package.py')
archive_package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_package)

GOOD = ('the selected algo type is AicpuAllGatherSoleMesh\n'
        '[CMD DONE] mpirun (exit=0)\n'
        '[CHECKER_RUN_SUMMARY] All Success (Total Op: 2, Big Graph: 1)\n')


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_evidence(self, name, text=GOOD, role="baseline"):
        root = self.root / name
        root.mkdir()
        log = root / 'case.log'
        log.write_text(text)
        row = evidence.parse_log(text)
        row.update(case_id='AI_CPU|allgather|int32|64|112', raw_log=str(log),
                   raw_log_sha256=evidence.digest(log), test_binary_sha256='test-bin',
                   case_environment={'HCCL_OP_EXPANSION_MODE': 'AI_CPU', 'HCCL_ENABLE_OPEN_CCU': ''})
        manifest = dict(role=role, dry_run=0, source_id='commit+diff',
                        declared_artifact=dict(schema_version=1, package_sha256=name,
                                               source_id='commit+diff', environment_id='test-env',
                                               build_log_sha256='log', build_command='build',
                                               package='pkg.run', build_exit_code=0,
                                               expected_libraries={'host': 'host', 'device': 'device'}),
                        installed_libraries=[{'kind': 'host', 'sha256': 'host'},
                                             {'kind': 'device', 'sha256': 'device'}],
                        expected_cases=[row['case_id']], environment={'HCCL_ALGO': ''},
                        cluster='cluster', iterations=1, warmup=0, runner=0)
        evidence.write_json(root / 'manifest.json', manifest)
        (root / 'results.jsonl').write_text(json.dumps(row) + '\n')
        return root

    def test_no_false_pass_from_earlier_summary_or_partial_graph(self):
        for text in (GOOD + '[CHECKER_RUN_SUMMARY] Failed (Total Op: 2, Big Graph: 1)\n',
                     GOOD.replace('Big Graph: 1', 'Big Graph: 0'),
                     GOOD.replace('exit=0', 'exit=1'),
                     GOOD + '[CMD DONE] mpirun (exit=0)\n'):
            self.assertEqual(evidence.parse_log(text)['verdict'], 'FAIL')

    def test_middle_algorithm_mismatch_is_not_hidden(self):
        text = GOOD + 'the selected algo type is Wrong\nthe selected algo type is AicpuAllGatherSoleMesh\n'
        self.assertEqual(evidence.parse_log(text, 'AicpuAllGatherSoleMesh')['verdict'], 'FAIL')

    def test_interleaved_process_prefix_is_not_part_of_algorithm(self):
        algorithm = 'AicpuScatterParallelRingRing'
        text = GOOD.replace('AicpuAllGatherSoleMesh', algorithm + '[info][PID:59631][TID:123]')
        result = evidence.parse_log(text, algorithm)
        self.assertEqual(result['verdict'], 'PASS')
        self.assertEqual(result['observed_algorithms'], [algorithm])
        self.assertEqual(result['algorithm_counts'], {algorithm: 1})

        truncated = text.replace(algorithm + '[info]', 'AicpuScatterParallelRingRin[info]')
        result = evidence.parse_log(truncated, algorithm)
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertEqual(result['observed_algorithms'], ['AicpuScatterParallelRingRin'])

    def test_incomplete_or_overwritten_baseline_is_rejected(self):
        root = self.run_evidence('base')
        evidence.validate(root)
        (root / 'case.log').write_text('overwritten')
        with self.assertRaisesRegex(ValueError, '日志丢失或被修改'):
            evidence.validate(root)
        root = self.run_evidence('missing')
        (root / 'results.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError, '遗漏'):
            evidence.validate(root)

    def test_compare_uses_case_keys_and_rejects_same_failure(self):
        b, c = self.run_evidence('b'), self.run_evidence('c', role='regression')
        evidence.compare(b, c)
        bad = GOOD.replace('All Success', 'Failed')
        b, c = self.run_evidence('bad-b', bad), self.run_evidence('bad-c', bad, role='regression')
        evidence.validate(b)  # 完整失败证据可以记录，但不能豁免通过要求。
        with self.assertRaisesRegex(ValueError, '不能以相同失败'):
            evidence.compare(b, c)

    def test_comm_override_and_environment_drift_rejected(self):
        b = self.run_evidence('base')
        c = self.run_evidence(
            'candidate',
            GOOD + '[FilterCmByHcclAlgo] use algo config: [allgather:sole{mesh}]\n',
            role='regression')
        with self.assertRaisesRegex(ValueError, '算法覆盖'):
            evidence.compare(b, c)
        c = self.run_evidence('config-drift', role='regression')
        m = evidence.load(c / 'manifest.json')
        m['environment']['HCCL_USE_NEW_SELECTOR'] = '1'
        evidence.write_json(c / 'manifest.json', m)
        with self.assertRaisesRegex(ValueError, '测试配置不同'):
            evidence.compare(b, c)

    def test_package_archive_survives_clean_and_refuses_overwrite(self):
        build = self.root / 'build_out'
        build.mkdir()
        pkg, log, rc = build / 'hccl.run', build / 'build.log', build / 'rc'
        pkg.write_bytes(b'package-bytes')
        log.write_text('build completed')
        rc.write_text('0')
        a = argparse.Namespace(package=str(pkg), build_log=str(log), exit_code_file=str(rc),
                               source_id='commit+diff', environment_id='cann-test',
                               build_command='build-command', output=str(self.root / 'archive'),
                               repo=str(self.root), clean_root=[], host_library=str(pkg), device_library=str(pkg))
        a.output = str(self.root / 'build/should-not-survive')
        with self.assertRaisesRegex(ValueError, '清理目录'):
            archive_package.archive(a)
        a.output = str(self.root / 'archive')
        path = archive_package.archive(a)
        with self.assertRaises(FileExistsError):
            archive_package.archive(a)
        pkg.unlink()
        m = evidence.load(path)
        self.assertEqual(evidence.digest(path.parent / m['package']), m['package_sha256'])

    def test_archive_identity_mismatch_rejected(self):
        root = self.run_evidence('wrong-installed')
        manifest = evidence.load(root / 'manifest.json')
        manifest['installed_libraries'][0]['sha256'] = 'different-host'
        evidence.write_json(root / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, '库缺失或与归档'):
            evidence.validate(root)

    def test_init_record_check_roundtrip(self):
        root = self.root / 'roundtrip'
        root.mkdir()
        cann, install = self.root / 'cann', self.root / 'install'
        host, device = cann / 'lib64/libhccl.so', install / 'lib/aarch64/libscatter_aicpu_kernel.so'
        for library in (host, device):
            library.parent.mkdir(parents=True)
            library.write_bytes(library.name.encode())
        package = self.root / 'package.run'
        package.write_bytes(b'archive')
        artifact = self.root / 'artifact.json'
        evidence.write_json(artifact, dict(schema_version=1, source_id='commit+snapshot',
            environment_id='cann-test', package=package.name, package_sha256=evidence.digest(package),
            build_log_sha256='build-log', build_command='build', build_exit_code=0,
            expected_libraries={'host': evidence.digest(host), 'device': evidence.digest(device)}))
        init = argparse.Namespace(run_dir=str(root), cann_home=str(cann), install_dir=str(install),
            artifact=str(artifact), role='baseline', phase='baseline', source_id='', modes='AI_CPU', ops='allgather',
            dtypes='int32', sizes='64', comms='112', cluster='cluster', iterations=1, warmup=0,
            runner=0, dry_run=0)
        evidence.initialize(init)
        log = root / 'case.log'
        log.write_text(GOOD)
        evidence.record(argparse.Namespace(run_dir=str(root), mode='AI_CPU', op='allgather',
            dtype='int32', size='64', comm='112', log=str(log), expected='', timed_out=False,
            duration=1, test_binary=str(host)))
        evidence.validate(root, require_pass=True)
        other = self.root / 'mismatch'
        other.mkdir()
        init.run_dir = str(other)
        host.write_bytes(b'new-version-with-old-artifact')
        with self.assertRaisesRegex(ValueError, '库缺失或与归档'):
            evidence.initialize(init)

    def test_failed_build_cannot_be_archived(self):
        rc = self.root / 'rc'
        rc.write_text('1')
        a = argparse.Namespace(package=str(self.root / 'absent.run'), build_log=str(self.root / 'absent.log'),
                               exit_code_file=str(rc), output=str(self.root / 'out'))
        with self.assertRaisesRegex(ValueError, '退出码非零'):
            archive_package.archive(a)

    def test_dry_runs_get_unique_directories_and_preserve_metadata(self):
        install = self.root / 'install'
        (install / 'bin').mkdir(parents=True)
        env = self.root / 'set_env.sh'
        env.write_text('export ASCEND_HOME_PATH=' + str(self.root) + '\n')
        command = ['bash', str(VM / 'run.sh'), '--dry-run', '--install-dir', str(install),
                   '--cann', str(env), '--log-dir', str(self.root / 'logs'), '-m', 'AI_CPU',
                   '-o', 'allgather', '--env', 'HCCL_USE_NEW_SELECTOR=1']
        bad = subprocess.run(command + ['-o', 'typo'], capture_output=True)
        self.assertNotEqual(bad.returncode, 0)
        for _ in range(2):
            subprocess.run(command, check=True, capture_output=True)
        runs = list((self.root / 'logs').glob('run_*'))
        self.assertEqual(len(runs), 2)
        m = evidence.load(runs[0] / 'manifest.json')
        self.assertEqual(m['environment']['HWLOC_COMPONENTS'], '-gl,-opencl')
        self.assertEqual(m['environment']['HCCL_USE_NEW_SELECTOR'], '1')
        with self.assertRaisesRegex(ValueError, 'dry-run'):
            evidence.validate(runs[0])

    def test_custom_phase_cannot_turn_ordinary_record_into_baseline(self):
        b, c = self.run_evidence('old-label', role='ordinary'), self.run_evidence('new-label', role='regression')
        with self.assertRaisesRegex(ValueError, '角色不匹配'):
            evidence.compare(b, c)
        a = argparse.Namespace(run_dir=str(self.root), cann_home=str(self.root), install_dir=str(self.root),
                               artifact='', role='baseline', phase='preinstall-default')
        with self.assertRaisesRegex(ValueError, '--artifact'):
            evidence.initialize(a)
        a.role = 'ordinary'
        a.phase = 'baseline-default'
        with self.assertRaisesRegex(ValueError, '--role'):
            evidence.initialize(a)
        a.role, a.phase = 'directed', 'smoke'
        with self.assertRaisesRegex(ValueError, '--expect-algo'):
            evidence.initialize(a)

    def test_failure_stage_does_not_claim_environment_root_cause(self):
        result = evidence.parse_log('InitHvmCommEnv failed\n')
        self.assertEqual(result['failure_stage'], 'initialization')
        self.assertEqual(result['verdict'], 'FAIL')
        result = evidence.parse_log(GOOD.replace('All Success', 'Failed'))
        self.assertEqual(result['failure_stage'], 'execution_or_checker')


if __name__ == '__main__':
    unittest.main()
