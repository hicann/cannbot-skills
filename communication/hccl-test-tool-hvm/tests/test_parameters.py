# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parameters import validate_dtypes
import plan


class ParameterTests(unittest.TestCase):
    def test_spelling_rejected_before_launch(self):
        validate_dtypes(['int32', 'bfp16'])
        with self.assertRaisesRegex(ValueError, 'bfp16'):
            validate_dtypes(['bf16'])

    def test_installed_source_is_authoritative(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'common/src/hccl_test_common.cc'
            source.parent.mkdir(parents=True)
            source.write_text('const char* test_typenames[] = {"int32", "newtype"};')
            validate_dtypes(['newtype'], root / 'bin')
            with self.assertRaises(ValueError):
                validate_dtypes(['fp32'], root / 'bin')
            source.write_text('unrecognized format')
            with self.assertRaisesRegex(ValueError, '格式变化'):
                validate_dtypes(['int32'], root / 'bin')

    def test_plan_rejects_invalid_dtype_and_preserves_iterations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch = dict(id='case', role='ordinary', covers=['smoke'], modes=['AI_CPU'],
                         ops=['reduce_scatter'], dtypes=['bf16'], sizes=[64], comms=['112'],
                         iterations=3, warmup=2, runner=True)
            doc = dict(schema_version=1, install_dir='install', cann='env', cluster='cluster', batches=[batch])
            path = root / 'plan.json'
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'dtype'):
                plan.read_plan(path)
            batch['dtypes'] = ['bfp16']
            path.write_text(json.dumps(doc))
            base, parsed = plan.read_plan(path)
            args = plan.command(base, parsed, batch, root / 'out')
            self.assertEqual(args[args.index('-n') + 1], '3')
            self.assertEqual(args[args.index('-w') + 1], '2')
            self.assertIn('--runner', args)


if __name__ == '__main__':
    unittest.main()
