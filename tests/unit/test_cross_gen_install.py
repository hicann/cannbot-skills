# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the route installer through its shell CLI with an isolated package."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = (
    REPO
    / "ops/ascendc-cross-gen-migration/routes/3510-to-920x/references/build-and-install.sh.template"
)


class RouteInstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vendor = self.root / "vendor-A"
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        tools = self.root / "tools"
        tools.mkdir()
        # nm/ldd are external tools; filesystem and installer behavior remain real.
        for name, body in {
            "nm": "printf '0 T aclnnProbe\\n0 T aclnnProbeGetWorkspaceSize\\n'\n",
            "ldd": "printf 'dependencies resolved\\n'\n",
        }.items():
            p = tools / name
            p.write_text("#!/bin/sh\n" + body)
            p.chmod(0o755)
        env_file = self.root / "set_env.sh"
        env_file.write_text("# isolated environment\n")
        (self.root / "build.sh").write_text("# mock repository marker\n")
        self.package = self.root / "package.run"
        self.package.write_text(
            'set -e\nmkdir -p "$PACKAGE_DEST"\n'
            'printf payload > "$PACKAGE_DEST/libopapi.so"\n'
            'printf tiling > "$PACKAGE_DEST/libtiling.so"\n'
        )
        package_hash = hashlib.sha256(self.package.read_bytes()).hexdigest()
        (self.evidence / "package_candidates.sha256").write_text(
            f"{package_hash}  {self.package}\n"
        )
        (self.evidence / "build_target.txt").write_text("ascend960dt:9201\n")
        self.env = dict(
            os.environ,
            PATH=f"{tools}:{os.environ['PATH']}",
            CANN_SET_ENV=str(env_file),
            PYTHON_PATH=sys.executable,
            OP_DIR=str(self.root),
            EVIDENCE_DIR=str(self.evidence),
            SOC="ascend960dt",
            TARGET_ARCH="9201",
            VENDOR_DIR=str(self.vendor),
            OPAPI_LIB=str(self.vendor / "libopapi.so"),
            TILING_LIB=str(self.vendor / "libtiling.so"),
            ACLNN_SYMBOL="aclnnProbe",
            PACKAGE_DEST=str(self.vendor),
        )

    def install(self):
        return subprocess.run(
            ["bash", str(TEMPLATE), "install", str(self.package)],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_old_libraries_cannot_validate_a_package_installed_elsewhere(self):
        self.vendor.mkdir()
        (self.vendor / "libopapi.so").write_text("old opapi")
        (self.vendor / "libtiling.so").write_text("old tiling")
        self.env["PACKAGE_DEST"] = str(self.root / "vendor-B")
        result = self.install()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual((self.vendor / "libopapi.so").read_text(), "old opapi")

    def test_new_vendor_publishes_only_current_installation(self):
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.vendor / "libopapi.so").read_text(), "payload")
        self.assertEqual((self.vendor / "libtiling.so").read_text(), "tiling")
        manifest = (self.evidence / "installed_libraries.sha256").read_text()
        self.assertIn(hashlib.sha256(b"payload").hexdigest(), manifest)
        self.assertTrue((self.evidence / "installed_package.sha256").read_text())

    def test_empty_expected_location_does_not_accept_another_directory(self):
        self.env["PACKAGE_DEST"] = str(self.root / "vendor-B")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not install the expected library", result.stderr)
        self.assertTrue((self.root / "vendor-B/libopapi.so").is_file())
        self.assertEqual((self.evidence / "installed_package.sha256").read_text(), "")
        self.assertEqual((self.evidence / "installed_libraries.sha256").read_text(), "")

    def test_failed_retry_invalidates_previous_success(self):
        self.assertEqual(self.install().returncode, 0)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("new vendor directory", result.stderr)
        self.assertEqual((self.evidence / "installed_package.sha256").read_text(), "")
        self.assertEqual((self.evidence / "installed_libraries.sha256").read_text(), "")

    def test_unchanged_payload_can_be_installed_into_a_new_directory(self):
        self.assertEqual(self.install().returncode, 0)
        old_hash = hashlib.sha256((self.vendor / "libopapi.so").read_bytes()).digest()
        new_vendor = self.root / "vendor-C"
        self.env.update(
            VENDOR_DIR=str(new_vendor),
            OPAPI_LIB=str(new_vendor / "libopapi.so"),
            TILING_LIB=str(new_vendor / "libtiling.so"),
            PACKAGE_DEST=str(new_vendor),
        )
        result = self.install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            hashlib.sha256((new_vendor / "libopapi.so").read_bytes()).digest(), old_hash
        )

    def test_parent_traversal_cannot_select_an_existing_outside_library(self):
        outside = self.root / "old.so"
        outside.write_text("old")
        self.env["OPAPI_LIB"] = str(self.vendor / ".." / "old.so")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the recorded vendor directory", result.stderr)

    def test_missing_symbols_do_not_publish_success_manifests(self):
        (self.root / "tools/nm").write_text(
            "#!/bin/sh\nprintf '0 T another_symbol\\n'\n"
        )
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing exact ACLNN symbol", result.stderr)
        self.assertEqual((self.evidence / "installed_package.sha256").read_text(), "")
        self.assertEqual((self.evidence / "installed_libraries.sha256").read_text(), "")


if __name__ == "__main__":
    unittest.main()
