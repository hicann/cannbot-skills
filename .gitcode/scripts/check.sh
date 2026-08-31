# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e
export PATH="$PATH:/root/.opencode/bin"
which opencode
opencode --version
git clone https://${cui_user}:${cui_token}@gitcode.com/Junren6415/cann_auto_reviewer.git -b cannbot_backend
cd cann_auto_reviewer
cat > .env << 'EOF'
export CANNBOT_API_KEY="${CANNBOT_API_KEY}"
export GITCODE_TOKEN="${cui_token}"
export GIT_USER_NAME="${cui_user}"
export CANNBOT_AUTH_URL="https://cannbot.hicann.cn/cannbot/api/auth/authenticate"
EOF
cd review_cannbot
opencode --version

./review_cannbot_pr.sh ${MERGE_ID}
opencode_check=$(find ${WORKSPACE}/cann_auto_reviewer/review_cannbot/reports -name "pr_${MERGE_ID}*.md" 2>/dev/null | head -n1)
verdict=$(tail -1 "$opencode_check" | grep '^REVIEW_VERDICT=' | cut -d= -f2)
echo "${opencode_check}"
mkdir -p ${WORKSPACE}/build_out
mv ${opencode_check}  ${WORKSPACE}/build_out/pr_check.md
case "$verdict" in
  PASS)       echo "✓ 检视通过，可以合并" ; echo "{\"opencode\": \"success\"}" > UT_Test_check_opencode.json ;exit 0 ;;
  CONDITIONAL) echo "⚠ 有条件通过，建议复核" ; echo "{\"opencode\": \"failed\"}" > UT_Test_check_opencode.json ;exit 0 ;;
  REJECT) echo "✗ 需要修改后再审"; echo "{\"opencode\": \"failed\"}" > UT_Test_check_opencode.json ;exit 0 ;;
  *)             echo "○ 检视未完成（崩溃或超时）"; echo "{\"opencode\": \"failed\"}" > UT_Test_check_opencode.json ;exit 0 ;;
esac