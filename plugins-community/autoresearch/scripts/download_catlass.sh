#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");

set -e

CATLASS_REPO_URL="${CATLASS_REPO_URL:-https://gitcode.com/cann/catlass.git}"
CATLASS_COMMIT="${CATLASS_COMMIT:-d60bf08c278c07d8fd1a74d3a4a4f590555d9ab9}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/thirdparty/catlass"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found on PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "${TARGET_DIR}")"

if [ -d "${TARGET_DIR}/.git" ]; then
  echo "catlass already exists at ${TARGET_DIR}; fetching refs"
  git -C "${TARGET_DIR}" fetch --tags origin
else
  if [ -d "${TARGET_DIR}" ] && [ -n "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: ${TARGET_DIR} is non-empty and is not a git repository" >&2
    exit 1
  fi
  git clone "${CATLASS_REPO_URL}" "${TARGET_DIR}"
fi

if ! git -C "${TARGET_DIR}" rev-parse --verify "${CATLASS_COMMIT}^{commit}" >/dev/null 2>&1; then
  git -C "${TARGET_DIR}" fetch origin "${CATLASS_COMMIT}" || true
fi

if git -C "${TARGET_DIR}" rev-parse --verify "${CATLASS_COMMIT}^{commit}" >/dev/null 2>&1; then
  git -C "${TARGET_DIR}" checkout "${CATLASS_COMMIT}"
else
  echo "WARNING: target commit is unavailable; keeping the current ref" >&2
fi

if [ ! -d "${TARGET_DIR}/include/catlass" ]; then
  echo "ERROR: ${TARGET_DIR}/include/catlass is missing" >&2
  exit 1
fi

echo "catlass installed at ${TARGET_DIR}"
