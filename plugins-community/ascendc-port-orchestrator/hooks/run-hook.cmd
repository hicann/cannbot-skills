: <<'WINDOWS_DISPATCH'
@echo off
setlocal EnableExtensions

REM -----------------------------------------------------------------------------------------------------------
REM Copyright (c) 2026 Huawei Technologies Co., Ltd.
REM Licensed under the CANN Open Software License Agreement Version 2.0.
REM See LICENSE in the root of the software repository for the full text of the License.
REM -----------------------------------------------------------------------------------------------------------

if "%~1"=="" goto :missing_argument

set "HOOK_SCRIPT=%~dp0%~1"
shift
set "BASH_EXE="

for %%B in ("C:\Program Files\Git\bin\bash.exe" "C:\Program Files (x86)\Git\bin\bash.exe") do (
    if not defined BASH_EXE if exist "%%~B" set "BASH_EXE=%%~B"
)

if not defined BASH_EXE (
    where.exe bash >nul 2>nul
    if not errorlevel 1 set "BASH_EXE=bash"
)

if not defined BASH_EXE exit /b 0

"%BASH_EXE%" "%HOOK_SCRIPT%" %1 %2 %3 %4 %5 %6 %7 %8
set "HOOK_STATUS=%ERRORLEVEL%"
endlocal & exit /b %HOOK_STATUS%

:missing_argument
>&2 echo run-hook.cmd: missing script name
exit /b 1
WINDOWS_DISPATCH

# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if [ "$#" -eq 0 ]; then
    printf '%s\n' 'run-hook.cmd: missing script name' >&2
    exit 1
fi

hook_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 1
hook_script=$1
shift
exec bash "$hook_root/$hook_script" "$@"
