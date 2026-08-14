:: Copyright (c) 2026 Huawei Technologies Co., Ltd.
:: This program is free software, you can redistribute it and/or modify it under the terms and conditions of
:: CANN Open Software License Agreement Version 2.0 (the "License").
:: Please refer to the License for details. You may not use this file except in compliance with the License.
:: THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
:: INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
:: See LICENSE in the root of the software repository for the full text of the License.
@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

:: CANNBot-Insight Windows startup script
:: Equivalent to start.sh for native Windows (cmd.exe)

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: ============================================================
:: Node.js version check (>= 20.0.0 required)
:: ============================================================
for /f "tokens=1 delims=." %%v in ('node -v 2^>nul') do set "NODE_VER=%%v"
set "NODE_MAJOR=%NODE_VER:v=%"
if %NODE_MAJOR% lss 20 (
    echo [setup] Node.js %NODE_VER% is not supported. Requires ^>= 20.0.0.
    echo [setup] Please install Node.js 20+ from https://nodejs.org
    exit /b 1
)
echo [setup] Node.js %NODE_VER% detected

:: ============================================================
:: Parse arguments
:: ============================================================
set "UPDATE=0"
set "CLI=0"
set "CLI_CMD="
set "KILL_EXISTING=0"
set "FRESH=0"
set "ADVANCED=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="-u" set "UPDATE=1"
if /i "%~1"=="-k" set "KILL_EXISTING=1"
if /i "%~1"=="-f" set "FRESH=1"
if /i "%~1"=="-a" set "ADVANCED=1"
if /i "%~1"=="-c" (
    set "CLI=1"
    set "CLI_CMD=%~2"
    shift
)
shift
goto :parse_args
:args_done

:: ============================================================
:: Install dependencies
:: ============================================================
if %UPDATE%==1 goto :do_install
if not exist "node_modules" goto :do_install
goto :install_done
:do_install
echo [setup] Installing dependencies...
call npm install
if !errorlevel! neq 0 (
    echo [setup] npm install failed. If you are behind a proxy, try:
    echo [setup]   npm config set proxy http://your-proxy:port
    echo [setup]   npm config set https-proxy http://your-proxy:port
    exit /b 1
)
:install_done

:: ============================================================
:: Ensure better-sqlite3 native module matches the running Node ABI
:: (node_modules is per-machine; switching Node 20<->24 leaves a
::  mismatched .node that crashes at runtime — auto-rebuild)
:: ============================================================
if not exist "node_modules\better-sqlite3" goto :sqlite_done
node -e "require('better-sqlite3')" >nul 2>&1
if !errorlevel! equ 0 goto :sqlite_done
echo [setup] better-sqlite3 native module mismatched for Node %NODE_VER% — rebuilding...
call npm rebuild better-sqlite3
if !errorlevel! neq 0 (
    echo [setup] better-sqlite3 rebuild failed. Install build tools or run: npm ci
    exit /b 1
)
node -e "require('better-sqlite3')" >nul 2>&1
if !errorlevel! neq 0 (
    echo [setup] better-sqlite3 still fails after rebuild.
    exit /b 1
)
echo [setup] better-sqlite3 rebuilt for Node %NODE_VER% OK
:sqlite_done

:: ============================================================
:: Fresh build: clear .next cache
:: ============================================================
if %FRESH%==1 (
    echo [setup] Clearing .next cache for fresh build...
    if exist ".next" rmdir /s /q ".next"
)

:: ============================================================
:: Ensure .env exists with DATABASE_URL
:: ============================================================
if not exist ".env" (
    echo [setup] Creating .env with DATABASE_URL...
    echo DATABASE_URL="file:./dev.db" > .env
)

:: ============================================================
:: Advanced tabs toggle
:: ============================================================
set "NEXT_PUBLIC_SHOW_ADVANCED_TABS=%ADVANCED%"
echo [setup] Advanced tabs: %ADVANCED% (use -a flag to enable)

:: ============================================================
:: Prisma: generate client + migrate
:: ============================================================
if not exist "node_modules\.prisma" (
    echo [setup] Generating Prisma client...
    call npx prisma generate
)
if %UPDATE%==1 goto :do_migrate
if not exist "prisma\dev.db" goto :do_migrate
goto :migrate_done
:do_migrate
echo [setup] Running Prisma migration...
call npx prisma migrate dev --name init
:migrate_done

:: ============================================================
:: Set DATABASE_URL
:: ============================================================
if not defined DATABASE_URL set "DATABASE_URL=file:%SCRIPT_DIR%prisma\dev.db"

:: ============================================================
:: Kill existing dev server via .next/dev/lock PID
:: ============================================================
set "LOCK_FILE=.next\dev\lock"
if not exist "%LOCK_FILE%" goto :lock_done
for /f "tokens=2 delims=:" %%p in ('findstr /r "\"pid\":" "%LOCK_FILE%" 2^>nul') do (
    for /f "tokens=1 delims=,} " %%n in ("%%p") do set "OLD_PID=%%n"
)
if not defined OLD_PID goto :lock_cleanup
echo [start] Stopping existing dev server ^(!OLD_PID!^)...
taskkill /pid !OLD_PID! /f >nul 2>&1
timeout /t 2 /nobreak >nul
:lock_cleanup
del /f /q "%LOCK_FILE%" 2>nul
:lock_done

:: ============================================================
:: Find available port (starting from 21025)
:: ============================================================
set "BASE_PORT=21025"
set "PORT=%BASE_PORT%"

if %KILL_EXISTING%==1 (
    echo [start] -k: Checking port %BASE_PORT%...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%BASE_PORT% " ^| findstr "LISTENING"') do (
        echo [start] -k: Killing process on port %BASE_PORT% ^(PID %%a^)...
        taskkill /pid %%a /f >nul 2>&1
    )
    set "PORT=%BASE_PORT%"
    goto :port_found
)

:: Scan ports 21025..21152
for /l %%i in (0,1,127) do (
    set /a "TEST_PORT=BASE_PORT+%%i"
    set "PORT_OCCUPIED=0"
    for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":!TEST_PORT! " ^| findstr "LISTENING"') do set "PORT_OCCUPIED=1"
    if "!PORT_OCCUPIED!"=="1" (
        echo [start] Port !TEST_PORT! is in use, trying next...
    ) else (
        set "PORT=!TEST_PORT!"
        goto :port_found
    )
)
echo [start] ERROR: No available port in range %BASE_PORT%-%BASE_PORT+127
exit /b 1

:port_found
echo [start] Launching CANNBot-Insight on port %PORT%...

:: ============================================================
:: Launch smart-agent (Python) in background (optional, AI audit v2)
:: Mirrors start.sh: auto-start if server.py exists and python present,
:: skip silently otherwise. Shares the console so it dies with the window.
:: ============================================================
set "AGENT_PORT=21026"
if not exist "%SCRIPT_DIR%smart-agent\server.py" goto :agent_done
netstat -ano 2>nul | findstr ":%AGENT_PORT% " | findstr "LISTENING" >nul 2>&1
if !errorlevel! equ 0 (
    echo [start] Port %AGENT_PORT% in use, assuming smart-agent already running
    goto :agent_done
)
where python >nul 2>&1
if !errorlevel! neq 0 (
    echo [start] Python not found - smart-agent \(AI audit v2\) skipped
    echo [start]   Install Python 3 from https://python.org to enable it
    goto :agent_done
)
echo [start] Launching smart-agent ^(Python^) on port %AGENT_PORT%...
pushd "%SCRIPT_DIR%smart-agent"
set "CANNBOT_AGENT_PORT=%AGENT_PORT%"
start /b python server.py
popd
set "CANNBOT_AGENT_URL=http://localhost:%AGENT_PORT%"
for /l %%i in (1,1,10) do (
    curl -s "http://localhost:%AGENT_PORT%/health" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [start] smart-agent ready at %CANNBOT_AGENT_URL%
        goto :agent_done
    )
    timeout /t 1 /nobreak >nul
)
:agent_done

:: ============================================================
:: CLI mode: start backend + CLI
:: ============================================================
if not %CLI%==1 goto :web_mode

set "SERVER_URL=http://localhost:%PORT%"

echo [start] Starting backend...
start /b npx next dev --port %PORT%

echo [start] Waiting for backend at %SERVER_URL%...
for /l %%i in (1,1,60) do (
    curl -s "%SERVER_URL%/api/observe/data?pageSize=1" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [start] Backend ready at %SERVER_URL%
        goto :cli_launch
    )
    timeout /t 1 /nobreak >nul
)

:cli_launch
echo [start] Launching CLI: %CLI_CMD%
call npx tsx src/cli/index.ts %CLI_CMD% --server %SERVER_URL%
echo [start] CLI exited, backend stopped
:: Stop backend + smart-agent by port (start /b children otherwise orphaned)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%PORT% " ^| findstr "LISTENING"') do taskkill /pid %%a /f >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%AGENT_PORT% " ^| findstr "LISTENING"') do taskkill /pid %%a /f >nul 2>&1
exit /b 0

:: ============================================================
:: Web mode: start dev server + open browser
:: ============================================================
:web_mode
echo [start] Starting Next.js dev server...
start /b npx next dev --port %PORT%

echo [start] Waiting for server at http://localhost:%PORT%...
for /l %%i in (1,1,30) do (
    curl -s "http://localhost:%PORT%" >nul 2>&1
    if !errorlevel! equ 0 (
        echo [start] Server ready - opening http://localhost:%PORT%
        start http://localhost:%PORT%
        goto :wait_server
    )
    timeout /t 1 /nobreak >nul
)

:wait_server
echo [start] Server running at http://localhost:%PORT%
echo [start] Close this window to stop the server and smart-agent.
echo [start] Or press Ctrl+C, then answer the prompt.
:agent_cleanup_loop
pause >nul
goto :agent_cleanup_loop