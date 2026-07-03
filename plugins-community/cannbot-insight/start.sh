#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Node.js version check (>= 20.0.0 required; v18.19.x fails with better-sqlite3 / Prisma 6)
NODE_MAJOR=$(node -e "console.log(process.versions.node.split('.')[0])" 2>/dev/null || echo "0")
if [ "$NODE_MAJOR" -lt 20 ]; then
  NODE_VERSION=$(node -v 2>/dev/null || echo "unknown")
  echo "[setup] Node.js $NODE_VERSION is not supported. Requires >= 20.0.0." >&2
  echo "[setup] v18.19.x fails to install better-sqlite3 and Prisma 6 native modules." >&2

  # Try auto-switch via nvm if available
  if command -v nvm &>/dev/null || [ -s "$HOME/.nvm/nvm.sh" ]; then
    [ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
    NVM_LTS=$(nvm ls-remote --lts=20 2>/dev/null | tail -1 | awk '{print $1}')
    if [ -n "$NVM_LTS" ]; then
      echo "[setup] Found nvm — installing Node.js $NVM_LTS..."
      nvm install "$NVM_LTS" && nvm use "$NVM_LTS"
      NODE_MAJOR=$(node -e "console.log(process.versions.node.split('.')[0])")
      if [ "$NODE_MAJOR" -ge 20 ]; then
        echo "[setup] Switched to Node.js $(node -v) ✓"
      else
        echo "[setup] nvm install failed, please run: nvm install 20 && nvm use 20" >&2
        exit 1
      fi
    else
      echo "[setup] nvm found but no Node 20 LTS available. Run: nvm install 20" >&2
      exit 1
    fi
  else
    echo "[setup] No nvm detected. Quick fix:" >&2
    echo "[setup]   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash" >&2
    echo "[setup]   source ~/.nvm/nvm.sh && nvm install 20 && nvm use 20" >&2
    echo "[setup] Or install Node 20+ directly: https://nodejs.org" >&2
    exit 1
  fi
fi

export DATABASE_URL="${DATABASE_URL:-file:$SCRIPT_DIR/prisma/dev.db}"

UPDATE=false
CLI=false
CLI_CMD=""
KILL_EXISTING=false
FRESH=false
ADVANCED=false
while getopts "uc:kfa" opt; do
  case $opt in
    u) UPDATE=true ;;
    c) CLI=true; CLI_CMD="$OPTARG" ;;
    k) KILL_EXISTING=true ;;
    f) FRESH=true ;;
    a) ADVANCED=true ;;
    *) echo "Usage: $0 [-u] [-k] [-f] [-a] [-c <command|tui>]  (-u: update; -k: kill port; -f: fresh build; -a: show advanced tabs; -c: CLI mode)" >&2; exit 1 ;;
  esac
done

if [ "$UPDATE" = true ] || [ ! -d "node_modules" ]; then
  echo "[setup] Installing dependencies..."
  npm install
fi

if [ "$FRESH" = true ]; then
  echo "[setup] Clearing .next cache for fresh build..."
  rm -rf .next
fi

# Ensure .env exists with DATABASE_URL
if [ ! -f ".env" ]; then
  echo "[setup] Creating .env with DATABASE_URL..."
  echo 'DATABASE_URL="file:./dev.db"' > .env
fi

# Advanced tabs toggle (subagents, interactions, AI workflow)
export NEXT_PUBLIC_SHOW_ADVANCED_TABS="${ADVANCED}"
echo "[setup] Advanced tabs: ${ADVANCED} (use -a flag to enable)"

if [ "$UPDATE" = true ] || [ ! -f "prisma/dev.db" ]; then
  echo "[setup] Running Prisma migration..."
  npx prisma migrate dev --name init
fi

LOCK_FILE=".next/dev/lock"
if [ -f "$LOCK_FILE" ]; then
  OLD_PID=$(python3 -c "import json; print(json.load(open('$LOCK_FILE'))['pid'])" 2>/dev/null || cat "$LOCK_FILE" | grep -o '"pid":[0-9]*' | grep -o '[0-9]*')
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[start] Stopping existing dev server (PID $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 2
  fi
  rm -f "$LOCK_FILE"
fi

BASE_PORT=21025
PORT=$BASE_PORT

# -k: kill any process occupying BASE_PORT, then use that port
if [ "$KILL_EXISTING" = true ]; then
  OCCUPIER_PID=$(lsof -ti :$BASE_PORT 2>/dev/null || true)
  if [ -n "$OCCUPIER_PID" ]; then
    echo "[start] -k: Killing process on port $BASE_PORT (PID $OCCUPIER_PID)..."
    kill $OCCUPIER_PID 2>/dev/null || true
    sleep 2
  else
    echo "[start] -k: Port $BASE_PORT is free, no process to kill"
  fi
  PORT=$BASE_PORT
else
  # Find next available port
  MAX_ATTEMPTS=128
  for i in $(seq 0 $((MAX_ATTEMPTS - 1))); do
    PORT=$((BASE_PORT + i))
    if (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
      echo "[start] Port $PORT is in use, trying next..."
      continue
    fi
    break
  done

  if [ $PORT -ge $((BASE_PORT + MAX_ATTEMPTS)) ]; then
    echo "[start] ERROR: No available port in range $BASE_PORT-$((BASE_PORT + MAX_ATTEMPTS - 1))" >&2
    exit 1
  fi
fi

echo "[start] Launching CANNBot-Insight on port $PORT..."

if [ "$CLI" = true ]; then
  SERVER_URL="http://localhost:$PORT"

  # Start backend in background
  npx next dev --port $PORT &
  BACKEND_PID=$!

  # Wait for backend to be ready
  echo "[start] Waiting for backend at $SERVER_URL..."
  for i in $(seq 1 60); do
    if curl -s "$SERVER_URL/api/observe/data?pageSize=1" > /dev/null 2>&1; then
      echo "[start] Backend ready at $SERVER_URL"
      break
    fi
    sleep 1
  done

  echo "[start] Launching CLI: $CLI_CMD"
  npx tsx src/cli/index.ts $CLI_CMD --server $SERVER_URL
  kill $BACKEND_PID 2>/dev/null || true
  echo "[start] CLI exited, backend stopped"
else
  npx next dev --port $PORT &
  NEXT_PID=$!

  echo "[start] Waiting for server at http://localhost:$PORT..."
  for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
      echo "[start] Server ready — opening http://localhost:$PORT"
      # WSL: use Windows cmd.exe to open browser
      if grep -qi microsoft /proc/version 2>/dev/null && [ -x /mnt/c/Windows/System32/cmd.exe ]; then
        /mnt/c/Windows/System32/cmd.exe /c start "http://localhost:$PORT" 2>/dev/null || true
      elif command -v xdg-open > /dev/null 2>&1; then
        xdg-open "http://localhost:$PORT" 2>/dev/null || true
      elif command -v open > /dev/null 2>&1; then
        open "http://localhost:$PORT" 2>/dev/null || true
      elif command -v sensible-browser > /dev/null 2>&1; then
        sensible-browser "http://localhost:$PORT" 2>/dev/null || true
      else
        echo "[start] Could not detect browser command. Open manually: http://localhost:$PORT"
      fi
      break
    fi
    sleep 1
  done

  wait $NEXT_PID
fi
