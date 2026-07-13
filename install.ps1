# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# Ensure TLS 1.2 (Windows PowerShell 5.1 defaults to TLS 1.0)
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
} catch {}

# Channel selection: latest (default), beta, or explicit version
# Set $env:CANNBOT_CHANNEL = "beta" before running to install beta
$Channel = if ($env:CANNBOT_CHANNEL) { $env:CANNBOT_CHANNEL } else { "latest" }

# Auto-fix ExecutionPolicy for npm global tools (.ps1)
$execPolicy = Get-ExecutionPolicy -Scope CurrentUser
if ($execPolicy -eq "Restricted") {
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-Host " [OK] ExecutionPolicy set to RemoteSigned (required for npm tools)" -ForegroundColor Green
}

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

trap {
    Write-Host ""
    Write-Host " [ERROR] $_" -ForegroundColor Red
    Write-Host ""
    Write-Host " Press Enter to exit..."
    Read-Host
    return
}

Write-Host ""
Write-Host "   ____    _    _   _ _   _ ____        _   " -ForegroundColor Cyan
Write-Host "  / ___|  / \  | \ | | \ | | __ )  ___ | |_ " -ForegroundColor Cyan
Write-Host " | |     / _ \ |  \| |  \| |  _ \ / _ \| __|" -ForegroundColor Cyan
Write-Host " | |___ / ___ \| |\  | |\  | |_) | (_) | |_ " -ForegroundColor Cyan
Write-Host "  \____/_/   \_\_| \_|_| \_|____/ \___/ \__|" -ForegroundColor Cyan
Write-Host ""
Write-Host " CANNBot Install Helper" -ForegroundColor Cyan
Write-Host ""

# 1. Detect platform
$Arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
$SubPkgName = "@cannbot-ai/install-helper-windows-$Arch"
$BinaryName = "install-helper.exe"

Write-Host " [OK] Platform: windows-$Arch" -ForegroundColor Green
Write-Host ""

# 2. Query version from npm registry (channel-specific)
Write-Host " Querying version ($Channel)..." -ForegroundColor Cyan

$Mirrors = @(
    "https://registry.npmjs.org",
    "https://registry.npmmirror.com"
)

$Version = $null
$Registry = $null
$Errors = @()

if ($Channel -eq "latest" -or $Channel -eq "beta") {
    foreach ($url in $Mirrors) {
        try {
            $resp = Invoke-RestMethod "$url/$SubPkgName/$Channel" -TimeoutSec 15
            if ($resp.version) {
                $Version = $resp.version
                $Registry = $url
            }
        } catch { $Errors += $_ }
        if ($Version) {
            break
        }
    }
} else {
    $Version = $Channel
    $Registry = $Mirrors[0]
}

if ($Version) {
    Write-Host " [OK] Version: $Version ($Channel)" -ForegroundColor Green
}

if (-not $Version) {
    Write-Host " [ERROR] Failed to query version from all registries" -ForegroundColor Red
    if ($Errors.Count -gt 0) {
        Write-Host " Errors:" -ForegroundColor Yellow
        foreach ($err in $Errors) {
            Write-Host "  - $($err.Exception.Message)" -ForegroundColor Yellow
        }
    }
    Write-Host ""
    Write-Host " Alternative (requires Node.js 18+):"
    Write-Host "  npm install -g @cannbot-ai/install-helper"
    return
}

# 3. Check if already installed and up-to-date
$InstallDir = Join-Path $env:USERPROFILE ".cannbot\bin"
$BinaryDest = Join-Path $InstallDir "install-helper.exe"
if (Test-Path $BinaryDest) {
    try {
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $localVersion = (& $BinaryDest --version 2>$null)
        $ErrorActionPreference = $prevEAP
        if ($localVersion -and $localVersion.Trim() -eq $Version) {
            Write-Host " [OK] Already up-to-date (v$Version)" -ForegroundColor Green
            Write-Host ""
            $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
            if (-not $UserPath) {
                [Environment]::SetEnvironmentVariable("Path", $InstallDir, "User")
            } elseif ($UserPath -notlike "*$InstallDir*") {
                [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
            }
            Write-Host " Launching install-helper..." -ForegroundColor Cyan
            Write-Host ""
            & $BinaryDest
            return
        }
        if ($localVersion) {
            Write-Host " [INFO] Updating: v$($localVersion.Trim()) -> v$Version" -ForegroundColor Yellow
        }
    } catch {}
}

# 4. Download tarball (with mirror fallback)
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$TarballName = "$($SubPkgName.Split('/')[1])-$Version.tgz"

Write-Host ""
Write-Host " Downloading $TarballName ..." -ForegroundColor Cyan

$TempFile = Join-Path $env:TEMP "ih-tarball.tgz"
$Downloaded = $false
$DownloadMirrors = @($Registry) + ($Mirrors | Where-Object { $_ -ne $Registry })
foreach ($url in $DownloadMirrors) {
    $TarballUrl = "$url/$SubPkgName/-/$TarballName"
    try {
        Invoke-WebRequest $TarballUrl -OutFile $TempFile -TimeoutSec 300 -UseBasicParsing
        $Downloaded = $true
        Write-Host " [OK] Download complete" -ForegroundColor Green
        break
    } catch {
        Write-Host " [WARN] Download from $url failed: $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

if (-not $Downloaded) {
    Write-Host " [ERROR] Download failed from all mirrors" -ForegroundColor Red
    Write-Host ""
    Write-Host " Alternative (requires Node.js 18+):"
    Write-Host "  npm install -g @cannbot-ai/install-helper"
    return
}

# 5. Extract
Write-Host " Installing v$Version..." -ForegroundColor Cyan

$TempExtract = Join-Path $env:TEMP "ih-extract-$(Get-Random)"
New-Item -ItemType Directory -Force -Path $TempExtract | Out-Null

try {
    & tar xzf $TempFile -C $TempExtract 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "tar exit code $LASTEXITCODE"
    }
} catch {
    Write-Host " [ERROR] Extraction failed. tar not found or error." -ForegroundColor Red
    Write-Host "  Windows 10 1803+ includes tar. For older Windows, use npm install."
    Remove-Item $TempExtract -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $TempFile -Force -ErrorAction SilentlyContinue
    return
}

$BinarySrc = Join-Path $TempExtract "package\$BinaryName"

if (-not (Test-Path $BinarySrc)) {
    Write-Host " [ERROR] Binary not found in tarball: $BinarySrc" -ForegroundColor Red
    Remove-Item $TempExtract -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item $TempFile -Force -ErrorAction SilentlyContinue
    return
}

Copy-Item $BinarySrc $BinaryDest -Force
Remove-Item $TempExtract -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $TempFile -Force -ErrorAction SilentlyContinue

Write-Host " [OK] Installation successful!" -ForegroundColor Green
Write-Host ""

# 6. PATH
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $UserPath) {
    [Environment]::SetEnvironmentVariable("Path", $InstallDir, "User")
    Write-Host " [OK] Added to PATH (restart terminal to take effect)" -ForegroundColor Green
    Write-Host ""
} elseif ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    Write-Host " [OK] Added to PATH (restart terminal to take effect)" -ForegroundColor Green
    Write-Host ""
}

# 7. Verify binary and launch
try {
    $null = & $BinaryDest --version 2>$null
    Write-Host " Launching install-helper..." -ForegroundColor Cyan
    Write-Host ""
    & $BinaryDest
} catch {
    Write-Host " [WARN] Binary failed to run" -ForegroundColor Yellow
    Write-Host ""
    Write-Host " Alternative (requires Node.js 18+):"
    Write-Host "  npm install -g @cannbot-ai/install-helper"
}
