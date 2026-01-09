#!/usr/bin/env powershell

# ShioRIS3 ビルドクリーンアップスクリプト

param(
    [switch]$Force,          # 強制削除（確認なし）
    [switch]$KeepCache,      # CMakeキャッシュを保持
    [switch]$Verbose         # 詳細表示
)

Write-Host "=== ShioRIS3 Build Cleanup ===" -ForegroundColor Green
Write-Host ""

# プロジェクトルートディレクトリの確認
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $projectRoot "CMakeLists.txt"))) {
    Write-Host "Error: CMakeLists.txt not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

Write-Host "Project root: $projectRoot" -ForegroundColor Cyan
Write-Host ""

# クリーンアップ対象の定義
$cleanupTargets = @{
    "build" = @{
        "path" = Join-Path $projectRoot "build"
        "description" = "Main build directory"
        "essential" = $true
    }
    "build-debug" = @{
        "path" = Join-Path $projectRoot "build-debug"
        "description" = "Debug build directory"
        "essential" = $false
    }
    "build-release" = @{
        "path" = Join-Path $projectRoot "build-release"
        "description" = "Release build directory"
        "essential" = $false
    }
    "out" = @{
        "path" = Join-Path $projectRoot "out"
        "description" = "Visual Studio out directory"
        "essential" = $false
    }
}

# 追加のクリーンアップファイル/ディレクトリ
$additionalCleanup = @(
    @{
        "pattern" = "*.vcxproj.user"
        "description" = "Visual Studio user files"
    }
    @{
        "pattern" = "*.sln"
        "description" = "Visual Studio solution files"
    }
    @{
        "pattern" = ".vs"
        "description" = "Visual Studio cache directory"
    }
)

# 存在するターゲットを確認
$existingTargets = @()
foreach ($target in $cleanupTargets.GetEnumerator()) {
    if (Test-Path $target.Value.path) {
        $size = 0
        try {
            $size = (Get-ChildItem $target.Value.path -Recurse -ErrorAction SilentlyContinue | 
                    Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
            $sizeStr = if ($size -gt 1GB) { 
                "{0:F1} GB" -f ($size / 1GB) 
            } elseif ($size -gt 1MB) { 
                "{0:F1} MB" -f ($size / 1MB) 
            } elseif ($size -gt 1KB) { 
                "{0:F1} KB" -f ($size / 1KB) 
            } else { 
                "$size B" 
            }
        } catch {
            $sizeStr = "Unknown size"
        }
        
        $existingTargets += @{
            "name" = $target.Key
            "path" = $target.Value.path
            "description" = $target.Value.description
            "size" = $sizeStr
            "essential" = $target.Value.essential
        }
    }
}

# 追加ファイルの確認
$additionalFiles = @()
foreach ($item in $additionalCleanup) {
    $files = Get-ChildItem $projectRoot -Filter $item.pattern -Recurse -ErrorAction SilentlyContinue
    if ($files) {
        $additionalFiles += @{
            "files" = $files
            "description" = $item.description
            "pattern" = $item.pattern
        }
    }
}

# 削除対象がない場合
if ($existingTargets.Count -eq 0 -and $additionalFiles.Count -eq 0) {
    Write-Host "No build artifacts found. Project is already clean." -ForegroundColor Green
    exit 0
}

# 削除対象の表示
if ($existingTargets.Count -gt 0) {
    Write-Host "Found build directories:" -ForegroundColor Yellow
    foreach ($target in $existingTargets) {
        $color = if ($target.essential) { "Red" } else { "Yellow" }
        Write-Host "  [$($target.size)] $($target.path)" -ForegroundColor $color
        Write-Host "    $($target.description)" -ForegroundColor Gray
    }
    Write-Host ""
}

if ($additionalFiles.Count -gt 0) {
    Write-Host "Found additional files:" -ForegroundColor Yellow
    foreach ($item in $additionalFiles) {
        Write-Host "  $($item.description) ($($item.files.Count) files)" -ForegroundColor Yellow
        if ($Verbose) {
            foreach ($file in $item.files) {
                Write-Host "    $($file.FullName)" -ForegroundColor Gray
            }
        }
    }
    Write-Host ""
}

# 確認または強制実行
if (-not $Force) {
    $confirmation = Read-Host "Do you want to delete these items? [y/N]"
    if ($confirmation -notmatch '^[Yy]') {
        Write-Host "Cleanup cancelled." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host "Starting cleanup..." -ForegroundColor Green
Write-Host ""

# ディレクトリの削除
$deletedCount = 0
$errorCount = 0

foreach ($target in $existingTargets) {
    try {
        Write-Host "Deleting $($target.path)..." -ForegroundColor Cyan
        
        if ($KeepCache -and $target.name -eq "build") {
            # CMakeキャッシュを保持する場合
            $cacheFiles = @("CMakeCache.txt", "CMakeFiles")
            $tempDir = Join-Path $env:TEMP "ShioRIS3_cache_$(Get-Random)"
            New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
            
            foreach ($cacheFile in $cacheFiles) {
                $cachePath = Join-Path $target.path $cacheFile
                if (Test-Path $cachePath) {
                    Copy-Item $cachePath $tempDir -Recurse -Force
                    if ($Verbose) {
                        Write-Host "  Backed up $cacheFile" -ForegroundColor Gray
                    }
                }
            }
            
            Remove-Item $target.path -Recurse -Force -ErrorAction Stop
            New-Item -ItemType Directory -Path $target.path -Force | Out-Null
            
            foreach ($cacheFile in $cacheFiles) {
                $backupPath = Join-Path $tempDir $cacheFile
                if (Test-Path $backupPath) {
                    Copy-Item $backupPath (Join-Path $target.path $cacheFile) -Recurse -Force
                    if ($Verbose) {
                        Write-Host "  Restored $cacheFile" -ForegroundColor Gray
                    }
                }
            }
            
            Remove-Item $tempDir -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  Directory cleaned (cache preserved)" -ForegroundColor Green
        } else {
            Remove-Item $target.path -Recurse -Force -ErrorAction Stop
            Write-Host "  Directory deleted successfully" -ForegroundColor Green
        }
        $deletedCount++
    } catch {
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        $errorCount++
    }
}

# 追加ファイルの削除
foreach ($item in $additionalFiles) {
    try {
        Write-Host "Deleting $($item.description)..." -ForegroundColor Cyan
        foreach ($file in $item.files) {
            Remove-Item $file.FullName -Force -ErrorAction Stop
            if ($Verbose) {
                Write-Host "  Deleted $($file.Name)" -ForegroundColor Gray
            }
        }
        Write-Host "  $($item.files.Count) files deleted" -ForegroundColor Green
        $deletedCount += $item.files.Count
    } catch {
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        $errorCount++
    }
}

Write-Host ""
Write-Host "=== Cleanup Summary ===" -ForegroundColor Green
Write-Host "Items processed: $($deletedCount + $errorCount)"
Write-Host "Successfully deleted: $deletedCount" -ForegroundColor Green
if ($errorCount -gt 0) {
    Write-Host "Errors: $errorCount" -ForegroundColor Red
}

if ($KeepCache) {
    Write-Host "CMake cache preserved" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Cleanup completed!" -ForegroundColor Green
