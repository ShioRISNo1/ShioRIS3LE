#!/usr/bin/env powershell

# ShioRIS3 ビルドスクリプト

param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$BuildType = "Release",     # ビルドタイプ
    
    [switch]$Clean,                     # ビルド前にクリーン実行
    [switch]$Rebuild,                   # 完全にリビルド
    [switch]$Verbose,                   # 詳細出力
    [switch]$Parallel,                  # 並列ビルド（デフォルトで有効）
    [switch]$SkipTests,                 # テストをスキップ
    [switch]$Deploy,                    # ビルド後にQt DLLを配布
    
    [string]$Generator = "",            # CMakeジェネレーター指定
    [string]$Toolchain = "",            # カスタムツールチェーン
    [string]$BuildDir = "",             # カスタムビルドディレクトリ
    
    [int]$Jobs = 0                      # 並列ジョブ数（0=自動）
)

# エラー時の動作設定
$ErrorActionPreference = "Stop"

Write-Host "=== ShioRIS3 Build Script ===" -ForegroundColor Green
Write-Host ""

# プロジェクトルートディレクトリの確認
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $projectRoot "CMakeLists.txt"))) {
    Write-Host "Error: CMakeLists.txt not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

Write-Host "Project root: $projectRoot" -ForegroundColor Cyan
Write-Host "Build type: $BuildType" -ForegroundColor Cyan

# ビルドディレクトリの設定
if ($BuildDir -eq "") {
    $BuildDir = Join-Path $projectRoot "build"
}
Write-Host "Build directory: $BuildDir" -ForegroundColor Cyan

# 環境確認
Write-Host ""
Write-Host "=== Environment Check ===" -ForegroundColor Yellow

# CMakeの確認
try {
    $cmakeVersion = & cmake --version 2>$null | Select-Object -First 1
    Write-Host "✓ CMake: $cmakeVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ CMake not found. Please install CMake." -ForegroundColor Red
    exit 1
}

# vcpkgの確認
if ($env:VCPKG_ROOT) {
    if (Test-Path (Join-Path $env:VCPKG_ROOT "vcpkg.exe")) {
        Write-Host "✓ vcpkg: $env:VCPKG_ROOT" -ForegroundColor Green
    } else {
        Write-Host "✗ vcpkg.exe not found in VCPKG_ROOT" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✗ VCPKG_ROOT not set" -ForegroundColor Red
    exit 1
}

# Qt6の確認
$qtFound = $false
$qtPaths = @("C:/Qt/6.9.1/msvc2022_64", "C:/Qt/6.8.1/msvc2022_64")
foreach ($qtPath in $qtPaths) {
    if (Test-Path $qtPath) {
        Write-Host "✓ Qt6: $qtPath" -ForegroundColor Green
        $qtFound = $true
        break
    }
}
if (-not $qtFound) {
    Write-Host "✗ Qt6 not found in expected locations" -ForegroundColor Red
    exit 1
}

# ビルド前のクリーンアップ
if ($Clean -or $Rebuild) {
    Write-Host ""
    Write-Host "=== Cleanup ===" -ForegroundColor Yellow
    
    if ($Rebuild) {
        Write-Host "Performing full rebuild cleanup..." -ForegroundColor Cyan
        if (Test-Path $BuildDir) {
            Remove-Item $BuildDir -Recurse -Force
            Write-Host "Build directory removed" -ForegroundColor Green
        }
    } elseif ($Clean) {
        Write-Host "Performing incremental cleanup..." -ForegroundColor Cyan
        if (Test-Path $BuildDir) {
            # CMakeファイルとビルド成果物を削除、キャッシュは保持
            $cleanItems = @("CMakeFiles", "*.vcxproj", "*.sln", "*.exe", "*.dll", "*.lib")
            foreach ($item in $cleanItems) {
                $itemPath = Join-Path $BuildDir $item
                if (Test-Path $itemPath) {
                    Remove-Item $itemPath -Recurse -Force -ErrorAction SilentlyContinue
                }
            }
            Write-Host "Build artifacts cleaned" -ForegroundColor Green
        }
    }
}

# ビルドディレクトリの作成
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
    Write-Host "Build directory created: $BuildDir" -ForegroundColor Green
}

# CMakeの設定準備
Write-Host ""
Write-Host "=== CMake Configuration ===" -ForegroundColor Yellow

$cmakeArgs = @()
$cmakeArgs += "-DCMAKE_BUILD_TYPE=$BuildType"

# ツールチェーンファイルの設定
if ($Toolchain -ne "") {
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$Toolchain"
} elseif ($env:VCPKG_ROOT) {
    $vcpkgToolchain = Join-Path $env:VCPKG_ROOT "scripts/buildsystems/vcpkg.cmake"
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
}

# ジェネレーターの設定
if ($Generator -ne "") {
    $cmakeArgs += "-G", $Generator
}

# 追加の最適化設定
if ($BuildType -eq "Release") {
    $cmakeArgs += "-DCMAKE_CXX_FLAGS_RELEASE=/O2 /DNDEBUG"
}

# ビルド時間計測開始
$buildStartTime = Get-Date

try {
    # CMake設定段階
    Write-Host "Running CMake configuration..." -ForegroundColor Cyan
    Write-Host "Command: cmake $($cmakeArgs -join ' ') .." -ForegroundColor Gray
    
    Push-Location $BuildDir
    
    if ($Verbose) {
        & cmake @cmakeArgs ".."
    } else {
        & cmake @cmakeArgs ".." | Out-Host
    }
    
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed with exit code $LASTEXITCODE"
    }
    
    Write-Host "✓ CMake configuration completed successfully" -ForegroundColor Green
    
    # ビルド段階
    Write-Host ""
    Write-Host "=== Build ===" -ForegroundColor Yellow
    
    $buildArgs = @("--build", ".", "--config", $BuildType)
    
    # 並列ビルドの設定
    if ($Parallel -or (-not $PSBoundParameters.ContainsKey('Parallel'))) {
        if ($Jobs -eq 0) {
            $Jobs = [Environment]::ProcessorCount
        }
        $buildArgs += "--parallel", $Jobs
        Write-Host "Using $Jobs parallel jobs" -ForegroundColor Cyan
    }
    
    if ($Verbose) {
        $buildArgs += "--verbose"
    }
    
    Write-Host "Running build..." -ForegroundColor Cyan
    Write-Host "Command: cmake $($buildArgs -join ' ')" -ForegroundColor Gray
    
    & cmake @buildArgs
    
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
    
    Write-Host "✓ Build completed successfully" -ForegroundColor Green
    
    # ビルド成果物の確認
    $executable = Join-Path $BuildDir "$BuildType/ShioRIS3.exe"
    if (-not (Test-Path $executable)) {
        $executable = Join-Path $BuildDir "ShioRIS3.exe"
    }
    
    if (Test-Path $executable) {
        $fileInfo = Get-Item $executable
        Write-Host "✓ Executable created: $($fileInfo.Name) ($($fileInfo.Length) bytes)" -ForegroundColor Green
        Write-Host "  Location: $($fileInfo.FullName)" -ForegroundColor Cyan
    } else {
        Write-Host "⚠ Executable not found in expected location" -ForegroundColor Yellow
    }
    
    # Qt DLLの配布
    if ($Deploy) {
        Write-Host ""
        Write-Host "=== Deployment ===" -ForegroundColor Yellow
        
        # windeployqtの検索
        $windeployqt = $null
        foreach ($qtPath in $qtPaths) {
            $windeployqtPath = Join-Path $qtPath "bin/windeployqt.exe"
            if (Test-Path $windeployqtPath) {
                $windeployqt = $windeployqtPath
                break
            }
        }
        
        if ($windeployqt -and (Test-Path $executable)) {
            Write-Host "Deploying Qt libraries..." -ForegroundColor Cyan
            & $windeployqt --verbose 2 --no-translations --no-system-d3d-compiler $executable
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✓ Qt libraries deployed successfully" -ForegroundColor Green
            } else {
                Write-Host "⚠ Qt deployment completed with warnings" -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠ windeployqt not found or executable missing" -ForegroundColor Yellow
        }
    }
    
} catch {
    Write-Host ""
    Write-Host "✗ Build failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

# ビルド時間の計算
$buildEndTime = Get-Date
$buildDuration = $buildEndTime - $buildStartTime

Write-Host ""
Write-Host "=== Build Summary ===" -ForegroundColor Green
Write-Host "Build type: $BuildType"
Write-Host "Build time: $($buildDuration.ToString('mm\:ss'))"
Write-Host "Build directory: $BuildDir"

if (Test-Path $executable) {
    Write-Host ""
    Write-Host "To run the application:" -ForegroundColor Cyan
    Write-Host "  $executable" -ForegroundColor Yellow
    
    if (-not $Deploy) {
        Write-Host ""
        Write-Host "To deploy Qt libraries, run:" -ForegroundColor Cyan
        Write-Host "  .\build.ps1 -Deploy" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Build completed successfully! 🎉" -ForegroundColor Green