# ShioRIS3 Windows Build Script (PowerShell)
# Place this file in the project root directory and execute

param(
    [string]$Configuration = "Release",
    [switch]$Clean = $false,
    [switch]$Run = $false
)

Write-Host "===========================================" -ForegroundColor Green
Write-Host "ShioRIS3 Windows Build Script (PowerShell)" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green

# Check environment variables
Write-Host "Checking environment variables..." -ForegroundColor Yellow

if (-not $env:Qt6_DIR) {
    Write-Host "ERROR: Qt6_DIR environment variable is not set" -ForegroundColor Red
    Write-Host "Example: `$env:Qt6_DIR = 'C:\Qt\6.8.0\msvc2022_64\lib\cmake\Qt6'" -ForegroundColor Yellow
    Read-Host "Press any key to continue"
    exit 1
}

if (-not $env:VCPKG_ROOT) {
    Write-Host "WARNING: VCPKG_ROOT environment variable is not set" -ForegroundColor Yellow
    Write-Host "Please set it if you want to use vcpkg" -ForegroundColor Yellow
    Write-Host "Example: `$env:VCPKG_ROOT = 'C:\vcpkg'" -ForegroundColor Yellow
}

Write-Host "Qt6_DIR: $env:Qt6_DIR" -ForegroundColor Green
Write-Host "VCPKG_ROOT: $env:VCPKG_ROOT" -ForegroundColor Green
Write-Host ""

# Setup Visual Studio environment
Write-Host "Setting up Visual Studio environment..." -ForegroundColor Yellow

$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vsPath)) {
    Write-Host "ERROR: Visual Studio 2022 not found" -ForegroundColor Red
    Write-Host "Path: $vsPath" -ForegroundColor Red
    Read-Host "Press any key to continue"
    exit 1
}

# Handle build directory
Write-Host "Processing build directory..." -ForegroundColor Yellow

if ($Clean -and (Test-Path "build")) {
    Write-Host "Removing existing build directory..." -ForegroundColor Yellow
    Remove-Item -Path "build" -Recurse -Force
}

if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Name "build" | Out-Null
}

Set-Location "build"

# Execute CMake
Write-Host "Configuring with CMake..." -ForegroundColor Yellow

$cmakeArgs = @(
    ".."
    "-G", "Visual Studio 17 2022"
    "-A", "x64"
    "-DCMAKE_BUILD_TYPE=$Configuration"
    "-DQt6_DIR=$env:Qt6_DIR"
)

if ($env:VCPKG_ROOT) {
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
}

try {
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
} catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Set-Location ..
    Read-Host "Press any key to continue"
    exit 1
}

# Execute build
Write-Host "Building..." -ForegroundColor Yellow

try {
    & cmake --build . --config $Configuration --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
} catch {
    Write-Host "ERROR: $_" -ForegroundColor Red
    Set-Location ..
    Read-Host "Press any key to continue"
    exit 1
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Green
Write-Host "Build completed successfully!" -ForegroundColor Green
Write-Host "Executable: build\$Configuration\ShioRIS3.exe" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green

# Check executable
$exePath = "$Configuration\ShioRIS3.exe"
if (Test-Path $exePath) {
    Write-Host "Executable created successfully" -ForegroundColor Green
    
    if ($Run) {
        Write-Host "Starting application..." -ForegroundColor Yellow
        Set-Location $Configuration
        & ".\ShioRIS3.exe"
        Set-Location ..
    } else {
        Write-Host ""
        $choice = Read-Host "Do you want to run the application? (Y/N)"
        if ($choice -eq "Y" -or $choice -eq "y") {
            Set-Location $Configuration
            & ".\ShioRIS3.exe"
            Set-Location ..
        }
    }
} else {
    Write-Host "WARNING: Executable not found" -ForegroundColor Yellow
}

Set-Location ..
Write-Host "Script completed" -ForegroundColor Green