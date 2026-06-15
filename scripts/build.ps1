# Windows build for bert-vits2-project. Mirrors scripts/build_macos.sh /
# build_linux.sh: detect ONNX Runtime under third_party/, configure CMake,
# build Release. Run from project root or scripts/ - both work.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\build.ps1 -Cpu
#
# Parameters:
#   -Cpu          force the CPU ONNX Runtime even if the GPU package exists
#   -BuildDir     build directory (default: build)
#   -Config       CMake config (default: Release)

[CmdletBinding()]
param(
    [switch]$Cpu,
    [string]$BuildDir = "build",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root      = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $Root

Write-Host "[build] platform: Windows ($([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture))"

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "cmake not found in PATH. Install from https://cmake.org/download or via 'winget install Kitware.CMake'."
}

# ORT auto-detection mirrors cmake/OnnxRuntime.cmake: prefer GPU nuget, fall
# back to the CPU nuget. -Cpu forces the CPU package.
$Candidates = @()
if (-not $Cpu) {
    $Candidates += "third_party\onnxruntime-gpu-windows-nuget"
}
$Candidates += "third_party\onnxruntime-nuget"

$OrtRoot = $null
foreach ($rel in $Candidates) {
    $abs = Join-Path $Root $rel
    if (Test-Path $abs) {
        $OrtRoot = $abs
        break
    }
}

$CMakeArgs = @(
    "-B", $BuildDir,
    "-S", ".",
    "-DCMAKE_BUILD_TYPE=$Config"
)
if ($OrtRoot) {
    Write-Host "[build] ONNX Runtime: $OrtRoot"
    $CMakeArgs += "-DONNXRUNTIME_ROOT=$OrtRoot"
} else {
    Write-Host "[build] ONNX Runtime: (none found under third_party\, relying on CMake auto-detect)"
}

& cmake @CMakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

& cmake --build $BuildDir --config $Config -j
if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }

$OutDir = if (Test-Path (Join-Path $Root "bin\windows")) {
    Join-Path $Root "bin\windows"
} else {
    Join-Path $Root "build-cl\windows"
}
$Exe = Join-Path $OutDir "bert-vits2-project.exe"
if (Test-Path $Exe) {
    Write-Host "[build] windows build ok ($Exe)"
} else {
    Write-Warning "[build] expected $Exe but it was not produced."
}
