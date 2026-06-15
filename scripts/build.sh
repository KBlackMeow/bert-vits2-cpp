#!/usr/bin/env bash
# Cross-platform build entry point. Detects the host OS and dispatches to the
# matching per-platform build script. Forwards any extra arguments to the
# delegate (e.g. `bash scripts/build.sh --gpu` -> build_linux.sh --gpu).
#
# Recognised platforms:
#   - Darwin (macOS, Apple Silicon / x86_64)  -> scripts/build_macos.sh
#   - Linux                                    -> scripts/build_linux.sh
#   - Windows via Git Bash / MSYS / Cygwin     -> scripts/build.ps1 (PowerShell)
#
# Environment variables forwarded to the delegate:
#   ONNXRUNTIME_GPU=1   build the CUDA flavour on Linux
#   ONNXRUNTIME_VERSION pin a specific ONNX Runtime release
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uname_s="$(uname -s 2>/dev/null || echo unknown)"

case "${uname_s}" in
    Darwin)
        echo "[build] platform: macOS ($(uname -m))"
        exec bash "${ROOT}/scripts/build_macos.sh" "$@"
        ;;
    Linux)
        echo "[build] platform: Linux ($(uname -m))"
        exec bash "${ROOT}/scripts/build_linux.sh" "$@"
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        echo "[build] platform: Windows (${uname_s})"
        if command -v powershell.exe >/dev/null 2>&1; then
            exec powershell.exe -NoProfile -ExecutionPolicy Bypass \
                -File "${ROOT}/scripts/build.ps1" "$@"
        elif command -v pwsh >/dev/null 2>&1; then
            exec pwsh -NoProfile -ExecutionPolicy Bypass \
                -File "${ROOT}/scripts/build.ps1" "$@"
        else
            echo "PowerShell not found. Run scripts\\build.ps1 from a Windows shell." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unsupported platform: ${uname_s}" >&2
        echo "Supported: Darwin (macOS), Linux, Windows (Git Bash/MSYS/Cygwin)." >&2
        exit 1
        ;;
esac
