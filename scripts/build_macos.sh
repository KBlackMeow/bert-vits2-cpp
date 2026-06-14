#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! command -v cmake >/dev/null; then
    echo "cmake not found. Install with: brew install cmake ninja" >&2
    exit 1
fi

ARCH="$(uname -m)"
case "${ARCH}" in
    arm64|aarch64) ORT_DIR="${ROOT}/third_party/onnxruntime-osx-arm64" ;;
    x86_64)        ORT_DIR="${ROOT}/third_party/onnxruntime-osx-x86_64" ;;
    *) echo "Unsupported macOS architecture: ${ARCH}" >&2; exit 1 ;;
esac

if [[ ! -d "${ORT_DIR}" ]]; then
    bash scripts/fetch_onnxruntime_macos.sh
fi

GENERATOR=()
if command -v ninja >/dev/null; then
    GENERATOR=(-G Ninja)
fi

cmake -B build-macos -S . "${GENERATOR[@]}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3 \
    -DONNXRUNTIME_ROOT="${ORT_DIR}"
cmake --build build-macos -j"$(sysctl -n hw.ncpu)"

if [[ -d "${ROOT}/bin" ]]; then
    OUTPUT_DIR="${ROOT}/bin/macos"
else
    OUTPUT_DIR="${ROOT}/build-cl/macos"
fi

test -f "${OUTPUT_DIR}/libonnxruntime.dylib"
test -f "${OUTPUT_DIR}/bert-vits2-project"
"${ROOT}/build/tests/test_jp_frontend" >/dev/null
echo "macOS build ok (${OUTPUT_DIR})"
