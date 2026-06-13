#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v cmake >/dev/null; then
    CMAKE_DIR="/tmp/cmake-3.28.3-linux-x86_64"
    if [[ ! -x "${CMAKE_DIR}/bin/cmake" ]]; then
        curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz \
            | tar xz -C /tmp
    fi
    export PATH="${CMAKE_DIR}/bin:${PATH}"
fi

ONNXRUNTIME_GPU="${ONNXRUNTIME_GPU:-0}"
if [[ "${ONNXRUNTIME_GPU}" == "1" ]]; then
    ORT_DIR="${ROOT}/third_party/onnxruntime-gpu-linux"
else
    ORT_DIR="${ROOT}/third_party/onnxruntime-linux"
fi

if [[ ! -d "${ORT_DIR}" ]]; then
    ONNXRUNTIME_GPU="${ONNXRUNTIME_GPU}" bash scripts/fetch_onnxruntime_linux.sh
fi

cmake -B build-linux -S . -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="${ORT_DIR}"
cmake --build build-linux -j"$(nproc)"

if [[ -d "${ROOT}/bin" ]]; then
    OUTPUT_DIR="${ROOT}/bin/linux"
else
    OUTPUT_DIR="${ROOT}/build-cl/linux"
fi

test -f "${OUTPUT_DIR}/libonnxruntime.so"
test -f "${OUTPUT_DIR}/bert-vits2-project"
./build/tests/test_jp_frontend
echo "linux build ok (${OUTPUT_DIR})"
