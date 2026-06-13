#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${ONNXRUNTIME_VERSION:-1.20.1}"
GPU="${ONNXRUNTIME_GPU:-0}"

if [[ "$GPU" == "1" ]]; then
    PACKAGE="onnxruntime-linux-x64-gpu-${VERSION}"
    DEST="${ROOT}/third_party/onnxruntime-gpu-linux"
else
    PACKAGE="onnxruntime-linux-x64-${VERSION}"
    DEST="${ROOT}/third_party/onnxruntime-linux"
fi

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PACKAGE}.tgz"
TMP="$(mktemp -d)"

cleanup() {
    rm -rf "$TMP"
}
trap cleanup EXIT

echo "Downloading ${URL}"
curl -fsSL "$URL" -o "${TMP}/onnxruntime.tgz"
rm -rf "$DEST"
mkdir -p "$DEST"
tar -xzf "${TMP}/onnxruntime.tgz" -C "$DEST" --strip-components=1

echo "Installed ONNX Runtime to ${DEST}"
echo "Configure with: cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=${DEST}"
