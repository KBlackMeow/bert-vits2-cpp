#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${ONNXRUNTIME_VERSION:-1.20.1}"
ARCH="${ARCH:-$(uname -m)}"

case "${ARCH}" in
    arm64|aarch64)
        PACKAGE="onnxruntime-osx-arm64-${VERSION}"
        DEST="${ROOT}/third_party/onnxruntime-osx-arm64"
        ;;
    x86_64)
        PACKAGE="onnxruntime-osx-x86_64-${VERSION}"
        DEST="${ROOT}/third_party/onnxruntime-osx-x86_64"
        ;;
    *)
        echo "Unsupported macOS architecture: ${ARCH}" >&2
        exit 1
        ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${PACKAGE}.tgz"
TMP="$(mktemp -d)"
cleanup() { rm -rf "${TMP}"; }
trap cleanup EXIT

echo "Downloading ${URL}"
curl -fsSL "${URL}" -o "${TMP}/onnxruntime.tgz"
rm -rf "${DEST}"
mkdir -p "${DEST}"
tar -xzf "${TMP}/onnxruntime.tgz" -C "${DEST}" --strip-components=1

echo "Installed ONNX Runtime to ${DEST}"
echo "CoreML (MLX) EP is bundled inside libonnxruntime.dylib for macOS."
echo "Configure with: cmake -B build-macos -S . -G Ninja -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=${DEST}"
