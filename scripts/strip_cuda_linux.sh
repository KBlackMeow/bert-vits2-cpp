#!/usr/bin/env bash
# Inverse of scripts/bundle_cuda_linux.sh.
#
# Removes CUDA Toolkit + cuDNN runtime libraries that were copied next to the
# Linux binary, so bin/linux/ stops shipping ~1.7-3 GB of bundled CUDA.
#
# By default keeps:
#   - bert-vits2-project (the executable)
#   - libonnxruntime.so* (ONNX Runtime main shared library)
#   - libonnxruntime_providers_shared.so
#   - libonnxruntime_providers_cuda.so   (~694 MB; lets the host system's
#                                         CUDA Toolkit + cuDNN drive --device cuda)
#
# Use --drop-ort-cuda to also remove libonnxruntime_providers_cuda.so and
# libonnxruntime_providers_tensorrt.so for a fully CPU-only deployment.
#
# Usage:
#   bash scripts/strip_cuda_linux.sh                 # keep ORT CUDA provider
#   bash scripts/strip_cuda_linux.sh --drop-ort-cuda # also drop ORT CUDA/TensorRT providers
set -euo pipefail

DROP_ORT_CUDA=0
for arg in "$@"; do
    case "$arg" in
        --drop-ort-cuda) DROP_ORT_CUDA=1 ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail/p' "$0" | sed '$d'
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "${ROOT}/bin" ]]; then
    DEST="${ROOT}/bin/linux"
else
    DEST="${ROOT}/build-cl/linux"
fi

if [[ ! -d "${DEST}" ]]; then
    echo "error: ${DEST} does not exist; nothing to strip" >&2
    exit 1
fi

echo "dest: ${DEST}"
echo "drop ORT CUDA/TensorRT providers: ${DROP_ORT_CUDA}"
echo

# Patterns covering everything bundle_cuda_linux.sh may have copied (full or
# minimal mode), matching the clean step in that script.
TOOLKIT_PATTERNS=(
    "libcudart.so*"
    "libcublas*.so*"
    "libcufft.so*"
    "libcurand.so*"
    "libnvJitLink.so*"
    "libnvrtc.so*"
    "libcudnn*.so*"
)

removed=0
shopt -s nullglob
for pattern in "${TOOLKIT_PATTERNS[@]}"; do
    for f in "${DEST}"/${pattern}; do
        rm -f "$f"
        echo "  - removed $(basename "$f")"
        removed=$((removed + 1))
    done
done

if [[ "${DROP_ORT_CUDA}" == "1" ]]; then
    for f in "${DEST}"/libonnxruntime_providers_cuda.so \
             "${DEST}"/libonnxruntime_providers_tensorrt.so; do
        if [[ -e "$f" ]]; then
            rm -f "$f"
            echo "  - removed $(basename "$f")"
            removed=$((removed + 1))
        fi
    done
fi
shopt -u nullglob

echo
echo "removed ${removed} file(s) from ${DEST}"
echo
echo "remaining contents:"
ls -lh "${DEST}" | sed 's/^/  /'

cat <<'EOF'

next steps:
  - To restore the full CUDA bundle:
      bash scripts/bundle_cuda_linux.sh           # full (~3 GB)
      bash scripts/bundle_cuda_linux.sh --minimal # minimal (~1.9 GB)
  - To restore ORT CUDA provider after --drop-ort-cuda, re-run the Linux build
    with the GPU ONNX Runtime package, e.g.:
      ONNXRUNTIME_GPU=1 bash scripts/build_linux.sh
EOF
