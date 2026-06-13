#!/usr/bin/env bash
# Copy CUDA Toolkit + cuDNN runtime libraries next to the Linux binary so the
# folder is self-contained (mirrors what Windows bundles in bin/windows/).
#
# libcuda.so.1 is intentionally NOT copied: it must come from the NVIDIA
# driver on the host system.
#
# Usage:
#   bash scripts/bundle_cuda_linux.sh             # full bundle (~3 GB)
#   bash scripts/bundle_cuda_linux.sh --minimal   # skip ops VITS/BERT don't use (~1.7 GB)
set -euo pipefail

MODE="full"
for arg in "$@"; do
    case "$arg" in
        --minimal) MODE="minimal" ;;
        --full)    MODE="full"    ;;
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
    echo "error: ${DEST} does not exist; build the Linux binary first" >&2
    exit 1
fi

CANDIDATE_DIRS=(
    "/usr/local/cuda/targets/x86_64-linux/lib"
    "/usr/local/cuda/lib64"
    "/usr/lib/x86_64-linux-gnu"
)

# Always-needed runtime libs for ONNX Runtime CUDA EP with VITS / BERT models.
LIB_PATTERNS_REQUIRED=(
    "libcudart.so.12*"
    "libcublas.so.12*"
    "libcublasLt.so.12*"
    "libnvJitLink.so.12*"
    "libnvrtc.so.12*"
    "libcudnn.so.9*"
    "libcudnn_cnn.so.9*"
    "libcudnn_ops.so.9*"
    "libcudnn_graph.so.9*"
    "libcudnn_engines_runtime_compiled.so.9*"
    "libcudnn_engines_tensor_ir.so.9*"
)

# Optional libs included only in --full mode (large, rarely exercised by TTS).
LIB_PATTERNS_OPTIONAL=(
    "libcufft.so.11*"                   # FFT — not used by VITS/BERT
    "libcurand.so.10*"                  # RNG — TTS samples on CPU
    "libcudnn_adv.so.9*"                # RNN / cuDNN MHA — not used
    "libcudnn_engines_precompiled.so.9*"  # 493 MB precompiled kernel zoo
    "libcudnn_heuristic.so.9*"          # engine selector heuristics
    "libcudnn_ext.so.9*"                # extensions
)

LIB_PATTERNS=("${LIB_PATTERNS_REQUIRED[@]}")
if [[ "$MODE" == "full" ]]; then
    LIB_PATTERNS+=("${LIB_PATTERNS_OPTIONAL[@]}")
fi

echo "mode: ${MODE}"
echo "dest: ${DEST}"
echo

echo "cleaning previously bundled CUDA / cuDNN files in ${DEST}..."
shopt -s nullglob
for f in "${DEST}"/libcudart.so* "${DEST}"/libcublas*.so* \
         "${DEST}"/libcufft.so* "${DEST}"/libcurand.so* \
         "${DEST}"/libnvJitLink.so* "${DEST}"/libnvrtc.so* \
         "${DEST}"/libcudnn*.so*; do
    rm -f "$f"
done
shopt -u nullglob

copied=0
missing=()
for pattern in "${LIB_PATTERNS[@]}"; do
    found=""
    for dir in "${CANDIDATE_DIRS[@]}"; do
        [[ -d "${dir}" ]] || continue
        match=$(compgen -G "${dir}/${pattern}" || true)
        if [[ -n "${match}" ]]; then
            found="${dir}"
            for f in ${match}; do
                base=$(basename "${f}")
                if [[ -L "${f}" ]]; then
                    cp -P "${f}" "${DEST}/${base}"
                else
                    cp -f "${f}" "${DEST}/${base}"
                fi
                copied=$((copied + 1))
            done
            break
        fi
    done
    if [[ -z "${found}" ]]; then
        missing+=("${pattern}")
    fi
done

echo "copied ${copied} CUDA / cuDNN files into ${DEST}"
if (( ${#missing[@]} > 0 )); then
    echo "warning: not found on system:" >&2
    for p in "${missing[@]}"; do
        echo "  - ${p}" >&2
    done
    echo "install with: sudo apt-get install -y libcudnn9-cuda-12 cuda-toolkit-12-4 (or similar)" >&2
fi

echo
echo "verification:"
ldd "${DEST}/bert-vits2-project" 2>/dev/null | grep -E 'cuda|cudnn|cublas' || true
echo
echo "test (should print device: cuda):"
echo "  ./bin/linux/bert-vits2-project --server --device cuda --cuda-device 0"
