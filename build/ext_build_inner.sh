#!/bin/bash
# runs INSIDE v41build (adapted from tonyd2wild build/build_stable_ext.sh): stable-only configure + build _C_stable_libtorch for sm_121a
set -o pipefail
export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=16
cd /src
echo "=== $(date -u +%FT%TZ) configure ==="
cp -n CMakeLists.txt CMakeLists.txt.orig; cp CMakeLists.txt.orig CMakeLists.txt
sed -i -E "s|^(\s*)include\(cmake/external_projects/|\1# STABLE-ONLY BUILD: include(cmake/external_projects/|" CMakeLists.txt
echo "stable-only includes commented: $(grep -c 'STABLE-ONLY BUILD' CMakeLists.txt)"
rm -rf /src/build/CMakeCache.txt /src/build/CMakeFiles
PYPATH=$(python3 -c "import sys;print(':'.join(p for p in sys.path if p))")
TORCH_PREFIX=$(python3 -c "import torch;print(torch.utils.cmake_prefix_path)")
NVRTC=$(ls /usr/local/cuda/lib64/libnvrtc.so /usr/local/cuda/lib64/libnvrtc.so.* /usr/local/lib/python3.12/dist-packages/nvidia/*/lib/libnvrtc.so* /usr/lib/aarch64-linux-gnu/libnvrtc.so* 2>/dev/null | head -1)
echo "nvrtc=$NVRTC torch_prefix=$TORCH_PREFIX"
cmake -S /src -B /src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE=$(which python3) -DVLLM_PYTHON_PATH="$PYPATH" \
  -DFETCHCONTENT_BASE_DIR=/src/build/_deps -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/src/build/_deps/cutlass-src \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DNVCC_THREADS=2 -DCUDA_nvrtc_LIBRARY="$NVRTC" 2>&1 | tail -20
rc=${PIPESTATUS[0]}; echo "configure rc=$rc"; [ $rc -ne 0 ] && { echo "CONFIGURE FAILED"; echo "BUILD_EXIT=2"; exit 2; }
echo "=== $(date -u +%FT%TZ) build _C_stable_libtorch (-j16, NVCC_THREADS=2) ==="
cmake --build /src/build --target _C_stable_libtorch -j 16 2>&1 | grep -E --line-buffered "^\[[0-9]+/[0-9]+\]|error|Error|FAILED|Linking" | awk 'NR%25==1 || /error|Error|FAILED|Linking/ {print; fflush()}'
rc=${PIPESTATUS[0]}; echo "build rc=$rc"
find /src/build -maxdepth 2 -name "_C_stable_libtorch*.so" -exec ls -la {} \;
echo "=== $(date -u +%FT%TZ) done ==="; echo "BUILD_EXIT=$rc"
