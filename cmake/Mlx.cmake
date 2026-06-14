# Fetch and build Apple's MLX for native Apple-Silicon GPU/ANE inference.
# Used by the bv2-mlx C++ runtime as an alternative to the ONNX-Runtime
# CoreML execution provider.
#
# Set BV2_MLX_TAG to pin a specific MLX release/tag.

include(FetchContent)
# MLX's CMakeLists.txt invokes `install(...)` with destinations that come from
# `GNUInstallDirs` (CMAKE_INSTALL_LIBDIR / INCLUDEDIR / BINDIR ...) AND
# `configure_package_config_file()` which asserts `CMAKE_INSTALL_LIBDIR` is
# defined. We don't actually `make install` MLX, but CMake still parses every
# install rule at configure time so the variables must exist - and must be
# non-empty. Force GNU defaults in regardless of any stale cache state from
# a previous failed configure (where these were sometimes left as "").
include(GNUInstallDirs)
set(_bv2_mlx_installdir_LIBDIR      "lib")
set(_bv2_mlx_installdir_INCLUDEDIR  "include")
set(_bv2_mlx_installdir_BINDIR      "bin")
set(_bv2_mlx_installdir_DATAROOTDIR "share")
foreach(_dir LIBDIR INCLUDEDIR BINDIR DATAROOTDIR)
    if(NOT CMAKE_INSTALL_${_dir})
        set(CMAKE_INSTALL_${_dir} "${_bv2_mlx_installdir_${_dir}}"
            CACHE PATH "GNUInstallDirs CMAKE_INSTALL_${_dir} (forced by MLX)"
            FORCE)
    endif()
endforeach()

set(BV2_MLX_TAG "v0.21.0" CACHE STRING "Apple MLX git tag to fetch")

# MLX is Apple-Silicon only (Metal backend). Don't try to build it on
# Windows / Linux / x86 macOS - the BV2 runtime falls back to the ONNX
# CoreML / CUDA / CPU backends there.
if(NOT APPLE)
    set(BERT_VITS2_MLX_AVAILABLE OFF CACHE INTERNAL "MLX available")
    return()
endif()

if(NOT CMAKE_OSX_ARCHITECTURES STREQUAL "" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    message(STATUS "MLX disabled (CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES})")
    set(BERT_VITS2_MLX_AVAILABLE OFF CACHE INTERNAL "MLX available")
    return()
endif()
if(CMAKE_HOST_SYSTEM_PROCESSOR AND NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "MLX disabled (host arch ${CMAKE_HOST_SYSTEM_PROCESSOR})")
    set(BERT_VITS2_MLX_AVAILABLE OFF CACHE INTERNAL "MLX available")
    return()
endif()

# MLX builds against Apple Metal-cpp; FetchContent it.
FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG        ${BV2_MLX_TAG}
    GIT_SHALLOW    TRUE
)

# Disable MLX's own examples / python bindings; we only want the C++ library.
set(MLX_BUILD_TESTS      OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES   OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_METAL      ON  CACHE BOOL "" FORCE)
set(MLX_BUILD_CPU        ON  CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(mlx)

set(BERT_VITS2_MLX_AVAILABLE ON  CACHE INTERNAL "MLX available")
set(BERT_VITS2_MLX_TARGET    mlx CACHE INTERNAL "MLX cmake target")
