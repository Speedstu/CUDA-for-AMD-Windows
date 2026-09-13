# Generic example: manually link CUDA-enabled LibTorch without requiring CMake to
# discover a full NVIDIA GPU/toolkit at runtime. Adapt library names/paths to your
# LibTorch release.

set(MANUAL_LIBTORCH_ROOT "" CACHE PATH "Path to libtorch root")
set(CUDA_HEADER_ROOT "" CACHE PATH "Path containing cuda_runtime_api.h")

if(NOT MANUAL_LIBTORCH_ROOT)
  message(FATAL_ERROR "Set -DMANUAL_LIBTORCH_ROOT=<libtorch>")
endif()

set(TORCH_INCLUDE_DIRS
  "${MANUAL_LIBTORCH_ROOT}/include"
  "${MANUAL_LIBTORCH_ROOT}/include/torch/csrc/api/include"
)

if(CUDA_HEADER_ROOT)
  list(APPEND TORCH_INCLUDE_DIRS "${CUDA_HEADER_ROOT}")
endif()

# Example target supplied by the parent project.
target_include_directories(my_cuda_facing_app PRIVATE ${TORCH_INCLUDE_DIRS})

set(TORCH_LIB "${MANUAL_LIBTORCH_ROOT}/lib")
target_link_libraries(my_cuda_facing_app PRIVATE
  "${TORCH_LIB}/c10.lib"
  "${TORCH_LIB}/torch.lib"
  "${TORCH_LIB}/torch_cpu.lib"
  "${TORCH_LIB}/c10_cuda.lib"
  "${TORCH_LIB}/torch_cuda.lib"
)

# Runtime DLLs still need to be placed on PATH / beside the executable.
# At runtime this repo stages ZLUDA's nvcuda.dll and the AMD-backed compatibility
# layer; the application remains CUDA-facing.
