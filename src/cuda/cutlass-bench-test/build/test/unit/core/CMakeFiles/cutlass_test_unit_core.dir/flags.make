# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# compile CUDA with /home/tgrogers-raid/a/common/cuda-11.7/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -DCUTLASS_TARGET_NAME=\"cutlass_test_unit_core\"

CUDA_INCLUDES = --options-file CMakeFiles/cutlass_test_unit_core.dir/includes_CUDA.rsp

CUDA_FLAGS = -O3 -DNDEBUG --generate-code=arch=compute_75,code=[sm_75] --generate-code=arch=compute_75,code=[compute_75] -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -std=c++17

CXX_DEFINES = -DCUTLASS_TARGET_NAME=\"cutlass_test_unit_core\"

CXX_INCLUDES = -I/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/include -I/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/common -I/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/include -I/home/tgrogers-raid/a/common/cuda-11.7/include -I/include -I/examples -I/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/tools/util/include -isystem /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/_deps/googletest-src/googletest/include -isystem /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/_deps/googletest-src/googletest

CXX_FLAGS = -O3 -DNDEBUG -fPIE -std=c++1z

