# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build

# Utility rule file for cutlass_test_unit_gemm_device.

# Include any custom commands dependencies for this target.
include test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/compiler_depend.make

# Include the progress variables for this target.
include test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/progress.make

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_simt
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm70
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm75
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_f16_sm80
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_f32_sm80
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_f32_tf32_sm80
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_f64
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_s32_sm80
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_wmma
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_planar_complex
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_sparse_tensorop_sm80
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemv_device
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemv_device_strided_batched
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_cluster_multicast_sm90

cutlass_test_unit_gemm_device: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device
cutlass_test_unit_gemm_device: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/build.make
.PHONY : cutlass_test_unit_gemm_device

# Rule to build all files generated by this target.
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/build: cutlass_test_unit_gemm_device
.PHONY : test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/build

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/clean:
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && $(CMAKE_COMMAND) -P CMakeFiles/cutlass_test_unit_gemm_device.dir/cmake_clean.cmake
.PHONY : test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/clean

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/depend:
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device.dir/depend

