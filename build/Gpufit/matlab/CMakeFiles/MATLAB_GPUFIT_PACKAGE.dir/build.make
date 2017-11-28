# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build

# Utility rule file for MATLAB_GPUFIT_PACKAGE.

# Include the progress variables for this target.
include Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/progress.make

Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Creating Gpufit Matlab package"
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab && /usr/local/bin/cmake -E remove_directory /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/./matlab
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab && /usr/local/bin/cmake -E make_directory /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/./matlab
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab && /usr/local/bin/cmake -E copy_if_different /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/matlab/EstimatorID.m /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/matlab/gpufit.m /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/matlab/gpufit_cuda_available.m /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/matlab/gpufit_version.m /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/matlab/ModelID.m /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/matlab/README.txt /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/./matlab
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab && /usr/local/bin/cmake -E copy_if_different /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/libGpufit.so /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/./matlab
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab && /usr/local/bin/cmake -E copy_if_different /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab/libGpufitMex.mexa64 /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/./matlab
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab && /usr/local/bin/cmake -E copy_if_different /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab/libGpufitCudaAvailableMex.mexa64 /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/./matlab

MATLAB_GPUFIT_PACKAGE: Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE
MATLAB_GPUFIT_PACKAGE: Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/build.make

.PHONY : MATLAB_GPUFIT_PACKAGE

# Rule to build all files generated by this target.
Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/build: MATLAB_GPUFIT_PACKAGE

.PHONY : Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/build

Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/clean:
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab && $(CMAKE_COMMAND) -P CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/cmake_clean.cmake
.PHONY : Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/clean

Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/depend:
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/matlab /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Gpufit/matlab/CMakeFiles/MATLAB_GPUFIT_PACKAGE.dir/depend

