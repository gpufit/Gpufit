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

# Include any dependencies generated for this target.
include Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/depend.make

# Include the progress variables for this target.
include Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/progress.make

# Include the compile flags for this target's objects.
include Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/flags.make

Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o: Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/flags.make
Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o: ../Gpufit/examples/Linear_Regression_Example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o"
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o -c /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/examples/Linear_Regression_Example.cpp

Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.i"
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/examples/Linear_Regression_Example.cpp > CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.i

Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.s"
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/examples/Linear_Regression_Example.cpp -o CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.s

Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o.requires:

.PHONY : Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o.requires

Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o.provides: Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o.requires
	$(MAKE) -f Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/build.make Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o.provides.build
.PHONY : Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o.provides

Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o.provides.build: Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o


# Object files for target Linear_Regression_Example
Linear_Regression_Example_OBJECTS = \
"CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o"

# External object files for target Linear_Regression_Example
Linear_Regression_Example_EXTERNAL_OBJECTS =

Linear_Regression_Example: Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o
Linear_Regression_Example: Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/build.make
Linear_Regression_Example: Gpufit/libGpufit.so
Linear_Regression_Example: /usr/local/cuda-6.5/lib64/libcudart_static.a
Linear_Regression_Example: /usr/lib/x86_64-linux-gnu/librt.so
Linear_Regression_Example: Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../Linear_Regression_Example"
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Linear_Regression_Example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/build: Linear_Regression_Example

.PHONY : Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/build

Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/requires: Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/Linear_Regression_Example.cpp.o.requires

.PHONY : Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/requires

Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/clean:
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && $(CMAKE_COMMAND) -P CMakeFiles/Linear_Regression_Example.dir/cmake_clean.cmake
.PHONY : Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/clean

Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/depend:
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/examples /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Gpufit/examples/CMakeFiles/Linear_Regression_Example.dir/depend

