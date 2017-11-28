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
include Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/depend.make

# Include the progress variables for this target.
include Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/progress.make

# Include the compile flags for this target's objects.
include Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/flags.make

Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o: Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/flags.make
Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o: ../Gpufit/examples/KineticModel_fitExample.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o"
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o -c /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/examples/KineticModel_fitExample.cpp

Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.i"
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/examples/KineticModel_fitExample.cpp > CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.i

Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.s"
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/examples/KineticModel_fitExample.cpp -o CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.s

Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o.requires:

.PHONY : Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o.requires

Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o.provides: Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o.requires
	$(MAKE) -f Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/build.make Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o.provides.build
.PHONY : Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o.provides

Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o.provides.build: Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o


# Object files for target KineticModel_fitExample
KineticModel_fitExample_OBJECTS = \
"CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o"

# External object files for target KineticModel_fitExample
KineticModel_fitExample_EXTERNAL_OBJECTS =

KineticModel_fitExample: Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o
KineticModel_fitExample: Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/build.make
KineticModel_fitExample: Gpufit/libGpufit.so
KineticModel_fitExample: /usr/local/cuda-6.5/lib64/libcudart_static.a
KineticModel_fitExample: /usr/lib/x86_64-linux-gnu/librt.so
KineticModel_fitExample: Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../KineticModel_fitExample"
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/KineticModel_fitExample.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/build: KineticModel_fitExample

.PHONY : Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/build

Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/requires: Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/KineticModel_fitExample.cpp.o.requires

.PHONY : Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/requires

Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/clean:
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples && $(CMAKE_COMMAND) -P CMakeFiles/KineticModel_fitExample.dir/cmake_clean.cmake
.PHONY : Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/clean

Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/depend:
	cd /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/Gpufit/examples /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples /media/MAXTOR/___backup/home/mscipio/Scrivania/GPUFIT_SRC/Gpufit/build/Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : Gpufit/examples/CMakeFiles/KineticModel_fitExample.dir/depend

