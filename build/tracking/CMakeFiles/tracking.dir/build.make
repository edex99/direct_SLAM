# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.4

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
CMAKE_SOURCE_DIR = /home/dexheimere/thesis_ws/src/tracking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dexheimere/thesis_ws/build/tracking

# Include any dependencies generated for this target.
include CMakeFiles/tracking.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/tracking.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tracking.dir/flags.make

CMakeFiles/tracking.dir/src/main.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/main.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tracking.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/main.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/main.cpp

CMakeFiles/tracking.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/main.cpp > CMakeFiles/tracking.dir/src/main.cpp.i

CMakeFiles/tracking.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/main.cpp -o CMakeFiles/tracking.dir/src/main.cpp.s

CMakeFiles/tracking.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/main.cpp.o.requires

CMakeFiles/tracking.dir/src/main.cpp.o.provides: CMakeFiles/tracking.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/main.cpp.o.provides

CMakeFiles/tracking.dir/src/main.cpp.o.provides.build: CMakeFiles/tracking.dir/src/main.cpp.o


CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/LieAlgebra.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/LieAlgebra.cpp

CMakeFiles/tracking.dir/src/LieAlgebra.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/LieAlgebra.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/LieAlgebra.cpp > CMakeFiles/tracking.dir/src/LieAlgebra.cpp.i

CMakeFiles/tracking.dir/src/LieAlgebra.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/LieAlgebra.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/LieAlgebra.cpp -o CMakeFiles/tracking.dir/src/LieAlgebra.cpp.s

CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o.requires

CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o.provides: CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o.provides

CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o.provides.build: CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o


CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/SparseAlignment.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/SparseAlignment.cpp

CMakeFiles/tracking.dir/src/SparseAlignment.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/SparseAlignment.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/SparseAlignment.cpp > CMakeFiles/tracking.dir/src/SparseAlignment.cpp.i

CMakeFiles/tracking.dir/src/SparseAlignment.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/SparseAlignment.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/SparseAlignment.cpp -o CMakeFiles/tracking.dir/src/SparseAlignment.cpp.s

CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o.requires

CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o.provides: CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o.provides

CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o.provides.build: CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o


CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/RGDBSimulator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/RGDBSimulator.cpp

CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/RGDBSimulator.cpp > CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.i

CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/RGDBSimulator.cpp -o CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.s

CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o.requires

CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o.provides: CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o.provides

CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o.provides.build: CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o


CMakeFiles/tracking.dir/src/PixelSelector.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/PixelSelector.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/PixelSelector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/tracking.dir/src/PixelSelector.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/PixelSelector.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/PixelSelector.cpp

CMakeFiles/tracking.dir/src/PixelSelector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/PixelSelector.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/PixelSelector.cpp > CMakeFiles/tracking.dir/src/PixelSelector.cpp.i

CMakeFiles/tracking.dir/src/PixelSelector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/PixelSelector.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/PixelSelector.cpp -o CMakeFiles/tracking.dir/src/PixelSelector.cpp.s

CMakeFiles/tracking.dir/src/PixelSelector.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/PixelSelector.cpp.o.requires

CMakeFiles/tracking.dir/src/PixelSelector.cpp.o.provides: CMakeFiles/tracking.dir/src/PixelSelector.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/PixelSelector.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/PixelSelector.cpp.o.provides

CMakeFiles/tracking.dir/src/PixelSelector.cpp.o.provides.build: CMakeFiles/tracking.dir/src/PixelSelector.cpp.o


CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/ImageProcessing.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/ImageProcessing.cpp

CMakeFiles/tracking.dir/src/ImageProcessing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/ImageProcessing.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/ImageProcessing.cpp > CMakeFiles/tracking.dir/src/ImageProcessing.cpp.i

CMakeFiles/tracking.dir/src/ImageProcessing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/ImageProcessing.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/ImageProcessing.cpp -o CMakeFiles/tracking.dir/src/ImageProcessing.cpp.s

CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o.requires

CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o.provides: CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o.provides

CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o.provides.build: CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o


CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/MatToPcConverter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/MatToPcConverter.cpp

CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/MatToPcConverter.cpp > CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.i

CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/MatToPcConverter.cpp -o CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.s

CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o.requires

CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o.provides: CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o.provides

CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o.provides.build: CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o


CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/PointCloudMap.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/PointCloudMap.cpp

CMakeFiles/tracking.dir/src/PointCloudMap.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/PointCloudMap.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/PointCloudMap.cpp > CMakeFiles/tracking.dir/src/PointCloudMap.cpp.i

CMakeFiles/tracking.dir/src/PointCloudMap.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/PointCloudMap.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/PointCloudMap.cpp -o CMakeFiles/tracking.dir/src/PointCloudMap.cpp.s

CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o.requires

CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o.provides: CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o.provides

CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o.provides.build: CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o


CMakeFiles/tracking.dir/src/DepthFilter.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/DepthFilter.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/DepthFilter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/tracking.dir/src/DepthFilter.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/DepthFilter.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/DepthFilter.cpp

CMakeFiles/tracking.dir/src/DepthFilter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/DepthFilter.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/DepthFilter.cpp > CMakeFiles/tracking.dir/src/DepthFilter.cpp.i

CMakeFiles/tracking.dir/src/DepthFilter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/DepthFilter.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/DepthFilter.cpp -o CMakeFiles/tracking.dir/src/DepthFilter.cpp.s

CMakeFiles/tracking.dir/src/DepthFilter.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/DepthFilter.cpp.o.requires

CMakeFiles/tracking.dir/src/DepthFilter.cpp.o.provides: CMakeFiles/tracking.dir/src/DepthFilter.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/DepthFilter.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/DepthFilter.cpp.o.provides

CMakeFiles/tracking.dir/src/DepthFilter.cpp.o.provides.build: CMakeFiles/tracking.dir/src/DepthFilter.cpp.o


CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o: CMakeFiles/tracking.dir/flags.make
CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o: /home/dexheimere/thesis_ws/src/tracking/src/FeatureDetector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o -c /home/dexheimere/thesis_ws/src/tracking/src/FeatureDetector.cpp

CMakeFiles/tracking.dir/src/FeatureDetector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/tracking.dir/src/FeatureDetector.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dexheimere/thesis_ws/src/tracking/src/FeatureDetector.cpp > CMakeFiles/tracking.dir/src/FeatureDetector.cpp.i

CMakeFiles/tracking.dir/src/FeatureDetector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/tracking.dir/src/FeatureDetector.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dexheimere/thesis_ws/src/tracking/src/FeatureDetector.cpp -o CMakeFiles/tracking.dir/src/FeatureDetector.cpp.s

CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o.requires:

.PHONY : CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o.requires

CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o.provides: CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o.requires
	$(MAKE) -f CMakeFiles/tracking.dir/build.make CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o.provides.build
.PHONY : CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o.provides

CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o.provides.build: CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o


# Object files for target tracking
tracking_OBJECTS = \
"CMakeFiles/tracking.dir/src/main.cpp.o" \
"CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o" \
"CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o" \
"CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o" \
"CMakeFiles/tracking.dir/src/PixelSelector.cpp.o" \
"CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o" \
"CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o" \
"CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o" \
"CMakeFiles/tracking.dir/src/DepthFilter.cpp.o" \
"CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o"

# External object files for target tracking
tracking_EXTERNAL_OBJECTS =

/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/main.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/PixelSelector.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/DepthFilter.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/build.make
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/libroscpp.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/librosconsole.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/librosconsole_log4cxx.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/librosconsole_backend_interface.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/liblog4cxx.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/libxmlrpcpp.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/libroscpp_serialization.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/librostime.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/libcpp_common.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_xfeatures2d.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_calib3d.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_objdetect.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_common.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_octree.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libOpenNI.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_io.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_kdtree.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_search.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_sample_consensus.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_filters.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_features.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_keypoints.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_segmentation.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_visualization.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_outofcore.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_registration.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_recognition.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libqhull.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_surface.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_people.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_tracking.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_apps.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libqhull.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libOpenNI.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkCharts.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libDBoW2.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/libroscpp.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_signals.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/librosconsole.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/librosconsole_log4cxx.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/librosconsole_backend_interface.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/liblog4cxx.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_regex.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/libxmlrpcpp.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/libroscpp_serialization.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/librostime.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /opt/ros/indigo/lib/libcpp_common.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_system.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_thread.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libpthread.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_common.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_octree.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libOpenNI.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_io.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_kdtree.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_search.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_sample_consensus.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_filters.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_features.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_keypoints.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_segmentation.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_visualization.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_outofcore.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_registration.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_recognition.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/x86_64-linux-gnu/libqhull.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_surface.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_people.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_tracking.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libpcl_apps.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libDBoW2.so
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_features2d.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_flann.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_shape.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_video.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_highgui.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_videoio.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_imgproc.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_ml.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_core.so.3.0.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/lib/libopencv_hal.a
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkViews.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkInfovis.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkWidgets.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkHybrid.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkParallel.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkVolumeRendering.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkRendering.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkGraphics.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkImaging.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkIO.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkFiltering.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtkCommon.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: /usr/lib/libvtksys.so.5.8.0
/home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking: CMakeFiles/tracking.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dexheimere/thesis_ws/build/tracking/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Linking CXX executable /home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tracking.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tracking.dir/build: /home/dexheimere/thesis_ws/devel/.private/tracking/lib/tracking/tracking

.PHONY : CMakeFiles/tracking.dir/build

CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/main.cpp.o.requires
CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/LieAlgebra.cpp.o.requires
CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/SparseAlignment.cpp.o.requires
CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/RGDBSimulator.cpp.o.requires
CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/PixelSelector.cpp.o.requires
CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/ImageProcessing.cpp.o.requires
CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/MatToPcConverter.cpp.o.requires
CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/PointCloudMap.cpp.o.requires
CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/DepthFilter.cpp.o.requires
CMakeFiles/tracking.dir/requires: CMakeFiles/tracking.dir/src/FeatureDetector.cpp.o.requires

.PHONY : CMakeFiles/tracking.dir/requires

CMakeFiles/tracking.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tracking.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tracking.dir/clean

CMakeFiles/tracking.dir/depend:
	cd /home/dexheimere/thesis_ws/build/tracking && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dexheimere/thesis_ws/src/tracking /home/dexheimere/thesis_ws/src/tracking /home/dexheimere/thesis_ws/build/tracking /home/dexheimere/thesis_ws/build/tracking /home/dexheimere/thesis_ws/build/tracking/CMakeFiles/tracking.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/tracking.dir/depend

