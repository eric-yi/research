# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yixiaobin/local/workspace/research/prj/cpp/games101

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build

# Include any dependencies generated for this target.
include assignment2/src/CMakeFiles/Rasterizer-2.dir/depend.make

# Include the progress variables for this target.
include assignment2/src/CMakeFiles/Rasterizer-2.dir/progress.make

# Include the compile flags for this target's objects.
include assignment2/src/CMakeFiles/Rasterizer-2.dir/flags.make

assignment2/src/CMakeFiles/Rasterizer-2.dir/main.cpp.o: assignment2/src/CMakeFiles/Rasterizer-2.dir/flags.make
assignment2/src/CMakeFiles/Rasterizer-2.dir/main.cpp.o: ../assignment2/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object assignment2/src/CMakeFiles/Rasterizer-2.dir/main.cpp.o"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Rasterizer-2.dir/main.cpp.o -c /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src/main.cpp

assignment2/src/CMakeFiles/Rasterizer-2.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Rasterizer-2.dir/main.cpp.i"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src/main.cpp > CMakeFiles/Rasterizer-2.dir/main.cpp.i

assignment2/src/CMakeFiles/Rasterizer-2.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Rasterizer-2.dir/main.cpp.s"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src/main.cpp -o CMakeFiles/Rasterizer-2.dir/main.cpp.s

assignment2/src/CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.o: assignment2/src/CMakeFiles/Rasterizer-2.dir/flags.make
assignment2/src/CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.o: ../assignment2/src/rasterizer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object assignment2/src/CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.o"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.o -c /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src/rasterizer.cpp

assignment2/src/CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.i"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src/rasterizer.cpp > CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.i

assignment2/src/CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.s"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src/rasterizer.cpp -o CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.s

assignment2/src/CMakeFiles/Rasterizer-2.dir/Triangle.cpp.o: assignment2/src/CMakeFiles/Rasterizer-2.dir/flags.make
assignment2/src/CMakeFiles/Rasterizer-2.dir/Triangle.cpp.o: ../assignment2/src/Triangle.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object assignment2/src/CMakeFiles/Rasterizer-2.dir/Triangle.cpp.o"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Rasterizer-2.dir/Triangle.cpp.o -c /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src/Triangle.cpp

assignment2/src/CMakeFiles/Rasterizer-2.dir/Triangle.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Rasterizer-2.dir/Triangle.cpp.i"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src/Triangle.cpp > CMakeFiles/Rasterizer-2.dir/Triangle.cpp.i

assignment2/src/CMakeFiles/Rasterizer-2.dir/Triangle.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Rasterizer-2.dir/Triangle.cpp.s"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src/Triangle.cpp -o CMakeFiles/Rasterizer-2.dir/Triangle.cpp.s

# Object files for target Rasterizer-2
Rasterizer__2_OBJECTS = \
"CMakeFiles/Rasterizer-2.dir/main.cpp.o" \
"CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.o" \
"CMakeFiles/Rasterizer-2.dir/Triangle.cpp.o"

# External object files for target Rasterizer-2
Rasterizer__2_EXTERNAL_OBJECTS =

../bin/Rasterizer-2: assignment2/src/CMakeFiles/Rasterizer-2.dir/main.cpp.o
../bin/Rasterizer-2: assignment2/src/CMakeFiles/Rasterizer-2.dir/rasterizer.cpp.o
../bin/Rasterizer-2: assignment2/src/CMakeFiles/Rasterizer-2.dir/Triangle.cpp.o
../bin/Rasterizer-2: assignment2/src/CMakeFiles/Rasterizer-2.dir/build.make
../bin/Rasterizer-2: /usr/local/lib/libopencv_gapi.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_stitching.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_aruco.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_bgsegm.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_bioinspired.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_ccalib.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_dnn_objdetect.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_dnn_superres.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_dpm.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_face.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_freetype.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_fuzzy.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_hfs.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_img_hash.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_line_descriptor.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_quality.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_reg.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_rgbd.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_saliency.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_sfm.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_stereo.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_structured_light.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_superres.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_surface_matching.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_tracking.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_videostab.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_xfeatures2d.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_xobjdetect.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_xphoto.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_highgui.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_shape.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_datasets.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_plot.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_text.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_dnn.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_ml.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_phase_unwrapping.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_optflow.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_ximgproc.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_video.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_videoio.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_imgcodecs.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_objdetect.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_calib3d.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_features2d.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_flann.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_photo.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_imgproc.4.2.0.dylib
../bin/Rasterizer-2: /usr/local/lib/libopencv_core.4.2.0.dylib
../bin/Rasterizer-2: assignment2/src/CMakeFiles/Rasterizer-2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../../../bin/Rasterizer-2"
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Rasterizer-2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
assignment2/src/CMakeFiles/Rasterizer-2.dir/build: ../bin/Rasterizer-2

.PHONY : assignment2/src/CMakeFiles/Rasterizer-2.dir/build

assignment2/src/CMakeFiles/Rasterizer-2.dir/clean:
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src && $(CMAKE_COMMAND) -P CMakeFiles/Rasterizer-2.dir/cmake_clean.cmake
.PHONY : assignment2/src/CMakeFiles/Rasterizer-2.dir/clean

assignment2/src/CMakeFiles/Rasterizer-2.dir/depend:
	cd /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/yixiaobin/local/workspace/research/prj/cpp/games101 /Users/yixiaobin/local/workspace/research/prj/cpp/games101/assignment2/src /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src /Users/yixiaobin/local/workspace/research/prj/cpp/games101/build/assignment2/src/CMakeFiles/Rasterizer-2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : assignment2/src/CMakeFiles/Rasterizer-2.dir/depend

