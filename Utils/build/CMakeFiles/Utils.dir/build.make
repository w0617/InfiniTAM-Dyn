# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/build

# Include any dependencies generated for this target.
include CMakeFiles/Utils.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Utils.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Utils.dir/flags.make

CMakeFiles/Utils.dir/FileUtils.o: CMakeFiles/Utils.dir/flags.make
CMakeFiles/Utils.dir/FileUtils.o: ../FileUtils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Utils.dir/FileUtils.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Utils.dir/FileUtils.o -c /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/FileUtils.cpp

CMakeFiles/Utils.dir/FileUtils.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Utils.dir/FileUtils.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/FileUtils.cpp > CMakeFiles/Utils.dir/FileUtils.i

CMakeFiles/Utils.dir/FileUtils.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Utils.dir/FileUtils.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/FileUtils.cpp -o CMakeFiles/Utils.dir/FileUtils.s

CMakeFiles/Utils.dir/FileUtils.o.requires:

.PHONY : CMakeFiles/Utils.dir/FileUtils.o.requires

CMakeFiles/Utils.dir/FileUtils.o.provides: CMakeFiles/Utils.dir/FileUtils.o.requires
	$(MAKE) -f CMakeFiles/Utils.dir/build.make CMakeFiles/Utils.dir/FileUtils.o.provides.build
.PHONY : CMakeFiles/Utils.dir/FileUtils.o.provides

CMakeFiles/Utils.dir/FileUtils.o.provides.build: CMakeFiles/Utils.dir/FileUtils.o


CMakeFiles/Utils.dir/ImageIOpfm.o: CMakeFiles/Utils.dir/flags.make
CMakeFiles/Utils.dir/ImageIOpfm.o: ../ImageIOpfm.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Utils.dir/ImageIOpfm.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Utils.dir/ImageIOpfm.o -c /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/ImageIOpfm.cpp

CMakeFiles/Utils.dir/ImageIOpfm.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Utils.dir/ImageIOpfm.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/ImageIOpfm.cpp > CMakeFiles/Utils.dir/ImageIOpfm.i

CMakeFiles/Utils.dir/ImageIOpfm.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Utils.dir/ImageIOpfm.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/ImageIOpfm.cpp -o CMakeFiles/Utils.dir/ImageIOpfm.s

CMakeFiles/Utils.dir/ImageIOpfm.o.requires:

.PHONY : CMakeFiles/Utils.dir/ImageIOpfm.o.requires

CMakeFiles/Utils.dir/ImageIOpfm.o.provides: CMakeFiles/Utils.dir/ImageIOpfm.o.requires
	$(MAKE) -f CMakeFiles/Utils.dir/build.make CMakeFiles/Utils.dir/ImageIOpfm.o.provides.build
.PHONY : CMakeFiles/Utils.dir/ImageIOpfm.o.provides

CMakeFiles/Utils.dir/ImageIOpfm.o.provides.build: CMakeFiles/Utils.dir/ImageIOpfm.o


# Object files for target Utils
Utils_OBJECTS = \
"CMakeFiles/Utils.dir/FileUtils.o" \
"CMakeFiles/Utils.dir/ImageIOpfm.o"

# External object files for target Utils
Utils_EXTERNAL_OBJECTS =

libUtils.a: CMakeFiles/Utils.dir/FileUtils.o
libUtils.a: CMakeFiles/Utils.dir/ImageIOpfm.o
libUtils.a: CMakeFiles/Utils.dir/build.make
libUtils.a: CMakeFiles/Utils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libUtils.a"
	$(CMAKE_COMMAND) -P CMakeFiles/Utils.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Utils.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Utils.dir/build: libUtils.a

.PHONY : CMakeFiles/Utils.dir/build

CMakeFiles/Utils.dir/requires: CMakeFiles/Utils.dir/FileUtils.o.requires
CMakeFiles/Utils.dir/requires: CMakeFiles/Utils.dir/ImageIOpfm.o.requires

.PHONY : CMakeFiles/Utils.dir/requires

CMakeFiles/Utils.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Utils.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Utils.dir/clean

CMakeFiles/Utils.dir/depend:
	cd /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/build /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/build /home/w/Desktop/Reconstruction/InfiniTAM-Dyn/Utils/build/CMakeFiles/Utils.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Utils.dir/depend

