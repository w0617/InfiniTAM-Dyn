cmake_minimum_required(VERSION 3.1)
project(InfiniTAM)

# TODO(andrei): This CMakeLists.txt file is not 100% appropriate for allowing InfiniTAM to be used
# as a library. InfiniTAM should provide an `infinitam-config.cmake` file to be used as a lib, as
# described on the CMake wiki:
# https://cmake.org/Wiki/CMake/Tutorials/How_to_create_a_ProjectConfig.cmake_file

# Using PROJECT_SOURCE_DIR ensures we can use InfiniTAM as a library via CMake.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

macro(CANONIFY_BOOL var)
  if (${var})
    set(${var} TRUE)
  else()
    set(${var} FALSE)
  endif()
endmacro()

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(CUDA QUIET)
CANONIFY_BOOL(CUDA_FOUND)
message(STATUS "CUDA found: " ${CUDA_FOUND})

find_package(OpenNI QUIET)
CANONIFY_BOOL(OPENNI_FOUND)
message(STATUS "OpenNI found: " ${OPENNI_FOUND})

find_package(LibRoyale QUIET)
CANONIFY_BOOL(LibRoyale_FOUND)
message(STATUS "LibRoyale found: " ${LibRoyale_FOUND})

find_package(libuvc QUIET)
CANONIFY_BOOL(libuvc_FOUND)
message(STATUS "libuvc found: ${libuvc_FOUND}")

find_package(RealSense QUIET)
CANONIFY_BOOL(REALSENSE_FOUND)
message(STATUS "librealsense found: ${REALSENSE_FOUND}")

find_package(PNG QUIET)
CANONIFY_BOOL(PNG_FOUND)
message(STATUS "libpng found: ${PNG_FOUND}")

OPTION(WITH_CUDA "Build with CUDA support?" ${CUDA_FOUND})
INCLUDE(${PROJECT_SOURCE_DIR}/cmake/UseOpenMP.cmake)

OPTION(WITH_REALSENSE "Build with Intel RealSense support?" ${REALSENSE_FOUND})
OPTION(WITH_LIBROYALE "Build with LibRoyale support?" ${LibRoyale_FOUND})

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  # TODO(andrei): Proper debug build config.
#  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fPIC -g ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wno-deprecated-declarations -Wno-unused-function")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libstdc++")
ENDIF()

# Set some additional common compilation flags, like C++11 feature support.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(ORUtils)
add_subdirectory(Utils)
add_subdirectory(Engine)
add_subdirectory(ITMLib)
add_subdirectory(Driver)

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

# These are the main InfiniTAM demo executables.

add_executable(InfiniTAM_cli InfiniTAM_cli.cpp ${EXTRA_EXECUTABLE_FLAGS})
target_link_libraries(InfiniTAM_cli Engine)
target_link_libraries(InfiniTAM_cli Utils)
target_link_libraries(InfiniTAM_cli ORUtils)
target_link_libraries(InfiniTAM_cli Driver)


add_executable(InfiniTAM InfiniTAM.cpp ${EXTRA_EXECUTABLE_FLAGS})
target_link_libraries(InfiniTAM Engine)
target_link_libraries(InfiniTAM Utils)
target_link_libraries(InfiniTAM ORUtils)
target_link_libraries(InfiniTAM Driver)
