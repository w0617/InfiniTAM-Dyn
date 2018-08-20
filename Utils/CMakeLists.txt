IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

IF(PNG_FOUND)
  include_directories(${PNG_INCLUDE_DIRS})
  add_definitions(${PNG_DEFINITIONS})
  add_definitions(-DUSE_LIBPNG)
ENDIF()

find_package(OpenCV 3.4.0 REQUIRED)
if(OpenCV_3.4.0_FOUND)
    message("OpenCV 3.4.0 Not Found ")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif ()

set(Utils_SOURCES
FileUtils.cpp
ImageIOpfm.cpp
)

set(Utils_HEADERS
FileUtils.h
NVTimer.h
ImageIOpfm.h
)

add_library(Utils ${Utils_HEADERS} ${Utils_SOURCES})

IF(PNG_FOUND)
  target_link_libraries(Utils ${PNG_LIBRARIES})
ENDIF()

#add_executable(ReadPFM ImageIOpfm.cpp ${EXTRA_EXECUTABLE_FLAGS})
#target_link_libraries(ReadPFM ${OpenCV_LIBS})




