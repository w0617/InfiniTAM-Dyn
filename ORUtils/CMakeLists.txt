#############################
# Specify the project files #
#############################

SET(ORUTILS_HEADERS
Vector.h
Matrix.h
Cholesky.h
MathUtils.h
Image.h
CUDADefines.h
LexicalCast.h
MemoryBlock.h
MemoryBlockPersister.h
PlatformIndependence.h
)

SET(ORUTILS_SOURCES
CUDADefines.cpp
)


#############################
# Specify the source groups #
#############################

SOURCE_GROUP("" FILES ${ORUTILS_HEADERS} ${ORUTILS_SOURCES})
add_library(ORUtils ${ORUTILS_HEADERS} ${ORUTILS_SOURCES})


##############################################################
# Specify the include directories, target and link libraries #
##############################################################

IF(WITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
ELSE()
  add_definitions(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

