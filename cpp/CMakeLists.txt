########################################################################
IF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    MESSAGE(FATAL_ERROR "Prevented in-tree built. This is bad practice.")
ENDIF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})


########################################################################
# Project setup
########################################################################
CMAKE_MINIMUM_REQUIRED(VERSION 3.5.0)
PROJECT(soft-isp CXX)
ENABLE_TESTING()

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if (POLICY CMP0071)
  cmake_policy(SET CMP0071 NEW)
endif()

 
set(CMAKE_MODULE_PATH APPEND "${CMAKE_SOURCE_DIR}/cmake/Modules/")




if (UNIX)
	message(STATUS "current platform: Linux ")
elseif (WIN32)
	message(STATUS "current platform: Windows")
endif()



message(STATUS "cpu type: ${CMAKE_SYSTEM_PROCESSOR} ")

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm") 
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mthumb -march=armv7-a -mfpu=neon -marm -funsafe-math-optimizations")
  set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -mthumb -march=armv7-a -mfpu=neon")
endif()
set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})

message("-- CFLAGS: ${CMAKE_C_FLAGS}")
message("-- CXXFLAGS: ${CMAKE_CXX_FLAGS}")

#set(CMAKE_C_FLAGS_Release "${CMAKE_C_FLAGS_Release} -DNDEBUG")
#set(CMAKE_CPP_FLAGS_Release "${CMAKE_CPP_FLAGS_Release} -DNDEBUG")

message(STATUS "GLOBAL_PLATFORM: ${GLOBAL_PLATFORM}")
message(STATUS "GLOBAL_BOARD_NAME: ${GLOBAL_BOARD_NAME}")
message(STATUS "GLOBAL_VERSION: ${GLOBAL_VERSION}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set(OpenCV_DIR /home/wy/src/opencv/opencv-4.6.0/out/lib/cmake/opencv4)


find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
	message("fail to find OpenCV")
else()
	include_directories(${OpenCV_INCLUDE_DIRS})
	set(LIBS ${LIBS} ${OpenCV_LIBS})
	message(STATUS "OpenCV library status:")
	message(STATUS "    version: ${OpenCV_VERSION}")
	message(STATUS "    include: ${OpenCV_INCLUDE_DIRS}")
	message(STATUS "    library: ${OpenCV_INSTALL_PATH}")
	message(STATUS "    libs: ${OpenCV_LIBS}")
endif()


# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../3rdparty/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/aarch64/librknnrt.so)
set(LIBS ${LIBS} ${RKNN_RT_LIB})
include_directories(${RKNN_API_PATH}/include)

# jpeg turbo
set(JPEG_PATH ${CMAKE_SOURCE_DIR}/../../3rdparty/jpeg_turbo)
set(JPEG_LIB ${JPEG_PATH}/Linux/aarch64/libturbojpeg.a)
set(JPEG_INCLUDES ${JPEG_PATH}/include)

set(LIBS ${LIBS} ${JPEG_LIB})
include_directories(${JPEG_INCLUDES})


# rga
set(RGA_PATH ${CMAKE_SOURCE_DIR}/../../3rdparty/librga)
set(RGA_LIB ${RGA_PATH}/Linux/aarch64/librga.a)
set(RGA_INCLUDES ${RGA_PATH}/include)

set(LIBS ${LIBS} ${RGA_LIB})
include_directories(${RGA_INCLUDES})

# # utils
# set(RKNN_UTILS_PATH ${CMAKE_SOURCE_DIR}/../../3rdparty/utils)
# set(RKNN_FILEUTILS_LIB ${RKNN_UTILS_PATH}/lib/libfileutils.a)
# set(RKNN_IMAGEUTILS_LIB ${RKNN_UTILS_PATH}/lib/libimageutils.a)
# set(RKNN_IMAGEDRAWING_LIB ${RKNN_UTILS_PATH}/lib/libimagedrawing.a)

# set(LIBS ${LIBS} ${RKNN_FILEUTILS_LIB})
# set(LIBS ${LIBS} ${RKNN_IMAGEUTILS_LIB})
# set(LIBS ${LIBS} ${RKNN_IMAGEDRAWING_LIB}) 

# link_directories(${RKNN_UTILS_PATH})

#set(LIBS ${LIBS} fileutils imageutils imagedrawing)
# include_directories(${CMAKE_SOURCE_DIR}/../../3rdparty/utils/include)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/libs)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/hls) 
include_directories(${PROJECT_BINARY_DIR}/include)


link_directories(${PROJECT_BINARY_DIR}/libs)


 
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/board)



























message(STATUS "Using install prefix: ${CMAKE_INSTALL_PREFIX}")
