cmake_minimum_required(VERSION 3.10)

project(sesr_rknn_demo)

set(CMAKE_SYSTEM_NAME "Linux")
set(TARGET_SOC "rk3588")

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../utils/ utils.out)

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

message(STATUS "RKNNRT library status:")
message(STATUS "    include: ${LIBRKNNRT_INCLUDES}")
message(STATUS "    libs: ${LIBRKNNRT}")

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)


add_executable(${PROJECT_NAME}
    sesr_rknn_demo.cpp
)

target_link_libraries(${PROJECT_NAME}
    fileutils
    imageutils
    imagedrawing
    ${LIBRKNNRT}
    ${LIBS}
    dl
)



if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)


