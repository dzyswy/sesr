
file(GLOB_RECURSE TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/sample_ini/include)
 
if (UNIX)
	set(LIBS ${LIBS} pthread)
endif ()

message(STATUS "LIBS: ${LIBS}")

foreach(SOURCE_FILE ${TEST_SRCS})
  # get file NAME
  get_filename_component(NAME ${SOURCE_FILE} NAME_WE)
    
  add_executable(${NAME} ${SOURCE_FILE})
  target_link_libraries(${NAME} ${LIBS})


  # install
  install(TARGETS ${NAME} DESTINATION bin)


endforeach()


