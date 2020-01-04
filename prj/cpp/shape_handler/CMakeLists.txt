cmake_minimum_required(VERSION 2.8.11.2)
project(shape_handler)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SHP_LIB ${CMAKE_CURRENT_SOURCE_DIR}/shp_lib)
include_directories(${SHP_LIB}/src)
file(GLOB LIB_SRC_FILES ${SHP_LIB}/src/*.c)

set(TargetName ${PROJECT_NAME})
add_executable(${TargetName}
  ${LIB_SRC_FILES}
  main.cc
  )
target_link_libraries(${TargetName} ${OpenCV_LIBS})