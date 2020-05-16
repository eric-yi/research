cmake_minimum_required(VERSION 3.15)
project(OpenGLResearch)
set(CMAKE_CONFIGURATION_TYPES "Debug" CACHE STRING "" FORCE)
add_compile_options(-Wno-deprecated-declarations)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

find_package(glfw3 3.3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})
message("==== GLFW_INCLUDE_DIRS: ${GLFW_INCLUDE_DIRS}")

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})

find_path(GLEW_INCLUDE_DIR GL/glew.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)

set(TARGET_NAME opengl-research)
add_executable(${TARGET_NAME} ${SOURCE_FILES})
target_link_libraries(${TARGET_NAME}
${GLFW_LIBRARIES} 
${OPENGL_LIBRARY}
${GLUT_LIBRARY}
${GLEW_LIBRARY}
)
