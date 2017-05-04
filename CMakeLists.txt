cmake_minimum_required (VERSION 2.6)
project (api)

add_subdirectory (nnl)
include(FindXercesC)
find_package(XercesC REQUIRED)
find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include_directories ("${PROJECT_SOURCE_DIR}/nnl")
#include_directories ("${PROJECT_SOURCE_DIR}/xercesc")
include_directories (${XercesC_INCLUDE_DIR})
include_directories (${JNI_INCLUDE_DIRS})

# add the executable
add_executable(api ${PROJECT_SOURCE_DIR}/api.cpp)
add_library(bridge ${PROJECT_SOURCE_DIR}/Bridge.cpp)
target_link_libraries (api nnl xerces-c)
target_link_libraries (bridge nnl xerces-c)
