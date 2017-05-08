cmake_minimum_required (VERSION 2.6)
project (api)
set (CMAKE_CXX_STANDARD 98)

include(FindXercesC)
find_package(XercesC REQUIRED)
find_package(Java REQUIRED)
find_package(JNI REQUIRED)

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})
include_directories (${PROJECT_SOURCE_DIR})
include_directories (${XercesC_INCLUDE_DIR})
include_directories (${JNI_INCLUDE_DIRS})

add_subdirectory(nnl)

add_library(ocr SHARED ocr/api.cpp)
target_link_libraries(ocr nnl ${XercesC_LIBRARIES})

add_executable(api_test ocr/test.cpp)
target_link_libraries(api_test ocr nnl ${XercesC_LIBRARIES})


SET(CMAKE_SWIG_FLAGS "-c++")
set_property(SOURCE ports/python/api.i PROPERTY CPLUSPLUS ON)

# libocr.so Containing just the API.

SWIG_ADD_MODULE(pyocr python ports/python/api.i ocr/api.cpp)
SWIG_LINK_LIBRARIES(pyocr ocr nnl ${XercesC_LIBRARIES} ${PYTHON_LIBRARIES})

# Java Stuff
#add_library(bridge ${PROJECT_SOURCE_DIR}/Bridge.cpp)
#target_link_libraries (bridge nnl xerces-c)
