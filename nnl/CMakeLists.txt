cmake_minimum_required (VERSION 2.6)
project (nnl)

# Find source files
file(GLOB SOURCES *.cpp)
find_package(XercesC REQUIRED)

# add the executable
add_library(nnl SHARED ${SOURCES})
target_link_libraries (nnl ${XercesC})
