cmake_minimum_required (VERSION 2.6)
project(ghosts)
file(GLOB SOURCES *.cpp *.c)
add_library(ghosts SHARED ${SOURCES})
