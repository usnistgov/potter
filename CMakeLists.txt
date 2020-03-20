cmake_minimum_required(VERSION 3.10)
project(potter)

set(CMAKE_CXX_STANDARD 17)

if (NOT MSVC)
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2")
else()
    # With MSVC, can't use Cuba, and use cubature instead
    add_definitions("-DNO_CUBA") 
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/cubature-master")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/ThreadPool2")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/mcx/pymcx/include")


macro(add_cubature target)
    target_sources(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/cubature-master/pcubature.c")
    target_sources(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/cubature-master/hcubature.c")
endmacro()

add_executable(potter "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cxx")
add_cubature(potter)

if (MSVC)
    target_sources(potter PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
else()
    target_link_libraries(potter PRIVATE "cuba")
endif()

#add_executable(cuba-test "${CMAKE_CURRENT_SOURCE_DIR}/src/cuba_test.cxx")
#target_link_libraries(cuba-test PRIVATE "cuba")

add_executable(catchtest "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/catch_runner.cxx")
add_cubature(catchtest)

macro(add_nix_libraries target)
    # See https://stackoverflow.com/a/29871891
    # Add DL and pthreads
    FIND_PACKAGE ( Threads REQUIRED )
    find_package(Threads REQUIRED)
    if(THREADS_HAVE_PTHREAD_ARG)
      target_compile_options(${target} PRIVATE "-pthread")
    endif()
    if(CMAKE_THREAD_LIBS_INIT)
      target_link_libraries(${target} PRIVATE "${CMAKE_THREAD_LIBS_INIT}")
    endif()
endmacro()

if (NOT MSVC)
    add_nix_libraries(potter)
endif()
