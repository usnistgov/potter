cmake_minimum_required(VERSION 3.10)
project(potter)

set(CMAKE_CXX_STANDARD 17)

if (NOT MSVC)
    set(ZIPFN "${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2.2.tar.gz")
    set(OUTFN "${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2.2/libcuba.a")
    if(NOT EXISTS ${OUTFN})
        execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${ZIPFN}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/externals)
        message(STATUS "Untarred Cuba 4.2")
        execute_process(COMMAND "./configure"
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2.2)
        execute_process(COMMAND make
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2.2)
        message(STATUS "Built Cuba 4.2")
    endif()
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2.2")
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2.2")
    add_definitions(-DENABLE_CUBA)
else()
    # With MSVC, can't use Cuba, and use cubature instead
    add_definitions("-DNO_CUBA") 
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/cubature-master")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/ThreadPool2")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/mcx/pymcx/include")

add_library(cubature STATIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/externals/cubature-master/pcubature.c" 
    "${CMAKE_CURRENT_SOURCE_DIR}/externals/cubature-master/hcubature.c")

add_executable(potter "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cxx")
target_link_libraries(potter PRIVATE cubature)
if (MSVC)
    target_sources(potter PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Eigen/debug/msvc/eigen.natvis")
else()
    target_link_libraries(potter PRIVATE "cuba")
endif()

if (NOT MSVC)
    add_executable(cuba-test "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/cuba_test.cxx")
    target_link_libraries(cuba-test PRIVATE "cuba")
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2")
    target_include_directories(cuba-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/externals/Cuba-4.2")
endif()

add_executable(catchtest "${CMAKE_CURRENT_SOURCE_DIR}/src/tests/catch_runner.cxx")
target_link_libraries(catchtest PRIVATE cubature)
if (NOT MSVC)
    target_link_libraries(catchtest PRIVATE "cuba")
endif()

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
